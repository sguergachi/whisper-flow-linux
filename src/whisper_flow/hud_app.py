"""The recording HUD: a glass pill on the compositor's overlay layer.

Run as its own process (``python -m whisper_flow.hud_app``) so a stall in GTK
can never reach the audio path.

Three Wayland pieces do the work a normal toplevel cannot:
  * layer-shell puts it on the overlay layer, anchored to the bottom of a
    chosen output - a plain xdg_toplevel cannot position itself at all, and
    would be given a server-side titlebar by the compositor.
  * ext-background-effect-v1 asks for the backdrop blur.
  * the audio level file drives the waveform.
"""

import ctypes
import math
import os
import signal
import struct
import sys
import threading
import time
from collections import deque

_T0 = time.monotonic()
_TIMING = bool(os.environ.get("WHISPER_FLOW_HUD_TIMING"))

IS_WINDOWS = sys.platform == "win32"


def _mark(label):
    """Log milliseconds since process start; this window is on the hot path."""
    if _TIMING:
        print(f"[HUD] +{(time.monotonic() - _T0) * 1000:6.1f}ms {label}", flush=True)

import cairo
import gi

gi.require_version("Gtk", "4.0")
# Named too, not left to Gtk to pull in: without it every overlay start
# printed a PyGIWarning about Gdk's version being unspecified, on the one
# stream anyone reads when the overlay has not appeared.
gi.require_version("Gdk", "4.0")
if sys.platform != "win32":
    # Before Gdk, so it can intercept surface creation: after it the window
    # silently becomes an ordinary toplevel with decorations.
    gi.require_version("Gtk4LayerShell", "1.0")

from gi.repository import Gdk, GLib, Gtk  # noqa: E402

if IS_WINDOWS:
    # No layer-shell on Windows: the pill is an ordinary undecorated window,
    # kept on top and given its acrylic and rounded corners by DWM instead.
    LayerShell = None
else:
    from gi.repository import Gtk4LayerShell as LayerShell  # noqa: E402

# Loaded by explicit path rather than imported. Going through the package would
# pull in the daemon, and with it pystray's GTK 3, which cannot coexist with
# GTK 4 in one process; putting this directory on sys.path instead would shadow
# the standard library's logging module with the package's own.
def _load_sibling(name):
    import importlib.util
    # A frozen build unpacks bundled files to _MEIPASS, not beside __file__.
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_hud_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if IS_WINDOWS:
    _win_blur = _load_sibling("blur_win")
else:
    enable_blur = _load_sibling("wayland_blur").enable_blur
# The processing sweep's arithmetic. Out here so it can be tested without a
# desktop; see hud_anim.
_anim = _load_sibling("hud_anim")
_mark("imports done")

# The pill was drawn at 268x54, and every measurement below is in terms of
# that. A different height scales all of them together rather than squashing
# the same contents into a shorter box - bars, dot and margins keep their
# proportions and the silhouette stays the shape it was designed as.
#
# 38 on both platforms now. Windows arrived at it out of necessity: there the
# pill's outline is whatever DWM rounds the window to, because a region will
# not clip the acrylic behind it, and DWM offers one radius - 8 logical
# pixels, take it or leave it. On a 54-tall pill that reads as a
# barely-softened rectangle, and the only way to make it look round was to
# make the pill smaller against that fixed 8.
#
# Linux was never under that constraint and kept the full 54, so the same
# overlay was two different sizes depending on where it ran. It comes down to
# meet Windows. It keeps its squircle either way - the cap shape is its own
# knob and not a function of the height.
BASE_WIDTH, BASE_HEIGHT = 268, 54
DEFAULT_HEIGHT = 38
HEIGHT = int(os.environ.get("WHISPER_FLOW_HUD_HEIGHT", str(DEFAULT_HEIGHT)))
SIZE_SCALE = HEIGHT / BASE_HEIGHT
WIDTH = round(BASE_WIDTH * SIZE_SCALE)
RADIUS = HEIGHT / 2
BOTTOM_MARGIN = int(os.environ.get("WHISPER_FLOW_HUD_BOTTOM_MARGIN", "28"))

DOT_X = 27 * SIZE_SCALE
DOT_R = 4.5 * SIZE_SCALE
WAVE_L = 48 * SIZE_SCALE
WAVE_R = WIDTH - 24 * SIZE_SCALE
BARS = 30                       # a count, not a measurement
BAR_W = 2.6 * SIZE_SCALE
BAR_MAX = 15.0 * SIZE_SCALE
BAR_OUTLINE = 0.9 * SIZE_SCALE   # dark ring hugging each bar
BAR_SHADOW_DX = 0.9 * SIZE_SCALE  # cast shadow offset, down and to the right
BAR_SHADOW_DY = 1.4 * SIZE_SCALE

# The processing state: after the key is let go and while the transcript is
# still being worked out. The pill stays up rather than vanishing into a gap
# where nothing says whether anything is happening.
#
# Blue, not the recording red, because the two states must not be mistaken
# for each other at a glance - the point of keeping the pill on screen is
# lost if it still looks like it is listening.
#
# The marker the daemon drops beside the level file. It is how an overlay
# with no command pipe - every non-Windows one, which is spawned per
# recording - learns the recording has ended and the waiting has begun.
PROCESSING_SUFFIX = ".processing"
PROCESS_RGB = (0.45, 0.72, 1.0)
PROCESS_RING_R = 1.55       # spinner radius, in dot radii
PROCESS_TINT = 0.75         # how far the bars move towards PROCESS_RGB
# The timings live in hud_anim, which is where they can be tested.

STROKE_W = 1.0              # outline width in surface-local units
STROKE_RGB = (0.42, 0.43, 0.47)
# The blur region is built from rectangles and cannot be antialiased, so its
# rounded ends are a staircase. Keeping it inside the outline hides that edge
# under the stroke instead of leaving a ragged rim of blur outside the pill.
# The margin also has to absorb fractional scaling: at 1.25x this surface is
# 67.5 physical pixels tall and the compositor rounds the region outward.
# Two pixels covers that; it was briefly wider while chasing the corner
# artifact, which turned out to be GTK's window shadow instead.
BLUR_INSET = 2
# Cap shape. n = 2 is a circle; just above 2 keeps the capsule silhouette while
# giving zero curvature where the cap meets the straight edge. Going much
# higher squares the ends off into a rounded rectangle.
SQUIRCLE_N = 2.3
SQUIRCLE_STEPS = 96
CAP_EXT = 1.12   # cap reaches this many half-heights along the edge
# Tint laid over the compositor's blur. Higher alpha reads darker but hides
# more of the frost, so this is the one knob that trades the two off.
MATERIAL_RGB = (0.03, 0.03, 0.04)
MATERIAL_ALPHA = 0.80
if IS_WINDOWS:
    # Opaque when there is nothing behind the pill to show. Translucency
    # over a raw desktop is not glass, it just looks like a bug. This is
    # the fallback for when acrylic could not be applied - with it, the
    # tint below is used instead.
    MATERIAL_ALPHA = 1.0
# Over acrylic instead. Barely a tint: the material is already blurred and
# tinted, and anything heavier turns it back into the flat slab it replaced.
ACRYLIC_TINT_ALPHA = 0.12
# What DWM rounds a window to, in logical pixels. With acrylic behind the
# pill this is its silhouette - a region cannot clip a material, so DWM's
# rounding is the only shape on offer.
DWM_CORNER_RADIUS = 8
# Rim masking the blur region's edge. Only just wider than the inset: it is
# darker than the glass, so every extra pixel reads as a heavy border.
EDGE_COVER = BLUR_INSET + 0.5
EDGE_COVER_ALPHA = 0.90
# Inner shadow, in the CSS sense: it holds full strength for SPREAD, then
# fades out over BLUR. Concentric strokes clipped to the pill stand in for a
# gradient - each covers half its line width inward, so keeping every radius
# at or beyond SPREAD leaves that band evenly covered and only the outer ones
# thin out, which is the falloff.
INNER_SHADOW_ALPHA = 0.20
INNER_SHADOW_SPREAD = 1
INNER_SHADOW_BLUR = 1
INNER_SHADOW_STEPS = 6   # a 2px band needs few bands to look smooth
INNER_SHADOW_RGB = (0.13, 0.13, 0.15)

FADE_MS = 200.0      # fade in and out duration
FADE_SCALE = 0.1     # shrink by this much at zero opacity, growing to full
LEVEL_EASE = 0.30    # how fast a bar chases its target height
PEAK_DECAY = 0.95    # adaptive gain, so quiet speech still reads
PEAK_FLOOR = 150.0   # below this the gain stops opening up, or idle noise dances
# The levels are the RMS of 16-bit audio, so nothing above this is a level at
# all - it is a misread of the file. Skipping those rather than letting them
# set the gain matters because the gain is sticky: one impossible value used
# to pin the peak in the millions and flatten the waveform for several seconds
# while it decayed back. Whatever produced the bad byte, the bars should not
# stop moving over it.
LEVEL_CEILING = 32767.0
LEVEL_GAMMA = 0.65   # loudness is perceptual; linear RMS leaves speech near flat
# Longer than the daemon's max recording; purely a stuck-overlay backstop.
MAX_LIFETIME_S = 600
# How often the level thread looks for new samples. Its own thread, so this
# costs nothing the render loop can feel.
LEVEL_POLL_S = 0.02

DRAG_SLOP = 4  # movement under this is a tap, not a drag
POSITION_FILE = os.path.join(
    os.path.expanduser("~"), ".config", "whisper-flow", "hud-position.json",
)


def _load_positions() -> dict:
    """Saved pill positions, keyed by monitor connector."""
    try:
        import json
        with open(POSITION_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_position(connector: str, x: int, y: int):
    """Remember where the user dragged the pill on this output."""
    try:
        import json
        positions = _load_positions()
        positions[connector or "default"] = [int(x), int(y)]
        os.makedirs(os.path.dirname(POSITION_FILE), exist_ok=True)
        tmp = POSITION_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(positions, f)
        os.replace(tmp, POSITION_FILE)
    except Exception as e:
        print(f"[HUD] could not save position: {e}", flush=True)

# GTK paints more than the background: the "decoration" node carries a drop
# shadow and a border, which land outside the pill at the surface bounds and
# read as a faint straight-edged smudge past the rounded ends. Everything the
# toolkit might draw has to be cleared, not just the background colour.
CSS = b"""
window, window.background, window.csd, window.solid-csd,
decoration, decoration:backdrop {
    background-color: transparent;
    background-image: none;
    border: none;
    border-radius: 0;
    box-shadow: none;
    outline: none;
    margin: 0;
    padding: 0;
}
drawingarea { background-color: transparent; background-image: none; }
"""


def _load_css(provider, css: bytes):
    """Feed CSS to a provider, whichever call this GTK offers.

    load_from_data() has always taken the length as a second argument. Older
    introspection data annotated it as the array's own length, so PyGObject
    filled it in and the one-argument call worked; newer data does not, and
    the same call raises TypeError instead. That is what left every window on
    Windows unbuilt: the overlay died inside GTK before the first frame, and
    the settings window threw inside an activate handler, where GObject prints
    the traceback and carries on with no window to show.

    load_from_string() has no length to get wrong and has been there since GTK
    4.12, so it is the call everywhere it exists. The fallback is for GTK
    before that, where the one-argument form is the correct one.
    """
    if hasattr(provider, "load_from_string"):
        provider.load_from_string(css.decode())
    else:
        provider.load_from_data(css)


def _unix_signal_add(sig, callback):
    """Route a POSIX signal through the main loop, across GLib versions."""
    try:
        gi.require_version("GLibUnix", "2.0")
        from gi.repository import GLibUnix
        return GLibUnix.signal_add(GLib.PRIORITY_HIGH, sig, callback)
    except (ValueError, ImportError, AttributeError):
        return GLib.unix_signal_add(GLib.PRIORITY_HIGH, sig, callback)


def _ease(t: float) -> float:
    """Smoothstep: eases out of rest and into rest, unlike a linear ramp."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _squircle_points(x, y, w, h, n=SQUIRCLE_N, cap=CAP_EXT, steps=SQUIRCLE_STEPS):
    """The capsule outline as a point list. See _squircle for the shape.

    Separate from the drawing so the window region can be cut from the very
    same geometry: a region that only approximated the pill would clip its
    caps or leave a rim of window outside them, and the two would drift
    apart the moment either was tuned.
    """
    b = h / 2.0
    rx = min(b * cap, w / 2.0)
    cy = y + b
    left_j, right_j = x + rx, x + w - rx
    e = 2.0 / n
    pts = []
    for base, lo in ((right_j, -math.pi / 2), (left_j, math.pi / 2)):
        for i in range(steps + 1):
            t = lo + math.pi * i / steps
            ct, st = math.cos(t), math.sin(t)
            pts.append((
                base + rx * math.copysign(abs(ct) ** e, ct),
                cy + b * math.copysign(abs(st) ** e, st),
            ))
    return pts


def _squircle(cr, x, y, w, h, n=SQUIRCLE_N, cap=CAP_EXT, steps=SQUIRCLE_STEPS):
    """Append a capsule whose end caps have continuous curvature.

    A circular cap meets the straight edge with matching tangents but a sudden
    jump in curvature, from zero to 1/r, and that discontinuity is what reads
    as a seam. A superellipse with n > 2 has zero curvature at its axis
    extremes, so the cap leaves the straight edge flat and rounds up smoothly -
    the same idea as Apple's continuous corners.

    The superellipse is applied per cap, not to the whole outline: stretching
    one across the full width would flatten the ends into a rounded rectangle
    instead of a capsule. Each cap reaches `cap` times the half-height along
    the edge, which is what gives the curvature room to ramp.
    """
    pts = _squircle_points(x, y, w, h, n, cap, steps)
    cr.move_to(*pts[0])
    for px, py in pts[1:]:
        cr.line_to(px, py)
    cr.close_path()


def _round_rect(cr, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


def _gobject_pointer(obj) -> int:
    """Address of the underlying GObject, via PyGObject's capsule."""
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ctypes.pythonapi.PyCapsule_GetPointer(obj.__gpointer__, None)


class HudWindow(Gtk.Window):
    """The pill itself."""

    def __init__(self, level_file: str, monitor_hint: str | None, on_quit=None,
                 resident: bool = False):
        super().__init__()
        self.level_file = level_file
        # Resident: stay alive between recordings and wait for orders on
        # stdin, instead of being spawned and killed for each one. Starting
        # a frozen process is the single largest cost between pressing the
        # hotkey and seeing anything.
        self._resident = resident
        self._hwnd = None
        # What Windows composition gave us; decides whether the window still
        # has to be cut to shape. See blur_win.apply_window_style.
        self._style = None
        # The pill's chrome, drawn once and blitted after that.
        self._chrome = None
        self._chrome_size = None
        # Guards targets and peak, which the level thread writes and the
        # render loop reads. Held only to hand over values, never across a
        # read of the file.
        self._levels_lock = threading.Lock()
        self._levels_done = None
        # The blurred picture of whatever is behind the pill, and its size.
        # Windows only: on Wayland the compositor supplies the real thing.
        # Gtk.Window.close() only emits close-request; on a layer-shell surface
        # that does not end the main loop, so the process would linger with the
        # pill still on screen. Always tear down through this instead.
        self._on_quit = on_quit
        self._quitting = False
        self.set_default_size(WIDTH, HEIGHT)
        self.set_resizable(False)

        provider = Gtk.CssProvider()
        _load_css(provider, CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        if LayerShell is not None:
            LayerShell.init_for_window(self)
            LayerShell.set_layer(self, LayerShell.Layer.OVERLAY)
            LayerShell.set_namespace(self, "whisper-flow-hud")
            LayerShell.set_keyboard_mode(self, LayerShell.KeyboardMode.NONE)
            # Negative: never reserve space, so windows do not get resized.
            LayerShell.set_exclusive_zone(self, -1)
        else:
            self.set_decorated(False)
            # Invisible, but EnumWindows finds the window by it.
            self.set_title("whisper-flow-hud")

        monitor = self._pick_monitor(monitor_hint)
        if monitor is not None and LayerShell is not None:
            LayerShell.set_monitor(self, monitor)
        self._monitor = monitor
        self._connector = (
            monitor.get_connector()
            if monitor is not None and hasattr(monitor, "get_connector")
            else "default")

        # A Wayland client cannot place its own window, but a layer surface's
        # margins are under our control: anchoring top-left turns them into
        # absolute coordinates, which is what makes dragging possible.
        saved = _load_positions().get(self._connector)
        self._pos = tuple(saved[:2]) if isinstance(saved, list) and len(saved) >= 2 else None
        self._drag_origin = None
        self._apply_position()
        print(f"[HUD] output={self._connector} position="
              f"{self._pos if self._pos else 'bottom-centre (default)'}", flush=True)

        self.alpha = 0.0
        self._fade_in_t0 = time.monotonic()
        self._fade_out_t0 = None
        self._fade_out_from = 1.0
        self.hover = 0.0
        self.want_hover = False
        self.peak = PEAK_FLOOR
        self.targets = deque([0.0] * BARS, maxlen=BARS)
        self.shown = [0.0] * BARS
        self.level_pos = 0
        self.start = time.monotonic()
        self._blur = None
        # Recording is over, the transcript is not back yet. The pill stays
        # up and says so, instead of disappearing into a silence the user
        # cannot tell from a failure.
        self.processing = False
        self._processing_t0 = 0.0

        area = Gtk.DrawingArea()
        area.set_content_width(WIDTH)
        area.set_content_height(HEIGHT)
        area.set_draw_func(self._draw)
        self.area = area
        self.set_child(area)

        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        drag.connect("drag-end", self._on_drag_end)
        area.add_controller(drag)

        motion = Gtk.EventControllerMotion()
        motion.connect("enter", self._on_enter)
        motion.connect("leave", self._on_leave)
        area.add_controller(motion)

        GLib.timeout_add(16, self._frame)
        # The lifetime cap is all that is left on the main loop; the levels
        # are followed on their own thread so disk never lands in a frame.
        GLib.timeout_add(500, self._watchdog)
        self._levels_done = threading.Event()
        self._levels_thread = threading.Thread(
            target=self._levels_loop, daemon=True,
            name="whisper-flow-hud-levels")
        self._levels_thread.start()
        # The daemon takes this overlay down with SIGTERM. Handle it through
        # the main loop so the pill can fade out instead of blinking away;
        # the manager waits a couple of seconds, which is ample.
        if not IS_WINDOWS:
            for sig in (signal.SIGTERM, signal.SIGINT):
                _unix_signal_add(sig, self._on_signal)
        self.connect("realize", self._on_realize)
        if IS_WINDOWS:
            # Position again once the window is on screen. Placing it at
            # realize is too early: GTK sets the surface's own position as
            # part of the map that follows, which overwrote ours - so the
            # pill appeared at Windows' default cascade spot in the top-left
            # corner rather than bottom-centre, on every recording. It was
            # never invisible, only never where anyone was looking.
            self.connect("map", self._on_map)
        _mark("window constructed")

    def _on_map(self, *_):
        """Re-apply the position after GTK has placed the window itself."""
        GLib.idle_add(self._reposition)

    def _reposition(self):
        self._apply_position()
        # Re-cut here too: at realize the surface has not been sized yet, so
        # the region built there is against a provisional rectangle.
        self._apply_shape_win32()
        # And re-claim the topmost band, which another window may have taken
        # since this overlay was last on screen.
        self._raise_win32()
        return GLib.SOURCE_REMOVE

    def _pick_monitor(self, hint: str | None):
        """Choose the output to show on.

        A Wayland client cannot ask where the pointer is or which window has
        focus, so the daemon passes what it knows: either a connector name, or
        a point (the centre of the window being dictated into). Falling back to
        the first monitor is a guess and will be wrong on multi-head setups,
        which is why the daemon works to supply one of these.
        """
        display = Gdk.Display.get_default()
        monitors = display.get_monitors()
        candidates = [monitors.get_item(i) for i in range(monitors.get_n_items())]
        if not candidates:
            return None

        if hint:
            for mon in candidates:
                if mon.get_connector() == hint:
                    return mon

        point = os.environ.get("WHISPER_FLOW_HUD_POINT", "")
        if point:
            try:
                px, py = (int(v) for v in point.split(",", 1))
                for mon in candidates:
                    g = mon.get_geometry()
                    if g.x <= px < g.x + g.width and g.y <= py < g.y + g.height:
                        return mon
            except ValueError:
                pass

        return candidates[0]

    def _on_realize(self, *_):
        """Attach platform window effects once there is a surface for them."""
        if IS_WINDOWS:
            self._realize_win32()
            return
        if os.environ.get("WHISPER_FLOW_HUD_NO_BLUR"):
            print("[HUD] blur disabled by env", flush=True)
            return
        try:
            display = Gdk.Display.get_default()
            surface = self.get_surface()
            lib = ctypes.CDLL("libgtk-4.so.1")
            lib.gdk_wayland_display_get_wl_display.restype = ctypes.c_void_p
            lib.gdk_wayland_display_get_wl_display.argtypes = [ctypes.c_void_p]
            lib.gdk_wayland_surface_get_wl_surface.restype = ctypes.c_void_p
            lib.gdk_wayland_surface_get_wl_surface.argtypes = [ctypes.c_void_p]

            wl_display = lib.gdk_wayland_display_get_wl_display(
                ctypes.c_void_p(_gobject_pointer(display)))
            wl_surface = lib.gdk_wayland_surface_get_wl_surface(
                ctypes.c_void_p(_gobject_pointer(surface)))
            _mark("realize: blur start")
            self._blur = enable_blur(
                wl_display, wl_surface, WIDTH, HEIGHT, SQUIRCLE_N, BLUR_INSET,
                active=False)
            _mark("realize: blur done")
            print(f"[HUD] blur {'enabled' if self._blur else 'unavailable'}", flush=True)
        except Exception as e:
            print(f"[HUD] blur setup failed: {e}", flush=True)

    # ------------------------------------------------------------- Windows
    def _win32_hwnd(self):
        """The window's HWND, via GDK first and EnumWindows as a fallback."""
        for dll in ("libgtk-4-1.dll", "libgtk-4.so.1"):
            try:
                lib = ctypes.CDLL(dll)
                get = lib.gdk_win32_surface_get_handle
                get.restype = ctypes.c_void_p
                get.argtypes = [ctypes.c_void_p]
                hwnd = get(ctypes.c_void_p(
                    _gobject_pointer(self.get_surface())))
                if hwnd:
                    return hwnd
            except (OSError, AttributeError):
                continue

        # The GDK name varies; find the window by title and owner instead.
        user32 = ctypes.windll.user32
        found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def each(hwnd, _):
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value != os.getpid():
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if buf.value == "whisper-flow-hud":
                    found.append(hwnd)
            return True

        user32.EnumWindows(each, None)
        return found[0] if found else None

    def _raise_win32(self):
        """Put the pill above everything, including the taskbar.

        Re-asserted on every show, not just at realize. WS_EX_TOPMOST is
        supposed to be sticky, but the band is ordered by whoever claimed it
        last: a shell surface or another topmost window that appears after
        us ends up in front, and the overlay spends the recording behind the
        taskbar. Claiming it again each time the pill is shown costs one
        call and settles it.

        argtypes are declared because HWND_TOPMOST is the sentinel -1 and
        this is a 64-bit process. Left untyped, ctypes marshals it as a
        32-bit int and what the callee reads for a pointer-sized handle is
        not -1 at all.
        """
        if not self._hwnd:
            return
        try:
            user32 = ctypes.windll.user32
            user32.SetWindowPos.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint,
            ]
            user32.SetWindowPos.restype = ctypes.c_bool
            HWND_TOPMOST = -1
            SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x1, 0x2, 0x10
            user32.SetWindowPos(
                ctypes.c_void_p(self._hwnd), ctypes.c_void_p(HWND_TOPMOST),
                0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
        except Exception as e:
            print(f"[HUD] could not raise the overlay: {e}", flush=True)

    def _realize_win32(self):
        """Topmost, positioned, acrylic and rounded - DWM does the styling."""
        try:
            self._hwnd = self._win32_hwnd()
            if not self._hwnd:
                print("[HUD] no HWND yet", flush=True)
                return
            self._raise_win32()
            self._apply_position()
            if not os.environ.get("WHISPER_FLOW_HUD_NO_BLUR"):
                self._style = _win_blur.apply_window_style(self._hwnd)
                print(f"[HUD] window style: {self._style or 'not applied'}",
                      flush=True)
                # The style decides the pill's shape and how much tint goes
                # over it, so anything cached before now is of another pill.
                self._chrome = None
            self._apply_shape_win32()
        except Exception as e:
            print(f"[HUD] Windows window setup failed: {e}", flush=True)

    # How far the region is pushed past the drawn outline. The cut has no
    # antialiasing, so landing it exactly on the outline would replace the
    # pill's soft rim with a staircase. One physical pixel out, the ragged
    # edge falls on fully transparent pixels and the drawn rim survives
    # intact - the same reasoning as BLUR_INSET on the Wayland side, in the
    # opposite direction, because there the region sits under the glass and
    # here it decides what exists at all.
    SHAPE_BLEED = 1

    def _apply_shape_win32(self):
        """Cut the window down to the pill, when nothing honours the alpha.

        A last resort, not the normal path. The region's edge has no
        antialiasing, so it replaces the pill's soft rim with a staircase;
        with per-pixel alpha working, the transparent corners cost nothing
        and the rim survives. This is what is left when the alpha channel is
        being thrown away and every transparent pixel comes out opaque.

        Measured from the window rather than computed from WIDTH and HEIGHT:
        GTK owns the surface size and applies the display scale itself, and a
        region built from what we assumed the size was would be wrong on
        every display that is not at 100%.
        """
        # Not with a material: a region does not clip one, so cutting the
        # window would leave the acrylic rectangular and the content capsule
        # shaped - the two disagreeing is worse than either alone.
        if not self._hwnd or self._style in ("per-pixel-alpha", "accent-acrylic"):
            return
        try:
            rect = (ctypes.c_long * 4)()
            if not ctypes.windll.user32.GetWindowRect(
                    ctypes.c_void_p(self._hwnd), ctypes.byref(rect)):
                return
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            if w <= 0 or h <= 0:
                return
            bleed = self.SHAPE_BLEED
            points = _squircle_points(-bleed, -bleed, w + 2 * bleed, h + 2 * bleed)
            applied = _win_blur.set_shape(self._hwnd, points)
            print(f"[HUD] window shape: {w}x{h} "
                  f"{'clipped to the pill' if applied else 'not applied'}",
                  flush=True)
        except Exception as e:
            print(f"[HUD] could not shape the window: {e}", flush=True)

    def _monitor_scale(self) -> float:
        """Logical-to-physical factor for the output the pill is on.

        GDK measures in logical units and SetWindowPos takes physical pixels,
        so every coordinate has to cross that boundary. Nothing converted, so
        on a 2x display the pill went to half its intended offset - the
        upper-left quadrant instead of bottom-centre, which read as the
        overlay simply not appearing.

        get_scale() before get_scale_factor(): the latter is an integer and
        reports 1 at 125% and 150%, where the real factor is fractional and
        the error is smaller but just as wrong.
        """
        if self._monitor is None:
            return 1.0
        for name in ("get_scale", "get_scale_factor"):
            getter = getattr(self._monitor, name, None)
            if getter is None:
                continue
            try:
                scale = float(getter())
            except Exception:
                continue
            if scale > 0:
                return scale
        return 1.0

    def _apply_position_win32(self, x: int, y: int):
        if not self._hwnd:
            return
        SWP_NOACTIVATE = 0x10
        SWP_NOZORDER = 0x4
        ctypes.windll.user32.SetWindowPos(
            self._hwnd, None, int(x), int(y), 0, 0,
            0x1 | SWP_NOZORDER | SWP_NOACTIVATE)          # NOSIZE

    def quit(self):
        """Take the overlay down and end the process.

        Fades out first, then exits outright. Neither Gtk.Window.close() nor
        MainLoop.quit() ends this process once a layer-shell surface is mapped
        - the loop returns but the process lives on - and dropping the Wayland
        connection is what actually removes the surface.
        """
        if self._quitting:
            return
        self._quitting = True
        if self._blur is not None:
            # Drop the blur while the pill is still opaque enough to hide the
            # change; otherwise it lingers as a blurred patch as alpha falls.
            self._blur.set_active(False)
        self._fade_out_from = self.alpha
        self._fade_out_t0 = time.monotonic()
        # Backstop in case the frame timer stops before the fade completes.
        GLib.timeout_add(int(FADE_MS) + 150, self._after_fade_out)

    def _after_fade_out(self):
        if self._resident:
            self._fade_out_t0 = None
            self.alpha = 0.0
            self.hide()
            self._quitting = False
            return False
        os._exit(0)

    # ------------------------------------------------------- resident mode
    def begin_processing(self):
        """Recording is over; keep the pill up while the words are worked out.

        Deliberately not a fresh fade-in: the pill is already on screen and
        stays there, changing what it says rather than blinking to say it.
        """
        if self.processing or self._quitting:
            return False
        self.processing = True
        self._processing_t0 = time.monotonic()
        print(f"[HUD] processing {time.time():.6f}", flush=True)
        return False

    def begin_show(self, level_file: str):
        """Show for a new recording. No process start, no window creation."""
        self.level_file = level_file
        self.processing = False
        self.level_pos = 0
        self.peak = PEAK_FLOOR
        self.targets = deque([0.0] * BARS, maxlen=BARS)
        self.shown = [0.0] * BARS
        self.start = time.monotonic()
        self.alpha = 0.0
        self._fade_in_t0 = time.monotonic()
        self._fade_out_t0 = None
        self._quitting = False
        self._apply_position()
        self.present()
        print(f"[HUD] visible {time.time():.6f}", flush=True)

    def begin_hide(self):
        """Fade out and hide, keeping the process and window for next time."""
        if self._quitting:
            return
        self.quit()

    def _check_lifetime(self):
        """Backstop against an overlay outliving whatever spawned it."""
        if time.monotonic() - self.start > MAX_LIFETIME_S:
            print("[HUD] lifetime cap reached, closing", flush=True)
            self.quit()
            return False
        return True

    def _on_signal(self, *_):
        self.quit()
        return GLib.SOURCE_REMOVE

    def _default_pos(self):
        """Bottom-centre of the chosen output, in monitor-local coordinates."""
        if self._monitor is None:
            return (0, 0)
        g = self._monitor.get_geometry()
        return ((g.width - WIDTH) // 2, g.height - HEIGHT - BOTTOM_MARGIN)

    def _apply_position(self):
        """Anchor bottom-centre by default, or top-left at a dragged position."""
        if LayerShell is None:
            # Win32 coordinates are absolute and physical; layer-shell margins
            # were local, and everything above this line is in GDK's logical
            # units. This is the one place the two meet.
            local = self._pos if self._pos is not None else self._default_pos()
            g = self._monitor.get_geometry() if self._monitor else None
            scale = self._monitor_scale()
            self._apply_position_win32(
                ((g.x if g else 0) + local[0]) * scale,
                ((g.y if g else 0) + local[1]) * scale)
            return
        if self._pos is None:
            LayerShell.set_anchor(self, LayerShell.Edge.LEFT, False)
            LayerShell.set_anchor(self, LayerShell.Edge.TOP, False)
            LayerShell.set_anchor(self, LayerShell.Edge.BOTTOM, True)
            LayerShell.set_margin(self, LayerShell.Edge.BOTTOM, BOTTOM_MARGIN)
            return
        x, y = self._pos
        LayerShell.set_anchor(self, LayerShell.Edge.BOTTOM, False)
        LayerShell.set_anchor(self, LayerShell.Edge.LEFT, True)
        LayerShell.set_anchor(self, LayerShell.Edge.TOP, True)
        LayerShell.set_margin(self, LayerShell.Edge.LEFT, int(x))
        LayerShell.set_margin(self, LayerShell.Edge.TOP, int(y))

    def _clamp(self, x, y):
        if self._monitor is None:
            return (x, y)
        g = self._monitor.get_geometry()
        return (max(0, min(int(x), g.width - WIDTH)),
                max(0, min(int(y), g.height - HEIGHT)))

    def _on_drag_begin(self, gesture, sx, sy):
        self._drag_origin = self._pos if self._pos is not None else self._default_pos()
        self._drag_start_xy = (sx, sy)

    def _on_drag_update(self, gesture, dx, dy):
        if self._drag_origin is None:
            return
        ox, oy = self._drag_origin
        self._pos = self._clamp(ox + dx, oy + dy)
        self._apply_position()

    def _on_drag_end(self, gesture, dx, dy):
        if self._drag_origin is None:
            return
        moved = abs(dx) + abs(dy)
        self._drag_origin = None
        if moved < DRAG_SLOP:
            # A tap. Only the close affordance dismisses; anywhere else would
            # make the pill vanish whenever a drag came up a pixel short.
            sx, sy = self._drag_start_xy
            if sx >= WIDTH - 34 * SIZE_SCALE:
                self.quit()
            return
        _save_position(self._connector, *self._pos)
        print(f"[HUD] position saved for {self._connector}: {self._pos}", flush=True)

    def _on_enter(self, *_):
        self.want_hover = True

    def _on_leave(self, *_):
        self.want_hover = False

    def _frame(self):
        if self._resident and not self.get_visible():
            return True                 # parked between recordings
        now = time.monotonic()
        if self._fade_out_t0 is None:
            self.alpha = _ease((now - self._fade_in_t0) * 1000.0 / FADE_MS)
            # Only once the glass is opaque: the compositor blur cannot be
            # faded, so switching it while the pill is still translucent shows
            # a bare blurred patch with hard edges instead of a fade.
            if self.alpha >= 0.999 and self._blur is not None:
                self._blur.set_active(True)
        else:
            t = (now - self._fade_out_t0) * 1000.0 / FADE_MS
            self.alpha = self._fade_out_from * (1.0 - _ease(t))
            if t >= 1.0:
                self._after_fade_out()
        self.hover += ((1.0 if self.want_hover else 0.0) - self.hover) * 0.25
        if self.processing:
            targets = self._processing_levels(now)
        else:
            with self._levels_lock:
                targets = list(self.targets)
        for i, target in enumerate(targets):
            self.shown[i] += (target - self.shown[i]) * LEVEL_EASE
        self.area.queue_draw()
        return True

    def _processing_levels(self, now):
        """Bar heights for the processing sweep. See hud_anim.sweep."""
        return _anim.sweep(now - self._processing_t0, BARS)

    def _levels_loop(self):
        """Follow the level file on its own thread, off the render path.

        This is disk I/O - an open, a seek and a read - and it used to run on
        the main loop between frames. Every stall on the filesystem landed
        squarely in a frame, on the one thread that has 16ms to do everything
        and is also the thread GTK draws on. Reading here leaves the main
        loop doing nothing but drawing what has already arrived.

        The main loop is never made to wait on this thread either: it takes
        the lock only to copy out the levels, and never holds it across a
        read.
        """
        while not self._levels_done.wait(LEVEL_POLL_S):
            path = self.level_file
            if not path:
                continue
            # The daemon deletes the level file when the recording ends. If
            # it is gone this overlay has been orphaned - it must never sit
            # on screen forever because whoever spawned it failed to.
            if not os.path.exists(path):
                print("[HUD] level file gone, closing", flush=True)
                self.level_file = ""
                GLib.idle_add(self.quit)
                continue
            # An overlay spawned per recording has no pipe to be told
            # anything on - that is Windows-only - so the daemon leaves a
            # marker beside the level file and this notices it. One poll of
            # latency, against a processing step measured in seconds.
            if not self.processing and os.path.exists(path + PROCESSING_SUFFIX):
                GLib.idle_add(self.begin_processing)
                continue
            if self.processing:
                continue        # no more levels are coming; do not read
            try:
                with open(path, "rb") as handle:
                    handle.seek(0, 2)
                    count = handle.tell() // 4
                    if count <= self.level_pos:
                        continue
                    handle.seek(self.level_pos * 4)
                    data = handle.read((count - self.level_pos) * 4)
                    self.level_pos = count
                values = struct.unpack("<%di" % (len(data) // 4), data)
            except Exception:
                continue

            with self._levels_lock:
                for value in values:
                    value = abs(value)
                    if value > LEVEL_CEILING:
                        continue
                    self.peak = max(self.peak * PEAK_DECAY, float(value),
                                    PEAK_FLOOR)
                    self.targets.append(
                        min(1.0, (value / self.peak) ** LEVEL_GAMMA))

    def _watchdog(self):
        """The lifetime cap, which no longer has a level timer to live on."""
        if not self._check_lifetime():
            return False
        return True

    def _outline(self, cr, x, y, w, h):
        """The pill's silhouette, which is not the same shape on Windows.

        With acrylic behind it the shape is not ours to choose: a window
        region does not clip a material, so whatever DWM rounds the window
        to is what the pill is - a rounded rectangle at the system radius.
        Drawing the capsule over that would put our outline inside the
        material's own edge, with frost showing past it at both ends.

        Everywhere else, the squircle: continuous curvature into the caps,
        which is the shape this overlay is meant to have.
        """
        if self._style == "accent-acrylic":
            _round_rect(cr, x, y, w, h, DWM_CORNER_RADIUS)
        else:
            _squircle(cr, x, y, w, h)

    def _paint_chrome(self, cr, w, h, a):
        """Everything about the pill that is not the dot or the waveform.

        Depends only on the alpha, which is why it can be cached: the fill,
        the sheen, the rim, the inner shadow and the outline are the same
        picture on every frame once the fade is over.

        The outline used to be drawn after the bars, to be certain nothing
        painted over it. It is safe here - the waveform stops well inside
        the glass - and it has to be here for the cache to hold all of it.
        """
        inner = STROKE_W / 2
        material = (ACRYLIC_TINT_ALPHA if self._style == "accent-acrylic"
                    else MATERIAL_ALPHA)
        self._outline(cr, inner, inner, w - 2 * inner, h - 2 * inner)
        cr.set_source_rgba(*MATERIAL_RGB, material * a)
        cr.fill_preserve()

        sheen = cairo.LinearGradient(0, 0, 0, h)
        sheen.add_color_stop_rgba(0.0, 1, 1, 1, 0.10 * a)
        sheen.add_color_stop_rgba(0.45, 1, 1, 1, 0.015 * a)
        sheen.add_color_stop_rgba(1.0, 0, 0, 0, 0.14 * a)
        cr.set_source(sheen)
        cr.fill()

        # Everything to the border is clipped to the pill, so strokes centred
        # on the outline only paint inwards.
        cr.save()
        self._outline(cr, 0, 0, w, h)
        cr.clip()

        # Opaque rim covering the blur region's staircase. The region is
        # built from rectangles and cannot be antialiased, so its curved ends
        # are ragged; this hides that boundary rather than leaving it on show.
        self._outline(cr, 0, 0, w, h)
        cr.set_line_width(2 * EDGE_COVER)
        cr.set_source_rgba(*MATERIAL_RGB, EDGE_COVER_ALPHA * a)
        cr.stroke()

        # Inner shadow. Layers composite as 1-(1-x)^n rather than adding, so
        # solve for the per-band alpha that lands on the target.
        band = 1.0 - (1.0 - INNER_SHADOW_ALPHA) ** (1.0 / INNER_SHADOW_STEPS)
        for i in range(INNER_SHADOW_STEPS):
            frac = i / max(1, INNER_SHADOW_STEPS - 1)
            reach = INNER_SHADOW_SPREAD + INNER_SHADOW_BLUR * frac
            self._outline(cr, 0, 0, w, h)
            cr.set_line_width(2 * reach)
            cr.set_source_rgba(*INNER_SHADOW_RGB, band * a)
            cr.stroke()
        cr.restore()

        # Solid and fully opaque: a translucent hairline picks up whatever is
        # behind the glass and reads as a ragged edge rather than a clean one.
        self._outline(cr, inner, inner, w - 2 * inner, h - 2 * inner)
        cr.set_line_width(STROKE_W)
        cr.set_source_rgba(*STROKE_RGB, a)
        cr.stroke()

    def _chrome_surface(self, w, h):
        """The chrome at full opacity, drawn once and kept."""
        size = (int(w), int(h))
        if self._chrome is not None and self._chrome_size == size:
            return self._chrome
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *size)
        context = cairo.Context(surface)
        context.set_antialias(cairo.ANTIALIAS_BEST)
        self._paint_chrome(context, w, h, 1.0)
        self._chrome = surface
        self._chrome_size = size
        return surface

    def _draw_spinner(self, cr, mid, a):
        """A turning arc where the recording dot was.

        Same place, same size envelope, so the pill does not appear to
        rearrange itself when the recording stops - only the thing in that
        spot changes what it is doing.
        """
        t = time.monotonic() - self._processing_t0
        radius = DOT_R * PROCESS_RING_R
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_width(1.8 * SIZE_SCALE)

        cr.new_path()
        cr.set_source_rgba(*PROCESS_RGB, 0.22 * a)
        cr.arc(DOT_X, mid, radius, 0, 2 * math.pi)
        cr.stroke()

        head = _anim.spinner_angle(t)
        cr.new_path()
        cr.set_source_rgba(*PROCESS_RGB, 0.95 * a)
        cr.arc(DOT_X, mid, radius, head, head + _anim.ARC)
        cr.stroke()

    def _draw(self, area, cr, w, h):
        if not getattr(self, "_drew", False):
            self._drew = True
            _mark("first draw")
        a = self.alpha
        mid = h / 2

        cr.save()
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()
        cr.restore()

        cr.set_antialias(cairo.ANTIALIAS_BEST)

        # Scale with the fade, about the centre, driven off the same eased
        # alpha so the two cannot drift apart. Exactly 1 at rest, so hit
        # testing against unscaled widget coordinates stays correct. Applied
        # after the clear, which must cover the whole surface untransformed.
        scale = 1.0 - FADE_SCALE * (1.0 - a)
        cr.save()
        cr.translate(w / 2.0, h / 2.0)
        cr.scale(scale, scale)
        cr.translate(-w / 2.0, -h / 2.0)

        # On Wayland the compositor blurs behind the surface and this tint
        # sits over the frost. On Windows it depends what the window got:
        # acrylic arrives already blurred and tinted, so the full material
        # over the top would flatten it back into the slab it replaced.
        # The chrome, from a cached image whenever it can be. None of it -
        # fill, sheen, rim, inner shadow, outline - depends on anything but
        # the alpha, and the alpha is 1 for all but the 200ms of a fade. It
        # was being rebuilt sixty times a second regardless: ten outline
        # paths and a gradient per frame, for a picture identical to the
        # last one.
        if a >= 0.999:
            cr.set_source_surface(self._chrome_surface(w, h), 0, 0)
            cr.paint()
        else:
            self._paint_chrome(cr, w, h, a)

        if self.processing:
            self._draw_spinner(cr, mid, a)
        else:
            elapsed = time.monotonic() - self.start
            breathe = 0.62 + 0.38 * (0.5 + 0.5 * math.sin(elapsed * 3.0))

            cr.set_source_rgba(1.0, 0.25, 0.28, 0.16 * breathe * a)
            cr.arc(DOT_X, mid, DOT_R * 2.4, 0, 2 * math.pi)
            cr.fill()
            cr.set_source_rgba(1.0, 0.27, 0.30, breathe * a)
            cr.arc(DOT_X, mid, DOT_R, 0, 2 * math.pi)
            cr.fill()

        # Bars as round-capped strokes rather than filled rounded rects.
        # A capsule BAR_W wide is exactly a line of width BAR_W with round
        # caps, drawn between the centres of its end caps - the same shape
        # from one path operation instead of four arcs. Ninety filled paths
        # a frame was the single largest cost in here.
        step = (WAVE_R - WAVE_L) / float(BARS)
        dim = 1.0 - 0.55 * self.hover
        half_cap = BAR_W / 2
        ends = []
        for i, level in enumerate(self.shown):
            reach = max(0.0, max(BAR_W, level * BAR_MAX * 2) / 2 - half_cap)
            ends.append((WAVE_L + i * step + half_cap, reach, level))

        cr.set_line_cap(cairo.LINE_CAP_ROUND)

        # Shadow and outline are one colour across every bar, so each is a
        # single path of thirty subpaths and one stroke, not thirty of each.
        cr.set_line_width(BAR_W)
        cr.set_source_rgba(0, 0, 0, 0.34 * dim * a)
        for x, reach, _ in ends:
            cr.move_to(x + BAR_SHADOW_DX, mid - reach + BAR_SHADOW_DY)
            cr.line_to(x + BAR_SHADOW_DX, mid + reach + BAR_SHADOW_DY)
        cr.stroke()

        cr.set_line_width(BAR_W + 2 * BAR_OUTLINE)
        cr.set_source_rgba(0, 0, 0, 0.46 * dim * a)
        for x, reach, _ in ends:
            cr.move_to(x, mid - reach)
            cr.line_to(x, mid + reach)
        cr.stroke()

        # The bars themselves carry a per-bar alpha, so these stay separate.
        # Processing tints them towards the spinner's blue: the sweep alone
        # reads as a quiet moment in the recording rather than as a different
        # state, and the colour is what makes it unmistakable.
        if self.processing:
            red, green, blue = (
                1.0 + (channel - 1.0) * PROCESS_TINT for channel in PROCESS_RGB)
        else:
            red = green = blue = 1.0
        cr.set_line_width(BAR_W)
        for x, reach, level in ends:
            cr.set_source_rgba(red, green, blue,
                               (0.34 + 0.66 * level) * dim * a)
            cr.move_to(x, mid - reach)
            cr.line_to(x, mid + reach)
            cr.stroke()

        if self.hover > 0.01:
            cx, r = w - 21 * SIZE_SCALE, 4.0 * SIZE_SCALE
            cr.set_source_rgba(1, 1, 1, 0.75 * self.hover * a)
            cr.set_line_width(1.4)
            cr.move_to(cx - r, mid - r)
            cr.line_to(cx + r, mid + r)
            cr.move_to(cx + r, mid - r)
            cr.line_to(cx - r, mid + r)
            cr.stroke()

        cr.restore()


def _command_loop(win: "HudWindow"):
    """Read daemon orders from stdin. End of stream means the daemon is
    gone, which is the guarantee a resident overlay is never left behind."""
    try:
        for line in sys.stdin:
            command = line.strip()
            if command.startswith("show"):
                _, _, path = command.partition(" ")
                GLib.idle_add(win.begin_show, path.strip())
            elif command == "processing":
                GLib.idle_add(win.begin_processing)
            elif command == "hide":
                GLib.idle_add(win.begin_hide)
            elif command == "quit":
                os._exit(0)
    except Exception:
        pass
    os._exit(0)


def main() -> int:
    level_file = os.environ.get("WHISPER_FLOW_HUD_LEVEL_FILE", "")
    monitor = os.environ.get("WHISPER_FLOW_HUD_MONITOR") or None
    # Only when the daemon says so: run by hand, stdin is inherited and would
    # reach end of stream at once, closing the overlay immediately.
    resident = os.environ.get("WHISPER_FLOW_HUD_RESIDENT") == "1"
    print(f"[HUD] starting level_file={level_file} monitor={monitor}",
          flush=True)

    # A plain window and main loop, not Gtk.Application: registering an
    # application id costs a D-Bus round trip before anything can be drawn,
    # and this window is opened every time the user starts talking.
    #
    # Which also means nothing here opens the display for us. GtkApplication
    # calls gtk_init() during startup; without one it is this process's job,
    # and GTK 4 does not check. Constructing a window with no display walked
    # off a null pointer deep inside GTK's style machinery, so the overlay
    # died with an access violation before its first frame - no traceback,
    # no window, and a daemon that went on believing it had one. Every test
    # that drives HudWindow called Gtk.init() itself, so this was the one
    # line no test covered.
    if not Gtk.init_check():
        print("[HUD] no display: GTK could not be initialised", flush=True)
        return 1

    loop = GLib.MainLoop()
    win = HudWindow(level_file, monitor, on_quit=loop.quit, resident=resident)
    win.connect("close-request", lambda *_: (win.quit(), False)[1])
    if resident:
        # Realize now so the window, its HWND and its styling all exist
        # before the first show; a show is then just a map, not a build.
        win.realize()
        threading.Thread(target=_command_loop, args=(win,), daemon=True,
                         name="whisper-flow-hud-commands").start()
    else:
        win.present()
    _mark("present() returned")
    loop.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
