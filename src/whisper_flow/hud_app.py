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
import time
from collections import deque

_T0 = time.monotonic()
_TIMING = bool(os.environ.get("WHISPER_FLOW_HUD_TIMING"))


def _mark(label):
    """Log milliseconds since process start; this window is on the hot path."""
    if _TIMING:
        print(f"[HUD] +{(time.monotonic() - _T0) * 1000:6.1f}ms {label}", flush=True)

import cairo
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402
from gi.repository import Gtk4LayerShell as LayerShell  # noqa: E402

# Loaded by explicit path rather than imported. Going through the package would
# pull in the daemon, and with it pystray's GTK 3, which cannot coexist with
# GTK 4 in one process; putting this directory on sys.path instead would shadow
# the standard library's logging module with the package's own.
def _load_sibling(name):
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_hud_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


enable_blur = _load_sibling("wayland_blur").enable_blur
_mark("imports done")

WIDTH = 268
HEIGHT = 54
RADIUS = HEIGHT / 2
BOTTOM_MARGIN = int(os.environ.get("WHISPER_FLOW_HUD_BOTTOM_MARGIN", "28"))

DOT_X = 27
DOT_R = 4.5
WAVE_L = 48
WAVE_R = WIDTH - 24
BARS = 30
BAR_W = 2.6
BAR_MAX = 15.0
BAR_OUTLINE = 0.9    # dark ring hugging each bar
BAR_SHADOW_DX = 0.9  # cast shadow offset, down and to the right
BAR_SHADOW_DY = 1.4

STROKE_W = 1.0              # outline width in surface-local units
STROKE_RGB = (0.42, 0.43, 0.47)
# The blur region is built from rectangles and cannot be antialiased, so its
# rounded ends are a staircase. Keeping it inside the outline hides that edge
# under the stroke instead of leaving a ragged rim of blur outside the pill.
# The margin also has to absorb fractional scaling: at 1.25x this surface is
# 67.5 physical pixels tall, and the compositor rounds the region outward, so
# too small an inset leaves blur poking past the corners.
BLUR_INSET = 4
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
EDGE_COVER = BLUR_INSET + 1.5   # opaque rim that hides the blur region's edge
# Inner shadow, in the CSS sense: it holds full strength for SPREAD, then
# fades out over BLUR. Concentric strokes clipped to the pill stand in for a
# gradient - each covers half its line width inward, so keeping every radius
# at or beyond SPREAD leaves that band evenly covered and only the outer ones
# thin out, which is the falloff.
INNER_SHADOW_ALPHA = 0.20
INNER_SHADOW_SPREAD = 14
INNER_SHADOW_BLUR = 14
INNER_SHADOW_STEPS = 18
INNER_SHADOW_RGB = (0.13, 0.13, 0.15)

FADE_MS = 200.0      # fade in and out duration
FADE_SCALE = 0.1     # shrink by this much at zero opacity, growing to full
LEVEL_EASE = 0.30    # how fast a bar chases its target height
PEAK_DECAY = 0.95    # adaptive gain, so quiet speech still reads
PEAK_FLOOR = 150.0   # below this the gain stops opening up, or idle noise dances
LEVEL_GAMMA = 0.65   # loudness is perceptual; linear RMS leaves speech near flat
# Longer than the daemon's max recording; purely a stuck-overlay backstop.
MAX_LIFETIME_S = 600

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
    b = h / 2.0
    rx = min(b * cap, w / 2.0)
    cy = y + b
    left_j, right_j = x + rx, x + w - rx
    e = 2.0 / n
    pts = []
    for j, (base, lo) in enumerate(((right_j, -math.pi / 2), (left_j, math.pi / 2))):
        for i in range(steps + 1):
            t = lo + math.pi * i / steps
            ct, st = math.cos(t), math.sin(t)
            pts.append((
                base + rx * math.copysign(abs(ct) ** e, ct),
                cy + b * math.copysign(abs(st) ** e, st),
            ))
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

    def __init__(self, level_file: str, monitor_hint: str | None, on_quit=None):
        super().__init__()
        self.level_file = level_file
        # Gtk.Window.close() only emits close-request; on a layer-shell surface
        # that does not end the main loop, so the process would linger with the
        # pill still on screen. Always tear down through this instead.
        self._on_quit = on_quit
        self._quitting = False
        self.set_default_size(WIDTH, HEIGHT)
        self.set_resizable(False)

        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        LayerShell.init_for_window(self)
        LayerShell.set_layer(self, LayerShell.Layer.OVERLAY)
        LayerShell.set_namespace(self, "whisper-flow-hud")
        LayerShell.set_keyboard_mode(self, LayerShell.KeyboardMode.NONE)
        # Negative: never reserve screen space, so windows do not get resized.
        LayerShell.set_exclusive_zone(self, -1)

        monitor = self._pick_monitor(monitor_hint)
        if monitor is not None:
            LayerShell.set_monitor(self, monitor)
        self._monitor = monitor
        self._connector = monitor.get_connector() if monitor else "default"

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
        GLib.timeout_add(33, self._read_levels)
        # The daemon takes this overlay down with SIGTERM. Handle it through
        # the main loop so the pill can fade out instead of blinking away;
        # the manager waits a couple of seconds, which is ample.
        for sig in (signal.SIGTERM, signal.SIGINT):
            _unix_signal_add(sig, self._on_signal)
        self.connect("realize", self._on_realize)
        _mark("window constructed")

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
        """Attach the blur once there is a wl_surface to attach it to."""
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
        GLib.timeout_add(int(FADE_MS) + 150, lambda: os._exit(0))

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
            if sx >= WIDTH - 34:
                self.quit()
            return
        _save_position(self._connector, *self._pos)
        print(f"[HUD] position saved for {self._connector}: {self._pos}", flush=True)

    def _on_enter(self, *_):
        self.want_hover = True

    def _on_leave(self, *_):
        self.want_hover = False

    def _frame(self):
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
                os._exit(0)
        self.hover += ((1.0 if self.want_hover else 0.0) - self.hover) * 0.25
        for i, target in enumerate(self.targets):
            self.shown[i] += (target - self.shown[i]) * LEVEL_EASE
        self.area.queue_draw()
        return True

    def _read_levels(self):
        if not self.level_file:
            # Nothing driving this window; still bound by the lifetime cap.
            return self._check_lifetime()

        # The daemon deletes the level file when the recording ends. If it is
        # gone, this overlay has been orphaned - never sit on screen forever
        # because whoever spawned us failed to take us down.
        if not os.path.exists(self.level_file):
            print("[HUD] level file gone, closing", flush=True)
            self.quit()
            return False
        if not self._check_lifetime():
            return False

        try:
            with open(self.level_file, "rb") as f:
                f.seek(0, 2)
                count = f.tell() // 4
                if count <= self.level_pos:
                    return True
                f.seek(self.level_pos * 4)
                data = f.read((count - self.level_pos) * 4)
                self.level_pos = count
            vals = struct.unpack("<%di" % (len(data) // 4), data)
        except Exception:
            return True

        for v in vals:
            v = abs(v)
            self.peak = max(self.peak * PEAK_DECAY, float(v), PEAK_FLOOR)
            self.targets.append(min(1.0, (v / self.peak) ** LEVEL_GAMMA))
        return True

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

        # A light tint only - the compositor blur behind the pill supplies the
        # frost, and a heavy fill would just mute it into flat grey.
        inner = STROKE_W / 2
        _squircle(cr, inner, inner, w - 2 * inner, h - 2 * inner)
        cr.set_source_rgba(*MATERIAL_RGB, MATERIAL_ALPHA * a)
        cr.fill_preserve()

        sheen = cairo.LinearGradient(0, 0, 0, h)
        sheen.add_color_stop_rgba(0.0, 1, 1, 1, 0.10 * a)
        sheen.add_color_stop_rgba(0.45, 1, 1, 1, 0.015 * a)
        sheen.add_color_stop_rgba(1.0, 0, 0, 0, 0.14 * a)
        cr.set_source(sheen)
        cr.fill()

        # Everything from here to the border is clipped to the pill, so strokes
        # centred on the outline only paint inwards.
        cr.save()
        _squircle(cr, 0, 0, w, h)
        cr.clip()

        # Opaque rim covering the blur region's staircase. The region is built
        # from rectangles and cannot be antialiased, so its curved ends are
        # ragged; this hides that boundary rather than leaving it on show.
        _squircle(cr, 0, 0, w, h)
        cr.set_line_width(2 * EDGE_COVER)
        cr.set_source_rgba(*MATERIAL_RGB, 0.95 * a)
        cr.stroke()

        # Inner shadow. Layers composite as 1-(1-x)^n rather than adding, so
        # solve for the per-band alpha that lands on the target.
        band = 1.0 - (1.0 - INNER_SHADOW_ALPHA) ** (1.0 / INNER_SHADOW_STEPS)
        for i in range(INNER_SHADOW_STEPS):
            frac = i / max(1, INNER_SHADOW_STEPS - 1)
            reach = INNER_SHADOW_SPREAD + INNER_SHADOW_BLUR * frac
            _squircle(cr, 0, 0, w, h)
            cr.set_line_width(2 * reach)
            cr.set_source_rgba(*INNER_SHADOW_RGB, band * a)
            cr.stroke()
        cr.restore()

        elapsed = time.monotonic() - self.start
        breathe = 0.62 + 0.38 * (0.5 + 0.5 * math.sin(elapsed * 3.0))

        cr.set_source_rgba(1.0, 0.25, 0.28, 0.16 * breathe * a)
        cr.arc(DOT_X, mid, DOT_R * 2.4, 0, 2 * math.pi)
        cr.fill()
        cr.set_source_rgba(1.0, 0.27, 0.30, breathe * a)
        cr.arc(DOT_X, mid, DOT_R, 0, 2 * math.pi)
        cr.fill()

        step = (WAVE_R - WAVE_L) / float(BARS)
        dim = 1.0 - 0.55 * self.hover
        for i, level in enumerate(self.shown):
            bar_h = max(BAR_W, level * BAR_MAX * 2)
            x = WAVE_L + i * step

            top = mid - bar_h / 2

            # Cast shadow, offset down-right, so the bars sit above the glass
            # rather than being painted flat onto it.
            cr.set_source_rgba(0, 0, 0, 0.34 * dim * a)
            _round_rect(cr, x + BAR_SHADOW_DX, top + BAR_SHADOW_DY,
                        BAR_W, bar_h, BAR_W / 2)
            cr.fill()

            # Tight outline around the bar itself. Doubles as contrast against
            # a light window behind the translucent glass.
            cr.set_source_rgba(0, 0, 0, 0.46 * dim * a)
            _round_rect(cr, x - BAR_OUTLINE, top - BAR_OUTLINE,
                        BAR_W + 2 * BAR_OUTLINE, bar_h + 2 * BAR_OUTLINE,
                        (BAR_W + 2 * BAR_OUTLINE) / 2)
            cr.fill()

            cr.set_source_rgba(1, 1, 1, (0.34 + 0.66 * level) * dim * a)
            _round_rect(cr, x, top, BAR_W, bar_h, BAR_W / 2)
            cr.fill()

        # Outline last, so nothing can paint over it. Solid and fully opaque:
        # a translucent hairline picks up whatever is behind the glass and
        # reads as a ragged edge rather than a clean one.
        _squircle(cr, inner, inner, w - 2 * inner, h - 2 * inner)
        cr.set_line_width(STROKE_W)
        cr.set_source_rgba(*STROKE_RGB, a)
        cr.stroke()

        if self.hover > 0.01:
            cx, r = w - 21, 4.0
            cr.set_source_rgba(1, 1, 1, 0.75 * self.hover * a)
            cr.set_line_width(1.4)
            cr.move_to(cx - r, mid - r)
            cr.line_to(cx + r, mid + r)
            cr.move_to(cx + r, mid - r)
            cr.line_to(cx - r, mid + r)
            cr.stroke()

        cr.restore()


def main() -> int:
    level_file = os.environ.get("WHISPER_FLOW_HUD_LEVEL_FILE", "")
    monitor = os.environ.get("WHISPER_FLOW_HUD_MONITOR") or None
    print(f"[HUD] starting level_file={level_file} monitor={monitor}", flush=True)

    # A plain window and main loop, not Gtk.Application: registering an
    # application id costs a D-Bus round trip before anything can be drawn,
    # and this window is opened every time the user starts talking.
    loop = GLib.MainLoop()
    win = HudWindow(level_file, monitor, on_quit=loop.quit)
    win.connect("close-request", lambda *_: (win.quit(), False)[1])
    win.present()
    _mark("present() returned")
    loop.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
