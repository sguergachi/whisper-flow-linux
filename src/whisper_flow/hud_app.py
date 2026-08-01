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

STROKE_W = 1.5              # outline width in surface-local units
STROKE_RGB = (0.93, 0.94, 0.97)
# The blur region is built from rectangles and cannot be antialiased, so its
# rounded ends are a staircase. Keeping it inside the outline hides that edge
# under the stroke instead of leaving a ragged rim of blur outside the pill.
BLUR_INSET = 2

FADE_STEP = 0.30     # opacity per frame at ~60fps; ~4 frames to fully opaque
LEVEL_EASE = 0.30    # how fast a bar chases its target height
PEAK_DECAY = 0.95    # adaptive gain, so quiet speech still reads
PEAK_FLOOR = 150.0   # below this the gain stops opening up, or idle noise dances
LEVEL_GAMMA = 0.65   # loudness is perceptual; linear RMS leaves speech near flat
# Longer than the daemon's max recording; purely a stuck-overlay backstop.
MAX_LIFETIME_S = 600

CSS = b"""
window, window.background { background-color: transparent; }
"""


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

        # Overlay layer, anchored bottom-centre. Anchoring only left+right
        # centres it horizontally; anchoring bottom pins it vertically.
        LayerShell.init_for_window(self)
        LayerShell.set_layer(self, LayerShell.Layer.OVERLAY)
        LayerShell.set_namespace(self, "whisper-flow-hud")
        LayerShell.set_anchor(self, LayerShell.Edge.BOTTOM, True)
        LayerShell.set_margin(self, LayerShell.Edge.BOTTOM, BOTTOM_MARGIN)
        LayerShell.set_keyboard_mode(self, LayerShell.KeyboardMode.NONE)
        # Negative: never reserve screen space, so windows do not get resized.
        LayerShell.set_exclusive_zone(self, -1)

        monitor = self._pick_monitor(monitor_hint)
        if monitor is not None:
            LayerShell.set_monitor(self, monitor)

        self.alpha = 0.0
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

        click = Gtk.GestureClick()
        click.connect("pressed", lambda *_: self.quit())
        area.add_controller(click)

        motion = Gtk.EventControllerMotion()
        motion.connect("enter", self._on_enter)
        motion.connect("leave", self._on_leave)
        area.add_controller(motion)

        GLib.timeout_add(16, self._frame)
        GLib.timeout_add(33, self._read_levels)
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
                wl_display, wl_surface, WIDTH, HEIGHT, RADIUS, BLUR_INSET)
            _mark("realize: blur done")
            print(f"[HUD] blur {'enabled' if self._blur else 'unavailable'}", flush=True)
        except Exception as e:
            print(f"[HUD] blur setup failed: {e}", flush=True)

    def quit(self):
        """Take the overlay down and end the process.

        Neither Gtk.Window.close() nor MainLoop.quit() ends this process once a
        layer-shell surface is mapped: the loop returns but the process lives
        on, leaving the pill stuck on screen. Exiting cannot be deferred to the
        main loop either, since quitting the loop is what stops it running.

        So leave immediately. This child owns nothing but its own window, and
        dropping the Wayland connection is what actually removes the surface.
        """
        if self._quitting:
            return
        self._quitting = True
        self.set_visible(False)
        os._exit(0)

    def _check_lifetime(self):
        """Backstop against an overlay outliving whatever spawned it."""
        if time.monotonic() - self.start > MAX_LIFETIME_S:
            print("[HUD] lifetime cap reached, closing", flush=True)
            self.quit()
            return False
        return True

    def _on_enter(self, *_):
        self.want_hover = True

    def _on_leave(self, *_):
        self.want_hover = False

    def _frame(self):
        self.alpha = min(1.0, self.alpha + FADE_STEP)
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

        # A light tint only - the compositor blur behind the pill supplies the
        # frost, and a heavy fill would just mute it into flat grey.
        inner = STROKE_W / 2
        _round_rect(cr, inner, inner, w - 2 * inner, h - 2 * inner, RADIUS - inner)
        cr.set_source_rgba(0.07, 0.07, 0.09, 0.42 * a)
        cr.fill_preserve()

        sheen = cairo.LinearGradient(0, 0, 0, h)
        sheen.add_color_stop_rgba(0.0, 1, 1, 1, 0.13 * a)
        sheen.add_color_stop_rgba(0.45, 1, 1, 1, 0.02 * a)
        sheen.add_color_stop_rgba(1.0, 0, 0, 0, 0.10 * a)
        cr.set_source(sheen)
        cr.fill()

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

            # Dark halo first: the glass is translucent, so white bars would
            # otherwise vanish against a light window behind it.
            cr.set_source_rgba(0, 0, 0, 0.38 * dim * a)
            _round_rect(cr, x - 1, mid - bar_h / 2 - 1,
                        BAR_W + 2, bar_h + 2, (BAR_W + 2) / 2)
            cr.fill()

            cr.set_source_rgba(1, 1, 1, (0.30 + 0.68 * level) * dim * a)
            _round_rect(cr, x, mid - bar_h / 2, BAR_W, bar_h, BAR_W / 2)
            cr.fill()

        # Outline last, so nothing can paint over it. Solid and fully opaque:
        # a translucent hairline picks up whatever is behind the glass and
        # reads as a ragged edge rather than a clean one.
        _round_rect(cr, inner, inner, w - 2 * inner, h - 2 * inner, RADIUS - inner)
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
