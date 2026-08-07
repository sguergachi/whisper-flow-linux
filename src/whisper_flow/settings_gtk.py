"""The settings window: GTK4 + libadwaita, with real backdrop blur on Wayland.

Runs as its own process (``python -m whisper_flow.settings_gtk``), like every
other window this app shows - the daemon's pystray loop and GTK each demand
their own main thread. One window on both platforms; there has been no
tkinter version of it for some time.

The blur is the HUD's own path: ext-background-effect-v1, with a region
tracing the window's rounded corners, and it only engages on Wayland with a
compositor that offers the protocol (KWin does). Everywhere else the window
keeps the stock opaque Adwaita look, which is the point of having a fallback
rather than a failure.

Saving writes the .env file and restarts the daemon so the new values apply
immediately. There is no "restart later" step: if a restart is needed, it
happens as part of Save.
"""

import contextlib
import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Measured from the click, not from this import. The daemon stamps the moment
# it launched us into the environment, so these numbers include the process
# start - which in a frozen build is a real part of what the user waits for
# and is invisible from inside. Wall clock rather than monotonic because it
# has to mean the same thing in two processes.
_T0 = float(os.environ.get("WHISPER_FLOW_TOOL_T0") or 0.0) or time.time()


def _mark(label: str, since: float | None = None) -> None:
    """Report a stage of opening this window, in ms since the click.

    To a file, not to stdout. A PyInstaller windowed build leaves sys.stdout
    as None, and print() silently returns when it is - so the first round of
    these went nowhere at all and the log came back with no sign the window
    had ever been opened. The daemon names the file and reads it back into
    the log the tray menu shows.

    `since` is for a window that was built before anyone asked for it: the
    stages of that build are measured from the process start as usual, but
    the number that matters afterwards is measured from the click, which
    happens minutes or hours later.
    """
    elapsed = (time.time() - (_T0 if since is None else since)) * 1000
    line = f"[SETTINGS] +{elapsed:6.0f}ms {label}"
    path = os.environ.get("WHISPER_FLOW_TOOL_LOG")
    if path:
        # One route or the other, never both. The daemon reads the file *and*
        # drains our stdout, so writing to each printed every stage twice in
        # the log - which reads as the window having been built twice.
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return
        except OSError:
            pass                # fall through and print; better than silence
    try:
        print(line, flush=True)     # a source checkout still has a console
    except Exception:
        pass


def _prefetch_the_config_stack() -> None:
    """Load the non-GTK imports while GTK is still loading.

    They are independent and were strictly serial: GTK first, then 720ms of
    pydantic-settings, measured, purely so Config can exist. Nothing about
    that ordering is required - the window needs both before it can draw and
    does not care which finishes first - so the slower half starts here and
    runs alongside the GTK import below.

    Python's per-module import locks make the duplicate import safe: whichever
    thread arrives second waits for the first rather than doing the work twice.
    Failures are ignored because the real import below will raise them again,
    in the place that can report them.
    """
    def work():
        try:
            import whisper_flow.backend      # noqa: F401
            import whisper_flow.config       # noqa: F401
        except Exception:
            pass

    threading.Thread(target=work, daemon=True,
                     name="whisper-flow-settings-prefetch").start()


_prefetch_the_config_stack()

import gi  # noqa: E402

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk  # noqa: E402

_mark("gtk imported")

from . import envfile, restart, settings_def, wayland_blur  # noqa: E402
from .backend import LocalBackend  # noqa: E402
from .config import Config, reload_config  # noqa: E402
from .logging import log, set_logging_enabled  # noqa: E402
from .version import build_version  # noqa: E402

_mark("imports done")

WIDTH, HEIGHT = 760, 820
CORNER_RADIUS = 12      # Adwaita's window corner radius

# Every icon this window names, in one place so the frozen build can check
# that the bundled theme actually has them. An icon GTK cannot find is not an
# error it reports - it silently draws the broken-image glyph, which is what
# the view switcher showed for three of its four tabs while the fourth, which
# happens to be in the set GTK compiles into libgtk, looked perfectly fine.
SECTION_ICONS = {
    "Speech": "audio-speakers-symbolic",
    "Hotkeys": "input-keyboard-symbolic",
    "Dictation": "audio-input-microphone-symbolic",
    "General": "emblem-system-symbolic",
}
# object-select-symbolic, not emblem-ok-symbolic: Adwaita 50 dropped the
# latter, and the running-daemon row would have drawn the broken-image glyph.
# The selftest caught it, which is the whole reason it checks these by name.
STATUS_ICONS = ("object-select-symbolic", "dialog-warning-symbolic")
# The copy button beside the version, on the About group.
ACTION_ICONS = ("edit-copy-symbolic",)
ICON_NAMES = tuple(SECTION_ICONS.values()) + STATUS_ICONS + ACTION_ICONS

CSS = b"""
.model-pill {
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 700;
}
.pill-recommended { background: #1b2c4a; color: #9ec2fc; }
.pill-current { background: #16301f; color: #4ade80; }
/* Violet, not another green. This sits directly beside "in use", which is
   green, and two green pills an inch apart read as one status repeated
   rather than as two different facts - which engine will run the model, and
   which model is running. The hue is the difference; nothing else is. */
.pill-gpu { background: #2e2148; color: #c4b5fd; }
.pill-warning { background: #3a2a12; color: #fbbf24; }
.model-progress { min-width: 110px; }
/* The model rows build their own title and subtitle, because the badge has
   to sit beside the name and AdwActionRow has no slot there. This is the
   one thing the row would otherwise have done for us. */
.model-subtitle { font-size: 0.9em; }
/* The device list, not the row: the row stays the width of the page and
   ellipsizes the one name it shows, while the popup you open to compare
   names is wide enough to read them. 520px is a long Windows device name
   with its host API suffix, inside a 760px window. Restored - it was lost
   resolving a merge, leaving the factory without the floor it needs for the
   case where the popover declines to grow past the row. */
.mic-row popover listview { min-width: 520px; }
/* Drawn level strip under Device (DrawingArea, not ProgressBar). */
.mic-meter {
    min-width: 160px;
    min-height: 14px;
    margin-left: 8px;
    margin-right: 8px;
}
.daemon-ok { color: #4ade80; }
.daemon-bad { color: #f87171; }
"""


# Only loaded once compositor blur is confirmed, so none of this can reach a
# window that has no frost behind it to show.
#
# Frosting the window alone was not enough to see: libadwaita paints the
# page, the viewport and every preferences card from its own named colours,
# and those are opaque. They cover all but a few pixels of margin, so the
# blur was real and invisible at the same time. Overriding the names frosts
# what the eye actually sees, rather than chasing widget selectors that
# change between libadwaita releases.
# Windows only, and applied before the window is realized rather than after.
#
# GTK keeps a transparent margin around the window to draw its own drop
# shadow into, and sizes the surface to include it. DWM knows nothing about
# that margin and fills the whole window rectangle, so the backdrop appears
# as a rectangle standing well outside the rounded window. Removing the
# shadow is what shrinks the surface to the window - and it has to be in
# place before realize, because that is when the extents are computed.
#
# Unscoped, deliberately: the .blurred class is only added once a backdrop
# is confirmed, which is after realize and therefore too late for this.
CSS_WIN_FRAME = b"""
decoration, decoration:backdrop {
    box-shadow: none;
    margin: 0;
    border-radius: 0;
}
"""

# The few nodes the shared stylesheet does not already clear. Deliberately
# not the window itself: its translucent fill is the frost. The blur comes
# from DWM behind it, exactly as the compositor's does on Wayland, and a
# fully transparent window would show a sharp desktop rather than glass.
CSS_WIN_BACKDROP = b"""
/* Every node between the surface and the content. One opaque box in this
   chain hides the whole backdrop, which is exactly what happened when this
   list was trimmed - the blur was there and covered. */
window.blurred > widget,
window.blurred windowhandle,
window.blurred box,
window.blurred toolbarview,
window.blurred .toolbar-view,
window.blurred stack,
window.blurred overlay { background-color: transparent; background-image: none; }

/* One radius, and it has to be DWM's. The backdrop fills the window
   rectangle, so a corner GTK rounds is a corner where the frost carries on
   past it - the sliver of material outside Adwaita's 12px arc. Painting
   square lets DWM's own rounding cut the window and the material together,
   which is also the radius every other window on this desktop uses. */
window.blurred,
window.blurred > .background { border-radius: 0; }

/* No tint at all, unlike Wayland.
   There the compositor hands back a raw blur and this fill is the entire
   material - 0.66 of near-black is what makes it glass. Acrylic arrives
   already tinted, by the alpha in ACCENT_TINT_DARK, so anything added here
   lands on a finished material and flattens it. The window has to be clear
   for the acrylic underneath to be the thing you see. */
window.blurred,
window.blurred > .background { background-color: transparent; }
"""

CSS_BLUR = b"""
@define-color window_bg_color rgba(22, 23, 29, 0.66);
@define-color view_bg_color rgba(22, 23, 29, 0.0);
@define-color headerbar_bg_color rgba(22, 23, 29, 0.0);
@define-color card_bg_color rgba(255, 255, 255, 0.055);
@define-color popover_bg_color rgba(30, 31, 38, 0.94);
@define-color dialog_bg_color rgba(24, 25, 31, 0.94);

window.blurred,
window.blurred > .background { background-color: rgba(22, 23, 29, 0.66); }

/* Each layer between the window and the cards draws its own background.
   Any one of them left opaque hides the frost under everything above it. */
window.blurred headerbar,
window.blurred .toolbar-view,
window.blurred scrolledwindow,
window.blurred viewport,
window.blurred preferencespage,
window.blurred preferencesgroup,
window.blurred clamp { background-color: transparent; background-image: none; }
window.blurred headerbar { box-shadow: none; border-color: transparent; }

/* Glass, not a slab: a hairline of light and a barely-there fill, so the
   card reads as sitting in the blur rather than on top of it. */
window.blurred .boxed-list,
window.blurred list.boxed-list,
window.blurred listview.boxed-list {
    background-color: rgba(255, 255, 255, 0.055);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: none;
}
window.blurred row,
window.blurred row.activatable { background-color: transparent; }
window.blurred row.activatable:hover {
    background-color: rgba(255, 255, 255, 0.05);
}

/* The switcher is chrome, not content: let it float on the frost. */
window.blurred viewswitcher button:not(:checked) { background-color: transparent; }
window.blurred viewswitcher button:checked {
    background-color: rgba(255, 255, 255, 0.10);
}

/* The unsaved-changes banner, on the frost rather than on a filled strip. */
window.blurred banner > revealer > widget {
    background-color: rgba(255, 255, 255, 0.09);
}
"""


def _load_css(provider, css: bytes):
    """Feed CSS to a provider, whichever call this GTK offers.

    load_from_data() has always taken the length as a second argument. Older
    introspection data annotated it as the array's own length, so PyGObject
    filled it in and the one-argument call worked; newer data does not, and
    the same call raises TypeError instead. This window builds itself inside
    an activate handler, where GObject prints the traceback and carries on -
    so on Windows the process sat there for as long as you left it, holding
    no window and reporting no error.

    load_from_string() has no length to get wrong and has been there since GTK
    4.12, so it is the call everywhere it exists. The fallback is for GTK
    before that, where the one-argument form is the correct one.
    """
    if hasattr(provider, "load_from_string"):
        provider.load_from_string(css.decode())
    else:
        provider.load_from_data(css)


def _gdk_key_to_hotkey_part(keyval: int) -> str | None:
    """Map a GDK keyval to the config name the hotkey listeners understand.

    Returns None for keys that cannot be part of a binding (unknown, pure
    dead keys, etc.). Modifiers and a-z / 0-9 / F-keys / a few named keys
    match what hotkey_evdev and hotkey_win parse.
    """
    keyval = Gdk.keyval_to_lower(keyval)
    if keyval in (Gdk.KEY_Control_L, Gdk.KEY_Control_R):
        return "ctrl"
    if keyval in (Gdk.KEY_Alt_L, Gdk.KEY_Alt_R):
        return "alt"
    if keyval in (Gdk.KEY_Shift_L, Gdk.KEY_Shift_R):
        return "shift"
    if keyval in (Gdk.KEY_Super_L, Gdk.KEY_Super_R,
                  Gdk.KEY_Hyper_L, Gdk.KEY_Hyper_R):
        return "super"
    # Meta is Super on many layouts; keep it out of the Alt bucket.
    if keyval in (Gdk.KEY_Meta_L, Gdk.KEY_Meta_R):
        return "super"
    if keyval == Gdk.KEY_space:
        return "space"
    if keyval in (Gdk.KEY_Escape,):
        return "escape"
    if keyval in (Gdk.KEY_BackSpace,):
        return "backspace"
    if keyval in (Gdk.KEY_Tab, Gdk.KEY_ISO_Left_Tab):
        return "tab"
    if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
        return "enter"
    if keyval == Gdk.KEY_Caps_Lock:
        return "capslock"
    # a-z
    if Gdk.KEY_a <= keyval <= Gdk.KEY_z:
        return chr(ord("a") + (keyval - Gdk.KEY_a))
    # 0-9 (top row and keypad)
    if Gdk.KEY_0 <= keyval <= Gdk.KEY_9:
        return chr(ord("0") + (keyval - Gdk.KEY_0))
    if Gdk.KEY_KP_0 <= keyval <= Gdk.KEY_KP_9:
        return chr(ord("0") + (keyval - Gdk.KEY_KP_0))
    # F1–F24
    if Gdk.KEY_F1 <= keyval <= Gdk.KEY_F12:
        return f"f{1 + (keyval - Gdk.KEY_F1)}"
    if Gdk.KEY_F13 <= keyval <= Gdk.KEY_F24:
        return f"f{13 + (keyval - Gdk.KEY_F13)}"
    return None


def _wide_list_factory(row) -> Gtk.SignalListItemFactory:
    """Dropdown rows that keep their full text, however long it is.

    AdwComboRow's stock factory ellipsizes to the width of the closed row.
    A plain Gtk.Label does not ellipsize at all, and GtkDropDown's popover
    propagates its natural width, so the popup grows to fit the widest entry
    instead of truncating all of them to the same useless prefix.

    The tick is the other half of what the stock factory does, and replacing
    the factory is what lost it: an open list of twenty near-identical device
    names with nothing marking the one in use. The row this list belongs to
    is asked which that is, rather than the list's own selection model - the
    popup builds a fresh one, so on the frame the list first appears its idea
    of the selection is not yet the row's.

    Hidden by opacity, not by visibility: an invisible tick takes no width,
    so every row would shift sideways as the selection moved.
    """
    factory = Gtk.SignalListItemFactory()
    shown = set()

    def mark(item):
        item.get_child().get_last_child().set_opacity(
            1.0 if item.get_position() == row.get_selected() else 0.0)

    def setup(_factory, item):
        box = Gtk.Box(spacing=12)
        box.append(Gtk.Label(xalign=0.0, hexpand=True))
        box.append(Gtk.Image(icon_name="object-select-symbolic"))
        item.set_child(box)

    def bind(_factory, item):
        value = item.get_item()
        item.get_child().get_first_child().set_text(
            value.get_string() if value else "")
        shown.add(item)
        mark(item)

    def unbind(_factory, item):
        shown.discard(item)

    def selection_changed(*_args):
        for item in shown:
            mark(item)

    factory.connect("setup", setup)
    factory.connect("bind", bind)
    factory.connect("unbind", unbind)
    row.connect("notify::selected", selection_changed)
    return factory


def _gobject_pointer(obj) -> int:
    """Address of the underlying GObject, via PyGObject's capsule."""
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ctypes.pythonapi.PyCapsule_GetPointer(obj.__gpointer__, None)


def _list_input_devices() -> list[tuple[int, str]]:
    """(index, name) for every capture device, or nothing if audio is down.

    The name carries the host API, because without it the list reads as
    duplicates: PortAudio enumerates each physical microphone once per
    Windows audio API - MME, DirectSound, WASAPI, WDM-KS - so one headset is
    four entries with the same name and four different indices. They are not
    interchangeable either. WASAPI is the modern path and the one that says
    what it will and will not resample; MME is the oldest and the loosest.
    Which was picked is exactly what a name repeated four times cannot say.
    """
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            devices = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) <= 0:
                    continue
                name = str(info.get("name", f"device {i}"))
                api = _host_api_name(pa, info)
                devices.append((i, f"{name} ({api})" if api else name))
            return devices
        finally:
            pa.terminate()
    except Exception as e:
        log(f"[SETTINGS] could not list microphones: {e}")
        return []


@contextlib.contextmanager
def _suppress_alsa_for_meter():
    """Quiet ALSA chatter while the Device meter opens a capture stream.

    Same problem the recorder has: PortAudio's ALSA backend prints probe
    noise on stderr for every device it walks, and that lands in the
    settings process log while nobody is diagnosing the audio stack.
    """
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def _open_meter_stream(pa, device):
    """Open a short capture stream for the Device-row level meter.

    Uses the device's own rate so WASAPI shared mode accepts the open - the
    same trap the recorder already documents. Levels are rate-agnostic RMS,
    so conversion is unnecessary here. The platform default (device is None)
    also has a native rate; hardcoding 16 kHz there fails or returns silence
    on arrays that only run at 44.1/48 kHz.

    Returns ``(stream, frames_per_read)`` or ``(None, 0)``.
    """
    import pyaudio

    rate = 16000
    try:
        if device is not None:
            info = pa.get_device_info_by_index(device)
        else:
            info = pa.get_default_input_device_info()
        native = int(info.get("defaultSampleRate") or 0)
        if native > 0:
            rate = native
    except Exception:
        pass
    # ~20 ms of audio per read: short enough that closing the window
    # releases the microphone within a frame, long enough that RMS is stable.
    chunk = max(256, rate // 50)
    try:
        with _suppress_alsa_for_meter():
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=device,
                frames_per_buffer=chunk,
            )
        return stream, chunk
    except Exception as e:
        log(f"[SETTINGS] mic meter could not open device {device}: {e}")
        return None, 0


def _close_meter_stream(stream) -> None:
    if stream is None:
        return
    try:
        stream.stop_stream()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass


def _host_api_name(pa, info) -> str:
    """Which Windows audio API serves a device, "" if it cannot be read.

    Shortened the way the system does: PortAudio calls them "Windows WASAPI"
    and "Windows DirectSound", and the word is on every entry in the list, so
    it is on none of them here.
    """
    try:
        name = str(pa.get_host_api_info_by_index(info["hostApi"])["name"])
    except Exception:
        return ""
    prefix = "Windows "
    return name[len(prefix):] if name.startswith(prefix) else name


# Spin adjustment (digits, step) per numeric field; anything not listed is a
# plain text row.
_SPIN = {
    "local_server_port": (0, 1),
    "live_interval": (1, 0.1),
    "silence_timeout": (1, 0.5),
    "auto_stop_silence_duration": (1, 0.5),
    "speedup_audio": (2, 0.25),
    "max_recording_duration": (0, 30),
    "notification_min_interval": (0, 1),
    "notification_timeout": (0, 500),
    "processing_lock_timeout": (0, 1),
    "watchdog_interval": (1, 0.5),
    "queue_request_timeout": (0, 5),
}

# Live Device-row meter. Floor is low enough that a quiet laptop mic still
# moves the bar; the HUD uses 150 because it has many bars and noise would
# dance - here one strip has to prove "this is the right device".
_MIC_METER_PEAK_FLOOR = 40.0
_MIC_METER_PEAK_DECAY = 0.92
_MIC_METER_LEVEL_GAMMA = 0.55
_MIC_METER_LEVEL_CEILING = 32767.0
_MIC_METER_POLL_MS = 50
# "off" means close the stream; None means the platform default input.
_MIC_METER_OFF = "off"


def _mic_level_from_rms(rms: float, peak: float) -> tuple[float, float]:
    """Map a frame level onto 0..1 and the peak used for adaptive gain.

    Pure so the test can pin the scaling without opening a microphone.
    `rms` may be peak-abs or true RMS; either is fine for a preview meter.
    """
    if rms <= 0:
        return 0.0, max(peak * _MIC_METER_PEAK_DECAY, _MIC_METER_PEAK_FLOOR)
    if rms > _MIC_METER_LEVEL_CEILING:
        rms = _MIC_METER_LEVEL_CEILING
    peak = max(peak * _MIC_METER_PEAK_DECAY, float(rms), _MIC_METER_PEAK_FLOOR)
    level = min(1.0, (rms / peak) ** _MIC_METER_LEVEL_GAMMA)
    return level, peak


class _MicLevelStrip(Gtk.DrawingArea):
    """A single green fill bar. ProgressBar/LevelBar both fought Adwaita CSS
    and read as empty even when the fraction was non-zero."""

    def __init__(self):
        super().__init__()
        self._level = 0.0
        self.set_content_width(160)
        self.set_content_height(14)
        self.set_hexpand(False)
        self.set_valign(Gtk.Align.CENTER)
        self.add_css_class("mic-meter")
        self.set_draw_func(self._draw)

    def set_level(self, level: float) -> None:
        level = max(0.0, min(1.0, float(level)))
        if abs(level - self._level) < 0.005:
            return
        self._level = level
        self.queue_draw()

    def get_level(self) -> float:
        return self._level

    @staticmethod
    def _draw(area, cr, width: int, height: int) -> None:
        # Trough
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.12)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        # Fill - bright green so speech is obvious on the dark theme
        fill = max(0.0, min(1.0, area._level)) * width
        if fill > 0.5:
            cr.set_source_rgb(0.29, 0.87, 0.50)
            cr.rectangle(0, 0, fill, height)
            cr.fill()


class SettingsWindow(Adw.ApplicationWindow):
    """The preferences window itself."""

    def __init__(self, application=None, config=None):
        super().__init__(title="WhisperFlow settings")
        if application is not None:
            # Registered, or closing the window leaves the process running.
            self.set_application(application)
        # Handed the one main() already built where there is one. Constructing
        # a second Config re-reads and re-validates the whole schema for an
        # answer this process is holding, and it happens on the path the user
        # is waiting on.
        self.config = config if config is not None else Config()
        self.backend = LocalBackend(self.config)
        self._current_model = self.backend.working_model()
        self._daemon_pid = restart.daemon_pid()
        self._show_t0 = _T0
        _mark("config and backend read")
        self._working = False
        self._rows: dict[str, Gtk.Widget] = {}
        # Stable ViewStack children that hold each PreferencesPage. Speech is
        # rebuilt in place inside its host so tab order never changes and the
        # stack never reparents Hotkeys/Dictation/General mid-session (that
        # path access-violates on Windows once those pages have been shown).
        self._page_hosts: dict[str, Gtk.Box] = {}
        self._mic_display: dict[str, str] = {}
        self._mic_meter: _MicLevelStrip | None = None
        self._mic_test_button: Gtk.Button | None = None
        # Only True while the user has pressed Test. The meter never opens
        # the microphone just because settings is on screen.
        self._mic_meter_active = False
        # GLib source id for the meter paint timer; only armed during Test.
        self._mic_meter_tick_id = 0
        # Shared with the capture thread. Main thread writes the wanted
        # device; the thread opens/closes and publishes the drawn level.
        # The ComboRow is NOT read from the paint tick - doing that every
        # 50ms while the device popover is open re-enters GTK and freezes
        # the window. Device is published on Test and on selection-changed.
        self._mic_want_device = _MIC_METER_OFF   # int | None | "off"
        self._mic_drawn_level = 0.0
        # One-shot error from the capture thread (open failed), shown as toast.
        self._mic_meter_error: str | None = None
        self._mic_meter_lock = threading.Lock()
        self._mic_meter_stop = threading.Event()
        self._mic_meter_thread: threading.Thread | None = None
        self._model_checks: dict[str, Gtk.CheckButton] = {}
        self._download_buttons: list[Gtk.Button] = []
        self._progress_bars: dict[str, Gtk.ProgressBar] = {}
        self._last_toast_title = ""
        self._blur = None

        Adw.StyleManager.get_default().set_color_scheme(
            Adw.ColorScheme.FORCE_DARK)
        self.set_default_size(WIDTH, HEIGHT)
        # Fixed: the blur region is in surface-local coordinates and would
        # be stale the moment the window changed size.
        self.set_resizable(False)

        provider = Gtk.CssProvider()
        _load_css(provider, CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        # Adwaita-specific icons (entry edit/apply affordances) live in
        # libadwaita's own resources; under a non-Adwaita icon theme - breeze
        # on KDE, for one - they are not on the lookup path and render as
        # the broken-image glyph on every text row.
        icon_theme = Gtk.IconTheme.get_for_display(Gdk.Display.get_default())
        icon_theme.add_resource_path("/org/gnome/Adwaita/icons")

        self._build()
        _mark("pages built")
        self._load()
        self._refresh_dirty()
        self._load_mics_in_background()
        GLib.timeout_add(400, self._refresh_dirty)
        GLib.timeout_add(2000, self._watch_the_daemon)
        # LevelBar paint timer is armed only for an active Test (see
        # _start_mic_test). Leaving it idle here keeps rebuilds free of
        # widget traffic on the Device row.
        self.connect("notify::visible", self._on_visible_for_mic_meter)
        if sys.platform == "win32":
            # Undecorated, and before realize. GTK's client-side decoration
            # carries a shadow the surface is sized to include; DWM knows
            # nothing of it and fills the whole window rectangle, so the
            # backdrop appears as a rectangle standing outside the window.
            # The header bar is Adwaita's own and is unaffected - this drops
            # only the frame GTK draws around it. DWMWCP_ROUND supplies the
            # corners in its place.
            self.set_decorated(False)
            provider = Gtk.CssProvider()
            _load_css(provider, CSS_WIN_FRAME)
            Gtk.StyleContext.add_provider_for_display(
                Gdk.Display.get_default(), provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 2)
        self.connect("realize", self._on_realize)

    # ------------------------------------------------------ shown on request
    def show_for_click(self, at: float | None = None) -> bool:
        """Put a window that was built in advance on screen.

        The whole point of building it in advance is that this costs a map
        and nothing else, so what happens here has to stay small. The reload
        is the exception, and it is not optional: this window may have been
        built at login and asked for hours later, by which time the daemon
        could have downloaded a model, installed the GPU engine, or written
        the .env itself. Showing what was true at login would be showing the
        user stale settings and then saving them back.

        Only on the way up from hidden, though. Clicking Settings while this
        window is already open is someone asking for it to be raised, and
        re-reading the config there would throw away edits they have made and
        not yet saved. Skipped while a download is running too, because that
        owns these rows and holds progress bars this would drop mid-flight.
        """
        self._show_t0 = at or time.time()
        if not self.get_visible() and not self._working:
            self._reload_from_disk()
        self._raise_to_front()
        return False                    # idle_add: run once

    def _raise_to_front(self) -> None:
        """Map, unminimize and take focus after a tray click.

        ``present()`` alone maps a hidden window but does not take focus when
        the click came from another process. On Wayland the compositor
        refuses focus-steal without an xdg-activation token the tray cannot
        hand us, so after present we ask the window manager (kdotool on KDE,
        wmctrl/xdotool elsewhere) to activate us by title.
        """
        try:
            if bool(self.get_property("minimized")):
                self.unminimize()
        except Exception:
            pass
        self.set_visible(True)
        self.present()
        # Once the surface is mapped (and again a moment later for slow WMs).
        GLib.idle_add(self._finish_raise)
        GLib.timeout_add(120, self._finish_raise)

    def _finish_raise(self) -> bool:
        try:
            self.present()
            if sys.platform == "win32":
                self._raise_win32_foreground()
            else:
                self._raise_linux_foreground()
        except Exception as e:
            log(f"[SETTINGS] could not raise the window: {e}")
        return False                    # idle/timeout: run once each

    def _raise_win32_foreground(self) -> None:
        """Ask Windows to put this HWND in front after a tray click."""
        try:
            from . import blur_win
            surface = self.get_surface()
            if surface is None:
                return
            hwnd = blur_win.surface_handle(_gobject_pointer(surface))
            if not hwnd:
                return
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            user32.SetForegroundWindow.argtypes = [wintypes.HWND]
            user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
            SW_RESTORE = 9
            user32.ShowWindow(wintypes.HWND(hwnd), SW_RESTORE)
            user32.SetForegroundWindow(wintypes.HWND(hwnd))
        except Exception as e:
            log(f"[SETTINGS] Win32 raise failed: {e}")

    def _raise_linux_foreground(self) -> None:
        """Activate this window via the compositor after a tray click.

        Runs off the GTK thread: kdotool/wmctrl can block briefly and must
        not stall the settings UI.
        """
        title = self.get_title() or "WhisperFlow settings"

        def work():
            try:
                self._activate_by_title(title)
            except Exception as e:
                log(f"[SETTINGS] compositor raise failed: {e}")

        threading.Thread(target=work, daemon=True,
                         name="whisper-flow-settings-raise").start()

    @staticmethod
    def _activate_by_title(title: str) -> None:
        """Best-effort focus for a window whose title matches `title`."""
        import shutil

        # KDE (Wayland + X11): same tool the paste path uses.
        if shutil.which("kdotool"):
            found = subprocess.run(
                ["kdotool", "search", "--name", title],
                capture_output=True, text=True, timeout=3, check=False,
            )
            for wid in found.stdout.splitlines():
                wid = wid.strip()
                if not wid:
                    continue
                subprocess.run(
                    ["kdotool", "windowactivate", wid],
                    capture_output=True, text=True, timeout=3, check=False,
                )
                return
        if shutil.which("wmctrl"):
            subprocess.run(
                ["wmctrl", "-a", title],
                capture_output=True, text=True, timeout=3, check=False,
            )
            return
        # X11 only; harmless no-op when DISPLAY is unset.
        if shutil.which("xdotool") and os.environ.get("DISPLAY"):
            subprocess.run(
                ["xdotool", "search", "--name", title, "windowactivate"],
                capture_output=True, text=True, timeout=3, check=False,
            )

    def _watch_the_daemon(self) -> bool:
        """Follow what the daemon does while this window is open.

        The Speech page answers "which model is being used", and it answered
        it once, when the page was built. Saving a different model and
        pressing Restart therefore left the pill reading the old one against
        a daemon that had already moved on - the page contradicting the app
        it describes, on the one question it exists to settle. The same goes
        for the daemon's own status row and the pid in it.

        Two seconds rather than the dirty poll's 400ms, and only while the
        window is on screen: this reads the .env and the models directory,
        which is cheap but not free, and nothing here changes faster than a
        person can restart a daemon.
        """
        if not self.get_visible():
            return True
        try:
            model = self.backend.working_model()
            pid = restart.daemon_pid()
        except Exception as e:
            log(f"[SETTINGS] could not check on the daemon: {e}")
            return True

        if model != self._current_model:
            # What the radio says now, before the rebuild rewrites it from
            # _current_model. An unsaved choice is the user's, and a daemon
            # restarting underneath is no reason to throw it away.
            chosen = self._selected_model()
            picked = chosen if chosen != self._current_model else None
            self._current_model = model
            self._rebuild_speech_page(focus=False)
            if picked and picked in self._model_checks:
                self._model_checks[picked].set_active(True)

        if pid != self._daemon_pid:
            self._daemon_pid = pid
            self._apply_daemon_status()
        return True

    def _reload_from_disk(self) -> None:
        """Catch up with everything that changed since this window was built."""
        try:
            self.config = reload_config()
            self.backend = LocalBackend(self.config)
            self._current_model = self.backend.working_model()
        except Exception as e:
            # A window showing what it read at login beats no window at all.
            log(f"[SETTINGS] could not re-read the config: {e}")
            return
        self._rebuild_speech_page()
        # In the same breath, so the status row is right on the first frame
        # rather than two seconds into it: the daemon that built this window
        # need not be the one running now.
        self._daemon_pid = restart.daemon_pid()
        self._apply_daemon_status()
        # After the rebuild, which carries edits across rather than reloading:
        # right for a download that lands mid-edit, wrong here, where the
        # window has been sitting unseen and has no edits worth keeping.
        self._load()
        self._refresh_dirty()
        # Devices come and go while this window waits - a headset paired after
        # login is exactly the case someone opens this window to deal with.
        self._load_mics_in_background()

    # ---------------------------------------------------------------- layout
    def _build(self):
        self._toasts = Adw.ToastOverlay()
        self.set_content(self._toasts)

        toolbar = Adw.ToolbarView()
        self._toasts.set_child(toolbar)

        header = Adw.HeaderBar()
        # The host's GTK settings can impose a foreign decoration layout -
        # KDE's adds a dead "menu" chevron and drops min/max. Ask for the
        # ordinary set instead.
        header.set_decoration_layout(":minimize,maximize,close")
        toolbar.add_top_bar(header)

        self._stack = Adw.ViewStack()
        switcher = Adw.ViewSwitcher()
        switcher.set_stack(self._stack)
        switcher.set_policy(Adw.ViewSwitcherPolicy.WIDE)
        header.set_title_widget(switcher)

        toolbar.set_content(self._stack)

        # No permanent bar for Save at all.
        #
        # A footer meant a rule across the window with a mostly empty strip
        # under it and one small button adrift at the right end - furniture
        # that was there on every page, most of the time with nothing to do,
        # and a separator inset from both edges so it spanned neither the
        # window nor the content column. The header was no better: beside the
        # view switcher the button read as a fifth tab.
        #
        # So it appears where it is relevant and not before: a banner under
        # the header, the moment there is something to save, saying what is
        # pending and offering the one action. Nothing changed means nothing
        # on screen, and the page runs clean to the bottom edge.
        self._banner = Adw.Banner()
        self._banner.set_button_label("Save")
        self._banner.connect("button-clicked", lambda *_: self._on_save())
        self._banner.set_revealed(False)
        toolbar.add_top_bar(self._banner)

        # Ctrl+S regardless, because the banner is only reachable by mouse
        # and this is a window people close by keyboard.
        save_action = Gio.SimpleAction.new("save", None)
        save_action.connect("activate", lambda *_: self._on_save())
        self.add_action(save_action)
        app = self.get_application()
        if app is not None:
            app.set_accels_for_action("win.save", ["<Control>s"])

        builders = {
            "Speech": self._build_speech_page,
            "Hotkeys": self._build_plain_page,
            "Dictation": self._build_plain_page,
            "General": self._build_general_page,
        }
        for section in settings_def.SECTIONS:
            name = section.lower()
            # Box stays in the stack for the life of the window; only its
            # child PreferencesPage is swapped when Speech is rebuilt.
            host = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            host.set_hexpand(True)
            host.set_vexpand(True)
            page = builders[section](section)
            host.append(page)
            self._page_hosts[name] = host
            # Without an icon the view switcher draws the broken-image glyph.
            self._stack.add_titled_with_icon(
                host, name, section, SECTION_ICONS[section])

    def _add_field_groups(self, page, section: str, lead_rows=None):
        """Render a section as titled groups, expert rows behind an expander.

        Every page used to be one untitled column holding everything in its
        section, which read as a wall: Dictation put the microphone, the
        live-typing interval, VAD aggressiveness, frame size and three
        different timeouts in a single list of ten. The schema carries the
        grouping, so the order and the headings are the same everywhere and
        a new field cannot land somewhere arbitrary.
        """
        ordered: dict[str, list] = {}
        for field in settings_def.FIELDS:
            if field.section == section:
                ordered.setdefault(field.group, []).append(field)

        for name, fields in ordered.items():
            group = Adw.PreferencesGroup(title=name)
            description = settings_def.GROUP_HELP.get((section, name), "")
            if description:
                group.set_description(description)
            page.add(group)

            for row in (lead_rows or {}).get(name, ()):
                group.add(row)
            for field in fields:
                if not field.advanced:
                    group.add(self._build_row(field))
                    # Level + Test sit on their own row under Device, not as
                    # ComboRow suffixes. Reparenting a ComboRow that held a
                    # LevelBar and a Button through ViewStack rebuilds was
                    # access-violating on Windows (and SIGSEGV here) when the
                    # Speech page rotated the stack.
                    if field.key == "mic_device_index":
                        group.add(self._build_mic_meter_row())

            # Per group rather than one bin at the foot of the page: sample
            # rate belongs with the microphone, not with a watchdog timer.
            advanced = [field for field in fields if field.advanced]
            if advanced:
                expander = Adw.ExpanderRow(
                    title="Advanced", subtitle=settings_def.ADVANCED_HELP)
                for field in advanced:
                    expander.add_row(self._build_row(field))
                group.add(expander)

    def _build_plain_page(self, section: str):
        page = Adw.PreferencesPage()
        self._add_field_groups(page, section)
        return page

    def _build_general_page(self, section: str):
        page = Adw.PreferencesPage()

        # Inside the Daemon group, not floating in an untitled card above
        # Notifications. It is the state of the thing that group configures,
        # and two "Daemon" headings separated by unrelated settings read as
        # two different subjects.
        row = Adw.ActionRow(title="Status")
        icon = Gtk.Image()
        row.add_prefix(icon)
        self._status_row, self._status_icon = row, icon
        self._apply_daemon_status()

        self._add_field_groups(page, section, lead_rows={"Daemon": [row]})
        page.add(self._about_group())
        return page

    def _about_group(self) -> Adw.PreferencesGroup:
        """Which build this is, at the foot of the last page.

        Nowhere in the app said so. The tray menu says it too, but that is a
        menu that closes the moment you look away from it - somebody writing
        a bug report needs a version they can read and copy, and the settings
        window is where people already go looking for one.
        """
        group = Adw.PreferencesGroup(title="About")
        self._version_row = Adw.ActionRow(
            title="Version", subtitle=build_version())
        # Selectable in the sense that matters: a button that puts it on the
        # clipboard. Label text in a row is not selectable by dragging.
        copy = Gtk.Button(icon_name="edit-copy-symbolic",
                          tooltip_text="Copy version")
        copy.set_valign(Gtk.Align.CENTER)
        copy.add_css_class("flat")
        copy.connect("clicked", lambda *_: self._copy_version())
        self._version_row.add_suffix(copy)
        group.add(self._version_row)
        return group

    def _copy_version(self) -> None:
        """Put the version on the clipboard, and say that it went there.

        Through a content provider rather than Gdk.Clipboard.set(), which is
        a varargs C convenience with no binding: the introspected pair is
        what every GTK4 version here offers.
        """
        version = build_version()
        display = self.get_display()
        if display is not None:
            provider = Gdk.ContentProvider.new_for_value(
                GObject.Value(str, version))
            display.get_clipboard().set_content(provider)
        self._toast(f"Copied {version}")

    def _apply_daemon_status(self) -> None:
        """Say whether the daemon is running, as of now."""
        pid = self._daemon_pid
        self._status_icon.set_from_icon_name(STATUS_ICONS[0 if pid else 1])
        # Both dropped first: this runs again when the daemon comes and goes,
        # and a green tick left carrying the red class is how a row ends up
        # saying one thing and looking like another.
        self._status_icon.remove_css_class("daemon-ok")
        self._status_icon.remove_css_class("daemon-bad")
        self._status_icon.add_css_class("daemon-ok" if pid else "daemon-bad")
        self._status_row.set_subtitle(
            f"running (pid {pid})" if pid else "not running")

    def _build_speech_page(self, section: str):
        page = Adw.PreferencesPage()

        # Say which one is in use, in words. A checked radio is easy to miss
        # among five rows, and says nothing at all when none is checked -
        # which is what a machine with no model downloaded yet looks like.
        if self._current_model:
            in_use = f"Using {self._current_model.replace('ggml-', '')}. "
        else:
            in_use = "None yet - download one below. "
        model_group = Adw.PreferencesGroup(
            title="Speech model",
            description=in_use + "Bigger is slower and more accurate.")
        page.add(model_group)
        model_group.add(self._engine_row())

        group_leader = None
        for item in self.backend.model_inventory():
            name = item["name"]
            # No title or subtitle on the row: they are built below and put in
            # as a prefix instead.
            #
            # The badge belongs immediately after the name - it answers which
            # model to pick, so it has to be read as part of the name rather
            # than found at the far side of the row - and AdwActionRow has no
            # slot there. It was Pango markup for a while, which does reach
            # that spot but cannot be a pill: markup paints its background to
            # the text's logical extents and has neither padding nor a corner
            # radius, so it read as clipped no matter how much the run was
            # padded out. A real widget has both, and the only way to place
            # one beside the name is to own that part of the row.
            row = Adw.ActionRow()
            names = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
            names.set_valign(Gtk.Align.CENTER)
            line = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            line.append(Gtk.Label(label=name.replace("ggml-", ""), xalign=0.0))
            if item["recommended"]:
                pill = Gtk.Label(label="recommended")
                pill.add_css_class("model-pill")
                pill.add_css_class("pill-recommended")
                pill.set_valign(Gtk.Align.CENTER)
                line.append(pill)
            names.append(line)
            subtitle = Gtk.Label(
                label=f"{item['size_mb']} MB · {item['wants']}", xalign=0.0)
            subtitle.add_css_class("dim-label")
            subtitle.add_css_class("model-subtitle")
            names.append(subtitle)

            check = Gtk.CheckButton()
            if group_leader is None:
                group_leader = check
            else:
                check.set_group(group_leader)
            if name == self._current_model:
                check.set_active(True)
            if not item["installed"]:
                # Stay visible but refuse the choice: an insensitive radio is
                # so dim on dark theme it looks like the row has none.
                check.connect("toggled", self._guard_uninstalled)
            # One prefix holding both, in the order they are to appear.
            #
            # Two add_prefix calls put the text to the *left* of the radio and
            # left every radio at a different distance in: add_prefix prepends
            # rather than appends, and the prefix area is sized to what is in
            # it, so each row's radio landed wherever that row's name pushed
            # it. Packing them here settles the order without depending on
            # which end add_prefix works from, and one prefix per row is the
            # same width in every row, so the radios line up again.
            #
            # Nothing here expands. The row's own title box is empty and takes
            # the slack, which is what keeps the suffix pills against the
            # right-hand edge.
            content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,
                              spacing=12)
            content.append(check)
            content.append(names)
            row.add_prefix(content)
            if item["installed"]:
                row.set_activatable_widget(check)

            suffix = Gtk.Box(spacing=6)
            suffix.set_valign(Gtk.Align.CENTER)
            # Which model uses the GPU, and whether this machine will give it
            # one. Both halves matter: "GPU" alone on a machine running the
            # CPU engine is the reassurance that hid the whole problem.
            if item["gpu_only"]:
                pill = Gtk.Label(label="GPU" if item["accelerated"]
                                 else "needs GPU")
                pill.add_css_class("model-pill")
                pill.add_css_class("pill-gpu" if item["accelerated"]
                                   else "pill-warning")
                pill.set_tooltip_text(
                    "Runs on the NVIDIA GPU." if item["accelerated"] else
                    "Needs the GPU engine; on CPU, tens of seconds "
                    "a sentence.")
                suffix.append(pill)
            if item["current"]:
                pill = Gtk.Label(label="in use")
                pill.add_css_class("model-pill")
                pill.add_css_class("pill-current")
                suffix.append(pill)
            if not item["installed"]:
                button = Gtk.Button(label="Download")
                button.add_css_class("flat")
                button.connect("clicked",
                               lambda _b, m=name: self._start_download(m))
                self._download_buttons.append(button)
                suffix.append(button)
                bar = Gtk.ProgressBar()
                bar.add_css_class("model-progress")
                bar.set_visible(False)
                self._progress_bars[name] = bar
                suffix.append(bar)
            row.add_suffix(suffix)

            self._model_checks[name] = check
            model_group.add(row)

        self._add_field_groups(page, section)
        return page

    def _engine_row(self) -> Gtk.Widget:
        """What the models above will actually run on, and how to improve it.

        The page listed five models and never said which engine would run
        them, so a machine with an NVIDIA card could download the GPU model,
        run it on the bundled CPU engine, and look exactly like a machine
        doing the right thing while taking twenty seconds a sentence. The
        engine is the other half of the answer, so it goes above the models
        rather than in a diagnostics report nobody opens.
        """
        row = Adw.ActionRow(title="Speech engine")
        summary = self.backend.engine_summary()
        row.set_subtitle(summary)

        accelerated = self.backend.engine_is_gpu()
        icon = Gtk.Image.new_from_icon_name(
            STATUS_ICONS[0] if accelerated else STATUS_ICONS[1])
        icon.add_css_class("daemon-ok" if accelerated else "daemon-bad")
        row.add_prefix(icon)

        if self.backend.needs_gpu_upgrade():
            row.set_subtitle(
                summary + " - 1.6GB, and makes the larger models usable")
            button = Gtk.Button(label="Install GPU engine")
            button.add_css_class("suggested-action")
            button.set_valign(Gtk.Align.CENTER)
            button.connect("clicked", lambda _b: self._start_engine_download())
            self._download_buttons.append(button)
            row.add_suffix(button)
            bar = Gtk.ProgressBar()
            bar.add_css_class("model-progress")
            bar.set_visible(False)
            bar.set_valign(Gtk.Align.CENTER)
            self._progress_bars["engine"] = bar
            row.add_suffix(bar)
        return row

    def _start_engine_download(self):
        """Fetch the cuBLAS engine, keeping whatever model is already here.

        install() compares the engine on disk against the one this machine
        wants, so the bundled CPU build no longer counts as "an engine is
        present" and the cuBLAS one is fetched. Naming the model in use keeps
        it from re-fetching gigabytes of weights that are already here.
        """
        if self._working:
            return
        self._working = True
        for button in self._download_buttons:
            button.set_sensitive(False)
        bar = self._progress_bars.get("engine")
        if bar:
            bar.set_visible(True)

        model = self._current_model or self._selected_model() or None

        def work():
            try:
                ok = self.backend.install(model, progress=self._on_progress)
            except Exception as e:
                log(f"[SETTINGS] engine download failed: {e}")
                ok = False
            GLib.idle_add(self._engine_download_done, ok)

        threading.Thread(target=work, daemon=True,
                         name="whisper-flow-engine-download").start()

    def _engine_download_done(self, ok: bool):
        self._working = False
        for button in self._download_buttons:
            button.set_sensitive(True)
        if not ok:
            self._toast("Could not install the GPU engine - check the "
                        "connection.")
            return
        self._rebuild_speech_page()
        self._restart_daemon(toast_start="Installed. Restarting...",
                             toast_ok="Installed and restarted.",
                             toast_fail_prefix="Installed, but could not "
                                               "restart")

    # ------------------------------------------------------------------ rows
    def _build_row(self, field) -> Gtk.Widget:
        if field.kind == "bool":
            row = Adw.ActionRow(title=field.label)
            if field.help:
                row.set_subtitle(field.help)
            switch = Gtk.Switch(valign=Gtk.Align.CENTER)
            row.add_suffix(switch)
            row.set_activatable_widget(switch)
            self._rows[field.key] = switch
            return row

        if field.kind == "choice":
            is_mic = field.key == "mic_device_index"
            choices = (self._mic_choices() if is_mic else list(field.choices))
            row = Adw.ComboRow(title=field.label,
                               model=Gtk.StringList.new(choices))
            if is_mic:
                # Capture device names are long and all alike at the front -
                # "Microphone Array (AMD Audio Device) (Windows WASAPI...)" -
                # so the default factory, which ellipsizes each row to the
                # width of the closed combo, cut every one of them off at
                # exactly the point where they start to differ. The list is
                # unreadable and the setting unusable. A factory whose labels
                # do not ellipsize lets the popup take its natural width, and
                # the CSS floor covers the case where the popover declines to
                # grow past the row it hangs off.
                row.set_list_factory(_wide_list_factory(row))
                row.add_css_class("mic-row")
                # Device choice for an in-progress Test - never polled from
                # the paint timer (that freezes the combo popover).
                row.connect("notify::selected", self._on_mic_device_changed)
            if field.help:
                row.set_subtitle(field.help)
            self._rows[field.key] = row
            return row

        if field.kind in ("int", "float") and field.key in _SPIN:
            digits, step = _SPIN[field.key]
            adjustment = Gtk.Adjustment(
                value=field.minimum or 0, lower=field.minimum or 0,
                upper=field.maximum or 100, step_increment=step,
                page_increment=step * 10, page_size=0)
            row = Adw.SpinRow(adjustment=adjustment, climb_rate=step,
                              digits=digits, title=field.label)
            if field.help:
                row.set_subtitle(field.help)
            self._rows[field.key] = row
            return row

        if field.key.startswith("hotkey_"):
            # Capture, do not type: free text never recorded a chord as the
            # user pressed it, so the field looked broken. Click, hold the
            # combination, and the row shows what the daemon will bind.
            #
            # EntryRow has no subtitle (ActionRow/ComboRow do) - calling
            # set_subtitle here raised AttributeError and the whole settings
            # window never opened, so tray → Settings looked dead.
            row = Adw.EntryRow(title=field.label)
            row.set_editable(False)
            # EntryRow has no set_subtitle on many libadwaita builds
            # (ActionRow/ComboRow do). A bare call raised AttributeError and
            # the whole settings window never opened — tray → Settings looked
            # dead. Prefer subtitle when present; otherwise a tooltip.
            if hasattr(row, "set_subtitle"):
                row.set_subtitle("Click, then press the keys")
            else:
                row.set_tooltip_text("Click, then press the keys")
            self._wire_hotkey_capture(row)
            self._rows[field.key] = row
            return row

        row = (Adw.PasswordEntryRow(title=field.label)
               if field.kind == "password"
               else Adw.EntryRow(title=field.label))
        self._rows[field.key] = row
        return row

    def _wire_hotkey_capture(self, row: Adw.EntryRow) -> None:
        """Record a chord into the row as keys go down, not as typed text."""
        held: set[str] = set()

        def pressed(_controller, keyval, _keycode, _state):
            part = _gdk_key_to_hotkey_part(keyval)
            if part is None:
                return True             # swallow unknowns; do not type junk
            if part in ("escape", "backspace"):
                held.clear()
                row.set_text("")
                return True
            # First key after a full release starts a new combination.
            if not held:
                held.clear()
            held.add(part)
            row.set_text(settings_def.format_hotkey(held))
            return True                 # do not let EntryRow insert a letter

        def released(_controller, keyval, _keycode, _state):
            part = _gdk_key_to_hotkey_part(keyval)
            if part is not None:
                held.discard(part)
            # Text stays at the last full chord so modifiers-only shortcuts
            # (super+alt) survive letting go of every key.
            return True

        controller = Gtk.EventControllerKey()
        controller.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        controller.connect("key-pressed", pressed)
        controller.connect("key-released", released)
        row.add_controller(controller)

    def _mic_choices(self) -> list[str]:
        """Display strings for the microphone dropdown, default first.

        Deliberately does not enumerate anything. Opening PortAudio walks
        every host API on the machine - MME, DirectSound, WASAPI and WDM-KS,
        forty-odd devices between them on a laptop with a headset paired -
        and the window cannot be shown until it returns. The real list
        arrives from a thread once there is a window to put it in; until then
        this is enough for the configured device to be selected correctly.
        """
        self._mic_display = {"Default": ""}
        configured = self.config.mic_device_index
        if configured is not None:
            self._mic_display[f"{configured}: ..."] = str(configured)
        return list(self._mic_display)

    def _load_mics_in_background(self):
        """Fill the microphone dropdown once the window is up."""
        def work():
            devices = _list_input_devices()
            GLib.idle_add(self._apply_mics, devices)

        threading.Thread(target=work, daemon=True,
                         name="whisper-flow-mic-list").start()

    def _apply_mics(self, devices: list) -> bool:
        """Swap in the real device list, keeping whatever is selected.

        One idle callback for the whole swap, so the dirty poll never sees a
        dropdown mid-rebuild and offers to save a change nobody made.

        The selection is read off the row rather than out of `_current`.
        `_current` is the saved baseline the dirty poll diffs against, not
        live widget state, so restoring from it would silently undo a choice
        made while the enumeration was still running - and the reason that
        enumeration is on a thread at all is that it is slow enough for
        someone to get there first.
        """
        row = self._rows.get("mic_device_index")
        if row is None:
            return False
        # get_selected() answers GTK_INVALID_LIST_POSITION when nothing is
        # selected, and get_string() of that is None rather than an error.
        position = row.get_selected()
        display = (row.get_model().get_string(position)
                   if position < row.get_model().get_n_items() else None)
        selected = self._mic_display.get(display, "") if display else ""

        self._mic_display = {"Default": ""}
        for index, name in devices:
            self._mic_display[f"{index}: {name}"] = str(index)
        # A device that is configured but no longer present keeps its row,
        # or selecting it would silently fall back to the default and read
        # as a change the user made.
        if selected and selected not in self._mic_display.values():
            self._mic_display[f"{selected}: (not connected)"] = selected

        displays = list(self._mic_display)
        row.set_model(Gtk.StringList.new(displays))
        for i, display in enumerate(displays):
            if self._mic_display[display] == selected:
                row.set_selected(i)
                break
        return False

    # -------------------------------------------------------- mic meter
    def _build_mic_meter_row(self) -> Gtk.Widget:
        """Level bar + Test/Stop, under the Device combo.

        Built as its own ActionRow so the Device ComboRow stays a plain
        dropdown. Suffix widgets on ComboRow did not survive ViewStack
        page rotation on Windows.
        """
        row = Adw.ActionRow(title="Level")
        row.set_subtitle("Press Test and speak to check this device")
        meter = _MicLevelStrip()
        meter.set_tooltip_text("Microphone activity")
        row.add_suffix(meter)
        self._mic_meter = meter
        test = Gtk.Button(label="Test", valign=Gtk.Align.CENTER)
        test.add_css_class("suggested-action")
        test.set_tooltip_text(
            "Listen to this device and show activity on the meter")
        test.connect("clicked", self._on_mic_test_toggle)
        row.add_suffix(test)
        self._mic_test_button = test
        return row

    def _on_visible_for_mic_meter(self, *_args) -> None:
        """Drop an in-progress Test when the window is hidden."""
        if not self.get_visible() and self._mic_meter_active:
            self._stop_mic_test()

    def _on_mic_test_toggle(self, *_args) -> None:
        """Test starts capture; Stop releases it."""
        if self._mic_meter_active:
            self._stop_mic_test()
        else:
            self._start_mic_test()

    def _on_mic_device_changed(self, *_args) -> None:
        """Follow the Device dropdown while a Test is running."""
        if not self._mic_meter_active:
            return
        with self._mic_meter_lock:
            self._mic_want_device = self._selected_mic_device()

    def _start_mic_test(self) -> None:
        self._mic_meter_active = True
        if self._mic_test_button is not None:
            self._mic_test_button.set_label("Stop")
        # Publish the device immediately. Never re-read the ComboRow from
        # the paint timer - that freezes the device list popover.
        with self._mic_meter_lock:
            self._mic_want_device = self._selected_mic_device()
            self._mic_drawn_level = 0.0
            self._mic_meter_error = None
        if self._mic_meter is not None:
            self._mic_meter.set_level(0.0)
        self._ensure_mic_meter_thread()
        if not self._mic_meter_tick_id:
            self._mic_meter_tick_id = GLib.timeout_add(
                _MIC_METER_POLL_MS, self._tick_mic_meter)

    def _stop_mic_test(self) -> None:
        self._mic_meter_active = False
        if self._mic_meter_tick_id:
            GLib.source_remove(self._mic_meter_tick_id)
            self._mic_meter_tick_id = 0
        with self._mic_meter_lock:
            self._mic_want_device = _MIC_METER_OFF
            self._mic_drawn_level = 0.0
            self._mic_meter_error = None
        if self._mic_meter is not None:
            self._mic_meter.set_level(0.0)
        if self._mic_test_button is not None:
            self._mic_test_button.set_label("Test")

    def _ensure_mic_meter_thread(self) -> None:
        """One capture thread for the life of the window."""
        thread = self._mic_meter_thread
        if thread is not None and thread.is_alive():
            return
        self._mic_meter_stop.clear()
        self._mic_meter_thread = threading.Thread(
            target=self._mic_meter_loop, daemon=True,
            name="whisper-flow-mic-meter")
        self._mic_meter_thread.start()

    def _selected_mic_device(self):
        """Index the Device row is showing, or None for the platform default.

        Main-thread only: it reads the ComboRow. The capture thread never
        calls this; it reads the copy `_mic_want_device` published on Test
        and on selection-changed.
        """
        row = self._rows.get("mic_device_index")
        if row is None:
            return None
        position = row.get_selected()
        model = row.get_model()
        if model is None or position >= model.get_n_items():
            return None
        display = model.get_string(position)
        raw = self._mic_display.get(display, "") if display else ""
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _tick_mic_meter(self) -> bool:
        """Paint the latest level. Does not touch the Device ComboRow.

        Reading the combo from this timer while its popover was open
        re-entered GTK and froze the settings window.

        Returns False to disarm the GLib source when Test is no longer
        running.
        """
        if (self._mic_meter is None
                or not self._mic_meter_active
                or not self.get_visible()):
            self._mic_meter_tick_id = 0
            return False

        self._ensure_mic_meter_thread()
        with self._mic_meter_lock:
            level = self._mic_drawn_level
            error = self._mic_meter_error
            self._mic_meter_error = None
        self._mic_meter.set_level(level)
        if error:
            self._toast(error)
            self._stop_mic_test()
            return False
        return True

    def _mic_meter_loop(self) -> None:
        """Open the chosen input and publish RMS levels until asked to stop.

        Holds the stream only while a Test is running. Changing the Device
        dropdown swaps streams on the next iteration so the bar follows the
        selection, not the saved config - which is the whole point of
        listening before you save.
        """
        try:
            import pyaudio
            import numpy as np
        except ImportError:
            with self._mic_meter_lock:
                self._mic_meter_error = (
                    "Microphone test needs PyAudio - it is not installed.")
            return

        pa = None
        stream = None
        chunk = 0
        open_device = _MIC_METER_OFF
        peak = _MIC_METER_PEAK_FLOOR
        open_failures = 0
        try:
            try:
                with _suppress_alsa_for_meter():
                    pa = pyaudio.PyAudio()
            except Exception as e:
                log(f"[SETTINGS] mic meter could not open audio: {e}")
                with self._mic_meter_lock:
                    self._mic_meter_error = (
                        "Could not open the audio system for a mic test.")
                return

            while not self._mic_meter_stop.is_set():
                with self._mic_meter_lock:
                    want = self._mic_want_device

                if want == _MIC_METER_OFF:
                    if stream is not None:
                        _close_meter_stream(stream)
                        stream = None
                        open_device = _MIC_METER_OFF
                        chunk = 0
                    with self._mic_meter_lock:
                        self._mic_drawn_level = 0.0
                    peak = _MIC_METER_PEAK_FLOOR
                    open_failures = 0
                    # Idle wait: no capture while Test is off.
                    self._mic_meter_stop.wait(0.1)
                    continue

                if stream is None or open_device != want:
                    if stream is not None:
                        _close_meter_stream(stream)
                        stream = None
                    stream, chunk = _open_meter_stream(pa, want)
                    open_device = want if stream is not None else _MIC_METER_OFF
                    peak = _MIC_METER_PEAK_FLOOR
                    if stream is None:
                        open_failures += 1
                        with self._mic_meter_lock:
                            self._mic_drawn_level = 0.0
                            if open_failures >= 2:
                                self._mic_meter_error = (
                                    "Could not open this microphone - it may "
                                    "be in use or unavailable.")
                        # Back off so a missing device does not spin the CPU.
                        self._mic_meter_stop.wait(0.4)
                        continue
                    open_failures = 0

                try:
                    data = stream.read(chunk, exception_on_overflow=False)
                except Exception:
                    _close_meter_stream(stream)
                    stream = None
                    open_device = _MIC_METER_OFF
                    chunk = 0
                    with self._mic_meter_lock:
                        self._mic_drawn_level = 0.0
                    continue

                try:
                    samples = np.frombuffer(data, dtype=np.int16)
                    if samples.size < 1:
                        continue
                    # Peak-abs moves more visibly than pure RMS on quiet
                    # laptop mics (room tone RMS ~20 never clears a high floor).
                    loud = float(np.max(np.abs(samples.astype(np.float32))))
                    rms = float(np.sqrt(
                        np.mean(samples.astype(np.float32) ** 2)))
                    measure = max(loud * 0.7, rms)
                except Exception:
                    continue

                level, peak = _mic_level_from_rms(measure, peak)
                with self._mic_meter_lock:
                    self._mic_drawn_level = level
        finally:
            if stream is not None:
                _close_meter_stream(stream)
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

    # -------------------------------------------------------------- values
    def _load(self):
        """Fill every row from the running config."""
        self._current = {}
        for field in settings_def.FIELDS:
            value = getattr(self.config, field.key, None)
            row = self._rows[field.key]
            if field.kind == "bool":
                row.set_active(bool(value))
                self._current[field.key] = bool(value)
                continue
            if field.kind == "choice":
                raw = "" if value is None else str(value)
                model = row.get_model()
                for i in range(model.get_n_items()):
                    display = model.get_string(i)
                    candidate = (self._mic_display.get(display, display)
                                 if field.key == "mic_device_index"
                                 else display)
                    if candidate == raw:
                        row.set_selected(i)
                        break
                self._current[field.key] = raw
                continue
            if field.kind in ("int", "float") and field.key in _SPIN:
                row.set_value(float(value if value is not None
                                    else field.minimum or 0))
                self._current[field.key] = str(value)
                continue
            row.set_text("" if value is None else str(value))
            self._current[field.key] = "" if value is None else str(value)

    def _refresh_dirty(self) -> bool:
        """Keep Save honest about whether there is anything to save.

        An always-enabled Save button on a page with nothing pending says
        nothing, and the only way to find out was to press it and read a
        "Nothing changed" toast. Polled rather than wired to every widget:
        four kinds of row each announce a change differently, and reading
        two dozen widgets four times a second costs nothing measurable.
        """
        if self._working:
            return True
        # Nothing to keep honest while nobody can see it. A window built at
        # login and not yet asked for would otherwise diff every row against
        # the config twice a second for as long as it waits, which is the
        # kind of cost that makes prewarming not worth having.
        if not self.get_visible():
            return True
        try:
            changed = dict(settings_def.updates_from(self._values(),
                                                     self._current))
            model = self._selected_model()
            if model and model != self._current_model:
                changed["WHISPER_FLOW_MODEL_NAME"] = model
        except Exception:
            return True         # a half-built row must not stop the timer

        # The banner is the whole state: revealed means there is something to
        # save and says what, hidden means there is not and takes its own
        # space back with it. No disabled control to explain, and no strip of
        # empty furniture on a page where nothing has been touched.
        if changed:
            self._banner.set_title(
                f"{len(changed)} unsaved change"
                f"{'s' if len(changed) != 1 else ''}")
        self._banner.set_revealed(bool(changed))
        return True

    def _apply_values(self, values: dict) -> None:
        """Put values back into the rows, after a page has been rebuilt.

        The mirror of _values(). _load() reads the running config instead,
        which is right when the window opens and wrong here: a model
        download must not quietly undo edits made before it started.
        """
        for field in settings_def.FIELDS:
            if field.key not in values or field.key not in self._rows:
                continue
            row = self._rows[field.key]
            raw = values[field.key]
            if field.kind == "bool":
                row.set_active(bool(raw))
            elif field.kind == "choice":
                model = row.get_model()
                for i in range(model.get_n_items()):
                    display = model.get_string(i)
                    candidate = (self._mic_display.get(display, display)
                                 if field.key == "mic_device_index"
                                 else display)
                    if candidate == str(raw):
                        row.set_selected(i)
                        break
            elif field.kind in ("int", "float") and field.key in _SPIN:
                try:
                    row.set_value(float(raw))
                except (TypeError, ValueError):
                    pass        # leave the row at its own default
            else:
                row.set_text("" if raw is None else str(raw))

    def _values(self) -> dict:
        values = {}
        for field in settings_def.FIELDS:
            row = self._rows[field.key]
            if field.kind == "bool":
                values[field.key] = bool(row.get_active())
            elif field.kind == "choice":
                display = row.get_model().get_string(row.get_selected())
                values[field.key] = (self._mic_display.get(display, display)
                                     if field.key == "mic_device_index"
                                     else display)
            elif field.kind in ("int", "float") and field.key in _SPIN:
                number = row.get_value()
                values[field.key] = (str(int(number)) if field.kind == "int"
                                     else f"{number:g}")
            else:
                values[field.key] = row.get_text()
        return values

    def _selected_model(self) -> str:
        for name, check in self._model_checks.items():
            if check.get_active():
                return name
        return self._current_model or ""

    @staticmethod
    def _guard_uninstalled(check: Gtk.CheckButton):
        """Snap a radio on an uninstalled model straight back off."""
        if check.get_active():
            check.set_active(False)

    # --------------------------------------------------------------- actions
    def _toast(self, message: str, button: str | None = None,
               on_button=None):
        toast = Adw.Toast.new(message)
        toast.set_timeout(8)
        if button and on_button:
            toast.set_button_label(button)
            toast.connect("button-clicked", lambda *_: on_button())
        self._last_toast_title = message
        self._toasts.add_toast(toast)

    def _on_save(self):
        if self._working:
            return
        values = self._values()
        problem = settings_def.validate(values)
        if problem:
            self._toast(problem)
            return

        updates = settings_def.updates_from(values, self._current)
        model = self._selected_model()
        if model and model != self._current_model:
            updates["WHISPER_FLOW_MODEL_NAME"] = model
        if not updates:
            self._toast("Nothing changed")
            return

        env_path = Path(self.config.config_dir) / ".env"
        try:
            envfile.set_values(env_path, updates)
        except Exception as e:
            self._toast(f"Could not save: {e}")
            return
        log(f"[SETTINGS] saved {sorted(updates)} to {env_path}")

        self._current = values
        self._current_model = model or self._current_model
        self._banner.set_revealed(False)
        # Settings only take effect in a new process. Do not ask - restart.
        self._restart_daemon(toast_start="Saved. Restarting...",
                             toast_ok="Saved and restarted.",
                             toast_fail_prefix="Saved, but could not restart")

    def _on_restart(self):
        """Manual restart (still available if something else needs one)."""
        self._restart_daemon(toast_start="Restarting...",
                             toast_ok="Restarted.",
                             toast_fail_prefix="Could not restart")

    def _restart_daemon(self, toast_start: str, toast_ok: str,
                        toast_fail_prefix: str) -> None:
        """Stop and start the daemon; report progress via toasts.

        The toast is drawn first, then the restart is deferred a frame so
        the user sees "Saved. Restarting..." before any work that can block
        the pipe to the daemon. The restart itself runs off the main thread.
        """
        if self._working:
            return
        self._working = True
        # Mic test holds a capture stream; release it before the daemon
        # comes back up and wants the same device.
        if self._mic_meter_active:
            self._stop_mic_test()
        self._toast(toast_start)
        # One frame for the toast to map, then hand off to a worker.
        GLib.timeout_add(80, self._restart_daemon_begin,
                         toast_ok, toast_fail_prefix)

    def _restart_daemon_begin(self, toast_ok: str,
                              toast_fail_prefix: str) -> bool:
        def work():
            ok, detail = restart.restart_daemon()

            def done():
                self._working = False
                if ok:
                    self._toast(toast_ok)
                else:
                    self._toast(f"{toast_fail_prefix}: {detail}")

            GLib.idle_add(done)

        threading.Thread(target=work, daemon=True,
                         name="whisper-flow-restart").start()
        return False                    # timeout_add: run once

    def _start_download(self, model: str):
        if self._working:
            return
        self._working = True
        for button in self._download_buttons:
            button.set_sensitive(False)
        bar = self._progress_bars.get(model)
        if bar:
            bar.set_visible(True)

        def work():
            try:
                ok = self.backend.install(model, progress=self._on_progress)
            except Exception as e:
                log(f"[SETTINGS] download failed: {e}")
                ok = False
            GLib.idle_add(self._download_done, model, ok)

        threading.Thread(target=work, daemon=True,
                         name="whisper-flow-model-download").start()

    def _download_done(self, model: str, ok: bool):
        self._working = False
        for button in self._download_buttons:
            button.set_sensitive(True)
        if ok:
            # Rebuild so the new model gains a radio and loses the button.
            self._rebuild_speech_page()
            self._model_checks[model].set_active(True)
            self._toast(f"Got {model.replace('ggml-', '')} - Save to use it.")
        else:
            self._toast("Download failed - check the connection.")

    def _rebuild_speech_page(self, focus: bool = True):
        """Redraw the Speech page after a download changed what is on disk.

        Every row on this page is replaced by that, and the new ones start
        empty: the port read 0, "Run the speech engine locally" read off, and
        the server URL blank - none of which the user touched. _current still
        held the real values, so Save saw the difference as deliberate and
        would have written it. What stopped it was the port failing validation
        at 0, which is why this surfaced as a confusing complaint about a port
        nobody had edited rather than as local transcription silently turning
        itself off.

        Carried across rather than reloaded from config, because a download
        does not discard edits made before it started.

        `focus` is for the callers this follows a click on: a download or an
        engine install happens on this page and the result belongs in front of
        the person who pressed the button. The daemon changing model underneath
        is not a click, and yanking someone off the Hotkeys page to show them
        a pill they did not ask about is not an improvement.

        Speech lives inside a host box that never leaves the ViewStack, so
        rebuilding swaps only that page's contents. The old approach removed
        and re-added stack pages to fix tab order (add_* always appends), and
        reparenting pages that had already been shown access-violated the
        view switcher on Windows.
        """
        keep = self._values()
        # The old page's widgets are about to be dropped; holding references
        # to them means a later download re-enables buttons that are no
        # longer on screen, and writes progress into a bar nobody can see.
        self._download_buttons = []
        self._progress_bars = {}
        self._model_checks = {}

        host = self._page_hosts.get("speech")
        if host is None:
            return
        old = host.get_first_child()
        if old is not None:
            host.remove(old)
        host.append(self._build_speech_page("Speech"))

        if focus:
            self._stack.set_visible_child_name("speech")
        self._apply_values(keep)

    def _on_progress(self, stage: str, fraction: float):
        GLib.idle_add(self._apply_progress, fraction)

    def _apply_progress(self, fraction: float):
        for bar in self._progress_bars.values():
            if bar.get_visible():
                bar.set_fraction(fraction)

    def _open_config_folder(self):
        path = str(self.config.config_dir)
        try:
            subprocess.Popen(["xdg-open", path])
        except Exception:
            self._toast(f"Config folder: {path}")

    # ------------------------------------------------------------------ blur
    def _frost(self):
        """Switch to the translucent stylesheet, once there is blur behind it.

        Only ever called after a backdrop is confirmed: these surfaces are
        see-through, and over an opaque window they would show the desktop
        grey of nothing at all rather than frosted glass.
        """
        provider = Gtk.CssProvider()
        _load_css(provider, CSS_BLUR)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 1)
        self.add_css_class("blurred")

    def _on_realize(self, *_):
        """Frost the window where the platform offers a backdrop."""
        if sys.platform == "win32":
            self._realize_win32()
            return
        if not (os.environ.get("WAYLAND_DISPLAY")
                or os.environ.get("XDG_SESSION_TYPE") == "wayland"):
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
            self._blur = wayland_blur.enable_window_blur(
                wl_display, wl_surface, WIDTH, HEIGHT, CORNER_RADIUS,
                inset=1, active=True)
            if not self._blur:
                return
            self._frost()
            log("[SETTINGS] backdrop blur enabled")
        except Exception as e:
            log(f"[SETTINGS] blur unavailable: {e}")

    def _realize_win32(self):
        """Acrylic behind the window, through DWM.

        The pill could not use this - the backdrop is the window's rectangle
        and ignores any attempt to shape it - but a settings window is a
        rectangle, so it is precisely the intended case. GTK's transparent
        pixels let it through; nothing else is needed.

        Worth knowing when this looks flat: Windows drops every acrylic and
        Mica surface to a solid fallback colour under battery saver, and
        again in a remote session. Nothing here is wrong when that happens.
        """
        try:
            from . import blur_win
        except Exception as e:                  # not a Windows build
            log(f"[SETTINGS] no Windows backdrop support: {e}")
            return
        try:
            surface = self.get_surface()
            hwnd = blur_win.surface_handle(_gobject_pointer(surface))
            if not hwnd:
                log("[SETTINGS] no HWND; window stays opaque")
                return
            # Before the backdrop, and not conditional on it: the icon is
            # what the taskbar draws, and a window with no backdrop still has
            # a taskbar button.
            self._set_window_icon(hwnd)
            applied = blur_win.apply_backdrop(hwnd, transient=True)
            if not applied:
                log("[SETTINGS] backdrop unavailable; window stays opaque")
                return
            self._frost()
            # GTK reserves a transparent margin around the window for its own
            # drop shadow. DWM does not know about it and paints the backdrop
            # across the whole window rectangle, so the material appears as a
            # grey rectangle standing out past the rounded corners. Take the
            # margin away and the window rectangle is the window; DWM rounds
            # the corners itself, which is what DWMWCP_ROUND was for.
            provider = Gtk.CssProvider()
            _load_css(provider, CSS_WIN_BACKDROP)
            Gtk.StyleContext.add_provider_for_display(
                Gdk.Display.get_default(), provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 2)
            log(f"[SETTINGS] backdrop blur enabled ({applied})")
        except Exception as e:
            log(f"[SETTINGS] blur unavailable: {e}")

    @staticmethod
    def _set_window_icon(hwnd: int) -> None:
        """Put the app's mark on the taskbar button.

        GTK sets a window's icon by name, out of the icon theme, and the
        theme shipped with this build is Adwaita's symbolic set - which has
        no entry for this application. So the window carried no icon at all
        and the taskbar drew a blank sheet beside a tray showing a
        microphone.

        Drawn here rather than shipped as a file, from the same code as the
        tray and the executable, which is the only arrangement in which the
        three cannot drift apart. It is the small sizes only and it costs a
        few milliseconds; the file goes as soon as Windows has read it.
        """
        import tempfile

        try:
            from . import blur_win, icon
        except Exception as e:                  # not a Windows build
            log(f"[SETTINGS] no window icon: {e}")
            return
        path = None
        try:
            fd, path = tempfile.mkstemp(suffix=".ico", prefix="whisper-flow-")
            os.close(fd)
            icon.write_ico(path, sizes=icon.WINDOW_ICO_SIZES)
            if not blur_win.set_window_icon(hwnd, path):
                log("[SETTINGS] the window icon was refused")
        except Exception as e:
            log(f"[SETTINGS] could not set the window icon: {e}")
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def _retire_if_unseen(window: "SettingsWindow") -> bool:
    """Leave, unless there is a window on screen that someone is using.

    End of stream means the daemon has gone, and a window nobody has asked
    for goes with it - that is what stops a prewarmed one, which has no tray
    icon and nothing on screen, from being stranded as an invisible process
    after the app it belongs to has quit.

    A window that is up is a different thing entirely. Save restarts the
    daemon from this very window; taking the window away mid-toast would
    look like a freeze. So EOF alone keeps a visible window until the user
    closes it. Tray Exit is not EOF: it writes the ``quit`` command first.
    """
    if window.get_visible():
        log("[SETTINGS] the daemon has gone; this window stays until closed")
        return False
    os._exit(0)


def _quit_settings(window: "SettingsWindow") -> bool:
    """Exit because the tray (or parent) asked, even if the window is open."""
    log("[SETTINGS] quit requested")
    try:
        app = window.get_application()
        if app is not None:
            app.quit()
    except Exception:
        pass
    os._exit(0)


def _command_loop(window: "SettingsWindow") -> None:
    """Read the daemon's orders from stdin, for a window built in advance."""
    try:
        for line in sys.stdin:
            command = line.strip()
            if command == "show":
                # Stamped here rather than in the callback: the click is what
                # the user is timing, and idle_add may not run for a frame.
                at = time.time()
                GLib.idle_add(window.show_for_click, at)
            elif command == "quit":
                GLib.idle_add(_quit_settings, window)
                return
    except Exception:
        pass
    # Asked on the main loop, because that is the thread that knows whether
    # anything is on screen.
    GLib.idle_add(_retire_if_unseen, window)


def main() -> int:
    # This window runs as its own process and never turned logging on, so
    # every log() in it went to the ring buffer and nowhere else - including
    # the line saying whether backdrop blur was applied, and the one saying
    # why it was not. Diagnosing "the blur is missing" meant re-running it by
    # hand with the flag forced; the daemon's own setting answers it now.
    config = None
    try:
        config = Config()
        set_logging_enabled(config.logging_enabled)
    except Exception:
        pass                    # a bad config must not cost us the window

    # Started before the click, to be shown when it comes. Only when the
    # daemon says so: run by hand, stdin is inherited and reaches end of
    # stream at once, which would close the window as soon as it opened.
    resident = os.environ.get("WHISPER_FLOW_SETTINGS_RESIDENT") == "1"

    app = Adw.Application(application_id="dev.whisperflow.settings")
    # A raise inside activate is not an error anyone sees. GObject catches
    # whatever a Python signal handler throws, prints it to a stderr nobody
    # is reading, and returns; the application then runs its main loop with
    # no window, forever. That is precisely how this failed on Windows - the
    # daemon watches the exit code, saw a process still running, and reported
    # a window it had opened. Take the process down with a code instead, so
    # the daemon's "no window was shown" path can do its job.
    failure = []

    def build(app):
        try:
            window = SettingsWindow(application=app, config=config)
            _mark("window constructed")
            # present() only queues the window. The frame clock is what says
            # something is actually on screen, which is the number that
            # matches what the user is waiting for.
            window.connect(
                "map", lambda *_: _mark("window mapped", window._show_t0))
            if resident:
                # No window on screen and none wanted yet. hold() is what
                # keeps the process alive meanwhile: GtkApplication ends its
                # main loop when the last window goes, and a window that has
                # never been shown is close enough to none for that to bite.
                app.hold()
                # And released when the window is closed, or the process
                # would go on running with nothing on screen: the daemon
                # adopts a newly downloaded model when this process exits,
                # so one that never exits is one that never hands anything
                # over. False lets the close proceed as it normally would.
                window.connect("close-request",
                               lambda *_: (app.release(), False)[1])
                # Realized now, so the surface, its HWND, the CSS and the
                # first style pass are all done before the click. What is
                # left for the click is a map.
                window.realize()
                _mark("built and waiting for the click")
                threading.Thread(
                    target=_command_loop, args=(window,), daemon=True,
                    name="whisper-flow-settings-commands").start()
                return
            window.present()
        except Exception:
            import traceback
            failure.append(traceback.format_exc())
            log(f"[SETTINGS] the window could not be built:\n{failure[0]}")
            app.quit()

    app.connect("activate", build)
    status = app.run(sys.argv[:1])
    if failure:
        print(failure[0], file=sys.stderr, flush=True)
        return 1
    return status


if __name__ == "__main__":
    sys.exit(main())
