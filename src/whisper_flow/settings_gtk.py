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

Saving writes the .env file; nothing here applies the values. The daemon
adopts a changed speech model as soon as this window closes - it reads the
models directory, not the running config - and everything else takes effect
on the restart the toast offers afterwards.
"""

import ctypes
import os
import subprocess
import sys
import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from . import envfile, restart, settings_def, wayland_blur  # noqa: E402
from .backend import LocalBackend  # noqa: E402
from .config import Config  # noqa: E402
from .logging import log, set_logging_enabled  # noqa: E402

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
ICON_NAMES = tuple(SECTION_ICONS.values()) + STATUS_ICONS

CSS = b"""
.model-pill {
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 700;
}
.pill-recommended { background: #1b2c4a; color: #9ec2fc; }
.pill-current { background: #16301f; color: #4ade80; }
.pill-gpu { background: #1d3220; color: #86efac; }
.pill-warning { background: #3a2a12; color: #fbbf24; }
.model-progress { min-width: 110px; }
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


def _gobject_pointer(obj) -> int:
    """Address of the underlying GObject, via PyGObject's capsule."""
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ctypes.pythonapi.PyCapsule_GetPointer(obj.__gpointer__, None)


def _list_input_devices() -> list[tuple[int, str]]:
    """(index, name) for every capture device, or nothing if audio is down."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            devices = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append((i, str(info.get("name", f"device {i}"))))
            return devices
        finally:
            pa.terminate()
    except Exception as e:
        log(f"[SETTINGS] could not list microphones: {e}")
        return []


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


class SettingsWindow(Adw.ApplicationWindow):
    """The preferences window itself."""

    def __init__(self, application=None):
        super().__init__(title="WhisperFlow settings")
        if application is not None:
            # Registered, or closing the window leaves the process running.
            self.set_application(application)
        self.config = Config()
        self.backend = LocalBackend(self.config)
        self._current_model = self.backend.working_model()
        self._working = False
        self._rows: dict[str, Gtk.Widget] = {}
        self._mic_display: dict[str, str] = {}
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
        self._load()
        self._refresh_dirty()
        self._load_mics_in_background()
        GLib.timeout_add(400, self._refresh_dirty)
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
            page = builders[section](section)
            # Without an icon the view switcher draws the broken-image glyph.
            self._stack.add_titled_with_icon(
                page, section.lower(), section, SECTION_ICONS[section])

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
        pid = restart.daemon_pid()
        icon = Gtk.Image.new_from_icon_name(STATUS_ICONS[0 if pid else 1])
        icon.add_css_class("daemon-ok" if pid else "daemon-bad")
        row.add_prefix(icon)
        row.set_subtitle(f"running (pid {pid})" if pid else
                         "not running - dictation will not work")

        self._add_field_groups(page, section, lead_rows={"Daemon": [row]})
        return page

    def _build_speech_page(self, section: str):
        page = Adw.PreferencesPage()

        # Say which one is in use, in words. A checked radio is easy to miss
        # among five rows, and says nothing at all when none is checked -
        # which is what a machine with no model downloaded yet looks like.
        if self._current_model:
            in_use = f"Using {self._current_model.replace('ggml-', '')}. "
        else:
            in_use = "No model on this machine yet - download one below. "
        model_group = Adw.PreferencesGroup(
            title="Speech model",
            description=in_use + "Bigger is more accurate and slower; "
                                 "the recommendation is sized to this machine.")
        page.add(model_group)
        model_group.add(self._engine_row())

        group_leader = None
        for item in self.backend.model_inventory():
            name = item["name"]
            row = Adw.ActionRow(
                title=name.replace("ggml-", ""),
                subtitle=f"{item['size_mb']} MB · {item['wants']}",
            )
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
            row.add_prefix(check)
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
                    "This model needs the GPU engine. On the CPU engine it "
                    "takes tens of seconds per sentence.")
                suffix.append(pill)
            if item["recommended"]:
                pill = Gtk.Label(label="recommended")
                pill.add_css_class("model-pill")
                pill.add_css_class("pill-recommended")
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
                summary + " - the GPU engine is a 1.6GB download and makes "
                          "the larger models usable")
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
            self._toast("Could not install the GPU engine. "
                        "Check the connection and try again.")
            return
        self._rebuild_speech_page()
        self._toast("GPU engine installed - restart the daemon to use it.",
                    button="Restart now", on_button=self._on_restart)

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
            choices = (self._mic_choices()
                       if field.key == "mic_device_index"
                       else list(field.choices))
            row = Adw.ComboRow(title=field.label,
                               model=Gtk.StringList.new(choices))
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

        row = (Adw.PasswordEntryRow(title=field.label)
               if field.kind == "password"
               else Adw.EntryRow(title=field.label))
        self._rows[field.key] = row
        return row

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
                f"{'s' if len(changed) != 1 else ''} - applies on restart")
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
        self._toast(
            "Saved. A new model applies when this window closes; "
            "everything else needs a restart.",
            button="Restart now", on_button=self._on_restart)

    def _on_restart(self):
        if self._working:
            return
        self._working = True
        self._toast("Restarting the daemon...")

        def work():
            ok, detail = restart.restart_daemon()
            GLib.idle_add(
                lambda: self._toast("Restarted." if ok
                                    else f"Could not restart: {detail}"))
            self._working = False

        threading.Thread(target=work, daemon=True,
                         name="whisper-flow-restart").start()

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
            self._toast(f"Downloaded {model.replace('ggml-', '')} - "
                        f"Save to use it.")
        else:
            self._toast("Download failed. Check the connection and try again.")

    def _rebuild_speech_page(self):
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
        """
        keep = self._values()
        # The old page's widgets are about to be dropped; holding references
        # to them means a later download re-enables buttons that are no
        # longer on screen, and writes progress into a bar nobody can see.
        self._download_buttons = []
        self._progress_bars = {}
        self._model_checks = {}
        self._stack.remove(self._stack.get_child_by_name("speech"))
        page = self._build_speech_page("Speech")
        # With the icon, as _build adds it: the view switcher draws the
        # broken-image glyph for a page that has none.
        self._stack.add_titled_with_icon(
            page, "speech", "Speech", SECTION_ICONS["Speech"])
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


def main() -> int:
    # This window runs as its own process and never turned logging on, so
    # every log() in it went to the ring buffer and nowhere else - including
    # the line saying whether backdrop blur was applied, and the one saying
    # why it was not. Diagnosing "the blur is missing" meant re-running it by
    # hand with the flag forced; the daemon's own setting answers it now.
    try:
        set_logging_enabled(Config().logging_enabled)
    except Exception:
        pass                    # a bad config must not cost us the window

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
            SettingsWindow(application=app).present()
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
