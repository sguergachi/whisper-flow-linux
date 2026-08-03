"""The settings window on Linux: GTK4 + libadwaita, with real backdrop blur.

Runs as its own process (``python -m whisper_flow.settings_gtk``), like every
other window this app shows - the daemon's pystray loop and GTK each demand
their own main thread. Windows uses the tkinter window in settings_ui.py
instead, the same split as the HUD.

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
from gi.repository import Adw, Gdk, GLib, Gtk, Pango  # noqa: E402

from . import envfile, restart, settings_def, wayland_blur  # noqa: E402
from .backend import LocalBackend  # noqa: E402
from .config import Config  # noqa: E402
from .logging import log, set_logging_enabled  # noqa: E402

WIDTH, HEIGHT = 760, 820
CORNER_RADIUS = 12      # Adwaita's window corner radius
# What AdwPreferencesPage clamps its own content to. The action bar uses the
# same number so Save sits on the right edge of the cards above it instead of
# out in the window margin, lined up with nothing.
CONTENT_WIDTH = 600

CSS = b"""
.model-pill {
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 700;
}
.pill-recommended { background: #1b2c4a; color: #9ec2fc; }
.pill-current { background: #16301f; color: #4ade80; }
.model-progress { min-width: 110px; }
.save-bar { border-top: 1px solid rgba(255, 255, 255, 0.08); }
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

/* One hairline separating the action bar from the page it commits, and no
   filled strip: the frost should run unbroken to the bottom edge. */
window.blurred .save-bar { background-color: transparent; }
"""


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
        provider.load_from_data(CSS)
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
        GLib.timeout_add(400, self._refresh_dirty)
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

        # Save lives on its own bar at the foot of the window, not packed
        # into the header beside the view switcher. There it sat immediately
        # to the right of the last tab, at the same size, on the same strip -
        # so it read as a fifth tab rather than the action that commits the
        # page. Navigation along the top, the action along the bottom.
        #
        # Clamped to the same column as the cards. Left to span the window it
        # pinned the button to the far corner and the hint to the far edge,
        # aligned with nothing and hard against the rounded corner, with the
        # width of the window between them.
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        # AdwPreferencesPage insets its content inside the same clamp; without
        # matching it the button overhangs the right edge of the cards by
        # about ten pixels, which is exactly the sort of near-miss that reads
        # as sloppy without being obvious enough to name.
        row.set_margin_start(10)
        row.set_margin_end(10)

        self._save_hint = Gtk.Label(xalign=0.0)
        self._save_hint.add_css_class("dim-label")
        self._save_hint.set_hexpand(True)
        self._save_hint.set_ellipsize(Pango.EllipsizeMode.END)
        row.append(self._save_hint)

        # No suggested-action here: _refresh_dirty owns that class, and adding
        # it up front means the window flashes an accented button for the
        # frame before the first check runs.
        self.save_button = Gtk.Button(label="Save")
        self.save_button.add_css_class("pill")
        self.save_button.connect("clicked", lambda *_: self._on_save())
        row.append(self.save_button)

        clamp = Adw.Clamp(maximum_size=CONTENT_WIDTH)
        clamp.set_child(row)
        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        actions.add_css_class("save-bar")
        actions.append(clamp)
        clamp.set_hexpand(True)
        actions.set_margin_start(12)
        actions.set_margin_end(12)
        actions.set_margin_top(12)
        actions.set_margin_bottom(16)
        toolbar.add_bottom_bar(actions)

        builders = {
            "Speech": self._build_speech_page,
            "Hotkeys": self._build_plain_page,
            "Dictation": self._build_plain_page,
            "General": self._build_general_page,
        }
        icons = {
            "Speech": "audio-speakers-symbolic",
            "Hotkeys": "input-keyboard-symbolic",
            "Dictation": "audio-input-microphone-symbolic",
            "General": "emblem-system-symbolic",
        }
        for section in settings_def.SECTIONS:
            page = builders[section](section)
            # Without an icon the view switcher draws the broken-image glyph.
            self._stack.add_titled_with_icon(
                page, section.lower(), section, icons[section])

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
        icon = Gtk.Image.new_from_icon_name(
            "emblem-ok-symbolic" if pid else "dialog-warning-symbolic")
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
        """Display strings for the microphone dropdown, default first."""
        self._mic_display = {"Default": ""}
        for index, name in _list_input_devices():
            self._mic_display[f"{index}: {name}"] = str(index)
        return list(self._mic_display)

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

        # The accent goes on only when pressing it would do something. An
        # insensitive suggested-action button renders as a muddy blue-grey
        # slab with grey text on it, which reads as broken rather than as
        # "nothing to save" - and it is the one control on the page, sitting
        # by itself, so there is nothing around it to read it against.
        dirty = bool(changed)
        self.save_button.set_sensitive(dirty)
        if dirty:
            self.save_button.add_css_class("suggested-action")
        else:
            self.save_button.remove_css_class("suggested-action")
        self._save_hint.set_text(
            f"{len(changed)} unsaved change"
            f"{'s' if len(changed) != 1 else ''} - applies on restart"
            if changed else "")
        return True

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
            self._stack.remove(self._stack.get_child_by_name("speech"))
            page = self._build_speech_page("Speech")
            self._stack.add_titled(page, "speech", "Speech")
            self._stack.set_visible_child_name("speech")
            self._model_checks[model].set_active(True)
            self._toast(f"Downloaded {model.replace('ggml-', '')} - "
                        f"Save to use it.")
        else:
            self._toast("Download failed. Check the connection and try again.")

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
    def _on_realize(self, *_):
        """Frost the window where the compositor offers backdrop blur."""
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
            provider = Gtk.CssProvider()
            provider.load_from_data(CSS_BLUR)
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 1)
            self.add_css_class("blurred")
            log("[SETTINGS] backdrop blur enabled")
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
    app.connect(
        "activate", lambda app: SettingsWindow(application=app).present())
    return app.run(sys.argv[:1])


if __name__ == "__main__":
    sys.exit(main())
