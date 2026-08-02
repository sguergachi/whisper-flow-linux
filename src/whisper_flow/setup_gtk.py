"""First-run setup window: one button to get a working speech model.

Runs as its own process, launched by the daemon with --setup (or
``python -m whisper_flow.setup_gtk``), because the daemon's pystray loop
and GTK each demand their own main thread. Same design language as the
settings window - libadwaita everywhere, one window toolkit in the app.

What it does is deliberately small: report what the machine can run, download
the matching engine and model behind one button, and record the choice. The
daemon re-checks the backend when this process exits, so the new model is in
use without a restart.
"""

import sys
import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from . import envfile  # noqa: E402
from .backend import (  # noqa: E402
    MODELS,
    LocalBackend,
    detect_accelerator,
    recommended_model,
    usable_cores,
)
from .config import Config  # noqa: E402
from .logging import log  # noqa: E402


class SetupWindow(Adw.ApplicationWindow):
    def __init__(self, application=None):
        super().__init__(title="WhisperFlow setup")
        if application is not None:
            self.set_application(application)
        self.config = Config()
        self.backend = LocalBackend(self.config)
        self.accelerator = detect_accelerator()
        self.has_gpu = self.accelerator.startswith("cuda")
        self.model = recommended_model(self.accelerator)
        self.size_mb, self.wants = MODELS.get(self.model, (0, ""))
        # Nothing to fetch when the recommended model already shipped.
        self.already_have = self.backend.model_path(self.model).exists()
        self._working = False
        self._done = False

        Adw.StyleManager.get_default().set_color_scheme(
            Adw.ColorScheme.FORCE_DARK)
        self.set_default_size(460, -1)
        self.set_resizable(False)
        self.connect("close-request", self._on_close_request)

        header = Adw.HeaderBar()
        header.set_show_end_title_buttons(True)
        header.set_decoration_layout("close")

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        box.append(header)

        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        body.set_margin_top(18)
        body.set_margin_bottom(20)
        body.set_margin_start(26)
        body.set_margin_end(26)
        box.append(body)
        self.set_content(box)

        icon = Gtk.Image.new_from_icon_name("audio-input-microphone-symbolic")
        icon.set_pixel_size(48)
        body.append(icon)

        headline, detail = self._copy()
        title = Gtk.Label(label=headline)
        title.add_css_class("title-2")
        body.append(title)
        subtitle = Gtk.Label(label=detail)
        subtitle.add_css_class("dim-label")
        subtitle.set_wrap(True)
        subtitle.set_justify(Gtk.Justification.CENTER)
        body.append(subtitle)

        group = Adw.PreferencesGroup()
        body.append(group)
        rows = (
            ("Model", self.model.replace("ggml-", "")),
            ("Download", "already installed" if self.already_have
             else f"{self.size_mb} MB" if self.size_mb else "unknown"),
            ("Uses", self.wants or "-"),
            ("Runs on", "GPU (CUDA)" if self.has_gpu
             else f"CPU, {usable_cores()} threads"),
        )
        for label, value in rows:
            row = Adw.ActionRow(title=label)
            row.add_suffix(Gtk.Label(label=value, valign=Gtk.Align.CENTER))
            group.add(row)

        self.status = Gtk.Label(label=" ")
        self.status.add_css_class("dim-label")
        body.append(self.status)

        self.progress = Gtk.ProgressBar()
        self.progress.set_visible(False)
        body.append(self.progress)

        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        buttons.set_halign(Gtk.Align.END)
        body.append(buttons)

        self.dismiss = Gtk.Button(label="Not now")
        self.dismiss.connect("clicked", lambda *_: self._on_close())
        buttons.append(self.dismiss)

        self.action = Gtk.Button(label=self._action_label())
        self.action.add_css_class("suggested-action")
        self.action.connect("clicked", lambda *_: self._on_action())
        buttons.append(self.action)

    # ------------------------------------------------------------------ copy
    def _copy(self) -> tuple[str, str]:
        short = self.model.replace("ggml-", "")
        if self.already_have:
            return ("Everything is ready.",
                    "The right model for this machine is already installed. "
                    "Hold your dictation hotkey and start talking.")
        if self.has_gpu:
            return ("An NVIDIA GPU was found.",
                    f"WhisperFlow can run {short} on it: much more accurate "
                    f"than the model included with the app, and still faster "
                    f"than you can speak.")
        # No CUDA. Say plainly why the GPU is not being used, because
        # "it is only using my CPU" otherwise reads as a bug.
        return ("Running on the processor.",
                f"whisper.cpp only publishes GPU builds for NVIDIA cards, so "
                f"integrated graphics cannot be used. {short} is sized to "
                f"this machine's CPU so it keeps up with speech.")

    def _action_label(self) -> str:
        if self.already_have:
            return "Done"
        return f"Download ({self.size_mb} MB)" if self.size_mb else "Download"

    # --------------------------------------------------------------- actions
    def _set_status(self, message: str, css: str = "dim-label"):
        def apply():
            for cls in ("dim-label", "success", "error"):
                self.status.remove_css_class(cls)
            self.status.add_css_class(css)
            self.status.set_label(message)
        GLib.idle_add(apply)

    def _on_action(self):
        if self._working:
            return
        if self._done or self.already_have:
            self._on_close()
            return

        self._working = True
        self.action.set_sensitive(False)
        self.action.set_label("Downloading...")
        self.dismiss.set_sensitive(False)
        self.progress.set_visible(True)
        self._set_status("Starting...")
        threading.Thread(target=self._download_worker, daemon=True,
                         name="whisper-flow-setup-download").start()

    def _download_worker(self):
        try:
            ok = self.backend.install(self.model, force_download=True,
                                      progress=self._on_progress)
        except Exception as e:                      # install swallows most
            log(f"[SETUP] download failed: {e}")
            ok = False

        if ok:
            self._save_choice()
        GLib.idle_add(self._finished if ok else self._failed)

    def _on_progress(self, stage: str, fraction: float):
        def apply():
            if self._done:
                return          # never let a late tick overwrite the result
            self.progress.set_fraction(fraction)
            label = "engine" if stage == "engine" else "model"
            self._set_status(
                f"Downloading the speech {label}... {int(fraction * 100)}%")
        GLib.idle_add(apply)

    def _save_choice(self):
        """Persist the model so the daemon picks it up when this exits."""
        try:
            envfile.set_values(Path(self.config.config_dir) / ".env",
                               {"WHISPER_FLOW_MODEL_NAME": self.model})
        except Exception as e:
            log(f"[SETUP] could not save the model choice: {e}")
            self._set_status(
                f"Downloaded, but the setting could not be saved: {e}",
                "error")

    def _finished(self):
        self._working = False
        self._done = True
        self.progress.set_fraction(1.0)
        where = "the GPU" if self.has_gpu else "your processor"
        self._set_status(f"Ready. WhisperFlow will transcribe on {where}.",
                         "success")
        self.action.set_sensitive(True)
        self.action.set_label("Done")
        self.dismiss.set_visible(False)

    def _failed(self):
        self._working = False
        if not self.status.get_label().startswith("Could not"):
            self._set_status("Could not download it. Check the connection "
                             "and try again.", "error")
        self.action.set_sensitive(True)
        self.action.set_label("Try again")
        self.dismiss.set_sensitive(True)

    def _on_close(self):
        if self._working:
            return                      # a half-finished download helps nobody
        self.backend.mark_setup_seen()
        self.close()

    def _on_close_request(self, *_):
        if self._working:
            return True                 # swallow the close
        self.backend.mark_setup_seen()
        return False


def main() -> int:
    app = Adw.Application(application_id="dev.whisperflow.setup")
    app.connect("activate",
                lambda app: SetupWindow(application=app).present())
    return app.run(sys.argv[:1])


if __name__ == "__main__":
    sys.exit(main())
