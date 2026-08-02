"""The settings window: every knob in one place, plus model management.

Runs as its own process, launched by the tray app with --settings (or
``python -m whisper_flow.settings_ui``), because the daemon's pystray loop
and Tk each demand their own main thread.

Saving writes the .env file; nothing in this process applies the values.
The daemon adopts a changed speech model as soon as this window closes - it
reads the models directory, not the running config - and everything else
takes effect on the restart the window offers afterwards.
"""

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

if __package__ in (None, ""):        # launched as a script by the frozen build
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_flow import envfile, restart, settings_def  # noqa: E402
from whisper_flow.backend import LocalBackend  # noqa: E402
from whisper_flow.config import Config  # noqa: E402
from whisper_flow.logging import log  # noqa: E402
from whisper_flow.setup_ui import (  # noqa: E402
    ACCENT,
    ACCENT_ACTIVE,
    BAD,
    BG,
    FG,
    GOOD,
    MUTED,
    PANEL,
)


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


class SettingsWindow:
    def __init__(self):
        self.config = Config()
        self.backend = LocalBackend(self.config)
        self._current_model = self.backend.working_model()
        self._working = False          # a download is in flight
        self._saved_since_restart = False

        self.root = tk.Tk()
        self.root.title("WhisperFlow settings")
        self.root.configure(bg=BG)
        self.root.minsize(620, 640)
        self._style()
        self._vars: dict[str, tk.Variable] = {}
        self._mic_display: dict[str, str] = {}   # display -> raw value
        self._build()
        self._load()
        self.root.attributes("-topmost", True)
        self.root.after(400, lambda: self.root.attributes("-topmost", False))

    # ------------------------------------------------------------- styling
    def _style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED,
                        padding=(16, 8), font=("Segoe UI", 9))
        style.map("TNotebook.Tab", background=[("selected", BG)],
                  foreground=[("selected", FG)])
        style.configure(
            "Settings.Horizontal.TProgressbar", troughcolor=PANEL,
            background=ACCENT, bordercolor=PANEL, lightcolor=ACCENT,
            darkcolor=ACCENT, thickness=6)
        # A read-only combobox ignores configure() colours; they only take
        # through map(), or the field stays white with faint text on it.
        style.configure("TCombobox", bordercolor=PANEL, arrowcolor=MUTED)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL)],
            background=[("readonly", PANEL)],
            foreground=[("readonly", FG)],
            selectbackground=[("readonly", PANEL)],
            selectforeground=[("readonly", FG)],
        )
        for option, value in (
            ("background", PANEL), ("foreground", FG),
            ("selectBackground", ACCENT), ("selectForeground", "white"),
        ):
            self.root.option_add(f"*TCombobox*Listbox.{option}", value)

    # ---------------------------------------------------------------- layout
    def _build(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self._models_frame = None
        for section in settings_def.SECTIONS:
            tab = tk.Frame(notebook, bg=BG)
            notebook.add(tab, text=f" {section} ")
            body = tk.Frame(tab, bg=BG)
            body.pack(fill="both", expand=True, padx=22, pady=18)
            if section == "Speech":
                self._build_model_section(body)
            if section == "Hotkeys":
                tk.Label(
                    body,
                    text="Key names joined with '+', e.g. super+alt or "
                         "ctrl+shift+space. Applies on restart.",
                    bg=BG, fg=MUTED, justify="left",
                    font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 12))
            if section == "General":
                self._build_daemon_status(body)
            for field in settings_def.FIELDS:
                if field.section == section:
                    self._build_field(body, field)

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=22, pady=(10, 16))

        self.status = tk.Label(footer, text="", bg=BG, fg=MUTED, anchor="w",
                               font=("Segoe UI", 9))
        self.status.pack(fill="x", pady=(0, 8))

        slot = tk.Frame(footer, bg=BG, height=14)
        slot.pack(fill="x", pady=(0, 10))
        slot.pack_propagate(False)
        self.progress = ttk.Progressbar(
            slot, style="Settings.Horizontal.TProgressbar",
            mode="determinate", maximum=100)

        buttons = tk.Frame(footer, bg=BG)
        buttons.pack(fill="x")
        self._button(buttons, "Open config folder", self._open_config_folder,
                   quiet=True).pack(side="left")
        self.restart_button = self._button(
            buttons, "Restart now", self._on_restart)
        self.save_button = self._button(buttons, "Save", self._on_save,
                                        primary=True)
        self.save_button.pack(side="right")
        self._button(buttons, "Close", self.root.destroy,
                     quiet=True).pack(side="right", padx=(0, 6))

    def _button(self, parent, text, command, primary=False, quiet=False):
        bg, fg = (ACCENT, "white") if primary else (BG, MUTED)
        return tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=fg, activebackground=ACCENT_ACTIVE if primary else BG,
            activeforeground="white" if primary else FG,
            disabledforeground="#555a61",
            relief="flat", bd=0, highlightthickness=0, cursor="hand2",
            font=("Segoe UI", 10, "bold") if primary else ("Segoe UI", 10),
            padx=16 if primary else 12, pady=8,
        )

    def _build_daemon_status(self, parent):
        pid = restart.daemon_pid()
        text = f"Daemon is running (pid {pid})" if pid else \
            "Daemon is not running - dictation will not work until it starts"
        tk.Label(parent, text=text, bg=BG,
                 fg=GOOD if pid else BAD,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 12))

    def _build_model_section(self, parent):
        tk.Label(parent, text="Speech model", bg=BG, fg=FG,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(
            parent,
            text="What listens to your voice. Bigger is more accurate and "
                 "slower; the recommendation is sized to this machine.",
            bg=BG, fg=MUTED, justify="left",
            font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))
        self._models_frame = tk.Frame(parent, bg=PANEL)
        self._models_frame.pack(fill="x", pady=(0, 16))
        self._model_var = tk.StringVar(value=self._current_model or "")
        self._build_model_rows()
        tk.Frame(parent, bg=PANEL, height=1).pack(fill="x", pady=(2, 14))

    def _build_model_rows(self):
        frame = self._models_frame
        for child in frame.winfo_children():
            child.destroy()
        for row in self.backend.model_inventory():
            line = tk.Frame(frame, bg=PANEL)
            line.pack(fill="x", padx=14, pady=6)
            name = row["name"]
            radio = tk.Radiobutton(
                line, variable=self._model_var, value=name,
                bg=PANEL, fg=FG, selectcolor=PANEL,
                activebackground=PANEL,
                state="normal" if row["installed"] else "disabled",
            )
            radio.pack(side="left")
            label = name.replace("ggml-", "")
            badges = ""
            if row["recommended"]:
                badges += "  · recommended"
            if row["current"]:
                badges += "  · in use"
            tk.Label(line, text=label + badges, bg=PANEL, fg=FG,
                     font=("Segoe UI", 9)).pack(side="left")
            right = tk.Frame(line, bg=PANEL)
            right.pack(side="right")
            tk.Label(right, text=f"{row['size_mb']} MB, {row['wants']}",
                     bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
            if not row["installed"]:
                tk.Button(
                    right, text="Download",
                    command=lambda m=name: self._start_download(m),
                    bg=PANEL, fg=ACCENT, activebackground=PANEL,
                    activeforeground=ACCENT_ACTIVE, relief="flat", bd=0,
                    highlightthickness=0, cursor="hand2",
                    font=("Segoe UI", 9, "bold"), padx=8,
                ).pack(side="left", padx=(8, 0))

    def _build_field(self, parent, field):
        if field.kind == "bool":
            var = tk.BooleanVar()
            self._vars[field.key] = var
            tk.Checkbutton(
                parent, text=field.label, variable=var,
                bg=BG, fg=FG, selectcolor=PANEL, activebackground=BG,
                activeforeground=FG, font=("Segoe UI", 9),
            ).pack(anchor="w", pady=4)
            return

        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=4)
        tk.Label(row, text=field.label, bg=BG, fg=MUTED, width=30,
                 anchor="w", font=("Segoe UI", 9)).pack(side="left")

        var = tk.StringVar()
        self._vars[field.key] = var
        if field.kind == "choice":
            choices = list(field.choices)
            if field.key == "mic_device_index":
                choices = self._mic_choices()
            widget = ttk.Combobox(row, textvariable=var, values=choices,
                                  state="readonly", width=28)
        else:
            widget = tk.Entry(
                row, textvariable=var, bg=PANEL, fg=FG, insertbackground=FG,
                relief="flat", highlightthickness=1,
                highlightcolor=ACCENT, highlightbackground=PANEL,
                show="•" if field.kind == "password" else "",
                font=("Segoe UI", 9),
            )
        widget.pack(side="left", fill="x", expand=True, ipady=3)
        if field.help:
            tk.Label(parent, text=field.help, bg=BG, fg=MUTED,
                     font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 2))

    def _mic_choices(self) -> list[str]:
        """Display strings for the microphone dropdown, current first."""
        self._mic_display = {"Default": ""}
        for index, name in _list_input_devices():
            self._mic_display[f"{index}: {name}"] = str(index)
        return list(self._mic_display)

    # -------------------------------------------------------------- values
    def _load(self):
        """Fill every widget from the running config."""
        self._current = {}
        for field in settings_def.FIELDS:
            value = getattr(self.config, field.key, None)
            if field.kind == "bool":
                self._vars[field.key].set(bool(value))
                self._current[field.key] = bool(value)
                continue
            if field.key == "mic_device_index":
                display = "Default"
                if value is not None:
                    for text, raw in self._mic_display.items():
                        if raw == str(value):
                            display = text
                            break
                    else:
                        # The configured device is gone; show it rather than
                        # silently rewriting the setting to Default.
                        display = f"{value} (unplugged)"
                        self._mic_display[display] = str(value)
                self._vars[field.key].set(display)
                self._current[field.key] = "" if value is None else str(value)
                continue
            text = "" if value is None else str(value)
            self._vars[field.key].set(text)
            self._current[field.key] = "" if value is None else str(value)

    def _values(self) -> dict:
        values = {}
        for field in settings_def.FIELDS:
            var = self._vars[field.key]
            if field.kind == "bool":
                values[field.key] = bool(var.get())
            elif field.key == "mic_device_index":
                values[field.key] = self._mic_display.get(var.get(), "")
            else:
                values[field.key] = var.get()
        return values

    # --------------------------------------------------------------- actions
    def _status(self, message: str, colour: str = MUTED):
        try:
            self.root.after(
                0, lambda: self.status.configure(text=message, fg=colour))
        except (RuntimeError, tk.TclError):
            pass

    def _on_save(self):
        if self._working:
            return
        values = self._values()
        problem = settings_def.validate(values)
        if problem:
            self._status(problem, BAD)
            return

        updates = settings_def.updates_from(values, self._current)
        model = self._model_var.get()
        if model and model != self._current_model:
            updates["WHISPER_FLOW_MODEL_NAME"] = model
        if not updates:
            self._status("Nothing changed")
            return

        env_path = Path(self.config.config_dir) / ".env"
        try:
            envfile.set_values(env_path, updates)
        except Exception as e:
            self._status(f"Could not save: {e}", BAD)
            return
        log(f"[SETTINGS] saved {sorted(updates)} to {env_path}")

        self._current = values
        self._current_model = model or self._current_model
        self._saved_since_restart = True
        self._status(
            "Saved. A new model applies when this window closes; "
            "everything else needs a restart.", GOOD)
        self.restart_button.pack(side="right", padx=(0, 6))

    def _on_restart(self):
        if self._working:
            return
        self._working = True
        self._status("Restarting the daemon...")

        def work():
            ok, detail = restart.restart_daemon()
            self._working = False
            self._status("Restarted." if ok else f"Could not restart: {detail}",
                         GOOD if ok else BAD)
            if ok:
                self.restart_button.after(
                    0, self.restart_button.pack_forget)

        threading.Thread(target=work, daemon=True,
                         name="whisper-flow-restart").start()

    def _start_download(self, model: str):
        if self._working:
            return
        self._working = True
        self.progress.pack(fill="x")
        self._status(f"Downloading {model.replace('ggml-', '')}...")

        def work():
            try:
                ok = self.backend.install(model, progress=self._on_progress)
            except Exception as e:
                log(f"[SETTINGS] download failed: {e}")
                ok = False

            def done():
                self._working = False
                if ok:
                    self._build_model_rows()
                    self._model_var.set(model)
                    self._status("Downloaded - Save to use it.", GOOD)
                else:
                    self._status("Download failed. Check the connection "
                                 "and try again.", BAD)
            try:
                self.root.after(0, done)
            except (RuntimeError, tk.TclError):
                pass

        threading.Thread(target=work, daemon=True,
                         name="whisper-flow-model-download").start()

    def _on_progress(self, stage: str, fraction: float):
        def apply():
            if not self.progress.winfo_ismapped():
                self.progress.pack(fill="x")
            self.progress["value"] = fraction * 100
            label = "engine" if stage == "engine" else "model"
            self.status.configure(
                text=f"Downloading the speech {label}... "
                     f"{int(fraction * 100)}%", fg=MUTED)
        try:
            self.root.after(0, apply)
        except (RuntimeError, tk.TclError):
            pass

    def _open_config_folder(self):
        path = str(self.config.config_dir)
        try:
            if sys.platform == "win32":
                os.startfile(path)          # noqa: S606 - the OS shell opens a folder
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            self._status(f"Config folder: {path}")

    def run(self):
        self.root.mainloop()


def main() -> int:
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass                            # not fatal, just a blurrier window
    try:
        SettingsWindow().run()
    except Exception as e:
        log(f"[SETTINGS] window failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
