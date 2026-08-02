"""First-run setup window: one button to get a working speech model.

Runs as its own process, launched by the tray app with --setup. The daemon
already owns a pystray event loop and Tk insists on being the main thread of
its own, so the two cannot share a process.

What it does is deliberately small: report what the machine can run, download
the matching engine and model behind one button, and record the choice. The
daemon re-checks the backend when this process exits, so the new model is in
use without a restart.
"""

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

if __package__ in (None, ""):        # launched as a script by the frozen build
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_flow import envfile  # noqa: E402
from whisper_flow.backend import (  # noqa: E402
    MODELS,
    LocalBackend,
    detect_accelerator,
    recommended_model,
    usable_cores,
)
from whisper_flow.config import Config  # noqa: E402
from whisper_flow.logging import log  # noqa: E402

BG = "#16171b"
PANEL = "#1e2026"
FG = "#e8e9ec"
MUTED = "#9aa0a6"
ACCENT = "#3b82f6"
ACCENT_ACTIVE = "#2f6fd0"
GOOD = "#4ade80"
BAD = "#f87171"


class SetupWindow:
    def __init__(self):
        self.config = Config()
        self.backend = LocalBackend(self.config, notify=self._status)
        self.accelerator = detect_accelerator()
        self.has_gpu = self.accelerator.startswith("cuda")
        self.model = recommended_model(self.accelerator)
        self.size_mb, self.wants = MODELS.get(self.model, (0, ""))
        # Nothing to fetch when the recommended model already shipped.
        self.already_have = self.backend.model_path(self.model).exists()
        self._working = False
        self._done = False

        self.root = tk.Tk()
        self.root.title("WhisperFlow setup")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._style_progressbar()
        self._build()
        self._centre()
        self.root.attributes("-topmost", True)
        self.root.after(400, lambda: self.root.attributes("-topmost", False))

    # ------------------------------------------------------------------ ui
    def _style_progressbar(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")     # the default theme ignores colours
        except tk.TclError:
            pass
        style.configure(
            "Setup.Horizontal.TProgressbar", troughcolor=PANEL, background=ACCENT,
            bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT, thickness=6,
        )

    def _build(self):
        pad = {"padx": 26}

        tk.Label(self.root, text="Speech model", bg=BG, fg=FG,
                 font=("Segoe UI", 17, "bold")).pack(anchor="w", pady=(24, 2), **pad)

        headline, detail = self._copy()
        tk.Label(self.root, text=headline, bg=BG, fg=FG,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", **pad)
        tk.Label(self.root, text=detail, bg=BG, fg=MUTED, justify="left",
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 16), **pad)

        panel = tk.Frame(self.root, bg=PANEL)
        panel.pack(fill="x", pady=(0, 14), **pad)
        rows = (
            ("Model", self.model.replace("ggml-", "")),
            ("Download", "already installed" if self.already_have
             else f"{self.size_mb} MB" if self.size_mb else "unknown"),
            ("Uses", self.wants or "-"),
            ("Runs on", "GPU (CUDA)" if self.has_gpu
             else f"CPU, {usable_cores()} threads"),
        )
        for label, value in rows:
            row = tk.Frame(panel, bg=PANEL)
            row.pack(fill="x", padx=16, pady=(8 if label == "Model" else 2,
                                              8 if label == "Runs on" else 2))
            tk.Label(row, text=label, bg=PANEL, fg=MUTED, width=9, anchor="w",
                     font=("Segoe UI", 9)).pack(side="left")
            tk.Label(row, text=value, bg=PANEL, fg=FG, anchor="w",
                     font=("Segoe UI", 9)).pack(side="left")

        self.status = tk.Label(self.root, text="", bg=BG, fg=MUTED, anchor="w",
                               font=("Segoe UI", 9))
        self.status.pack(fill="x", **pad)

        # The bar lives in a slot of fixed height, reserved whether or not it
        # is showing: appearing mid-download must not resize the window under
        # the pointer, and packing it later would put it below the buttons.
        slot = tk.Frame(self.root, bg=BG, height=14)
        slot.pack(fill="x", pady=(6, 0), **pad)
        slot.pack_propagate(False)
        self.progress = ttk.Progressbar(
            slot, style="Setup.Horizontal.TProgressbar",
            mode="determinate", maximum=100,
        )

        buttons = tk.Frame(self.root, bg=BG)
        buttons.pack(fill="x", pady=(16, 22), **pad)

        self.action = tk.Button(
            buttons, text=self._action_label(), command=self._on_action,
            bg=ACCENT, fg="white", activebackground=ACCENT_ACTIVE,
            activeforeground="white", disabledforeground="#8fa8cc",
            relief="flat", bd=0, highlightthickness=0,
            font=("Segoe UI", 10, "bold"), padx=20, pady=9, cursor="hand2",
        )
        self.action.pack(side="right")

        self.dismiss = tk.Button(
            buttons, text="Not now", command=self._on_close,
            bg=BG, fg=MUTED, activebackground=BG, activeforeground=FG,
            relief="flat", bd=0, highlightthickness=0,
            disabledforeground="#555a61", font=("Segoe UI", 10),
            padx=14, pady=9, cursor="hand2",
        )
        self.dismiss.pack(side="right", padx=(0, 6))

    def _copy(self) -> tuple[str, str]:
        short = self.model.replace("ggml-", "")
        if self.already_have:
            return ("Everything is ready.",
                    "The right model for this machine is already installed.\n"
                    "Hold your dictation hotkey and start talking.")
        if self.has_gpu:
            return ("An NVIDIA GPU was found.",
                    f"WhisperFlow can run {short} on it: much more accurate than "
                    f"the\nmodel included with the app, and still faster than you "
                    f"can speak.")
        # No CUDA. Say plainly why the GPU is not being used, because
        # "it is only using my CPU" otherwise reads as a bug.
        return ("Running on the processor.",
                f"whisper.cpp only publishes GPU builds for NVIDIA cards, so "
                f"integrated\ngraphics cannot be used. {short} is sized to this "
                f"machine's CPU so it\nkeeps up with speech.")

    def _action_label(self) -> str:
        if self.already_have:
            return "Done"
        return f"Download ({self.size_mb} MB)" if self.size_mb else "Download"

    def _centre(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 3
        self.root.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    # --------------------------------------------------------------- work
    def _status(self, message: str, colour: str = MUTED):
        """Safe to call from the worker thread: Tk is touched on its own."""
        try:
            self.root.after(
                0, lambda: self.status.configure(text=message, fg=colour))
        except (RuntimeError, tk.TclError):
            pass                        # window already gone

    def _on_progress(self, stage: str, fraction: float):
        def apply():
            if self._done:
                return          # never let a late tick overwrite the result
            if not self.progress.winfo_ismapped():
                self.progress.pack(fill="x")
            self.progress["value"] = fraction * 100
            label = "engine" if stage == "engine" else "model"
            self.status.configure(
                text=f"Downloading the speech {label}... {int(fraction * 100)}%",
                fg=MUTED)
        try:
            self.root.after(0, apply)
        except (RuntimeError, tk.TclError):
            pass

    def _on_action(self):
        if self._working:
            return
        if self._done or self.already_have:
            self._on_close()
            return

        self._working = True
        self.action.configure(state="disabled", text="Downloading...")
        self.dismiss.configure(state="disabled")
        self._status("Starting...")
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
        self.root.after(0, self._finished if ok else self._failed)

    def _save_choice(self):
        """Persist the model so the daemon picks it up when this exits."""
        try:
            envfile.set_values(Path(self.config.config_dir) / ".env",
                               {"WHISPER_FLOW_MODEL_NAME": self.model})
        except Exception as e:
            log(f"[SETUP] could not save the model choice: {e}")
            self._status(f"Downloaded, but the setting could not be saved: {e}", BAD)

    def _finished(self):
        self._working = False
        self._done = True
        self.progress["value"] = 100
        where = "the GPU" if self.has_gpu else "your processor"
        self.status.configure(
            text=f"Ready. WhisperFlow will transcribe on {where}.", fg=GOOD)
        self.action.configure(state="normal", text="Done")
        self.dismiss.pack_forget()

    def _failed(self):
        self._working = False
        if not self.status.cget("text").startswith("Could not"):
            self.status.configure(text="Could not download it. Check the "
                                       "connection and try again.", fg=BAD)
        else:
            self.status.configure(fg=BAD)
        self.action.configure(state="normal", text="Try again")
        self.dismiss.configure(state="normal")

    def _on_close(self):
        if self._working:
            return                      # a half-finished download helps nobody
        self.backend.mark_setup_seen()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


def main() -> int:
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass                            # not fatal, just a blurrier window
    try:
        SetupWindow().run()
    except Exception as e:
        log(f"[SETUP] window failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
