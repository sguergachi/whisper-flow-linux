"""The settings window: every knob in one place, plus model management.

Runs as its own process, launched by the tray app with --settings (or
``python -m whisper_flow.settings_ui``), because the daemon's pystray loop
and Tk each demand their own main thread.

Tk's stock checkboxes, radios and notebook tabs look like 1998 and there is
no styling them out of it, so the interactive pieces here are drawn by hand:
toggle switches on small canvases, a segmented tab bar, and clickable model
cards. Still pure tkinter - no new dependency, and the frozen build already
ships it.

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
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk

if __package__ in (None, ""):        # launched as a script by the frozen build
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_flow import envfile, restart, settings_def  # noqa: E402
from whisper_flow.backend import LocalBackend  # noqa: E402
from whisper_flow.config import Config  # noqa: E402
from whisper_flow.logging import log  # noqa: E402

BG = "#14151a"          # window
PANEL = "#1d1f26"       # cards and inputs
PANEL_HOVER = "#252833" # card under the pointer
BORDER = "#2c2f3a"      # hairlines and input edges
FG = "#e9eaee"
MUTED = "#969ba6"
ACCENT = "#3b82f6"
ACCENT_ACTIVE = "#5c9bf8"
GOOD = "#4ade80"
BAD = "#f87171"
TRACK_OFF = "#3a3e4a"


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


def _pick_family(root) -> str:
    """The nicest sans this machine has; Tk picks per platform otherwise."""
    try:
        families = set(tkfont.families(root))
    except tk.TclError:
        return "TkDefaultFont"
    for candidate in ("Segoe UI", "Ubuntu", "Cantarell", "Noto Sans",
                      "DejaVu Sans"):
        if candidate in families:
            return candidate
    return "TkDefaultFont"


class Toggle(tk.Canvas):
    """A boolean switch drawn by hand, because Tk's checkbutton is unsightly.

    Backed by an ordinary BooleanVar so the rest of the window (and the
    tests) treat it like any other variable.
    """

    W, H, KNOB = 40, 22, 16

    def __init__(self, parent, variable: tk.BooleanVar, bg: str):
        super().__init__(parent, width=self.W, height=self.H, bd=0,
                         highlightthickness=0, bg=bg, cursor="hand2")
        self._var = variable
        self._pos = 1.0 if variable.get() else 0.0
        self._anim = None
        self.bind("<Button-1>", self._flip)
        variable.trace_add("write", lambda *_: self._slide())
        self._draw()

    def _flip(self, _event):
        self._var.set(not self._var.get())

    def _slide(self):
        """Ease the knob towards the new state over a handful of frames."""
        target = 1.0 if self._var.get() else 0.0
        if self._anim is not None:
            self.after_cancel(self._anim)
            self._anim = None

        def step():
            self._pos += (target - self._pos) * 0.4
            if abs(target - self._pos) < 0.03:
                self._pos = target
                self._anim = None
            else:
                self._anim = self.after(16, step)
            self._draw()
        step()

    def _draw(self):
        self.delete("all")
        on = self._pos >= 0.5
        track = ACCENT if on else TRACK_OFF
        r = self.H // 2
        self.create_oval(1, 1, self.H - 1, self.H - 1,
                         fill=track, outline="")
        self.create_oval(self.W - self.H + 1, 1, self.W - 1, self.H - 1,
                         fill=track, outline="")
        self.create_rectangle(r, 1, self.W - r, self.H - 1,
                              fill=track, outline="")
        x = 4 + self._pos * (self.W - self.KNOB - 8)
        self.create_oval(x, 3, x + self.KNOB, 3 + self.KNOB,
                         fill="#f2f3f7", outline="")


class TabBar(tk.Frame):
    """A segmented tab control: quiet text, accent underline for the active."""

    def __init__(self, parent, bg: str, font):
        super().__init__(parent, bg=bg)
        self._bg = bg
        self._font = font
        self._bar = tk.Frame(self, bg=bg)
        self._bar.pack(fill="x", padx=20, pady=(14, 0))
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self._container = tk.Frame(self, bg=bg)
        self._container.pack(fill="both", expand=True)
        self._tabs: dict[str, tuple[tk.Frame, tk.Label, tk.Frame]] = {}

    def add(self, name: str) -> tk.Frame:
        holder = tk.Frame(self._bar, bg=self._bg)
        holder.pack(side="left", padx=(0, 4))
        label = tk.Label(holder, text=name, bg=self._bg, fg=MUTED,
                         font=self._font, padx=12, pady=6, cursor="hand2")
        label.pack()
        underline = tk.Frame(holder, bg=ACCENT, height=2)
        label.bind("<Button-1>", lambda _e, n=name: self.select(n))
        label.bind("<Enter>", lambda _e, n=name: self._hover(n, True))
        label.bind("<Leave>", lambda _e, n=name: self._hover(n, False))
        page = tk.Frame(self._container, bg=self._bg)
        self._tabs[name] = (page, label, underline)
        return page

    def _hover(self, name: str, inside: bool):
        _page, label, underline = self._tabs[name]
        if underline.winfo_ismapped():
            return                      # the active tab keeps its colour
        label.configure(fg=FG if inside else MUTED)

    def select(self, name: str):
        for tab_name, (page, label, underline) in self._tabs.items():
            active = tab_name == name
            label.configure(fg=FG if active else MUTED,
                            font=(self._font[0], self._font[1],
                                  "bold" if active else "normal"))
            underline.pack_forget()
            if active:
                underline.pack(fill="x")
                page.pack(fill="both", expand=True, padx=26, pady=20)
                page.tkraise()


class SettingsWindow:
    def __init__(self):
        self.config = Config()
        self.backend = LocalBackend(self.config)
        self._current_model = self.backend.working_model()
        self._working = False          # a download is in flight

        self.root = tk.Tk()
        self.root.title("WhisperFlow settings")
        self.root.configure(bg=BG)
        self.root.minsize(660, 700)
        self._family = _pick_family(self.root)
        self._style_ttk()
        self._vars: dict[str, tk.Variable] = {}
        self._mic_display: dict[str, str] = {}   # display -> raw value
        self._card_widgets: list[list] = []      # per model: widgets to tint
        self._build()
        self._load()
        self.root.attributes("-topmost", True)
        self.root.after(400, lambda: self.root.attributes("-topmost", False))

    def f(self, size: int, weight: str = "normal") -> tuple:
        return (self._family, size, weight)

    # ------------------------------------------------------------- styling
    def _style_ttk(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Settings.Horizontal.TProgressbar", troughcolor=PANEL,
            background=ACCENT, bordercolor=PANEL, lightcolor=ACCENT,
            darkcolor=ACCENT, thickness=5)
        # A read-only combobox ignores configure() colours; they only take
        # through map(), or the field stays white with faint text on it.
        style.configure("TCombobox", bordercolor=BORDER, arrowcolor=MUTED)
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
        self.tabs = TabBar(self.root, BG, self.f(10))
        self.tabs.pack(fill="both", expand=True)

        pages = {}
        for section in settings_def.SECTIONS:
            page = self.tabs.add(section)
            pages[section] = page
            if section == "Speech":
                self._build_model_section(page)
            if section == "Hotkeys":
                tk.Label(
                    page,
                    text="Key names joined with '+', e.g. super+alt or "
                         "ctrl+shift+space. Applies on restart.",
                    bg=BG, fg=MUTED, justify="left",
                    font=self.f(9)).pack(anchor="w", pady=(0, 14))
            if section == "General":
                self._build_daemon_status(page)
            for field in settings_def.FIELDS:
                if field.section == section:
                    self._build_field(page, field)
        self.tabs.select("Speech")

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=24, pady=(6, 18))

        self.status = tk.Label(footer, text=" ", bg=BG, fg=MUTED, anchor="w",
                               font=self.f(9))
        self.status.pack(fill="x", pady=(0, 8))

        slot = tk.Frame(footer, bg=BG, height=8)
        slot.pack(fill="x", pady=(0, 12))
        slot.pack_propagate(False)
        self.progress = ttk.Progressbar(
            slot, style="Settings.Horizontal.TProgressbar",
            mode="determinate", maximum=100)

        buttons = tk.Frame(footer, bg=BG)
        buttons.pack(fill="x")
        self._ghost_button(buttons, "Open config folder",
                           self._open_config_folder).pack(side="left")
        self.restart_button = self._outline_button(
            buttons, "Restart now", self._on_restart)
        self.save_button = self._primary_button(buttons, "Save", self._on_save)
        self.save_button.pack(side="right")
        self._ghost_button(buttons, "Close", self.root.destroy).pack(
            side="right", padx=(0, 4))

    # -------------------------------------------------------------- buttons
    def _primary_button(self, parent, text, command):
        btn = tk.Button(
            parent, text=text, command=command, padx=26, pady=9,
            bg=ACCENT, fg="white", activebackground=ACCENT_ACTIVE,
            activeforeground="white", disabledforeground="#8fa8cc",
            relief="flat", bd=0, highlightthickness=0, cursor="hand2",
            font=self.f(10, "bold"),
        )
        btn.bind("<Enter>", lambda _e: btn.configure(bg=ACCENT_ACTIVE))
        btn.bind("<Leave>", lambda _e: btn.configure(
            bg=ACCENT if str(btn["state"]) != "disabled" else ACCENT))
        return btn

    def _ghost_button(self, parent, text, command):
        btn = tk.Button(
            parent, text=text, command=command, padx=12, pady=9,
            bg=BG, fg=MUTED, activebackground=PANEL, activeforeground=FG,
            relief="flat", bd=0, highlightthickness=0, cursor="hand2",
            font=self.f(10),
        )
        btn.bind("<Enter>", lambda _e: btn.configure(fg=FG, bg=PANEL))
        btn.bind("<Leave>", lambda _e: btn.configure(fg=MUTED, bg=BG))
        return btn

    def _outline_button(self, parent, text, command):
        return tk.Button(
            parent, text=text, command=command, padx=18, pady=8,
            bg=BG, fg=ACCENT, activebackground=PANEL,
            activeforeground=ACCENT_ACTIVE, relief="flat", bd=0,
            highlightthickness=1, highlightbackground=ACCENT,
            cursor="hand2", font=self.f(10, "bold"),
        )

    # ------------------------------------------------------------------ tabs
    def _build_daemon_status(self, parent):
        pid = restart.daemon_pid()
        dot = tk.Canvas(parent, width=10, height=10, bd=0,
                        highlightthickness=0, bg=BG)
        dot.create_oval(1, 1, 9, 9, fill=GOOD if pid else BAD, outline="")
        dot.pack(side="left", anchor="n", pady=(3, 0))
        text = (f"Daemon is running (pid {pid})" if pid else
                "Daemon is not running - dictation will not work")
        tk.Label(parent, text=f"  {text}", bg=BG,
                 fg=GOOD if pid else BAD,
                 font=self.f(9, "bold")).pack(side="left", pady=(0, 12))

    def _build_model_section(self, parent):
        tk.Label(parent, text="Speech model", bg=BG, fg=FG,
                 font=self.f(12, "bold")).pack(anchor="w")
        tk.Label(
            parent,
            text="What listens to your voice. Bigger is more accurate and "
                 "slower; the recommendation is sized to this machine.",
            bg=BG, fg=MUTED, justify="left",
            font=self.f(9)).pack(anchor="w", pady=(2, 12))
        self._models_frame = tk.Frame(parent, bg=BG)
        self._models_frame.pack(fill="x", pady=(0, 18))
        self._model_var = tk.StringVar(value=self._current_model or "")
        self._model_var.trace_add("write", lambda *_: self._tint_cards())
        self._build_model_rows()
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=(0, 16))

    def _pill(self, parent, text: str, fg: str, bg: str):
        return tk.Label(parent, text=f" {text} ", bg=bg, fg=fg,
                        font=self.f(8, "bold"), padx=5, pady=1)

    def _build_model_rows(self):
        frame = self._models_frame
        for child in frame.winfo_children():
            child.destroy()
        self._cards: dict[str, tuple[tk.Frame, list]] = {}
        for row in self.backend.model_inventory():
            name = row["name"]
            installed = row["installed"]
            card = tk.Frame(frame, bg=PANEL, cursor="hand2",
                            highlightthickness=1,
                            highlightbackground=BORDER)
            card.pack(fill="x", pady=3)
            inner = tk.Frame(card, bg=PANEL)
            inner.pack(fill="x", padx=12, pady=9)

            radio = tk.Canvas(inner, width=18, height=18, bd=0,
                              highlightthickness=0, bg=PANEL,
                              cursor="hand2")
            radio.pack(side="left", pady=2)

            text_box = tk.Frame(inner, bg=PANEL)
            text_box.pack(side="left", padx=(10, 0))
            title = row["name"].replace("ggml-", "")
            tk.Label(text_box, text=title, bg=PANEL,
                     fg=FG if installed else MUTED,
                     font=self.f(10, "bold")).pack(anchor="w")
            tk.Label(text_box,
                     text=f"{row['size_mb']} MB  ·  {row['wants']}",
                     bg=PANEL, fg=MUTED,
                     font=self.f(8)).pack(anchor="w")

            right = tk.Frame(inner, bg=PANEL)
            right.pack(side="right")
            if row["recommended"]:
                self._pill(right, "recommended", "#9ec2fc", "#1b2c4a").pack(
                    side="left", padx=3)
            if row["current"]:
                self._pill(right, "in use", GOOD, "#16301f").pack(
                    side="left", padx=3)
            if not installed:
                tk.Button(
                    right, text="Download", padx=10, pady=2,
                    command=lambda m=name: self._start_download(m),
                    bg=PANEL, fg=ACCENT, activebackground=PANEL_HOVER,
                    activeforeground=ACCENT_ACTIVE, relief="flat", bd=0,
                    highlightthickness=1, highlightbackground=ACCENT,
                    cursor="hand2", font=self.f(9, "bold"),
                ).pack(side="left", padx=(8, 0))

            widgets = [card, inner, text_box, right, radio]
            self._cards[name] = (card, widgets, radio)
            for widget in widgets + list(text_box.winfo_children()):
                widget.bind("<Button-1>",
                            lambda _e, n=name, ok=installed: self._pick(n, ok))
                widget.bind("<Enter>",
                            lambda _e, n=name: self._hover_card(n, True))
                widget.bind("<Leave>",
                            lambda _e, n=name: self._hover_card(n, False))
        self._tint_cards()

    def _pick(self, name: str, installed: bool):
        if installed:
            self._model_var.set(name)

    def _hover_card(self, name: str, inside: bool):
        card, widgets, _radio = self._cards[name]
        colour = PANEL_HOVER if inside else PANEL
        for widget in widgets + [w for w in widgets[2].winfo_children()]:
            widget.configure(bg=colour)

    def _tint_cards(self):
        """Redraw the radio dots and the selected card's outline."""
        selected = self._model_var.get()
        for name, (card, widgets, radio) in getattr(
                self, "_cards", {}).items():
            active = name == selected
            card.configure(highlightbackground=ACCENT if active else BORDER)
            radio.delete("all")
            radio.create_oval(1, 1, 17, 17,
                              outline=ACCENT if active else MUTED, width=2)
            if active:
                radio.create_oval(5, 5, 13, 13, fill=ACCENT, outline="")

    def _build_field(self, parent, field):
        if field.kind == "bool":
            var = tk.BooleanVar()
            self._vars[field.key] = var
            row = tk.Frame(parent, bg=BG)
            row.pack(fill="x", pady=5)
            Toggle(row, var, BG).pack(side="left")
            tk.Label(row, text=f"   {field.label}", bg=BG, fg=FG,
                     font=self.f(9)).pack(side="left")
            return

        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=5)
        tk.Label(row, text=field.label, bg=BG, fg=MUTED, width=28,
                 anchor="w", font=self.f(9)).pack(side="left")

        var = tk.StringVar()
        self._vars[field.key] = var
        if field.kind == "choice":
            choices = (self._mic_choices()
                       if field.key == "mic_device_index"
                       else list(field.choices))
            widget = ttk.Combobox(row, textvariable=var, values=choices,
                                  state="readonly", width=26,
                                  font=self.f(9))
        else:
            widget = tk.Entry(
                row, textvariable=var, bg=PANEL, fg=FG, insertbackground=FG,
                relief="flat", highlightthickness=1,
                highlightcolor=ACCENT, highlightbackground=BORDER,
                show="•" if field.kind == "password" else "",
                font=self.f(9),
            )
        widget.pack(side="left", fill="x", expand=True, ipady=5)
        if field.help:
            tk.Label(parent, text=field.help, bg=BG, fg=MUTED,
                     font=self.f(8)).pack(anchor="w", pady=(0, 2))

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
        self._status(
            "Saved. A new model applies when this window closes; "
            "everything else needs a restart.", GOOD)
        self.restart_button.pack(side="right", padx=(0, 8))

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
                self.restart_button.after(0, self.restart_button.pack_forget)

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
