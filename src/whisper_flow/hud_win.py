"""The recording HUD on Windows.

Same contract as the GTK version in hud_app.py - reads the audio level file,
honours the same environment variables, exits when that file disappears - but
built on tkinter, which ships with Python on Windows, plus a Win32 region for
the rounded outline.

Windows 11 22H2 or later only - see blur_win for why.

What differs from Linux, and why:

  * Blur, rounded corners and the border all come from DWM rather than being
    drawn or masked here. The window keeps a uniform alpha so the acrylic
    shows through what is painted over it; an opaque canvas would hide it.
  * Fading uses the window's uniform alpha rather than per-pixel alpha.
"""

import ctypes
import json
import math
import os
import queue
import struct
import sys
import threading
import time
import tkinter as tk
from collections import deque


def _load_sibling(name):
    """Load a module next to this file by path.

    The supervisor runs this with PYTHONSAFEPATH=1, which keeps the script's
    own directory off sys.path, so a plain import would not find it. Going
    through the package instead would drag in the daemon and pystray.
    """
    import importlib.util
    # A frozen build unpacks bundled files to _MEIPASS, not beside __file__.
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_hud_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    # Normal import in a frozen build or an installed package.
    from whisper_flow import blur_win as _blur
except ImportError:
    # Running this file directly, where there is no package context.
    _blur = _load_sibling("blur_win")

WIDTH = 268
HEIGHT = 54
BOTTOM_MARGIN = int(os.environ.get("WHISPER_FLOW_HUD_BOTTOM_MARGIN", "28"))

DOT_X = 27
DOT_R = 5
WAVE_L = 48
WAVE_R = WIDTH - 24
BARS = 30
BAR_W = 3
BAR_MAX = 15

# Painted behind everything. Kept dark but relied on being seen through the
# window's alpha, so the acrylic blur underneath still reads.
BG = "#0a0a0c"
BAR_COLOUR = "#f2f2f5"
DOT_COLOUR = "#ff4548"
OUTLINE = "#6b6c73"

FADE_MS = 200.0
# Held below 1 so the acrylic behind the window reads through the panel.
TARGET_ALPHA = 0.78
FRAME_MS = 16
LEVEL_MS = 33
LEVEL_EASE = 0.30
PEAK_DECAY = 0.95
PEAK_FLOOR = 150.0
LEVEL_GAMMA = 0.65
MAX_LIFETIME_S = 600
DRAG_SLOP = 4
COMMAND_MS = 8          # how often a resident overlay checks for orders

def _config_dir() -> str:
    """Where settings live, matching config.default_config_dir().

    This said it used the same location as Config and then hard-coded the
    Unix one, so on Windows the overlay saved its position to ~/.config
    while every other setting lived under LOCALAPPDATA. Config cannot be
    imported here - it would pull pydantic into the overlay process - so the
    rule is repeated rather than shared.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "whisper-flow")
    return os.path.join(os.path.expanduser("~"), ".config", "whisper-flow")


POSITION_FILE = os.path.join(_config_dir(), "hud-position.json")

# Where the position used to be written on Windows. Read only, so an existing
# placement survives the move.
LEGACY_POSITION_FILE = os.path.join(
    os.path.expanduser("~"), ".config", "whisper-flow", "hud-position.json",
)


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _load_positions() -> dict:
    for path in (POSITION_FILE, LEGACY_POSITION_FILE):
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return data
        except Exception:
            continue
    return {}


def _save_position(key: str, x: int, y: int) -> None:
    try:
        positions = _load_positions()
        positions[key] = [int(x), int(y)]
        os.makedirs(os.path.dirname(POSITION_FILE), exist_ok=True)
        tmp = POSITION_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(positions, f)
        os.replace(tmp, POSITION_FILE)
    except Exception as e:
        print(f"[HUD] could not save position: {e}", flush=True)


class HudWindow:
    def __init__(self, level_file: str, resident: bool = False):
        self.level_file = level_file
        # Resident: stay alive between recordings and wait for orders,
        # instead of being spawned and killed for each one. Starting a
        # frozen process is the single largest cost on the path between
        # pressing the hotkey and seeing anything.
        self._resident = resident
        self._visible = not resident
        self._commands = queue.Queue()
        self._announced = False
        self.root = tk.Tk()
        self.root.overrideredirect(True)          # no title bar or border
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.0)       # faded in by _frame
        self.root.configure(bg=BG)

        self.x, self.y = self._start_position()
        self.root.geometry(f"{WIDTH}x{HEIGHT}+{self.x}+{self.y}")

        self.canvas = tk.Canvas(
            self.root, width=WIDTH, height=HEIGHT,
            bg=BG, highlightthickness=0, bd=0,
        )
        self.canvas.pack()

        self.alpha = 0.0
        self.fade_in_t0 = time.monotonic()
        self.fade_out_t0 = None
        self.start = time.monotonic()
        self.peak = PEAK_FLOOR
        self.targets = deque([0.0] * BARS, maxlen=BARS)
        self.shown = [0.0] * BARS
        self.level_pos = 0
        self._level_handle = None
        self._drag_from = None
        self._quitting = False
        self.style = None

        self._build_items()
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self.root.after(0, self._apply_window_style)
        if resident:
            # Built, styled and hidden now, so a later show is a deiconify.
            self.root.withdraw()
            threading.Thread(target=self._read_commands, daemon=True,
                             name="whisper-flow-hud-commands").start()
            self.root.after(COMMAND_MS, self._poll_commands)
        else:
            self.root.after(FRAME_MS, self._frame)
            self.root.after(LEVEL_MS, self._read_levels)

    # -- window shape and placement --------------------------------------
    def _apply_window_style(self):
        """Hand the backdrop, corners and border to DWM."""
        try:
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            self.style = _blur.apply_window_style(hwnd)
            print(f"[HUD] window style: {self.style or 'not applied'}", flush=True)
        except Exception as e:
            print(f"[HUD] window style failed: {e}", flush=True)

    def _start_position(self):
        saved = _load_positions().get("default")
        if isinstance(saved, list) and len(saved) >= 2:
            return int(saved[0]), int(saved[1])
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        return (sw - WIDTH) // 2, sh - HEIGHT - BOTTOM_MARGIN - 40

    # -- interaction ------------------------------------------------------
    def _on_press(self, event):
        self._drag_from = (event.x_root, event.y_root, self.x, self.y)

    def _on_drag(self, event):
        if not self._drag_from:
            return
        sx, sy, ox, oy = self._drag_from
        self.x = ox + (event.x_root - sx)
        self.y = oy + (event.y_root - sy)
        self.root.geometry(f"{WIDTH}x{HEIGHT}+{self.x}+{self.y}")

    def _on_release(self, event):
        if not self._drag_from:
            return
        sx, sy, _ox, _oy = self._drag_from
        moved = abs(event.x_root - sx) + abs(event.y_root - sy)
        self._drag_from = None
        if moved < DRAG_SLOP:
            if event.x >= WIDTH - 34:      # the close affordance
                self.quit()
            return
        _save_position("default", self.x, self.y)

    def _close_level_handle(self):
        if self._level_handle is not None:
            try:
                self._level_handle.close()
            except Exception:
                pass
            self._level_handle = None

    # -- resident mode ----------------------------------------------------
    def _read_commands(self):
        """Read commands from the daemon on stdin, on a thread of its own.

        Entered only when the daemon asked for it, never when this is run by
        hand: an inherited stdin would hit EOF at once and close the overlay.

        End of stream means the daemon is gone, which is the guarantee that
        a resident overlay can never be left behind on the screen.
        """
        try:
            for line in sys.stdin:
                self._commands.put(line.strip())
        except Exception:
            pass
        self._commands.put("quit")

    def _poll_commands(self):
        """Drain the command queue on Tk's own thread."""
        try:
            while True:
                command = self._commands.get_nowait()
                if command.startswith("show"):
                    _, _, path = command.partition(" ")
                    self._begin(path.strip())
                elif command == "hide":
                    self._end()
                elif command == "quit":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(COMMAND_MS, self._poll_commands)

    def _begin(self, level_file: str):
        """Show for a new recording. No process start, no window creation."""
        self._close_level_handle()
        self.level_file = level_file
        self.level_pos = 0
        self.peak = PEAK_FLOOR
        self.targets = deque([0.0] * BARS, maxlen=BARS)
        self.shown = [0.0] * BARS
        self.start = time.monotonic()
        self.fade_in_t0 = time.monotonic()
        self.fade_out_t0 = None
        self._quitting = False
        self._visible = True
        self._announced = True
        try:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.update_idletasks()
        except tk.TclError:
            return
        print(f"[HUD] visible {time.time():.6f}", flush=True)
        self.root.after(FRAME_MS, self._frame)
        self.root.after(LEVEL_MS, self._read_levels)

    def _end(self):
        """Fade out and hide, keeping the process and window for next time."""
        if not self._visible or self._quitting:
            return
        self._quitting = True
        self.fade_out_t0 = time.monotonic()

    def _retire(self):
        """Reached at the end of a fade: withdraw rather than exit."""
        self._close_level_handle()
        self.level_file = ""
        self._visible = False
        self._quitting = False
        self.fade_out_t0 = None
        try:
            self.root.attributes("-alpha", 0.0)
            self.root.withdraw()
        except tk.TclError:
            pass

    def quit(self):
        if self._quitting:
            return
        self._quitting = True
        self.fade_out_t0 = time.monotonic()

    # -- animation --------------------------------------------------------
    def _frame(self):
        now = time.monotonic()
        if self.fade_out_t0 is None:
            self.alpha = _ease((now - self.fade_in_t0) * 1000.0 / FADE_MS)
        else:
            t = (now - self.fade_out_t0) * 1000.0 / FADE_MS
            self.alpha = 1.0 - _ease(t)
            if t >= 1.0:
                if self._resident:
                    self._retire()          # stay alive for the next press
                else:
                    self._close_level_handle()
                    self.root.destroy()
                return
        try:
            self.root.attributes("-alpha", self.alpha * TARGET_ALPHA)
        except tk.TclError:
            return
        if not self._announced:
            self._announced = True
            print(f"[HUD] visible {time.time():.6f}", flush=True)

        for i, target in enumerate(self.targets):
            self.shown[i] += (target - self.shown[i]) * LEVEL_EASE
        self._draw()
        if self._visible:
            self.root.after(FRAME_MS, self._frame)

    def _read_levels(self):
        if self.level_file:
            # The daemon deletes this when the recording ends; treat its
            # absence as the signal to go, so the overlay cannot be orphaned.
            if not os.path.exists(self.level_file):
                print("[HUD] level file gone, closing", flush=True)
                self._close_level_handle()
                self._end() if self._resident else self.quit()
                return
            try:
                # Opened once and held. Reopening thirty times a second put
                # every read through the whole Windows filesystem filter
                # stack, antivirus included, for a file this process already
                # had open a moment earlier.
                if self._level_handle is None:
                    self._level_handle = open(self.level_file, "rb")
                f = self._level_handle
                f.seek(0, 2)
                count = f.tell() // 4
                if count > self.level_pos:
                    f.seek(self.level_pos * 4)
                    data = f.read((count - self.level_pos) * 4)
                    self.level_pos = count
                    vals = struct.unpack(f"<{len(data) // 4}i", data)
                    for v in vals:
                        v = abs(v)
                        self.peak = max(self.peak * PEAK_DECAY, float(v), PEAK_FLOOR)
                        self.targets.append(
                            min(1.0, (v / self.peak) ** LEVEL_GAMMA))
            except Exception:
                # Drop the handle so the next tick reopens rather than
                # spinning on a file descriptor that has gone bad.
                self._close_level_handle()

        # The cap guards against an orphan overlay. A resident one is held
        # open by the daemon's pipe instead, and a recording may legitimately
        # run to the configured maximum.
        if not self._resident and time.monotonic() - self.start > MAX_LIFETIME_S:
            print("[HUD] lifetime cap reached, closing", flush=True)
            self.quit()
            return
        if self._visible:
            self.root.after(LEVEL_MS, self._read_levels)

    def _build_items(self):
        """Create every canvas item once.

        The previous version cleared the canvas and rebuilt all thirty-three
        items on every frame - about two thousand item allocations a second,
        each re-tessellated by Tk - for a drawing whose geometry is the only
        thing that changes. Moving existing items is far cheaper, and this
        overlay is on screen precisely when the machine is also recording
        audio and running transcription passes.
        """
        c = self.canvas
        mid = HEIGHT / 2
        # Flat fill: DWM rounds and clips the window, so drawing a rounded
        # outline here would sit inside its curve and read as a double edge.
        c.create_rectangle(0, 0, WIDTH, HEIGHT, fill=BG, outline="")
        self.glow_item = c.create_oval(
            DOT_X - DOT_R * 2, mid - DOT_R * 2, DOT_X + DOT_R * 2,
            mid + DOT_R * 2, fill=DOT_COLOUR, outline="")
        c.create_oval(DOT_X - DOT_R, mid - DOT_R, DOT_X + DOT_R, mid + DOT_R,
                      fill=DOT_COLOUR, outline="")
        step = (WAVE_R - WAVE_L) / float(BARS)
        self.bar_items = [
            c.create_rectangle(WAVE_L + i * step, mid - BAR_W / 2,
                               WAVE_L + i * step + BAR_W, mid + BAR_W / 2,
                               fill=BAR_COLOUR, outline="")
            for i in range(BARS)
        ]

    def _draw(self):
        c = self.canvas
        mid = HEIGHT / 2

        breathe = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(
            (time.monotonic() - self.start) * 3.0))
        glow = int(60 + 40 * breathe)
        c.itemconfig(self.glow_item, fill=f"#{glow:02x}1416")

        step = (WAVE_R - WAVE_L) / float(BARS)
        for i, level in enumerate(self.shown):
            bar_h = max(BAR_W, level * BAR_MAX * 2)
            x = WAVE_L + i * step
            c.coords(self.bar_items[i], x, mid - bar_h / 2, x + BAR_W,
                     mid + bar_h / 2)

    def run(self):
        self.root.mainloop()


def main() -> int:
    level_file = os.environ.get("WHISPER_FLOW_HUD_LEVEL_FILE", "")
    # Only when the daemon says so: run by hand, stdin is inherited and would
    # reach end of stream at once, closing the overlay immediately.
    resident = os.environ.get("WHISPER_FLOW_HUD_RESIDENT") == "1"
    print(f"[HUD] starting level_file={level_file}", flush=True)
    if not _blur.is_supported():
        # Carry on without the acrylic. The blur is decoration; the overlay is
        # the only sign that recording started, and refusing to draw it left
        # the user with no feedback at all - including when the build check
        # itself failed, since RtlGetVersion reports build 0 on any error.
        print(f"[HUD] {_blur.unsupported_reason()}; drawing without blur",
              flush=True)
    try:
        # Per-monitor DPI aware, or the overlay is scaled and blurry.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    HudWindow(level_file, resident=resident).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
