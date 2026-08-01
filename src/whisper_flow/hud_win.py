"""The recording HUD on Windows.

Same contract as the GTK version in hud_app.py - reads the audio level file,
honours the same environment variables, exits when that file disappears - but
built on tkinter, which ships with Python on Windows, plus a Win32 region for
the rounded outline.

What differs from Linux, and why:

  * Blur comes from DWM rather than ext-background-effect - see blur_win -
    and the tint is the compositor's acrylic instead of one this draws. The
    window keeps a uniform alpha so the blur shows through what is painted
    over it; an opaque canvas would hide the effect entirely.
  * Corners come from SetWindowRgn, which is a hard-edged mask with no
    antialiasing. The alternative, per-pixel alpha through UpdateLayeredWindow,
    means giving up tkinter's drawing entirely for a fairly small gain.
  * Fading uses the window's uniform alpha rather than per-pixel alpha.
"""

import ctypes
import json
import math
import os
import struct
import sys
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


enable_blur = _load_sibling("blur_win").enable_blur

WIDTH = 268
HEIGHT = 54
RADIUS = HEIGHT // 2
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
TARGET_ALPHA = 0.78   # with blur behind it
OPAQUE_ALPHA = 0.97   # without
FRAME_MS = 16
LEVEL_MS = 33
LEVEL_EASE = 0.30
PEAK_DECAY = 0.95
PEAK_FLOOR = 150.0
LEVEL_GAMMA = 0.65
MAX_LIFETIME_S = 600
DRAG_SLOP = 4

# Same location the GTK overlay and Config use, so a checkout shared over a
# home directory keeps one set of settings.
POSITION_FILE = os.path.join(
    os.path.expanduser("~"), ".config", "whisper-flow", "hud-position.json",
)


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _load_positions() -> dict:
    try:
        with open(POSITION_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
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
    def __init__(self, level_file: str):
        self.level_file = level_file
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
        self._drag_from = None
        self._quitting = False
        self._blur = None

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self.root.after(0, self._round_corners)
        self.root.after(0, self._enable_blur)
        self.root.after(FRAME_MS, self._frame)
        self.root.after(LEVEL_MS, self._read_levels)

    # -- window shape and placement --------------------------------------
    def _round_corners(self):
        """Clip the window to a rounded rectangle via a Win32 region."""
        try:
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            region = ctypes.windll.gdi32.CreateRoundRectRgn(
                0, 0, WIDTH + 1, HEIGHT + 1, RADIUS * 2, RADIUS * 2,
            )
            ctypes.windll.user32.SetWindowRgn(hwnd, region, True)
        except Exception as e:
            print(f"[HUD] could not round corners: {e}", flush=True)

    def _enable_blur(self):
        """Ask DWM to blur the backdrop, and note whether it agreed."""
        try:
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            self._blur = enable_blur(hwnd)
            print(f"[HUD] blur: {self._blur or 'unavailable, drawing opaque'}",
                  flush=True)
        except Exception as e:
            print(f"[HUD] blur setup failed: {e}", flush=True)

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
                self.root.destroy()
                return
        try:
            # Blurred: hold the window translucent so the acrylic shows
            # through. Not blurred: go opaque, or the pill is just a dim
            # rectangle over whatever is behind it.
            ceiling = TARGET_ALPHA if self._blur else OPAQUE_ALPHA
            self.root.attributes("-alpha", self.alpha * ceiling)
        except tk.TclError:
            return

        for i, target in enumerate(self.targets):
            self.shown[i] += (target - self.shown[i]) * LEVEL_EASE
        self._draw()
        self.root.after(FRAME_MS, self._frame)

    def _read_levels(self):
        if self.level_file:
            # The daemon deletes this when the recording ends; treat its
            # absence as the signal to go, so the overlay cannot be orphaned.
            if not os.path.exists(self.level_file):
                print("[HUD] level file gone, closing", flush=True)
                self.quit()
                return
            try:
                with open(self.level_file, "rb") as f:
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
                pass

        if time.monotonic() - self.start > MAX_LIFETIME_S:
            print("[HUD] lifetime cap reached, closing", flush=True)
            self.quit()
            return
        self.root.after(LEVEL_MS, self._read_levels)

    def _draw(self):
        c = self.canvas
        c.delete("all")
        mid = HEIGHT / 2

        c.create_rectangle(0, 0, WIDTH, HEIGHT, fill=BG, outline="")
        # The region already rounds the window; this traces the same curve so
        # the edge reads as an outline rather than a cut.
        c.create_arc(1, 1, RADIUS * 2, HEIGHT - 1, start=90, extent=180,
                     style="arc", outline=OUTLINE)
        c.create_arc(WIDTH - RADIUS * 2, 1, WIDTH - 1, HEIGHT - 1,
                     start=270, extent=180, style="arc", outline=OUTLINE)
        c.create_line(RADIUS, 1, WIDTH - RADIUS, 1, fill=OUTLINE)
        c.create_line(RADIUS, HEIGHT - 1, WIDTH - RADIUS, HEIGHT - 1, fill=OUTLINE)

        breathe = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(
            (time.monotonic() - self.start) * 3.0))
        glow = int(60 + 40 * breathe)
        c.create_oval(DOT_X - DOT_R * 2, mid - DOT_R * 2,
                      DOT_X + DOT_R * 2, mid + DOT_R * 2,
                      fill=f"#{glow:02x}1416", outline="")
        c.create_oval(DOT_X - DOT_R, mid - DOT_R, DOT_X + DOT_R, mid + DOT_R,
                      fill=DOT_COLOUR, outline="")

        step = (WAVE_R - WAVE_L) / float(BARS)
        for i, level in enumerate(self.shown):
            bar_h = max(BAR_W, level * BAR_MAX * 2)
            x = WAVE_L + i * step
            c.create_rectangle(x, mid - bar_h / 2, x + BAR_W, mid + bar_h / 2,
                               fill=BAR_COLOUR, outline="")

    def run(self):
        self.root.mainloop()


def main() -> int:
    level_file = os.environ.get("WHISPER_FLOW_HUD_LEVEL_FILE", "")
    print(f"[HUD] starting level_file={level_file}", flush=True)
    try:
        # Per-monitor DPI aware, or the overlay is scaled and blurry.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    HudWindow(level_file).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
