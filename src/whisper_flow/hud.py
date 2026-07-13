"""HUD overlay for whisper-flow showing recording status with audio waveform."""

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time


HUD_SCRIPT = textwrap.dedent("""\
    import os
    import struct
    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk, Gdk, GLib
    import math

    LEVEL_FILE = os.environ.get("WHISPER_FLOW_HUD_LEVEL_FILE", "")

    WIDTH = 320
    HEIGHT = 100
    WAVE_HEIGHT = 40
    WAVE_BARS = 60
    TOP_HEIGHT = 32

    class RecordingHUD(Gtk.Window):
        def __init__(self):
            Gtk.Window.__init__(self)
            self.set_title("")
            self.set_decorated(False)
            self.set_resizable(False)
            self.set_keep_above(True)
            self.set_accept_focus(False)
            self.set_focus_on_map(False)
            self.set_skip_taskbar_hint(True)
            self.set_skip_pager_hint(True)
            self.set_type_hint(Gdk.WindowTypeHint.TOOLTIP)

            screen = Gdk.Screen.get_default()
            visual = screen.get_rgba_visual()
            if visual:
                self.set_visual(visual)
            self.set_app_paintable(True)

            self.set_size_request(WIDTH, HEIGHT)

            self.connect("draw", self._on_draw)
            self.connect("map", self._center)

            self.levels = [0.0] * WAVE_BARS
            self.blink = True
            self.level_file_pos = 0
            self.opacity = 0.0

            GLib.timeout_add(600, self._tick_blink)
            GLib.timeout_add(40, self._tick_levels)
            GLib.timeout_add(20, self._fade_in)

            self.set_opacity(0.0)

        def _tick_blink(self):
            self.blink = not self.blink
            self.queue_draw()
            return True

        def _tick_levels(self):
            if not LEVEL_FILE:
                return True
            try:
                with open(LEVEL_FILE, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    if size < 4:
                        return True
                    count = size // 4
                    read_start = max(0, (count - WAVE_BARS) * 4)
                    f.seek(read_start)
                    data = f.read(WAVE_BARS * 4)
                    raw = struct.unpack(f"<{len(data)//4}i", data) if len(data) >= 4 else []
                    if raw:
                        max_val = max(abs(v) for v in raw) or 1
                        self.levels = [min(1.0, abs(v) / 1000.0) for v in raw]
                        if len(self.levels) < WAVE_BARS:
                            self.levels = [0.0] * (WAVE_BARS - len(self.levels)) + self.levels
                        self.levels = self.levels[-WAVE_BARS:]
                        self.queue_draw()
            except OSError as e:
                import sys
                sys.stdout.write(f"[HUD] level file OSError: {e}\\n")
                sys.stdout.flush()
            except Exception as e:
                import sys
                sys.stdout.write(f"[HUD] level file error: {e}\\n")
                sys.stdout.flush()
            return True

        def _fade_in(self):
            self.opacity = min(1.0, self.opacity + 0.08)
            self.set_opacity(self.opacity)
            return self.opacity < 1.0

        def _center(self, *args):
            display = Gdk.Display.get_default()
            monitor = display.get_primary_monitor()
            if monitor is None:
                monitor = display.get_monitor(0)
            if monitor is None:
                return
            geometry = monitor.get_geometry()
            x = geometry.x + (geometry.width - WIDTH) // 2
            y = geometry.y + 36
            self.move(x, y)

        def _on_draw(self, widget, cr):
            w = widget.get_allocated_width()
            h = widget.get_allocated_height()

            cr.set_source_rgba(0.08, 0.08, 0.08, 0.88)
            _rounded_rect(cr, 0, 0, w, h, 12)
            cr.fill()

            dot_color = "#ff3333" if self.blink else "#cc0000"
            cr.set_source_rgba(1.0, 0.2, 0.2, 1.0 if self.blink else 0.7)
            cr.arc(24, 16, 5, 0, 2 * 3.14159)
            cr.fill()

            cr.set_source_rgba(1, 1, 1, 0.9)
            cr.select_font_face("Sans", 0, 1)
            cr.set_font_size(13)
            cr.move_to(38, 21)
            cr.show_text("Listening…")

            bar_w = (w - 24) / WAVE_BARS
            mid_y = TOP_HEIGHT + WAVE_HEIGHT // 2
            max_h = WAVE_HEIGHT // 2 - 2

            for i, level in enumerate(self.levels):
                bar_h = max(1, int(level * max_h))
                x = 12 + i * bar_w
                r, g, b = _level_color(level)
                cr.set_source_rgba(r, g, b, 0.9)
                cr.rectangle(x, mid_y - bar_h, bar_w - 1, bar_h * 2)
                cr.fill()

    def _level_color(level):
        if level < 0.3:
            return 0.2, 1.0, 0.3
        if level < 0.6:
            return 1.0, 0.9, 0.2
        return 1.0, 0.3, 0.2

    def _rounded_rect(cr, x, y, w, h, r):
        cr.arc(x+r, y+r, r, 180*3.14159/180, 270*3.14159/180)
        cr.arc(x+w-r, y+r, r, 270*3.14159/180, 360*3.14159/180)
        cr.arc(x+w-r, y+h-r, r, 0, 90*3.14159/180)
        cr.arc(x+r, y+h-r, r, 90*3.14159/180, 180*3.14159/180)
        cr.close_path()

    import sys
    sys.stdout.write(f"[HUD] Starting with LEVEL_FILE={LEVEL_FILE}\\n")
    sys.stdout.flush()
    win = RecordingHUD()
    win.show_all()
    Gtk.main()
""")


class HUD:
    """Manages a HUD overlay subprocess for recording status."""

    def __init__(self):
        self._process = None
        self._log_path = None

    def show(self, level_file: str = ""):
        """Show the recording HUD overlay.

        Args:
            level_file: Path to a file containing audio level data (int16 values)

        """
        self.hide()

        env = os.environ.copy()
        # Ensure display environment is available for HUD subprocess
        env.setdefault("WAYLAND_DISPLAY", "wayland-0")
        env.setdefault("GDK_BACKEND", "wayland")
        env.setdefault("NO_AT_BRIDGE", "1")
        if level_file:
            env["WHISPER_FLOW_HUD_LEVEL_FILE"] = level_file

        fd, path = tempfile.mkstemp(suffix=".py", prefix="whisper-flow-hud-")
        with os.fdopen(fd, "w") as f:
            f.write(HUD_SCRIPT)

        # Write HUD stderr to a log file (avoids pipe buffer deadlock)
        fd2, self._log_path = tempfile.mkstemp(suffix=".log", prefix="whisper-flow-hud-")
        os.close(fd2)

        self._process = subprocess.Popen(
            [sys.executable, path],
            stdout=open(self._log_path, "a"),
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

        # Wait briefly and check if the process is still alive
        time.sleep(0.3)
        if self._process and self._process.poll() is not None:
            with open(self._log_path) as f:
                stderr_out = f.read()
            print(f"[HUD] Failed to start HUD: {stderr_out}", flush=True)

    def hide(self):
        """Hide the recording HUD overlay."""
        if self._process:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                self._process.wait(timeout=2)
            except Exception:
                pass
            self._process = None
        if self._log_path:
            try:
                os.unlink(self._log_path)
            except OSError:
                pass
            self._log_path = None
