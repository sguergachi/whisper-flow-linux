"""Supervises the HUD overlay subprocess.

The overlay itself lives in whisper_flow.hud_app and runs as its own process,
so that GTK - which is not thread-safe and can block on the compositor - can
never stall the audio capture loop.
"""

import os
import signal
import subprocess
import sys
import tempfile


LAYER_SHELL_CANDIDATES = (
    "/usr/lib/libgtk4-layer-shell.so",
    "/usr/lib64/libgtk4-layer-shell.so",
    "/usr/lib/x86_64-linux-gnu/libgtk4-layer-shell.so",
)


def _layer_shell_library() -> str | None:
    """Path to libgtk4-layer-shell, or None if it is not installed."""
    for path in LAYER_SHELL_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


class HUD:
    """Manages a HUD overlay subprocess for recording status."""

    def __init__(self):
        self._process = None
        self._log_path = None
        self._log_file = None

    def show(self, level_file: str = "", monitor: str | None = None,
             point: tuple[int, int] | None = None):
        """Show the recording HUD overlay.

        Returns as soon as the subprocess is spawned. This runs on the path
        that starts a recording, so it must not block: any wait here delays
        the microphone opening and stalls the caller.

        Args:
            level_file: Path to a file containing audio level data (int16 values)
            monitor: Connector name of the output to show on, e.g. "DP-1"

        """
        self.hide()

        env = os.environ.copy()
        # The overlay is a Wayland layer-shell surface; it has no X11 path.
        env.setdefault("WAYLAND_DISPLAY", "wayland-0")
        env["GDK_BACKEND"] = "wayland"
        env.setdefault("NO_AT_BRIDGE", "1")
        # The pill is a few hundred pixels of 2D cairo drawing. GTK4's GPU
        # renderers spend ~450ms building a GL/Vulkan context for it before the
        # first frame; the software renderer draws it in a fraction of that.
        env.setdefault("GSK_RENDERER", "cairo")
        # Keep the script's own directory off sys.path: this package ships a
        # logging.py that would otherwise shadow the standard library's.
        env["PYTHONSAFEPATH"] = "1"
        # gtk4-layer-shell must be loaded ahead of libwayland-client or it
        # cannot intercept surface creation, and the window silently falls
        # back to an ordinary toplevel.
        preload = _layer_shell_library()
        if preload:
            existing = env.get("LD_PRELOAD", "")
            env["LD_PRELOAD"] = f"{preload}:{existing}" if existing else preload
        if level_file:
            env["WHISPER_FLOW_HUD_LEVEL_FILE"] = level_file
        if monitor:
            env["WHISPER_FLOW_HUD_MONITOR"] = monitor
        if point:
            # Only a placement hint: a malformed one must never stop the HUD
            # from appearing, it just falls back to the first monitor.
            try:
                env["WHISPER_FLOW_HUD_POINT"] = f"{int(point[0])},{int(point[1])}"
            except (TypeError, ValueError, IndexError, KeyError):
                pass

        # Launched by path rather than with -m: importing the package would
        # pull in the daemon and with it pystray's GTK 3, and one process
        # cannot hold both GTK 3 and the overlay's GTK 4.
        hud_app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hud_app.py")

        fd, self._log_path = tempfile.mkstemp(suffix=".log", prefix="whisper-flow-hud-")
        os.close(fd)

        try:
            self._log_file = open(self._log_path, "a")
            self._process = subprocess.Popen(
                [sys.executable, hud_app],
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,
            )
        except Exception as e:
            print(f"[HUD] Failed to spawn HUD: {e}", flush=True)
            self._cleanup_files()

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
        self._cleanup_files()

    def _cleanup_files(self):
        """Close and remove the temp files backing the HUD subprocess."""
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        if self._log_path:
            try:
                os.unlink(self._log_path)
            except OSError:
                pass
            self._log_path = None
