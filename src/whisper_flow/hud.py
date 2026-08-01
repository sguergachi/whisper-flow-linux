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
import threading

IS_WINDOWS = sys.platform == "win32"


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
        # show() runs on the thread starting a recording, hide() on the one
        # handling the hotkey release. Without this, a hide landing inside
        # show()'s Popen finds no process to kill and leaks an overlay that
        # nothing will ever take down. Reentrant because show() calls hide().
        self._lock = threading.RLock()

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
        with self._lock:
            self._show_locked(level_file, monitor, point)

    def _show_locked(self, level_file, monitor, point):
        self.hide()

        env = os.environ.copy()
        if not IS_WINDOWS:
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
        preload = None if IS_WINDOWS else _layer_shell_library()
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
        # hud_app is GTK4 plus Wayland layer-shell; hud_win is tkinter. Same
        # contract either way: level file in, overlay out.
        argv = self._overlay_command()

        fd, self._log_path = tempfile.mkstemp(suffix=".log", prefix="whisper-flow-hud-")
        os.close(fd)

        try:
            self._log_file = open(self._log_path, "a")
            # setsid groups the child on POSIX so the whole overlay can be
            # signalled; Windows has no equivalent and no such argument.
            extra = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                     if IS_WINDOWS else {"preexec_fn": os.setsid})
            self._process = subprocess.Popen(
                argv,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=env,
                **extra,
            )
        except Exception as e:
            print(f"[HUD] Failed to spawn HUD: {e}", flush=True)
            self._cleanup_files()

    @staticmethod
    def _overlay_command() -> list[str]:
        """How to launch the overlay, source tree or frozen build.

        A frozen build has no source files and sys.executable is the app
        itself, so the overlay ships as its own executable beside it.
        """
        if getattr(sys, "frozen", False):
            exe = "whisper-flow-hud.exe" if IS_WINDOWS else "whisper-flow-hud"
            return [os.path.join(os.path.dirname(sys.executable), exe)]
        module = "hud_win.py" if IS_WINDOWS else "hud_app.py"
        return [sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), module)]

    def hide(self):
        """Hide the recording HUD overlay."""
        with self._lock:
            self._hide_locked()

    def _hide_locked(self):
        if self._process:
            proc = self._process
            self._process = None
            try:
                if IS_WINDOWS:
                    proc.terminate()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            # Do not wait here. This runs on the thread dispatching hotkey
            # callbacks, and the overlay takes a fade to exit; blocking would
            # hold up every press that lands during it. Reap in the background
            # so the child does not linger as a zombie.
            threading.Thread(
                target=self._reap, args=(proc,), daemon=True,
                name="whisper-flow-hud-reap",
            ).start()
        self._cleanup_files()

    @staticmethod
    def _reap(proc):
        """Collect the exited overlay, forcing it if the fade never finishes."""
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                if IS_WINDOWS:
                    proc.kill()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=2)
            except Exception:
                pass

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
