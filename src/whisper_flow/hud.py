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

from .logging import log

IS_WINDOWS = sys.platform == "win32"

# Keep one overlay process alive and command it, instead of starting one per
# recording. Starting a frozen process is the largest cost between the hotkey
# and anything appearing on screen, and no amount of making our own code
# faster removes it.
#
# Windows only for now. The GTK overlay there is a Wayland layer-shell
# surface with its own lifecycle, and it already appears fast enough that
# this would be risk without reward.
RESIDENT = IS_WINDOWS


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

    def _overlay_env(self, level_file, monitor=None, point=None):
        """Environment for the overlay process."""
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
        return env

    def _show_locked(self, level_file, monitor, point):
        # A resident overlay is already running and already built; showing it
        # is a message down a pipe rather than a process start. Everything
        # below - spawning, loading Python, creating a window - is what made
        # the overlay take a visible moment to appear on Windows.
        if RESIDENT and self._command(f"show {level_file}"):
            return
        self.hide()

        env = self._overlay_env(level_file, monitor, point)
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
            log(f"[HUD] Failed to spawn overlay {argv}: {e}")
            self._cleanup_files()

    def crash_log(self) -> str:
        """Whatever the overlay printed before dying, for a failure report.

        The overlay is a separate process writing to a temp file that is
        normally deleted, so without this a crashing HUD is indistinguishable
        from one that simply did not appear.
        """
        try:
            if self._process and self._process.poll() not in (None, 0):
                path = self._log_path
                if path and os.path.exists(path):
                    text = open(path, errors="replace").read().strip()
                    if text:
                        return (f"overlay exited {self._process.poll()}\n"
                                + text[-2000:])
                return f"overlay exited {self._process.poll()} with no output"
        except Exception as e:
            return f"could not read the overlay log: {e}"
        return ""

    @staticmethod
    def _overlay_command() -> list[str]:
        """How to launch the overlay, source tree or frozen build.

        A frozen build has no source files and sys.executable is the app
        itself, so the overlay ships as its own executable beside it.
        """
        if getattr(sys, "frozen", False):
            # The same binary, told to be the overlay. One executable ships,
            # so there is no question about which one to run.
            return [sys.executable, "--hud"]
        module = "hud_win.py" if IS_WINDOWS else "hud_app.py"
        return [sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), module)]

    def prewarm(self) -> None:
        """Start the resident overlay now, so the first press is fast too.

        Without this it starts on the first recording, which means the first
        press of a session still pays process startup - the very cost the
        resident overlay exists to avoid. The daemon runs from login, so
        doing it here means the overlay has been ready for a long time by
        the time anyone dictates.

        On a thread: starting a process is quick, but the overlay taking
        longer to appear must not hold up the tray icon.
        """
        if not RESIDENT:
            return

        def work():
            try:
                with self._lock:
                    if self._resident_process():
                        log("[HUD] overlay ready")
            except Exception as e:
                log(f"[HUD] could not pre-start the overlay: {e}")

        threading.Thread(target=work, daemon=True,
                         name="whisper-flow-hud-prewarm").start()

    def _command(self, line: str) -> bool:
        """Send one order to a resident overlay. False if there is not one.

        Starts one if there is none - after a crash, or before prewarm has
        finished - so a missing overlay costs one slow press rather than
        being permanently absent.
        """
        for attempt in (1, 2):
            process = self._resident_process()
            if not process:
                return False
            try:
                process.stdin.write(f"{line}\n")
                process.stdin.flush()
                return True
            except (OSError, ValueError):
                # It died between the check and the write. Drop it and let
                # the second attempt start a fresh one.
                log(f"[HUD] resident overlay went away on attempt {attempt}")
                self._discard_resident()
        return False

    def _resident_process(self):
        """The live overlay process, started if it is not running."""
        if self._process and self._process.poll() is None:
            return self._process
        self._discard_resident()

        env = self._overlay_env("")
        env["WHISPER_FLOW_HUD_RESIDENT"] = "1"
        fd, self._log_path = tempfile.mkstemp(suffix=".log", prefix="whisper-flow-hud-")
        os.close(fd)
        try:
            self._log_file = open(self._log_path, "a")
            self._process = subprocess.Popen(
                self._overlay_command(),
                stdin=subprocess.PIPE,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                               if IS_WINDOWS else 0),
            )
        except Exception as e:
            log(f"[HUD] could not start the resident overlay: {e}")
            self._process = None
            self._cleanup_files()
        return self._process

    def _discard_resident(self):
        """Forget a dead overlay without touching a live one."""
        if self._process and self._process.poll() is None:
            return
        self._process = None
        self._cleanup_files()

    def shutdown(self):
        """Stop the overlay for good. Called when the daemon exits.

        Closing stdin is the ordinary path: the overlay reads end of stream
        and leaves. That is also the safety net if the daemon dies without
        calling this - the pipe closes either way, so a resident overlay
        cannot be stranded on screen.
        """
        with self._lock:
            process = self._process
            self._process = None
            if not process:
                return
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            try:
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            self._cleanup_files()

    def hide(self):
        """Hide the recording HUD overlay."""
        with self._lock:
            if RESIDENT and self._process and self._process.poll() is None:
                self._command("hide")
                return
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
