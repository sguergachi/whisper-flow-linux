"""Restarting the daemon, from a process that is not the daemon.

Settings that change hotkeys, audio or notifications are only read when the
daemon starts, so saving them has to be followed by a restart. The daemon
cannot easily respawn itself - on Windows its replacement would trip the
single-instance mutex while the old process still holds it - so the settings
window does it from outside: stop the old process, wait for it to die, then
start a new one.

Where systemd supervises the daemon this is just a unit restart.
"""

import os
import subprocess
import sys
import time

from .logging import log
from .paths import pid_file

UNIT = "whisper-flow.service"


def systemd_unit_active() -> bool:
    """Whether systemd is currently supervising the daemon (Linux only)."""
    if sys.platform == "win32":
        return False
    if not _which("systemctl"):
        return False
    result = subprocess.run(
        ["systemctl", "--user", "is-active", UNIT],
        capture_output=True, check=False,
    )
    return result.stdout.strip() == b"active"


def _which(name: str) -> bool:
    from shutil import which
    return which(name) is not None


def daemon_pid() -> int | None:
    """The pid of the running daemon, or None."""
    try:
        return int(pid_file().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def respawn_command() -> list[str]:
    """How to start a fresh daemon from this process."""
    if getattr(sys, "frozen", False):
        # The one executable, with no arguments, is the daemon.
        return [sys.executable]
    return [sys.executable, "-m", "whisper_flow.cli", "daemon", "--foreground"]


def _wait_for_exit(pid: int, timeout: float = 10.0) -> bool:
    """Block until the process is gone. False if it outlasted the timeout."""
    if sys.platform == "win32":
        import ctypes
        SYNCHRONIZE = 0x100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return True             # already gone
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(
                handle, int(timeout * 1000)) == 0
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)         # sig 0: existence check, no signal sent
        except OSError:
            return True
        time.sleep(0.1)
    return False


def restart_daemon() -> tuple[bool, str]:
    """Stop the daemon and start it again. Returns (ok, what happened).

    Must not take the settings window down with it, and must not block the
    settings UI for the whole restart. The settings process is a child of
    the daemon (prewarmed at login); systemd's default KillMode=control-group
    would kill that child on ``systemctl restart``. So under systemd we
    signal only the main process, then start the unit again.

    Waits for the old process are short and always release the GIL
    (``time.sleep``): a multi-second block with the GIL held froze the
    settings window until restart finished.
    """
    if systemd_unit_active():
        # --kill-who=main: only the daemon, not its settings/HUD children.
        result = subprocess.run(
            ["systemctl", "--user", "kill", "-s", "SIGTERM",
             "--kill-who=main", UNIT],
            capture_output=True, check=False,
        )
        if result.returncode != 0:
            return False, result.stderr.decode(errors="replace").strip() or \
                "systemctl refused to stop the daemon"
        # Brief pause so the old main is gone; do not wait the full cleanup
        # (which used to include a multi-second wait on the settings child).
        pid = daemon_pid()
        if pid:
            _wait_for_exit(pid, timeout=1.5)
        result = subprocess.run(
            ["systemctl", "--user", "start", UNIT],
            capture_output=True, check=False,
        )
        if result.returncode == 0:
            return True, "restarting via systemd"
        return False, result.stderr.decode(errors="replace").strip() or \
            "systemctl refused to start the daemon"

    pid = daemon_pid()
    if pid:
        try:
            # SIGTERM on POSIX; on Windows os.kill maps to TerminateProcess,
            # which is also how `whisper-flow stop` has always done it.
            os.kill(pid, 15)
        except OSError as e:
            log(f"[SETTINGS] could not signal daemon {pid}: {e}")
        # Short wait only - settings is still open and must stay responsive.
        if not _wait_for_exit(pid, timeout=1.5):
            log(f"[SETTINGS] daemon {pid} still running after SIGTERM; "
                "starting a replacement anyway")

    try:
        subprocess.Popen(
            respawn_command(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=sys.platform != "win32",
        )
    except Exception as e:
        return False, f"could not start the daemon: {e}"
    return True, "daemon restarted"
