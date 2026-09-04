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
import re
import subprocess
import sys
import time
from pathlib import Path

from .logging import log
from .paths import pid_file

UNIT = "whisper-flow.service"


def systemd_unit_active() -> bool:
    """Whether systemd is currently supervising the daemon (Linux only)."""
    if sys.platform == "win32":
        return False
    if not _which("systemctl"):
        return False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", UNIT],
            capture_output=True, check=False, timeout=2,
        )
    except subprocess.TimeoutExpired:
        return False
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


def _desktop_exec() -> str | None:
    """The Exec= path from the first-run desktop entry, if still on disk.

    First-run integration writes a stable AppImage path into
    ~/.local/share/applications/whisper-flow.desktop. Useful when APPIMAGE
    is missing (extracted runs) but the menu entry still points at the
    right file.
    """
    data_home = os.environ.get("XDG_DATA_HOME") or str(
        Path.home() / ".local" / "share")
    desktop = Path(data_home) / "applications" / "whisper-flow.desktop"
    try:
        text = desktop.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith("Exec="):
            continue
        # Exec="/path/with spaces/app.AppImage"  or  Exec=/path/app.AppImage
        raw = line[5:].strip()
        match = re.match(r'^"([^"]+)"|^(\S+)', raw)
        if not match:
            continue
        path = match.group(1) or match.group(2)
        if path and Path(path).is_file():
            return path
    return None


def frozen_binary() -> str:
    """Stable path of the frozen app binary, if this is a frozen build.

    Under AppImage, ``sys.executable`` is the FUSE mount
    (``/tmp/.mount_*/whisper-flow``). That path dies the moment the
    process that held the mount is killed - which is exactly what Save
    does to the daemon before respawning it. The AppImage runtime
    exports ``APPIMAGE`` as the real file; that path survives.
    """
    if not getattr(sys, "frozen", False):
        return sys.executable
    # Prefer the real AppImage file over the mount.
    try:
        from .desktop_install import appimage_path
        img = appimage_path()
        if img:
            return img
    except Exception:
        pass
    if os.path.isfile(sys.executable):
        return sys.executable
    desktop = _desktop_exec()
    if desktop:
        return desktop
    return sys.executable


def respawn_command() -> list[str]:
    """How to start a fresh daemon from this process."""
    if getattr(sys, "frozen", False):
        # The one executable, with no arguments, is the daemon.
        return [frozen_binary()]
    return [sys.executable, "-m", "whisper_flow.cli", "daemon", "--foreground"]


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` still exists. A failed query counts as gone."""
    if sys.platform == "win32":
        import ctypes
        # PROCESS_QUERY_LIMITED_INFORMATION is enough to read the exit
        # code. WaitForSingleObject was used here first; ctypes holds the
        # GIL for the whole wait, so a 0.8s Save restart froze every
        # Python callback in the settings window.
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(exit_code))
            return bool(ok) and exit_code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)             # sig 0: existence check, no signal sent
        return True
    except OSError:
        return False


def _wait_for_exit(pid: int, timeout: float = 10.0) -> bool:
    """Block until the process is gone. False if it outlasted the timeout.

    Sleeps between probes so the GIL is released; a ctypes wait that
    held it froze the settings window on Save.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


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

    Never raises: callers in the settings UI treat any exception as a
    frozen/broken Save path, so every failure is a ``(False, reason)``.
    """
    try:
        return _restart_daemon()
    except Exception as e:
        log(f"[SETTINGS] restart_daemon failed: {e}")
        return False, str(e) or e.__class__.__name__


def _restart_daemon() -> tuple[bool, str]:
    if systemd_unit_active():
        # --kill-who=main: only the daemon, not its settings/HUD children.
        # Every systemctl call is hard-capped so a stuck dbus cannot freeze
        # the settings worker (and, if the GIL were held, the UI).
        try:
            result = subprocess.run(
                ["systemctl", "--user", "kill", "-s", "SIGTERM",
                 "--kill-who=main", UNIT],
                capture_output=True, check=False, timeout=2,
            )
        except subprocess.TimeoutExpired:
            return False, "systemctl kill timed out"
        except OSError as e:
            return False, f"systemctl kill failed: {e}"
        if result.returncode != 0:
            return False, result.stderr.decode(errors="replace").strip() or \
                "systemctl refused to stop the daemon"
        # Brief pause so the old main is gone; keep it short.
        pid = daemon_pid()
        if pid:
            _wait_for_exit(pid, timeout=0.8)
        try:
            result = subprocess.run(
                ["systemctl", "--user", "start", UNIT],
                capture_output=True, check=False, timeout=3,
            )
        except subprocess.TimeoutExpired:
            # Start may still be proceeding; treat as success so Save returns.
            return True, "restarting via systemd (start still running)"
        except OSError as e:
            return False, f"systemctl start failed: {e}"
        if result.returncode == 0:
            return True, "restarting via systemd"
        return False, result.stderr.decode(errors="replace").strip() or \
            "systemctl refused to start the daemon"

    # Resolve the binary *before* killing the daemon. Under AppImage the
    # mount path in sys.executable can vanish the moment the process that
    # held the FUSE mount exits; APPIMAGE / the desktop entry do not.
    cmd = respawn_command()
    if not cmd or not os.path.isfile(cmd[0]):
        return False, (
            f"cannot find the daemon binary to restart "
            f"({cmd[0] if cmd else 'empty command'}). "
            f"Start WhisperFlow from the AppImage again."
        )

    pid = daemon_pid()
    if pid:
        try:
            # SIGTERM on POSIX; on Windows os.kill maps to TerminateProcess,
            # which is also how `whisper-flow stop` has always done it.
            os.kill(pid, 15)
        except OSError as e:
            log(f"[SETTINGS] could not signal daemon {pid}: {e}")
        # Wait for the old process to actually die: respawning while it still
        # holds the single-instance mutex makes the replacement exit at once
        # ("already running") and the tray never comes back.
        if not _wait_for_exit(pid, timeout=3.0):
            log(f"[SETTINGS] daemon {pid} still running after SIGTERM; "
                "starting a replacement anyway")

    try:
        # CREATE_NO_WINDOW: a source checkout respawns python.exe; without
        # this a Save-triggered restart flashes a console on Windows.
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=sys.platform != "win32",
            creationflags=flags,
        )
    except Exception as e:
        return False, f"could not start the daemon: {e}"
    # Popen succeeding only means a process started, not that the daemon
    # stayed up: an instant exit (stale mutex, platform check) used to be
    # reported as success while the tray never returned.
    if _replacement_is_up(pid):
        return True, "daemon restarted"
    return False, ("the daemon did not stay up after restart "
                   "(tray missing?) — launch WhisperFlow once by hand, "
                   "then use Copy log from its tray menu")


def _replacement_is_up(old_pid, timeout: float = 6.0) -> bool:
    """Whether the pid file now points at a different, living process."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new_pid = daemon_pid()
        if new_pid and new_pid != old_pid and _pid_alive(new_pid):
            # Give it another beat: a process that dies during startup would
            # still look alive at this instant.
            time.sleep(1.0)
            if _pid_alive(daemon_pid() or 0):
                return True
        time.sleep(0.25)
    return False
