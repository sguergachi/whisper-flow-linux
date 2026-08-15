"""Filesystem locations shared between processes.

Kept tiny and dependency-free: the settings and setup windows import this,
and they must not drag the daemon (and with it pystray, pyaudio and PIL)
into a process that only needs a path.
"""

from pathlib import Path


def pid_file() -> Path:
    """Where the daemon writes its process id.

    Historically ~/.config/whisper-flow on every platform, including Windows,
    so it stays here rather than moving under LOCALAPPDATA and stranding the
    stop/status commands that already look here.
    """
    return Path.home() / ".config" / "whisper-flow" / "daemon.pid"


def lock_file() -> Path:
    """Exclusive lock so a second daemon cannot start.

    The pid file is not enough: a second copy overwrites it, then fails to
    grab the keyboards the first copy already holds, and `whisper-flow stop`
    kills the broken one. The lock is released when the process dies.
    """
    return pid_file().with_name("daemon.lock")
