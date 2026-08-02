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
