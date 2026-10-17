"""Filesystem locations shared between processes.

Kept tiny and dependency-free: the settings and setup windows import this,
and they must not drag the daemon (and with it pystray, pyaudio and PIL)
into a process that only needs a path.
"""

from pathlib import Path
import os
import re


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


def find_wayland_socket() -> str | None:
    """The Wayland socket a window on this desktop should address.

    Most sessions name theirs wayland-0, but no rule says so - KWin hands out
    wayland-1 on some setups - and a HUD pointed at a socket that does not
    exist simply never appears. The session's runtime dir is the registry:
    whatever sockets it holds are numbered wayland-N, beside their .lock
    files, which the exact shape filters out.

    An inherited WAYLAND_DISPLAY only stands as long as its socket actually
    exists there: systemd units pinned to wayland-0 have outlived sessions
    on wayland-1, and re-publishing the same answer as the session does
    keeps every listener pointing at one place.
    """
    claimed = os.environ.get("WAYLAND_DISPLAY")
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if claimed and runtime and os.path.exists(os.path.join(runtime, claimed)):
        return None
    try:
        names = os.listdir(runtime) if runtime else []
    except OSError:
        return None
    for name in sorted(names):
        if _SOCKET_NAME.fullmatch(name) and os.path.exists(
                os.path.join(runtime, name)):
            return name
    return None


# The seats' sockets are numbered wayland-N; their .lock files are their name
# with a suffix, and nothing else wayland-* belongs to the session.
_SOCKET_NAME = re.compile(r"wayland-[0-9]+")
