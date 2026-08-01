"""Logging utilities for whisper-flow.

Lines are always kept in a small ring buffer, whether or not logging is
switched on. Printing is the part that is optional; remembering is not.

The frozen Windows build runs with console=False, so anything printed there
goes nowhere at all. Without this buffer a failure notification could say
that something went wrong but never what, which is as useful as silence when
the report has to travel through a person.
"""

import threading
from collections import deque
from datetime import datetime

# Global logging state
_logging_enabled = False

# Enough to cover a whole recording attempt and the startup before it, and
# small enough to paste into a message.
_RING_SIZE = 300
_ring: deque[str] = deque(maxlen=_RING_SIZE)
_ring_lock = threading.Lock()


def set_logging_enabled(enabled: bool):
    """Set whether logging is enabled globally."""
    global _logging_enabled
    _logging_enabled = enabled


def log(*args, **kwargs):
    """Record a line, and print it when logging is enabled."""
    try:
        line = " ".join(str(a) for a in args)
        with _ring_lock:
            _ring.append(f"{datetime.now():%H:%M:%S} {line}")
    except Exception:
        pass                    # logging must never be the thing that fails
    if _logging_enabled:
        print(*args, **kwargs, flush=True)


def recent_log(limit: int = 80) -> str:
    """The most recent lines, oldest first, for a failure report."""
    with _ring_lock:
        lines = list(_ring)
    return "\n".join(lines[-limit:])


def clear_log() -> None:
    """Drop what has been recorded. Only tests should need this."""
    with _ring_lock:
        _ring.clear()
