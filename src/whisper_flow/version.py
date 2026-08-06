"""Which build this is, in one place.

The daemon writes it into failure reports, the tray menu shows it, the
settings window shows it and `whisper-flow --version` prints it. It lived in
daemon.py while the log was the only thing that wanted it; importing the
daemon to answer "which version is this" costs the tray stack and the whole
config layer, which is why it moved here. Nothing but sys and pathlib.
"""

import sys
from pathlib import Path

from . import __version__


def build_version() -> str:
    """Which build this is, not which release it was branched from.

    __version__ is 0.4.0 in every build ever made from this commit series,
    so every log said 0.4.0 whether it was build 155 or build 170 - and a
    log that cannot say which build it came from cannot say whether a fix is
    in it. Three builds failed and published nothing while that was true,
    and the answer to "none of the fixes worked" turned out to be that none
    of them had shipped.

    The packaged version is written beside the executable at package time,
    where nothing but the build can invent it. A source checkout has no such
    file and says so.
    """
    try:
        stamp = Path(sys.executable).with_name("BUILD.txt")
        if stamp.exists():
            packaged = stamp.read_text(encoding="utf-8").strip()
            if packaged:
                return packaged
    except OSError:
        pass
    return f"{__version__} (source)"
