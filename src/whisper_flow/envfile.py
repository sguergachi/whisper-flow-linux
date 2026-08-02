"""Reading and writing the .env settings file.

Settings are plain KEY=value lines, hand-editable and shared between the
daemon, the CLI and the settings window. Writes are surgical: existing lines
keep their place and their comments, keys being removed disappear, new keys
append at the end, and the file is replaced atomically so a crash mid-write
cannot leave it half-written.
"""

import os
from pathlib import Path


def get(path: Path, key: str) -> str | None:
    """The value of one key, or None if it is not set."""
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1]
    except OSError:
        pass
    return None


def set_values(path: Path, updates: dict[str, str | None]) -> None:
    """Apply updates to the file, preserving everything else.

    A value of None removes the key - the only way to unset something like
    MIC_DEVICE_INDEX, where an empty value would fail to parse as an int.
    """
    path = Path(path)
    lines = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    remaining = dict(updates)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key and not line.lstrip().startswith("#") and key in remaining:
            value = remaining.pop(key)
            if value is not None:
                out.append(f"{key}={value}")
            # None: the line is dropped, unsetting the key
        else:
            out.append(line)

    for key, value in remaining.items():
        if value is not None:
            out.append(f"{key}={value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.replace(tmp, path)
