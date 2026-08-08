"""First-run desktop integration for the frozen Linux build.

The AppImage is a portable file: double-clicking it runs the daemon and
nothing else. The first time it does, this module registers the app with
the desktop — app menu entry, icon, login autostart — so WhisperFlow
behaves like a normally installed program from then on. The entry is
re-pointed when the AppImage moves or is updated, and source runs are left
alone (the venv installer owns those).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_MENU_ENTRY = """\
[Desktop Entry]
Type=Application
Version=1.0
Name=WhisperFlow
GenericName=Voice typing
Comment=Hold a key, talk, and the words appear where you type
Exec={exec}
Icon=whisper-flow
Terminal=false
Categories=Utility;Accessibility;
Keywords=voice;dictation;speech;whisper;transcription;
StartupNotify=false
"""

_AUTOSTART_ENTRY = """\
[Desktop Entry]
Type=Application
Name=WhisperFlow
GenericName=Voice typing
Comment=Hold a key, talk, and the words appear where you type
Exec={exec}
Icon=whisper-flow
Terminal=false
StartupNotify=false
X-GNOME-Autostart-enabled=true
"""


def appimage_path() -> str | None:
    """Path of the AppImage this process is running from, if any.

    The AppImage runtime exports APPIMAGE; without it (extracted runs,
    plain onedir builds) there is no stable file to point a desktop entry
    at and integration is skipped.
    """
    img = os.environ.get("APPIMAGE")
    if img and Path(img).is_file():
        return str(Path(img).resolve())
    return None


def _icon_source() -> Path | None:
    """The icon shipped next to the binary, if present."""
    if not getattr(sys, "frozen", False):
        return None
    candidates = [
        Path(sys.executable).parent / "whisper-flow.png",
        Path(sys._MEIPASS) / "whisper-flow.png",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _install_if_different(path: Path, content: str) -> bool:
    """Write a file, atomically, only when its content changed."""
    try:
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            return False
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.chmod(tmp, 0o644)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        return False
    return True


def ensure_desktop_integration() -> None:
    """Register the AppImage with the desktop unless it already points here."""
    if not getattr(sys, "frozen", False):
        return
    if os.environ.get("WHISPER_FLOW_SKIP_DESKTOP") == "1":
        return
    img = appimage_path()
    if not img:
        return

    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")

    apps_dir = Path(data_home) / "applications"
    icons_dir = Path(data_home) / "icons" / "hicolor" / "256x256" / "apps"
    autostart_dir = Path(config_home) / "autostart"
    for d in (apps_dir, icons_dir, autostart_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

    source = _icon_source()
    icon_dst = icons_dir / "whisper-flow.png"
    if source:
        try:
            if not icon_dst.is_file() or icon_dst.stat().st_mtime != source.stat().st_mtime:
                icon_dst.write_bytes(source.read_bytes())
        except OSError:
            pass

    _install_if_different(
        apps_dir / "whisper-flow.desktop",
        _MENU_ENTRY.format(exec=f'"{img}"'),
    )
    _install_if_different(
        autostart_dir / "whisper-flow.desktop",
        _AUTOSTART_ENTRY.format(exec=f'"{img}"'),
    )
