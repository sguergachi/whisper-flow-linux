# PyInstaller spec for the Windows build.
#
# Two things are easy to get wrong here:
#
#   * The HUD is launched as a separate process by path (hud.py runs
#     `sys.executable <path>/hud_win.py`). Inside a frozen build there is no
#     source tree, and sys.executable is the exe itself, so hud_win is built
#     as its own executable and the supervisor is told to use it.
#   * The Linux modules must not be pulled in. evdev, gi and pynput have no
#     Windows wheels, and importing them is what a hidden import would force.
import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

EXCLUDES = [
    "evdev", "gi", "gi.repository", "pynput", "cairo",
    "whisper_flow.hotkey_evdev", "whisper_flow.hud_app",
    "whisper_flow.wayland_blur", "whisper_flow.hotkey_kde",
]

hidden = collect_submodules("pydantic") + [
    "whisper_flow.hotkey_win",
    "whisper_flow.system_win",
    "pystray._win32",
    "PIL._tkinter_finder",
]

daemon = Analysis(
    ["../../src/whisper_flow/__main__win__.py"],
    pathex=["../../src"],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    excludes=EXCLUDES,
    cipher=block_cipher,
)

hud = Analysis(
    ["../../src/whisper_flow/hud_win.py"],
    pathex=["../../src"],
    binaries=[],
    datas=[],
    hiddenimports=[],
    excludes=EXCLUDES + ["openai", "requests", "numpy"],
    cipher=block_cipher,
)

MERGE((daemon, "whisper-flow", "whisper-flow"), (hud, "whisper-flow-hud", "whisper-flow-hud"))

daemon_pyz = PYZ(daemon.pure, daemon.zipped_data, cipher=block_cipher)
daemon_exe = EXE(
    daemon_pyz, daemon.scripts, [],
    exclude_binaries=True,
    name="whisper-flow",
    console=False,          # tray app; a console window would sit there
    icon=None,
)

hud_pyz = PYZ(hud.pure, hud.zipped_data, cipher=block_cipher)
hud_exe = EXE(
    hud_pyz, hud.scripts, [],
    exclude_binaries=True,
    name="whisper-flow-hud",
    console=False,
    icon=None,
)

COLLECT(
    daemon_exe, daemon.binaries, daemon.zipfiles, daemon.datas,
    hud_exe, hud.binaries, hud.zipfiles, hud.datas,
    strip=False, upx=False, name="whisper-flow",
)
