# PyInstaller spec for the Windows build.
#
# One executable. It runs the tray daemon normally, the overlay when launched
# with --hud (which hud.py does per recording), and the model setup window
# with --setup. Shipping those as separate .exes only raised the question of
# which one to run.
#
# The Linux modules must not be pulled in: evdev, gi and pynput have no
# Windows wheels, and a hidden import would force exactly that.
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
    "whisper_flow.blur_win",
    "whisper_flow.hud_win",
    "whisper_flow.setup_win",
    "pystray._win32",
    "PIL._tkinter_finder",
    # blur_win reaches for these, and ctypes.wintypes is a submodule that
    # importing ctypes does not bring along.
    "ctypes.wintypes",
    "tkinter",
    "tkinter.constants",
    "tkinter.ttk",          # the setup window's progress bar
]

app = Analysis(
    ["../../src/whisper_flow/__main__win__.py"],
    pathex=["../../src"],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    excludes=EXCLUDES,
    cipher=block_cipher,
)

pyz = PYZ(app.pure, app.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, app.scripts, [],
    exclude_binaries=True,
    name="whisper-flow",
    console=False,          # tray app; a console window would sit there
    icon=None,
)

COLLECT(
    exe, app.binaries, app.zipfiles, app.datas,
    strip=False, upx=False, name="whisper-flow",
)
