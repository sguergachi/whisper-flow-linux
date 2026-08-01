"""Tests that only mean anything on real Windows.

Everywhere else the Windows modules are exercised with ctypes.WinDLL stubbed,
which proves the Python logic and nothing about the Win32 calls underneath.
These run on a windows-latest runner and touch the real thing: the clipboard,
the key-state API, the version query, and the overlay process.

The clipboard in the failure reports shipped broken because the only tests
covering it were mocks of a method that did not exist. This file is the
answer to that.
"""

import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="exercises real Win32 APIs")


# --------------------------------------------------------------- clipboard
def test_the_clipboard_really_round_trips():
    from whisper_flow import system_win

    text = "whisper-flow ❌ report → mic gave nothing (can't open device)"
    assert system_win.copy_to_clipboard(text) is True
    assert _read_clipboard() == text


def test_the_clipboard_handles_a_large_report():
    """Failure reports carry 120 log lines; nothing may be truncated."""
    from whisper_flow import system_win

    text = "\n".join(f"{i:04d} some log line with detail" for i in range(400))
    assert system_win.copy_to_clipboard(text) is True
    assert _read_clipboard() == text


def test_system_manager_reaches_the_windows_clipboard(tmp_path):
    """The exact call the daemon makes when reporting a failure."""
    from whisper_flow.config import Config
    from whisper_flow.system import SystemManager

    manager = SystemManager(Config(config_dir=tmp_path))
    text = "report from SystemManager ✅"
    assert manager.copy_to_clipboard(text) is True
    assert _read_clipboard() == text


def _read_clipboard() -> str:
    """Read CF_UNICODETEXT back.

    The signatures matter as much here as in the code under test: without
    them ctypes truncates the 64-bit handle from GetClipboardData to an int
    and this reads nothing back - which is exactly how this helper first
    reported an empty clipboard for text that had been copied correctly.
    """
    import ctypes
    import ctypes.wintypes as wintypes

    CF_UNICODETEXT = 13
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]

    if not user32.OpenClipboard(None):
        return ""
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        locked = kernel32.GlobalLock(handle)
        if not locked:
            return ""
        try:
            return ctypes.c_wchar_p(locked).value or ""
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


# ------------------------------------------------------------- key state
def test_key_state_can_be_read_without_a_hook():
    """The listener polls rather than hooking; prove the call works."""
    from whisper_flow.hotkey_win import WinHotkeyListener

    listener = WinHotkeyListener()
    state = listener._user32.GetAsyncKeyState(0x11)     # ctrl
    assert isinstance(state, int)
    # Nothing is held on a runner, so nothing should read as down.
    assert listener._held() == set()


def test_every_configured_hotkey_maps_to_real_virtual_keys():
    from whisper_flow.config import default_hotkeys
    from whisper_flow.hotkey_win import WinHotkeyListener

    for combination in default_hotkeys().values():
        codes = WinHotkeyListener.parse_keys(combination)
        assert codes, f"{combination} parsed to nothing on Windows"
        assert len(codes) == len(combination.split("+"))


# --------------------------------------------------------------- platform
def test_the_build_number_is_real():
    """windows_build() returns 0 on failure, which would disable the blur."""
    from whisper_flow import blur_win

    assert blur_win.windows_build() > 0


def test_the_windows_modules_import_without_the_linux_ones():
    import whisper_flow.blur_win          # noqa: F401
    import whisper_flow.hotkey_win        # noqa: F401
    import whisper_flow.hud_win           # noqa: F401
    import whisper_flow.system_win        # noqa: F401

    for absent in ("evdev", "gi", "pynput"):
        assert absent not in sys.modules


# ---------------------------------------------------------------- overlay
def test_the_overlay_starts_and_stops():
    """It is spawned on the path that begins a recording, so it must run."""
    env = dict(os.environ, WHISPER_FLOW_HUD_LEVEL_FILE="")
    process = subprocess.Popen(
        [sys.executable, "-m", "whisper_flow.hud_win"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        time.sleep(4)
        assert process.poll() is None, (
            f"overlay exited early:\n{process.communicate()[0]}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def test_importing_the_overlay_stays_cheap():
    """The overlay is what the user waits for; it must not pull in the app.

    A budget rather than a fixed number: a runner is slower and noisier than
    a desktop, but importing the openai SDK and pyaudio would blow past this
    by an order of magnitude.
    """
    code = (
        "import time; t=time.perf_counter();"
        "import whisper_flow.hud_win;"
        "print((time.perf_counter()-t)*1000)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    elapsed_ms = float(out.stdout.strip().splitlines()[-1])
    assert elapsed_ms < 1500, f"overlay import took {elapsed_ms:.0f}ms"


def test_the_package_does_not_import_the_world():
    code = "import whisper_flow, sys; print(','.join(sorted(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    loaded = set(out.stdout.strip().split(","))
    assert not {"openai", "pyaudio", "pystray", "PIL"} & loaded
