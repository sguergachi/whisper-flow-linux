"""Tests for the Windows typing, clipboard and notification helpers.

Imported directly rather than through the package so they run on Linux: the
module only touches ctypes.windll inside functions, so everything below can
be exercised anywhere.
"""

import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="module")
def system_win():
    """Load the module with the Win32 loader stubbed, so this runs anywhere."""
    import ctypes
    from unittest.mock import Mock

    original = getattr(ctypes, "WinDLL", None)
    ctypes.WinDLL = Mock()              # resolved at import on Windows only
    try:
        return importlib.import_module("whisper_flow.system_win")
    finally:
        if original is None:
            del ctypes.WinDLL
        else:
            ctypes.WinDLL = original


# ------------------------------------------------------------ powershell quoting
def test_an_apostrophe_cannot_end_the_powershell_string(system_win):
    """'can't open device' used to terminate the literal and run as code."""
    quoted = system_win._ps_literal("can't open device")
    assert quoted == "can''t open device"


def test_quotes_are_doubled_not_dropped(system_win):
    assert system_win._ps_literal("'; Remove-Item C:\\ ;'") == (
        "''; Remove-Item C:\\ ;''")


def test_newlines_are_flattened(system_win):
    """A newline would end the command line mid-script."""
    assert "\n" not in system_win._ps_literal("first\nsecond\r\nthird")
    assert system_win._ps_literal("first\nsecond") == "first second"


def test_quoting_survives_non_string_input(system_win):
    assert system_win._ps_literal(42) == "42"


def test_a_quoted_message_still_reads_as_itself(system_win):
    """Escaping must not mangle ordinary text."""
    assert system_win._ps_literal("Recording failed (transcribe)") == (
        "Recording failed (transcribe)")


# ------------------------------------------------------------------- clipboard
def test_the_clip_exe_fallback_sends_a_byte_order_mark(system_win, monkeypatch):
    """Without a BOM clip.exe reads the bytes as the console code page."""
    sent = {}

    def fake_run(cmd, input=None, **kwargs):
        sent["cmd"], sent["input"] = cmd, input
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(system_win.subprocess, "run", fake_run)
    assert system_win._copy_via_clip_exe("héllo") is True
    assert sent["cmd"] == ["clip.exe"]
    assert sent["input"].startswith(b"\xff\xfe")          # UTF-16 LE BOM
    assert sent["input"].decode("utf-16") == "héllo"


def test_the_fallback_reports_failure_rather_than_raising(system_win, monkeypatch):
    monkeypatch.setattr(system_win.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no clip")))
    assert system_win._copy_via_clip_exe("x") is False


def test_utf16_round_trips_through_the_fallback(system_win, monkeypatch):
    """Failure reports carry arrows and emoji; they must survive intact."""
    sent = {}
    monkeypatch.setattr(
        system_win.subprocess, "run",
        lambda cmd, input=None, **k: sent.update(input=input) or
        type("P", (), {"returncode": 0})())
    text = "❌ Recording failed → check the microphone"
    system_win._copy_via_clip_exe(text)
    assert sent["input"].decode("utf-16") == text


# ------------------------------------------------------------ struct layout
def test_the_union_is_not_padded_to_a_guessed_width(system_win):
    """The exact byte counts only mean anything on Windows - wintypes.DWORD
    is 8 bytes here and 4 there - so this asserts the relationship that
    caused the bug instead: the union must be at least as large as its
    largest member. It was padded to 24, the 32-bit width, while MOUSEINPUT
    on x64 is 32, so SendInput was handed a cbSize 8 bytes short and
    rejected every call with ERROR_INVALID_PARAMETER.
    """
    import ctypes

    assert ctypes.sizeof(system_win._INPUTunion) >= ctypes.sizeof(
        system_win.MOUSEINPUT)
    assert ctypes.sizeof(system_win._INPUTunion) >= ctypes.sizeof(
        system_win.KEYBDINPUT)
    assert ctypes.sizeof(system_win.INPUT) > ctypes.sizeof(system_win._INPUTunion)


def test_a_key_event_carries_the_tag_as_a_number(system_win):
    """dwExtraInfo is ULONG_PTR - an integer, not a pointer to one."""
    event = system_win._key_event(0, 65, system_win.KEYEVENTF_UNICODE)
    assert event.union.ki.dwExtraInfo == system_win.INJECTED_TAG
    assert event.union.ki.wScan == 65


# ----------------------------------------------------------------- focus
def test_focus_is_restored_through_the_foreground_thread(system_win, monkeypatch):
    """SetForegroundWindow refuses a caller that is not already in front.

    Which is exactly this caller. Attaching to the foreground thread's input
    queue first is the documented way round it; without the attach the call
    is a silent no-op and the transcript still lands in the wrong window.
    """
    from unittest.mock import Mock

    user32 = Mock()
    user32.IsWindow.return_value = True
    foreground = [4321]
    user32.GetForegroundWindow.side_effect = lambda: foreground[0]
    user32.GetWindowThreadProcessId.return_value = 77
    user32.AttachThreadInput.return_value = 1

    def set_foreground(hwnd):
        foreground[0] = 1234
        return 1

    user32.SetForegroundWindow.side_effect = set_foreground
    monkeypatch.setattr(system_win, "_user32", user32)
    kernel32 = Mock()
    kernel32.GetCurrentThreadId.return_value = 11
    monkeypatch.setattr(system_win, "_kernel32", kernel32)

    assert system_win.focus_window(1234) is True
    assert user32.AttachThreadInput.call_args_list[0][0][2] is True
    assert user32.AttachThreadInput.call_args_list[-1][0][2] is False, (
        "the input queues were left attached")


def test_a_window_already_in_front_is_left_alone(system_win, monkeypatch):
    from unittest.mock import Mock

    user32 = Mock()
    user32.IsWindow.return_value = True
    user32.GetForegroundWindow.return_value = 1234
    monkeypatch.setattr(system_win, "_user32", user32)

    assert system_win.focus_window(1234) is True
    user32.SetForegroundWindow.assert_not_called()


def test_a_window_that_has_closed_is_not_chased(system_win, monkeypatch):
    from unittest.mock import Mock

    user32 = Mock()
    user32.IsWindow.return_value = False
    monkeypatch.setattr(system_win, "_user32", user32)

    assert system_win.focus_window(1234) is False
    user32.SetForegroundWindow.assert_not_called()


def test_no_saved_window_is_not_an_error(system_win):
    assert system_win.focus_window(0) is False


# --------------------------------------------------- blinding the listener
def test_the_listener_is_blinded_for_exactly_the_injection(system_win):
    """Not for a guess at how long the injection will take.

    The guess was 0.2s plus a millisecond per character. Too short for a long
    commit, so the race it exists to close could reopen; and far too long for
    a short one, leaving the listener ignoring the keyboard for up to half a
    second after the last word - which is when the next hotkey gets pressed,
    and why holding it appeared to do nothing.
    """
    from whisper_flow import hotkey_win

    hotkey_win._typing_depth = 0
    hotkey_win._settled_at = 0.0

    assert not hotkey_win._suppressed()
    with system_win._hotkeys_blinded():
        assert hotkey_win._suppressed(), (
            "the keyboard was still being read while we typed into it")
    # Still blind through the settling window: the events we queued are not
    # necessarily readable the instant SendInput returns.
    assert hotkey_win._suppressed()
    hotkey_win._settled_at = 0.0
    assert not hotkey_win._suppressed(), (
        "suppression outlived the typing; the hotkey would stop responding")


def test_nested_injections_do_not_uncover_each_other(system_win):
    from whisper_flow import hotkey_win

    hotkey_win._typing_depth = 0
    hotkey_win._settled_at = 0.0

    with system_win._hotkeys_blinded():
        with system_win._hotkeys_blinded():
            assert hotkey_win._suppressed()
        assert hotkey_win._typing_depth == 1
        assert hotkey_win._suppressed()
    assert hotkey_win._typing_depth == 0


def test_the_listener_is_uncovered_even_when_typing_throws(system_win):
    from whisper_flow import hotkey_win

    hotkey_win._typing_depth = 0
    with pytest.raises(RuntimeError):
        with system_win._hotkeys_blinded():
            raise RuntimeError("SendInput exploded")
    assert hotkey_win._typing_depth == 0, (
        "a failed injection left the keyboard permanently ignored")
