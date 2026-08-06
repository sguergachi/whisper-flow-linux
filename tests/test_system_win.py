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


# ------------------------------------------------- modifiers we hold ourselves
@pytest.fixture
def held_keys(system_win, monkeypatch):
    """Track what is pressed and released, with nothing physically held."""
    from whisper_flow import hotkey_win

    monkeypatch.setattr(hotkey_win, "_active_listener", None, raising=False)
    system_win._injected_down.clear()
    batches = []
    monkeypatch.setattr(system_win, "_send",
                        lambda events: (batches.append(list(events)), True)[1])
    yield system_win, batches
    system_win._injected_down.clear()


def _events(batches, flags):
    return [event.union.ki.wVk for batch in batches for event in batch
            if event.union.ki.wVk in system_win_modifiers() and
            event.union.ki.dwFlags == flags]


def system_win_modifiers():
    from whisper_flow import system_win
    return system_win.MODIFIERS


def test_a_restored_modifier_is_released_again_at_the_end(held_keys):
    """The stuck Super key, and why the desktop started minimising itself.

    Typing releases the push-to-talk modifiers, then presses back the ones
    the user is still holding. If they let go mid-injection their own key-up
    lands on a key already released, and our press has no release coming -
    so Windows believes Super is down and the next Shift is Win+Shift.
    """
    system_win, batches = held_keys
    system_win.restore_modifiers((system_win.VK_LWIN, system_win.VK_MENU))
    assert system_win._injected_down == {system_win.VK_LWIN,
                                         system_win.VK_MENU}
    assert _events(batches, 0) == [system_win.VK_LWIN, system_win.VK_MENU]

    batches.clear()
    released = system_win.release_injected_modifiers()

    assert set(released) == {system_win.VK_LWIN, system_win.VK_MENU}
    assert set(_events(batches, system_win.KEYEVENTF_KEYUP)) == {
        system_win.VK_LWIN, system_win.VK_MENU}, (
        "a key we pressed was never released; it stays down system-wide")
    assert system_win._injected_down == set()


def test_releasing_twice_sends_nothing_the_second_time(held_keys):
    system_win, batches = held_keys
    system_win.restore_modifiers((system_win.VK_CONTROL,))
    system_win.release_injected_modifiers()
    batches.clear()
    assert system_win.release_injected_modifiers() == ()
    assert batches == []


def test_nothing_held_leaves_nothing_to_release(held_keys):
    system_win, batches = held_keys
    system_win.restore_modifiers(())
    assert system_win._injected_down == set()
    assert system_win.release_injected_modifiers() == ()


def test_only_the_hotkeys_own_modifiers_are_pressed_back(system_win, monkeypatch):
    """A Shift the user happens to be holding is not ours to re-press.

    The snapshot covers every modifier that was down. Pressing all of them
    back makes this app responsible for releasing keys it had no business
    touching, and each one is another key that can be left stuck.
    """
    from whisper_flow import hotkey_win

    system_win._injected_down.clear()
    batches = []
    monkeypatch.setattr(system_win, "_send",
                        lambda events: (batches.append(list(events)), True)[1])

    class Listener:
        @staticmethod
        def triggered_keys():
            # super+alt, as configured on the machine that reported this
            return frozenset({hotkey_win.NAME_TO_VK["super"],
                              hotkey_win.NAME_TO_VK["alt"]})

    monkeypatch.setattr(hotkey_win, "_active_listener", Listener,
                        raising=False)
    try:
        system_win.restore_modifiers(
            (system_win.VK_LWIN, system_win.VK_MENU, system_win.VK_SHIFT))
        assert system_win._injected_down == {system_win.VK_LWIN,
                                             system_win.VK_MENU}
        assert system_win.VK_SHIFT not in _events(batches, 0)
    finally:
        system_win._injected_down.clear()


def test_the_right_windows_key_counts_as_super(system_win, monkeypatch):
    """The binding names 0x5B; the key the user held may be 0x5C."""
    from whisper_flow import hotkey_win

    system_win._injected_down.clear()
    monkeypatch.setattr(system_win, "_send", lambda events: True)

    class Listener:
        @staticmethod
        def triggered_keys():
            return frozenset({hotkey_win.NAME_TO_VK["super"]})

    monkeypatch.setattr(hotkey_win, "_active_listener", Listener,
                        raising=False)
    try:
        system_win.restore_modifiers((system_win.VK_RWIN,))
        assert system_win._injected_down == {system_win.VK_RWIN}, (
            "the right-hand Windows key was not recognised as the one the "
            "hotkey names, so the recording would end at the first word")
    finally:
        system_win._injected_down.clear()


def test_nothing_is_restored_while_no_hotkey_is_held(system_win, monkeypatch):
    """Pasting outside a push-to-talk holds nothing down on anyone's behalf."""
    from whisper_flow import hotkey_win

    system_win._injected_down.clear()
    monkeypatch.setattr(system_win, "_send", lambda events: True)

    class Listener:
        @staticmethod
        def triggered_keys():
            return frozenset()

    monkeypatch.setattr(hotkey_win, "_active_listener", Listener,
                        raising=False)
    try:
        system_win.restore_modifiers((system_win.VK_CONTROL,))
        assert system_win._injected_down == set()
    finally:
        system_win._injected_down.clear()


# ------------------------------------------------------- the Start menu
def _no_modifiers_held(system_win, monkeypatch):
    """A keyboard with nothing on it, so the escape path can be read plainly."""
    monkeypatch.setattr(system_win._user32, "GetAsyncKeyState", lambda vk: 0)


def _batches(system_win, monkeypatch) -> list:
    sent = []
    monkeypatch.setattr(system_win, "_send",
                        lambda events: (sent.append(list(events)), True)[1])
    return sent


def _in_front(system_win, monkeypatch, *answers):
    """Successive answers from start_menu_in_front, the last one repeating."""
    answers = list(answers)
    monkeypatch.setattr(
        system_win, "start_menu_in_front",
        lambda: answers.pop(0) if len(answers) > 1 else answers[0])


def test_the_start_menu_is_closed_with_escape(system_win, monkeypatch):
    """Once it is up it holds the foreground, and the dictation has nowhere
    to go for as long as the user keeps talking."""
    _no_modifiers_held(system_win, monkeypatch)
    sent = _batches(system_win, monkeypatch)
    _in_front(system_win, monkeypatch, True, False)

    assert system_win.dismiss_start_menu() is True

    escapes = [event.union.ki.dwFlags for batch in sent for event in batch
               if event.union.ki.wVk == system_win.VK_ESCAPE]
    assert escapes == [0, system_win.KEYEVENTF_KEYUP]


def test_a_menu_that_is_not_there_is_left_alone(system_win, monkeypatch):
    """This is asked on every live pass; the ordinary one must send nothing."""
    _no_modifiers_held(system_win, monkeypatch)
    sent = _batches(system_win, monkeypatch)
    _in_front(system_win, monkeypatch, False)

    assert system_win.dismiss_start_menu() is False
    assert sent == []


def test_a_menu_that_will_not_close_says_so(system_win, monkeypatch):
    """The caller reports what is in front instead, rather than typing into it."""
    _no_modifiers_held(system_win, monkeypatch)
    _batches(system_win, monkeypatch)
    _in_front(system_win, monkeypatch, True)
    monkeypatch.setattr(system_win.time, "sleep", lambda _s: None)

    assert system_win.dismiss_start_menu() is False


def test_the_modifiers_come_off_before_the_escape(system_win, monkeypatch):
    """The user is still holding Alt, and Alt+Escape cycles windows."""
    held = {system_win.VK_MENU, system_win.VK_LWIN}
    monkeypatch.setattr(system_win._user32, "GetAsyncKeyState",
                        lambda vk: -0x8000 if vk in held else 0)
    sent = _batches(system_win, monkeypatch)
    _in_front(system_win, monkeypatch, True, False)

    system_win.dismiss_start_menu()

    order = []
    for index, batch in enumerate(sent):
        for event in batch:
            if (event.union.ki.wVk in held
                    and event.union.ki.dwFlags & system_win.KEYEVENTF_KEYUP):
                order.append(("released", index))
            if event.union.ki.wVk == system_win.VK_ESCAPE:
                order.append(("escape", index))
    releases = [i for what, i in order if what == "released"]
    escapes = [i for what, i in order if what == "escape"]
    assert releases and escapes and max(releases) < min(escapes)


def test_the_keys_the_user_holds_go_back_down_afterwards(system_win,
                                                         monkeypatch):
    """Leaving them up tells the listener the push-to-talk was let go."""
    from whisper_flow import hotkey_win

    held = {system_win.VK_MENU, system_win.VK_LWIN}
    monkeypatch.setattr(system_win._user32, "GetAsyncKeyState",
                        lambda vk: -0x8000 if vk in held else 0)
    sent = _batches(system_win, monkeypatch)
    _in_front(system_win, monkeypatch, True, False)

    class Listener:
        @staticmethod
        def triggered_keys():
            return frozenset(held)

    monkeypatch.setattr(hotkey_win, "_active_listener", Listener,
                        raising=False)
    system_win._injected_down.clear()
    try:
        system_win.dismiss_start_menu()
        assert system_win._injected_down == held
        pressed = {event.union.ki.wVk for event in sent[-1]
                   if event.union.ki.dwFlags == 0}
        assert held <= pressed
    finally:
        system_win._injected_down.clear()


def test_the_listener_is_blind_while_the_escape_is_sent(system_win,
                                                        monkeypatch):
    """It polls the same state table; our Escape would read as the user's,
    and cancel the dictation this is trying to rescue."""
    from whisper_flow import hotkey_win

    _no_modifiers_held(system_win, monkeypatch)
    _in_front(system_win, monkeypatch, True, False)
    depths = []
    monkeypatch.setattr(
        system_win, "_send",
        lambda events: (depths.append(hotkey_win._typing_depth), True)[1])

    system_win.dismiss_start_menu()
    assert depths and all(depth > 0 for depth in depths)


def test_only_the_shell_is_ever_escaped(system_win):
    """Every UWP application shares the Start menu's window class, and
    pressing Escape into one of those because it happened to be in front is
    exactly what this must never do."""
    assert "startmenuexperiencehost.exe" in system_win.START_MENU_PROCESSES
    assert "searchhost.exe" in system_win.START_MENU_PROCESSES
    assert not any(name.endswith(".corewindow")
                   for name in system_win.START_MENU_PROCESSES)


def test_an_unreadable_window_is_not_the_start_menu(system_win, monkeypatch):
    """window_process_name answers "" when Win32 will not say, and "" must
    not match anything."""
    monkeypatch.setattr(system_win, "foreground_window", lambda: 0)
    assert system_win.start_menu_in_front() is False
    assert "" not in system_win.START_MENU_PROCESSES
