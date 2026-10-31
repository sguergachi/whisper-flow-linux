"""Tests for the Windows hotkey listener.

It polls GetAsyncKeyState rather than installing a hook, so everything here
can run anywhere with the user32 call stubbed.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture
def listener():
    import ctypes

    original = getattr(ctypes, "WinDLL", None)
    ctypes.WinDLL = Mock()              # needed for the import and the ctor
    try:
        module = importlib.import_module("whisper_flow.hotkey_win")
        listener = module.WinHotkeyListener()
    finally:
        if original is None:
            del ctypes.WinDLL
        else:
            ctypes.WinDLL = original

    listener._module = module
    held = set()

    def get_state(vk):
        return module.HELD_MASK if vk in held else 0

    listener._user32 = Mock()
    listener._user32.GetAsyncKeyState.side_effect = get_state
    listener._held_set = held
    return listener


def test_the_right_windows_key_works_too(listener):
    """Windows has no combined code for it, so it needed an explicit alias."""
    module = listener._module
    listener.register_hotkey("transcribe", "super+alt", lambda: None)

    listener._held_set.update({module.VK_RWIN, module.NAME_TO_VK["alt"]})
    assert listener._held() == {module.NAME_TO_VK["super"],
                                module.NAME_TO_VK["alt"]}


def test_the_left_windows_key_still_works(listener):
    module = listener._module
    listener.register_hotkey("transcribe", "super+alt", lambda: None)

    listener._held_set.update({module.NAME_TO_VK["super"],
                               module.NAME_TO_VK["alt"]})
    assert listener._held() == {module.NAME_TO_VK["super"],
                                module.NAME_TO_VK["alt"]}


def test_the_alternate_key_is_actually_polled(listener):
    """An alias is useless if the physical key is never sampled."""
    module = listener._module
    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    assert module.VK_RWIN in listener._watched


def test_a_binding_without_super_does_not_watch_the_windows_key(listener):
    module = listener._module
    listener.register_hotkey("auto", "ctrl+alt+space", lambda: None)
    assert module.VK_RWIN not in listener._watched


def test_the_most_specific_binding_wins(listener):
    """Releasing shift from ctrl+shift+alt must not fall through to ctrl+alt."""
    module = listener._module
    fired = []
    listener.register_hotkey("small", "ctrl+alt", lambda: fired.append("small"))
    listener.register_hotkey("big", "ctrl+shift+alt", lambda: fired.append("big"))

    down = {module.NAME_TO_VK[n] for n in ("ctrl", "shift", "alt")}
    listener._check_bindings(down, rising=True)
    name, kind, cb = listener._callbacks.get_nowait()
    cb()
    assert fired == ["big"]


def test_a_combination_only_arms_while_keys_are_going_down(listener):
    """Otherwise releasing one key of a larger combination starts a dictation."""
    module = listener._module
    fired = []
    listener.register_hotkey("t", "ctrl+alt", lambda: fired.append("press"))

    down = {module.NAME_TO_VK["ctrl"], module.NAME_TO_VK["alt"]}
    listener._check_bindings(down, rising=False)
    assert listener._callbacks.empty()
    assert fired == []


def test_escape_fires_even_during_a_held_combination(listener):
    """Cancel must work while push-to-talk keys are held.

    Bindings resolve most-specific-wins, so a lone Escape could never win
    against a held combination; it is tracked beside the bindings instead.
    """
    import threading
    import time

    module = listener._module
    listener.escape_callback = lambda: None
    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener._held_set.update({module.NAME_TO_VK["super"],
                               module.NAME_TO_VK["alt"],
                               module.NAME_TO_VK["esc"]})

    listener._running = True
    thread = threading.Thread(target=listener._poll_loop, daemon=True)
    thread.start()
    time.sleep(0.1)
    listener._running = False
    thread.join(timeout=2)

    names = []
    while not listener._callbacks.empty():
        names.append(listener._callbacks.get_nowait()[0])
    assert "escape" in names



def _fake_system_win(monkeypatch, spoil, dismiss=None, strip=None):
    """Replace whisper_flow.system_win for `from . import system_win`.

    Patching sys.modules alone is not enough: once the package has the
    submodule as an attribute - which it does on Windows, where it imports -
    `from . import system_win` reads that attribute and never consults
    sys.modules. On Linux only the sys.modules entry matters. Both are set.
    """
    import types

    import whisper_flow

    fake = types.ModuleType("whisper_flow.system_win")
    fake.spoil_start_menu = spoil
    fake.dismiss_start_menu = dismiss or (lambda: False)
    fake.strip_super = strip or (lambda: ())
    monkeypatch.setitem(sys.modules, "whisper_flow.system_win", fake)
    monkeypatch.setattr(whisper_flow, "system_win", fake, raising=False)
    return fake


# ------------------------------------------------------- the Start menu
def test_a_super_hotkey_suppresses_the_start_menu(listener, monkeypatch):
    """Releasing Super otherwise opens Start, and the dictation lands there."""
    module = listener._module
    calls = []
    _fake_system_win(monkeypatch, lambda: calls.append(1))

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    down = {module.NAME_TO_VK["super"], module.NAME_TO_VK["alt"]}
    listener._check_bindings(down, rising=True)
    assert calls == [1]


def test_a_hotkey_without_super_leaves_the_keyboard_alone(listener, monkeypatch):
    module = listener._module
    calls = []
    _fake_system_win(monkeypatch, lambda: calls.append(1))

    listener.register_hotkey("auto", "ctrl+alt+space", lambda: None)
    down = {module.NAME_TO_VK[n] for n in ("ctrl", "alt", "space")}
    listener._check_bindings(down, rising=True)
    assert calls == []


def test_the_start_menu_is_disarmed_again_while_the_key_is_held(
        listener, monkeypatch):
    """One keystroke at the press covers the press, not the forty seconds after.

    Windows arms Start on a Windows-key press and disarms it on any other
    key. Nothing presses Super deliberately during a dictation, but the key
    chatters and hands shift their grip - blips shorter than the poll
    interval, so the recording never notices - and each one arms the menu
    again. The next blip opens it, which is the bug that only showed on long
    holds.
    """
    module = listener._module
    calls = []
    _fake_system_win(monkeypatch, lambda: calls.append(1))
    monkeypatch.setattr(module, "SPOIL_INTERVAL", 0.0)

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    down = {module.NAME_TO_VK["super"], module.NAME_TO_VK["alt"]}
    listener._check_bindings(down, rising=True)
    at_the_press = len(calls)

    for _ in range(3):
        listener._keep_start_menu_spoiled(down)
    assert len(calls) == at_the_press + 3, (
        "the menu was disarmed once and then left armed for the rest of the "
        "hold")


def test_it_is_not_disarmed_on_every_poll(listener, monkeypatch):
    """The poll runs at 60Hz and this is a keystroke every application sees."""
    module = listener._module
    calls = []
    _fake_system_win(monkeypatch, lambda: calls.append(1))

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    down = {module.NAME_TO_VK["super"], module.NAME_TO_VK["alt"]}
    listener._press_triggered.add("transcribe")

    for _ in range(10):
        listener._keep_start_menu_spoiled(down)
    assert len(calls) == 1, "one per cadence, not one per poll"


def test_the_users_own_windows_key_is_left_alone(listener, monkeypatch):
    """Nothing of ours is held, so their Start menu is theirs to open."""
    module = listener._module
    calls = []
    _fake_system_win(monkeypatch, lambda: calls.append(1))

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener._keep_start_menu_spoiled({module.NAME_TO_VK["super"]})
    assert calls == []


def test_a_combination_without_super_sends_nothing(listener, monkeypatch):
    module = listener._module
    calls = []
    _fake_system_win(monkeypatch, lambda: calls.append(1))

    listener.register_hotkey("auto", "ctrl+alt+space", lambda: None)
    listener._press_triggered.add("auto")
    listener._keep_start_menu_spoiled(
        {module.NAME_TO_VK[n] for n in ("ctrl", "alt", "space")})
    assert calls == []


def test_a_key_no_longer_down_stops_the_disarming(listener, monkeypatch):
    """Belt and braces: the press is triggered but Super has gone."""
    module = listener._module
    calls = []
    _fake_system_win(monkeypatch, lambda: calls.append(1))

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener._press_triggered.add("transcribe")
    listener._keep_start_menu_spoiled({module.NAME_TO_VK["alt"]})
    assert calls == []


def test_the_poll_loop_keeps_disarming_it(listener, monkeypatch):
    """The wiring, not the method: a held combination must reach it."""
    import threading
    import time

    module = listener._module
    calls = []
    _fake_system_win(monkeypatch, lambda: calls.append(1))
    monkeypatch.setattr(module, "SPOIL_INTERVAL", 0.0)
    monkeypatch.setattr(module, "_typing_depth", 0)
    monkeypatch.setattr(module, "_settled_at", 0.0)

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener._held_set.update({module.NAME_TO_VK["super"],
                               module.NAME_TO_VK["alt"]})

    listener._running = True
    thread = threading.Thread(target=listener._poll_loop, daemon=True)
    thread.start()
    time.sleep(0.1)
    listener._running = False
    thread.join(timeout=2)

    assert len(calls) > 1, (
        "the hold was disarmed at the press and never again")


def test_suppression_failing_does_not_stop_the_hotkey(listener, monkeypatch):
    module = listener._module
    def explode():
        raise OSError("no user32")

    _fake_system_win(monkeypatch, explode)

    fired = []
    listener.register_hotkey("transcribe", "super+alt",
                             lambda: fired.append("press"))
    down = {module.NAME_TO_VK["super"], module.NAME_TO_VK["alt"]}
    listener._check_bindings(down, rising=True)
    name, kind, cb = listener._callbacks.get_nowait()
    cb()
    assert fired == ["press"]


# --------------------------------------- closing one that opened anyway
def test_start_menu_opening_on_hotkey_release_is_still_checked(listener, monkeypatch):
    import time
    module = listener._module
    closed = []
    _fake_system_win(monkeypatch, lambda: None,
                     dismiss=lambda: (closed.append(1), True)[1])
    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener._check_bindings({module.NAME_TO_VK["super"], module.NAME_TO_VK["alt"]}, rising=True)
    listener._check_bindings(set(), rising=False)
    assert not listener.triggered_keys()
    assert listener._close_start_menu_if_it_opened(set()) is True
    listener._start_release_until = time.monotonic() - 1
    listener._start_menu_check_due = 0
    assert listener._close_start_menu_if_it_opened(set()) is False
    assert len(closed) == 1


def test_a_start_menu_that_opened_is_closed_during_the_hold(listener,
                                                            monkeypatch):
    """The typing path only notices when it has words to deliver.

    Holds that commit nothing - a long pause, a quiet room - have no pass to
    notice with, so the menu stayed up, kept the foreground, and everything
    said afterwards had nowhere to go.
    """
    module = listener._module
    closed = []
    _fake_system_win(monkeypatch, lambda: None,
                     dismiss=lambda: (closed.append(1), True)[1])
    monkeypatch.setattr(module, "START_MENU_CHECK", 0.0)

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener._press_triggered.add("transcribe")
    down = {module.NAME_TO_VK["super"], module.NAME_TO_VK["alt"]}

    assert listener._close_start_menu_if_it_opened(down) is True
    assert closed == [1]


def test_the_foreground_is_not_read_on_every_poll(listener, monkeypatch):
    """It is three Win32 calls, and the loop runs at 60Hz."""
    module = listener._module
    looked = []
    _fake_system_win(monkeypatch, lambda: None,
                     dismiss=lambda: (looked.append(1), False)[1])

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener._press_triggered.add("transcribe")
    down = {module.NAME_TO_VK["super"], module.NAME_TO_VK["alt"]}

    for _ in range(10):
        listener._close_start_menu_if_it_opened(down)
    assert len(looked) == 1


def test_nobody_elses_start_menu_is_touched(listener, monkeypatch):
    """No hotkey of ours is held, so an open Start menu is the user's."""
    module = listener._module
    looked = []
    _fake_system_win(monkeypatch, lambda: None,
                     dismiss=lambda: (looked.append(1), True)[1])

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener._close_start_menu_if_it_opened({module.NAME_TO_VK["super"]})
    assert looked == []


def test_a_hotkey_without_super_never_looks(listener, monkeypatch):
    module = listener._module
    looked = []
    _fake_system_win(monkeypatch, lambda: None,
                     dismiss=lambda: (looked.append(1), True)[1])

    listener.register_hotkey("auto", "ctrl+alt+space", lambda: None)
    listener._press_triggered.add("auto")
    listener._close_start_menu_if_it_opened(
        {module.NAME_TO_VK[n] for n in ("ctrl", "alt", "space")})
    assert looked == []


def test_closing_it_does_not_cancel_the_dictation(listener, monkeypatch):
    """Escape closes the menu and Escape is also the cancel key.

    The poll that follows the close must not read our own Escape out of the
    key state and cancel the recording it was rescuing.
    """
    import threading
    import time

    module = listener._module
    _fake_system_win(monkeypatch, lambda: None, dismiss=lambda: True)
    monkeypatch.setattr(module, "START_MENU_CHECK", 0.0)
    monkeypatch.setattr(module, "SPOIL_INTERVAL", 0.0)
    monkeypatch.setattr(module, "_typing_depth", 0)
    monkeypatch.setattr(module, "_settled_at", 0.0)

    listener.escape_callback = lambda: None
    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener._held_set.update({module.NAME_TO_VK["super"],
                               module.NAME_TO_VK["alt"],
                               module.NAME_TO_VK["esc"]})   # Escape still down

    listener._running = True
    thread = threading.Thread(target=listener._poll_loop, daemon=True)
    thread.start()
    time.sleep(0.1)
    listener._running = False
    thread.join(timeout=2)

    fired = []
    while not listener._callbacks.empty():
        fired.append(listener._callbacks.get_nowait()[0])
    assert "escape" not in fired, (
        "the Escape that closed the Start menu cancelled the dictation")


# ------------------------------------ the Windows key while words are typed
def test_a_super_chord_releases_the_windows_key_at_the_press(
        listener, monkeypatch):
    """A held Windows key turns live text into shell shortcuts.

    Win+Down minimises the focused window, Win+D shows the desktop: every
    character the app types while Super is still down is read as a shell
    combination instead of text. The press is the only moment it can be
    released without a re-press to follow.
    """
    module = listener._module
    stripped = []
    _fake_system_win(
        monkeypatch, lambda: None,
        strip=lambda: (stripped.append(1), (module.NAME_TO_VK["super"],))[1])

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    down = {module.NAME_TO_VK["super"], module.NAME_TO_VK["alt"]}
    listener._check_bindings(down, rising=True)

    assert stripped == [1], "the Windows key was left down for the whole hold"
    # The poll must not read the release we just sent as the user letting go.
    assert module.NAME_TO_VK["super"] in listener._held()


def test_the_release_is_only_sent_once_per_hold(listener, monkeypatch):
    """A release on every poll would be a keystroke storm, not a fix."""
    module = listener._module
    stripped = []
    _fake_system_win(
        monkeypatch, lambda: None,
        strip=lambda: (stripped.append(1), (module.NAME_TO_VK["super"],))[1])

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    down = {module.NAME_TO_VK["super"], module.NAME_TO_VK["alt"]}
    for _ in range(5):
        listener._check_bindings(down, rising=True)
    assert stripped == [1]


def test_the_chord_ends_when_the_other_key_comes_up(listener, monkeypatch):
    """The release is read from the rest of the chord, not from Super.

    GetAsyncKeyState cannot tell the user's Windows-key release from the one
    the app sent, so a stripped Super would keep the recording running
    forever if it were the only key holding the chord together.
    """
    module = listener._module
    super_vk = module.NAME_TO_VK["super"]
    _fake_system_win(monkeypatch, lambda: None, strip=lambda: (super_vk,))

    listener.register_hotkey("transcribe", "super+alt", lambda: None,
                             lambda: None)
    listener._check_bindings({super_vk, module.NAME_TO_VK["alt"]},
                             rising=True)

    listener._check_bindings({super_vk}, rising=False)

    # The release callback runs on the dispatch thread, which no test starts.
    queued = []
    while not listener._callbacks.empty():
        name, kind, cb = listener._callbacks.get_nowait()
        if kind == "release":
            queued.append(name)
            cb()
    assert queued == ["transcribe"], "letting go of Alt did not end the recording"
    assert not listener._synthetic_down, "the virtual hold outlived the chord"


def test_super_coming_back_is_taken_out_again(listener, monkeypatch):
    """The key chatters, or is pressed again, during the hold.

    Until it is out again, every character is a shell shortcut. Releasing it
    once at the press is not enough for a key that comes back.
    """
    module = listener._module
    stripped = []
    _fake_system_win(
        monkeypatch, lambda: None,
        strip=lambda: (stripped.append(1), (module.NAME_TO_VK["super"],))[1])

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    down = {module.NAME_TO_VK["super"], module.NAME_TO_VK["alt"]}
    listener._check_bindings(down, rising=True)

    listener._keep_super_out({module.NAME_TO_VK["alt"]})     # it stays out
    listener._keep_super_out(down)                          # ...then comes back
    assert stripped == [1, 1], "Super was left down after coming back"

    listener._keep_super_out(down)
    listener._keep_super_out(down)
    assert stripped == [1, 1], (
        "a state table that keeps reporting the key down is retried on every "
        "poll, which is sixty keystroke pairs a second into the target app")


def test_a_super_only_chord_is_left_alone(listener, monkeypatch):
    """With nothing else in the chord there is no release left to read.

    Releasing the only key would be a recording that nothing can end, which
    is worse than the shortcuts it avoids.
    """
    module = listener._module
    stripped = []
    _fake_system_win(
        monkeypatch, lambda: None,
        strip=lambda: (stripped.append(1), (module.NAME_TO_VK["super"],))[1])

    listener.register_hotkey("transcribe", "super", lambda: None)
    listener._check_bindings({module.NAME_TO_VK["super"]}, rising=True)
    assert stripped == []


def test_a_failed_release_does_not_stop_the_dictation(listener, monkeypatch):
    """SendInput can be refused. The recording still has to start."""
    module = listener._module
    def explode():
        raise OSError("SendInput refused")

    _fake_system_win(monkeypatch, lambda: None, strip=explode)
    fired = []
    listener.register_hotkey("transcribe", "super+alt",
                             lambda: fired.append(1))

    listener._check_bindings({module.NAME_TO_VK["super"],
                              module.NAME_TO_VK["alt"]}, rising=True)
    name, kind, cb = listener._callbacks.get_nowait()
    cb()

    assert fired == [1]
    assert not listener._synthetic_down


def test_no_keystrokes_are_spent_once_super_is_out_of_the_hold(
        listener, monkeypatch):
    """The key that the cadence exists to spoil is no longer down.

    Every one of those keystrokes goes into the window being dictated into,
    ten times a second, for as long as the chord is held - which is exactly
    the kind of thing the user reports as the desktop doing things on its
    own.
    """
    module = listener._module
    super_vk = module.NAME_TO_VK["super"]
    calls = []
    _fake_system_win(monkeypatch, lambda: calls.append(1),
                     strip=lambda: (super_vk,))

    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    down = {super_vk, module.NAME_TO_VK["alt"]}
    listener._check_bindings(down, rising=True)
    calls.clear()

    for _ in range(3):
        listener._keep_start_menu_spoiled(down)

    assert calls == [], "the menu was disarmed again with Super already up"


def test_the_hold_survives_the_windows_key_leaving_and_ends_on_alt(
        listener, monkeypatch):
    """The wiring, not the methods: poll, press, stripped hold, and release.

    The whole risk in taking Super out of the state table is that the poll
    then reads the chord as released and stops the recording at the first
    word. It must keep the press alive until the *other* key comes up.
    """
    import threading
    import time

    module = listener._module
    super_vk = module.NAME_TO_VK["super"]
    alt_vk = module.NAME_TO_VK["alt"]
    stripped = []

    def strip():
        # SendInput takes the key out of the state table, so the next poll
        # reads it as up - as Windows does.
        listener._held_set.discard(super_vk)
        stripped.append(1)
        return (super_vk,)

    _fake_system_win(monkeypatch, lambda: None, strip=strip)
    listener.register_hotkey("transcribe", "super+alt",
                             lambda: None, lambda: None)
    listener._held_set.update({super_vk, alt_vk})

    listener._running = True
    thread = threading.Thread(target=listener._poll_loop, daemon=True)
    thread.start()
    time.sleep(0.12)
    assert stripped == [1], "the Windows key was not released at the press"
    assert listener._press_triggered == {"transcribe"}, (
        "the recording ended when the Windows key left the state table")

    listener._held_set.discard(alt_vk)      # the user lets go
    time.sleep(0.12)
    listener._running = False
    thread.join(timeout=2)

    kinds = []
    while not listener._callbacks.empty():
        name, kind, cb = listener._callbacks.get_nowait()
        kinds.append(kind)
        cb()
    assert "press" in kinds and "release" in kinds
    assert not listener._synthetic_down
