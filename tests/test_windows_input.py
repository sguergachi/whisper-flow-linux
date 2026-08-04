"""Typing must not tell Windows the user let go of the hotkey.

Live transcription types committed words while the push-to-talk keys are
still physically down. Typing releases every modifier first, so a held Alt
cannot turn the text into shortcuts - and the hotkey listener reads the very
state table those releases write to. It concluded the key had been let go and
ended the recording at the first committed word, every time, with the key
still held. The overlay disappearing mid-sentence was this.
"""

import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="the Windows input path")


@pytest.fixture()
def win(monkeypatch):
    from whisper_flow import system_win

    batches = []
    monkeypatch.setattr(system_win, "_send",
                        lambda events: (batches.append(list(events)), True)[1])
    return system_win, batches


def _flags_for(batch, vk):
    """The flags of every event in this batch that names this key."""
    return [event.union.ki.dwFlags for event in batch
            if event.union.ki.wVk == vk]


def test_typing_puts_back_the_modifiers_the_user_is_holding(win, monkeypatch):
    system_win, batches = win
    held = {system_win.VK_MENU, system_win.VK_LWIN}
    monkeypatch.setattr(
        system_win._user32, "GetAsyncKeyState",
        lambda vk: -0x8000 if vk in held else 0)

    assert system_win.type_text("hello")

    # The last thing sent has to be the two keys going back down, or the
    # listener polls a moment later and sees a push-to-talk that ended.
    restored = batches[-1]
    for vk in held:
        flags = _flags_for(restored, vk)
        assert flags == [0], (
            f"key {vk:#x} was not pressed back down after typing; the "
            f"listener will read it as released and stop the recording")


def test_nothing_is_pressed_that_was_not_already_held(win, monkeypatch):
    """A key the user is not holding must never be pushed down by us.

    Otherwise a recording that ends while text is being typed leaves a
    modifier stuck down for every application on the desktop.
    """
    system_win, batches = win
    monkeypatch.setattr(system_win._user32, "GetAsyncKeyState", lambda vk: 0)

    assert system_win.type_text("hello")

    presses = [event for batch in batches for event in batch
               if event.union.ki.wVk in system_win.MODIFIERS
               and event.union.ki.dwFlags == 0]
    assert not presses, (
        "a modifier was pressed that the user was not holding")


def test_typing_blinds_the_listener_for_the_duration(win, monkeypatch):
    """Restoring the keys is not enough on its own.

    The poll runs every 16ms and the gap between releasing the modifiers and
    putting them back is a few milliseconds wide - narrow, and caught often
    enough that the recording still ended at the first word after the
    restore was added. The listener has to not look at all.
    """
    from whisper_flow import hotkey_win

    system_win, _ = win
    monkeypatch.setattr(system_win._user32, "GetAsyncKeyState", lambda vk: 0)
    monkeypatch.setattr(hotkey_win, "_suppressed_until", 0.0)

    assert not hotkey_win._suppressed()
    system_win.type_text("hello")
    assert hotkey_win._suppressed(), (
        "the listener was still reading the keyboard while we typed into it")


def test_suppression_is_not_permanent(monkeypatch):
    """A window that never closes is a hotkey that never works again."""
    import time

    from whisper_flow import hotkey_win

    # suppress() extends rather than replaces, so an earlier test's window
    # would still be open here.
    monkeypatch.setattr(hotkey_win, "_suppressed_until", 0.0)
    hotkey_win.suppress(0.05)
    assert hotkey_win._suppressed()
    time.sleep(0.12)
    assert not hotkey_win._suppressed(), (
        "suppression outlived its window; the hotkey would stop responding")


def test_a_genuine_release_still_ends_the_recording(win, monkeypatch):
    """Restoring must be of the snapshot, not unconditional.

    The whole point is that letting go still works; a restore that ignored
    what was actually held would make the hotkey impossible to release.
    """
    system_win, batches = win
    monkeypatch.setattr(system_win._user32, "GetAsyncKeyState", lambda vk: 0)

    system_win.type_text("hello")
    restored = [event for batch in batches for event in batch
                if event.union.ki.wVk in system_win.MODIFIERS
                and event.union.ki.dwFlags == 0]
    assert not restored, (
        "nothing was held, so nothing may be restored - otherwise the "
        "hotkey can never be released")
