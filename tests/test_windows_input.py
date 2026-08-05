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


def _unsuppressed(monkeypatch):
    """Clear both routes to suppression, so a test starts from looking.

    Both are module state and a preceding type_text leaves them set, so
    zeroing only the one a test happens to read is how it would pass or fail
    on the order it ran in.
    """
    from whisper_flow import hotkey_win

    monkeypatch.setattr(hotkey_win, "_settled_at", 0.0)
    monkeypatch.setattr(hotkey_win, "_typing_depth", 0)
    return hotkey_win


def test_typing_blinds_the_listener_while_it_types(win, monkeypatch):
    """Restoring the keys is not enough on its own.

    The poll runs every 16ms and the gap between releasing the modifiers and
    putting them back is a few milliseconds wide - narrow, and caught often
    enough that the recording still ended at the first word after the
    restore was added. The listener has to not look at all.

    Observed from inside the injection rather than after it. What matters is
    that the keyboard is ignored for every event we send, and asserting that
    from the outside afterwards measures the settling window instead - which
    is deliberately short, and would make this a race.
    """
    hotkey_win = _unsuppressed(monkeypatch)
    system_win, _ = win
    monkeypatch.setattr(system_win._user32, "GetAsyncKeyState", lambda vk: 0)

    seen = []
    monkeypatch.setattr(
        system_win, "_send",
        lambda events: (seen.append(hotkey_win._suppressed()), True)[1])

    assert not hotkey_win._suppressed()
    system_win.type_text("hello")

    assert seen, "nothing was sent"
    assert all(seen), (
        "the listener was still reading the keyboard while we typed into it")


def test_the_listener_is_looking_again_once_typing_is_over(win, monkeypatch):
    """A window that never closes is a hotkey that never works again."""
    import time

    hotkey_win = _unsuppressed(monkeypatch)
    system_win, _ = win
    monkeypatch.setattr(system_win._user32, "GetAsyncKeyState", lambda vk: 0)

    system_win.type_text("hello")
    assert hotkey_win._typing_depth == 0, "the injection did not release it"
    time.sleep(hotkey_win.SETTLE_SECONDS * 3)
    assert not hotkey_win._suppressed(), (
        "suppression outlived the typing; the hotkey would stop responding")


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


# --------------------------------------------------- the Start menu
def _order_of(batches, vk):
    """Indices of the batches that carry an event for this key."""
    return [i for i, batch in enumerate(batches) if _flags_for(batch, vk)]


def test_letting_go_of_the_injected_super_does_not_open_start(win):
    """The release we send has to be spoiled, exactly as the first one is.

    restore_modifiers presses Super back down while the user is still
    holding it, and Windows sees that as a fresh press. The release at the
    end of the dictation then completes a lone Super tap and opens Start -
    which takes focus, so the rest of a live transcription was typed into
    its search box.
    """
    system_win, batches = win
    system_win._injected_down.clear()
    system_win._injected_down.update({system_win.VK_LWIN, system_win.VK_MENU})

    system_win.release_injected_modifiers()

    noop = _order_of(batches, system_win.VK_NONAME)
    supers = _order_of(batches, system_win.VK_LWIN)
    assert noop, "nothing marked the Super press as used"
    assert supers, "Super was never released"
    assert min(noop) < min(supers), (
        "the no-op key must be sent while Windows still believes Super is "
        "down; after the release it is too late and Start is already armed")


def test_nothing_is_sent_when_we_are_holding_nothing(win):
    """No keys of ours means no keystrokes at all, spoiler included."""
    system_win, batches = win
    system_win._injected_down.clear()

    assert system_win.release_injected_modifiers() == ()
    assert batches == []


def test_the_keys_are_still_released(win):
    """The spoiler must not have displaced the thing it guards."""
    system_win, batches = win
    system_win._injected_down.clear()
    system_win._injected_down.update({system_win.VK_LWIN, system_win.VK_MENU})

    released = system_win.release_injected_modifiers()

    assert set(released) == {system_win.VK_LWIN, system_win.VK_MENU}
    for vk in (system_win.VK_LWIN, system_win.VK_MENU):
        ups = [f for batch in batches for f in _flags_for(batch, vk)
               if f & system_win.KEYEVENTF_KEYUP]
        assert ups, f"{vk:#x} was never released"
    assert not system_win._injected_down, "the keys must not stay recorded"
