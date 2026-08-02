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



def _fake_system_win(monkeypatch, spoil):
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
