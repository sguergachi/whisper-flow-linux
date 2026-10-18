"""Robustness tests for the app's own typing injector.

The injector is the last line of defense when the host's injector is
disabled, so it must never hang, never leave a key held, and degrade
gracefully when the device cannot be opened.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import whisper_flow.type_inject as ti


class _FakeDevice:
    """A uinput device that records writes and can be made to fail."""

    def __init__(self, fail_on=None):
        self.writes = []
        self._fail_on = fail_on  # code to raise on, or None
        self.closed = False

    def write(self, etype, code, value):
        if self._fail_on is not None and code == self._fail_on:
            raise OSError("device disappeared")
        self.writes.append((etype, code, value))

    def syn(self):
        self.writes.append(("syn",))

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_device_state():
    """Each test starts with no cached device."""
    ti._DEVICE = None
    ti._DEVICE_FAILED = False
    yield
    ti._DEVICE = None
    ti._DEVICE_FAILED = False


def _patch_evdev(monkeypatch, device_factory=_FakeDevice):
    """Make evdev importable with a fake UInput."""
    fake_evdev = MagicMock()
    fake_evdev.UInput = lambda caps, name=None, bustype=None: device_factory()
    fake_ecodes = MagicMock()
    # Map the names type_inject uses to stable integers.
    for attr in ("KEY_LEFTSHIFT", "KEY_LEFTCTRL", "KEY_SPACE", "KEY_ENTER", "EV_KEY"):
        setattr(fake_ecodes, attr, hash(attr) & 0xFFFF)
    # _codes_for uses getattr(ecodes, "KEY_A") etc.; provide them.
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        setattr(fake_ecodes, f"KEY_{c}", ord(c))
    # _PLAIN and _SHIFTED reference ecodes at module load (already done with
    # real ecodes). Patch the module-level names to the fakes for _device().
    monkeypatch.setattr(ti, "UInput", fake_evdev.UInput, raising=False)
    monkeypatch.setattr(ti, "ecodes", fake_ecodes, raising=False)
    monkeypatch.setattr(ti, "HAVE_EVDEV", True)
    monkeypatch.setattr(ti.time, "sleep", lambda _: None)
    # Make 'import evdev' inside _device return our fake.
    monkeypatch.setitem(__import__("sys").modules, "evdev", fake_evdev)
    # Patch the module-level ecodes that _codes_for uses.
    monkeypatch.setattr(ti, "_PLAIN", {
        " ": fake_ecodes.KEY_SPACE,
        "\n": fake_ecodes.KEY_ENTER,
    }, raising=False)
    monkeypatch.setattr(ti, "_SHIFTED", {}, raising=False)
    monkeypatch.setattr(ti, "_MODIFIER_KEYS", (fake_ecodes.KEY_LEFTSHIFT,), raising=False)
    return fake_evdev, fake_ecodes


def test_type_text_returns_false_on_empty(monkeypatch):
    _patch_evdev(monkeypatch)
    assert ti.type_text("") is False
    assert ti.type_text(None) is False


def test_type_text_returns_false_when_device_cannot_open(monkeypatch):
    fake_evdev = MagicMock()
    fake_evdev.UInput = MagicMock(side_effect=OSError("no uinput"))
    fake_ecodes = MagicMock()
    monkeypatch.setattr(ti, "UInput", fake_evdev.UInput, raising=False)
    monkeypatch.setattr(ti, "ecodes", fake_ecodes, raising=False)
    monkeypatch.setattr(ti, "HAVE_EVDEV", True)
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "evdev", fake_evdev)

    assert ti.type_text("hello") is False
    assert ti._DEVICE is None
    assert ti._DEVICE_FAILED is True
    # A second call does not retry (avoids hammering a broken device).
    assert ti.type_text("world") is False


def test_type_text_releases_modifiers_on_write_failure(monkeypatch):
    fake_dev = _FakeDevice(fail_on=ord("X"))  # fail when typing 'X'
    fake_evdev, fake_ecodes = _patch_evdev(monkeypatch)
    fake_evdev.UInput = lambda caps, name=None, bustype=None: fake_dev
    monkeypatch.setattr(ti, "UInput", fake_evdev.UInput, raising=False)
    monkeypatch.setattr(ti, "UInput", fake_evdev.UInput, raising=False)

    # 'X' is uppercase -> shift down, then KEY_X (ord('X')) fails.
    with pytest.raises(ti.PartialTypingError) as caught:
        ti.type_text("aXb")
    # Partial success: 'a' was already delivered, so we must not claim total
    # failure (the caller would retry and duplicate it). Modifiers are still
    # released, which is what this test is really pinning.
    assert caught.value.typed == 1
    # The shift that was pressed for 'X' must have been released.
    shift_code = fake_ecodes.KEY_LEFTSHIFT
    # Find the last shift write: should be a release (0).
    shift_writes = [w for w in fake_dev.writes if len(w) == 3 and w[1] == shift_code]
    assert shift_writes, "no shift writes recorded"
    assert len(shift_writes) >= 2, "expected release at start and on failure"
    assert shift_writes[-1][2] == 0, "modifier left held after failure"


def test_release_modifiers_is_noop_without_device(monkeypatch):
    _patch_evdev(monkeypatch)
    ti._DEVICE = None
    ti._DEVICE_FAILED = True
    # Must not raise.
    ti.release_modifiers()


def test_release_modifiers_writes_releases(monkeypatch):
    fake_dev = _FakeDevice()
    fake_evdev, fake_ecodes = _patch_evdev(monkeypatch)
    fake_evdev.UInput = lambda caps, name=None, bustype=None: fake_dev
    monkeypatch.setattr(ti, "UInput", fake_evdev.UInput, raising=False)
    # Force device creation.
    dev = ti._device()
    assert dev is fake_dev
    ti.release_modifiers()
    shift_writes = [w for w in fake_dev.writes if len(w) == 3 and w[1] == fake_ecodes.KEY_LEFTSHIFT]
    assert shift_writes and shift_writes[-1][2] == 0


def test_type_text_does_not_create_device_for_empty(monkeypatch):
    fake_evdev = MagicMock()
    fake_evdev.UInput = MagicMock()
    fake_ecodes = MagicMock()
    monkeypatch.setattr(ti, "UInput", fake_evdev.UInput, raising=False)
    monkeypatch.setattr(ti, "ecodes", fake_ecodes, raising=False)
    monkeypatch.setattr(ti, "HAVE_EVDEV", True)
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "evdev", fake_evdev)

    ti.type_text("")
    fake_evdev.UInput.assert_not_called()


def test_type_text_refuses_untypable_chars_before_any_input(monkeypatch):
    fake_dev = _FakeDevice()
    fake_evdev, fake_ecodes = _patch_evdev(monkeypatch)
    fake_evdev.UInput = lambda caps, name=None, bustype=None: fake_dev
    monkeypatch.setattr(ti, "UInput", fake_evdev.UInput, raising=False)

    # Refuse the whole sentence, so a fallback can preserve the em dash.
    result = ti.type_text("a\u2014b")
    assert result is False
    assert fake_dev.writes == []
    assert ti._DEVICE is None


def test_failed_key_up_releases_the_key_and_discards_device(monkeypatch):
    fake_dev = _FakeDevice()
    _, codes = _patch_evdev(monkeypatch)
    monkeypatch.setattr(ti, "UInput", lambda *a, **kw: fake_dev)
    original = fake_dev.write
    failed = False

    def fail_once(etype, code, value):
        nonlocal failed
        if code == codes.KEY_A and value == 0 and not failed:
            failed = True
            raise OSError("key-up failed")
        original(etype, code, value)

    fake_dev.write = fail_once
    with pytest.raises(ti.PartialTypingError) as caught:
        ti.type_text("ab")
    assert caught.value.typed == 1  # don't retry the already sent prefix
    assert fake_dev.writes[-2:] == [(codes.EV_KEY, codes.KEY_A, 0), ("syn",)]
    assert fake_dev.closed
    assert ti._DEVICE is None


def test_interruption_cleans_up_held_keys(monkeypatch):
    fake_dev = _FakeDevice()
    _, codes = _patch_evdev(monkeypatch)
    monkeypatch.setattr(ti, "UInput", lambda *a, **kw: fake_dev)
    original = fake_dev.write

    def interrupt(etype, code, value):
        if code == codes.KEY_A and value == 0:
            raise KeyboardInterrupt
        original(etype, code, value)

    fake_dev.write = interrupt
    with pytest.raises(KeyboardInterrupt):
        ti.type_text("A")
    assert fake_dev.closed
    assert ti._DEVICE is None


def test_release_does_not_create_an_input_device(monkeypatch):
    create = MagicMock()
    monkeypatch.setattr(ti, "UInput", create, raising=False)
    ti.release_modifiers()
    create.assert_not_called()


def test_clipboard_paste_uses_the_owned_keyboard(monkeypatch):
    device = _FakeDevice()
    _, codes = _patch_evdev(monkeypatch)
    monkeypatch.setattr(ti, "UInput", lambda *a, **kw: device)
    assert ti.paste_clipboard()
    assert (codes.EV_KEY, codes.KEY_LEFTCTRL, 1) in device.writes
    assert (codes.EV_KEY, codes.KEY_V, 1) in device.writes
    assert (codes.EV_KEY, codes.KEY_LEFTCTRL, 0) in device.writes


def test_modifier_release_cannot_interleave_with_typing(monkeypatch):
    import threading

    device = _FakeDevice()
    _, codes = _patch_evdev(monkeypatch)
    monkeypatch.setattr(ti, "UInput", lambda *a, **kw: device)
    pressed, finish, attempted, released = [threading.Event() for _ in range(4)]
    original = device.write

    def pause_after_press(etype, code, value):
        original(etype, code, value)
        if code == codes.KEY_A and value == 1:
            pressed.set()
            assert finish.wait(5)

    def release():
        attempted.set()
        ti.release_modifiers()
        released.set()

    device.write = pause_after_press
    typing = threading.Thread(target=lambda: ti.type_text("A"))
    releasing = threading.Thread(target=release)
    typing.start()
    try:
        assert pressed.wait(5)
        releasing.start()
        assert attempted.wait(5)
        assert not released.wait(.02)
    finally:
        finish.set()
        typing.join(5)
        if releasing.ident is not None:
            releasing.join(5)
    assert released.is_set()
    assert not typing.is_alive()


def test_type_text_no_evdev_returns_false(monkeypatch):
    monkeypatch.setattr(ti, "HAVE_EVDEV", False)
    ti._DEVICE = None
    ti._DEVICE_FAILED = False
    assert ti.type_text("hello") is False
