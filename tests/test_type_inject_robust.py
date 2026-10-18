"""Robustness tests for the app's own typing injector.

The injector is the last line of defense when the host's injector is
disabled, so it must never hang, never leave a key held, and degrade
gracefully when the device cannot be opened.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    for attr in ("KEY_LEFTSHIFT", "KEY_SPACE", "KEY_ENTER", "EV_KEY"):
        setattr(fake_ecodes, attr, hash(attr) & 0xFFFF)
    # _codes_for uses getattr(ecodes, "KEY_A") etc.; provide them.
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        setattr(fake_ecodes, f"KEY_{c}", ord(c))
    # _PLAIN and _SHIFTED reference ecodes at module load (already done with
    # real ecodes). Patch the module-level names to the fakes for _device().
    monkeypatch.setattr(ti, "UInput", fake_evdev.UInput, raising=False)
    monkeypatch.setattr(ti, "ecodes", fake_ecodes, raising=False)
    monkeypatch.setattr(ti, "HAVE_EVDEV", True)
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
    result = ti.type_text("aXb")
    # Partial success: 'a' was already delivered, so we must not claim total
    # failure (the caller would retry and duplicate it). Modifiers are still
    # released, which is what this test is really pinning.
    assert result is True
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


def test_type_text_skips_untypable_chars(monkeypatch):
    fake_dev = _FakeDevice()
    fake_evdev, fake_ecodes = _patch_evdev(monkeypatch)
    fake_evdev.UInput = lambda caps, name=None, bustype=None: fake_dev
    monkeypatch.setattr(ti, "UInput", fake_evdev.UInput, raising=False)

    # "\u2014" (em dash) has no US key -> skipped. "ab" typed.
    result = ti.type_text("a\u2014b")
    assert result is True
    # Only 'a' and 'b' produced key writes (2 keys * 2 = 4 down/up + syns).
    key_writes = [w for w in fake_dev.writes if len(w) == 3]
    assert len(key_writes) == 5  # modifier release + a-down, a-up, b-down, b-up


def test_type_text_no_evdev_returns_false(monkeypatch):
    monkeypatch.setattr(ti, "HAVE_EVDEV", False)
    ti._DEVICE = None
    ti._DEVICE_FAILED = False
    assert ti.type_text("hello") is False
