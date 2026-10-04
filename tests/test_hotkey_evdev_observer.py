"""Safety contract for the production Wayland hotkey backend."""

from unittest.mock import Mock

import pytest

evdev = pytest.importorskip("evdev")
from evdev import ecodes

from whisper_flow.hotkey_evdev_observer import EvdevHotkeyObserver


class FakeEvent:
    type = ecodes.EV_KEY

    def __init__(self, code, value):
        self.code = code
        self.value = value


class FakeDevice:
    def __init__(self, path):
        self.path = path
        self.fd = 42
        self.closed = False
        self.grab_calls = 0

    def grab(self):
        self.grab_calls += 1
        raise AssertionError("production observer must never call EVIOCGRAB")

    def close(self):
        self.closed = True

    def active_keys(self):
        return []


def _drain(observer):
    while not observer._callbacks.empty():
        item = observer._callbacks.get()
        if item is None:
            continue
        _name, _kind, callback = item
        callback()


def test_opening_keyboards_never_grabs_them(monkeypatch):
    observer = EvdevHotkeyObserver()
    devices = {
        "/dev/input/event3": FakeDevice("/dev/input/event3"),
        "/dev/input/event7": FakeDevice("/dev/input/event7"),
    }
    monkeypatch.setattr(observer, "_find_keyboard_devices", lambda: list(devices))
    monkeypatch.setattr(
        "whisper_flow.hotkey_evdev_observer.evdev.InputDevice",
        lambda path: devices[path],
    )

    assert observer._open_devices() is True
    assert all(device.grab_calls == 0 for device in devices.values())
    assert observer.status_snapshot()["grabbed"] == 0


def test_hotkey_callbacks_need_no_proxy_or_forwarded_events():
    observer = EvdevHotkeyObserver()
    fired = []
    observer.register_hotkey(
        "transcribe", "cmd+alt",
        lambda: fired.append("press"),
        lambda: fired.append("release"),
        release_modifiers=True,
    )

    observer._handle_key(FakeEvent(ecodes.KEY_LEFTMETA, 1))
    observer._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 1))
    _drain(observer)
    observer._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 0))
    observer._handle_key(FakeEvent(ecodes.KEY_LEFTMETA, 0))
    _drain(observer)

    assert fired == ["press", "release"]
    assert not hasattr(observer, "_uinput")


def test_kernel_reconciliation_releases_only_internal_callback_state():
    observer = EvdevHotkeyObserver()
    fired = []
    observer.register_hotkey(
        "transcribe", "cmd+alt",
        lambda: fired.append("press"),
        lambda: fired.append("release"),
    )
    observer._handle_key(FakeEvent(ecodes.KEY_LEFTMETA, 1))
    observer._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 1))
    _drain(observer)
    assert fired == ["press"]

    observer._kbd_devices = [Mock(active_keys=Mock(return_value=[]))]
    assert observer._reconcile_with_kernel() == 2
    _drain(observer)
    assert fired == ["press", "release"]
    assert not hasattr(observer, "_uinput")


def test_manager_selects_the_fail_open_observer(monkeypatch):
    from whisper_flow.hotkey_manager import HotkeyManager, HotkeyMode

    built = []

    class FakeObserver:
        def __init__(self):
            built.append(self)
            self.registered = []
            self.escape_callback = None
            self.on_emergency = None

        def register_hotkey(self, *args, **kwargs):
            self.registered.append((args, kwargs))

        def start(self):
            pass

        def is_alive(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(
        "whisper_flow.hotkey_evdev_observer.EvdevHotkeyObserver", FakeObserver)
    manager = HotkeyManager()
    manager.register_hotkey(
        "transcribe", "super+alt", HotkeyMode.PUSH_TO_TALK,
        lambda: None, lambda: None,
    )
    manager._start_evdev()

    assert manager._evdev_listener is built[0]
    assert built[0].registered[0][1]["release_modifiers"] is False
