"""Safety contract for the production Wayland hotkey backend.

These tests are intentionally more defensive than ordinary unit tests.  A
hotkey may fail to fire, but this backend must never be able to interrupt the
physical keyboard's path to the compositor.
"""

import ast
import inspect
import itertools
import threading
from unittest.mock import Mock

import pytest

evdev = pytest.importorskip("evdev")
from evdev import ecodes

from whisper_flow import hotkey_evdev_observer
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


def _event(code, value):
    return FakeEvent(code, value)


def test_module_has_no_keyboard_interception_or_injection_primitives():
    """Make the fail-open property an enforced source-level invariant."""
    source = inspect.getsource(hotkey_evdev_observer)
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert "EVIOCGRAB" not in names | attrs
    assert "/dev/uinput" not in source
    assert "hotkey_evdev" not in source.replace("hotkey_evdev_observer", "")
    assert "UInput" not in names | attrs
    assert not ({"grab", "ungrab", "write", "syn"} & attrs)
    device_opens = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "InputDevice"
    ]
    assert device_opens
    assert all(
        any(keyword.arg == "readonly"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords)
        for call in device_opens
    )


def test_all_left_right_modifier_and_press_release_orders():
    """Every side/order combination fires once and leaves no held state."""
    metas = (ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA)
    alts = (ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT)

    for meta, alt in itertools.product(metas, alts):
        for down_order in itertools.permutations((meta, alt)):
            for up_order in itertools.permutations((meta, alt)):
                observer = EvdevHotkeyObserver()
                fired = []
                observer.register_hotkey(
                    "transcribe", "cmd+alt",
                    lambda fired=fired: fired.append("press"),
                    lambda fired=fired: fired.append("release"),
                )
                for code in down_order:
                    observer._handle_key(_event(code, 1))
                    observer._handle_key(_event(code, 2))
                for code in up_order:
                    observer._handle_key(_event(code, 0))
                _drain(observer)

                assert fired == ["press", "release"]
                assert observer._key_state == set()
                assert observer._press_triggered == set()


def test_exhaustive_short_event_sequences_match_reference_model():
    """Exercise 9,331 sequences including bounce, repeats, and stray ups."""
    meta = ecodes.KEY_LEFTMETA
    alt = ecodes.KEY_LEFTALT
    actions = (
        (meta, 0), (meta, 1), (meta, 2),
        (alt, 0), (alt, 1), (alt, 2),
    )

    for length in range(6):
        for sequence in itertools.product(actions, repeat=length):
            observer = EvdevHotkeyObserver()
            actual = []
            observer.register_hotkey(
                "transcribe", "cmd+alt",
                lambda actual=actual: actual.append("press"),
                lambda actual=actual: actual.append("release"),
            )
            model_state = set()
            model_active = False
            expected = []
            for code, value in sequence + ((meta, 0), (alt, 0)):
                observer._handle_key(_event(code, value))
                if value == 2:
                    continue
                if value == 1:
                    model_state.add(code)
                else:
                    model_state.discard(code)
                satisfied = model_state == {meta, alt}
                if value == 1 and satisfied and not model_active:
                    expected.append("press")
                    model_active = True
                elif not satisfied and model_active:
                    expected.append("release")
                    model_active = False
            _drain(observer)

            assert actual == expected, sequence
            assert observer._key_state == set(), sequence
            assert observer._press_triggered == set(), sequence


def test_duplicate_down_and_repeat_never_retrigger_active_chord():
    observer = EvdevHotkeyObserver()
    fired = []
    observer.register_hotkey(
        "transcribe", "cmd+alt",
        lambda: fired.append("press"),
        lambda: fired.append("release"),
    )
    for event in (
        _event(ecodes.KEY_LEFTMETA, 1),
        _event(ecodes.KEY_LEFTALT, 1),
        _event(ecodes.KEY_LEFTALT, 1),
        _event(ecodes.KEY_LEFTALT, 2),
        _event(ecodes.KEY_LEFTMETA, 2),
        _event(ecodes.KEY_LEFTALT, 0),
        _event(ecodes.KEY_LEFTALT, 0),
        _event(ecodes.KEY_LEFTMETA, 0),
    ):
        observer._handle_key(event)
    _drain(observer)

    assert fired == ["press", "release"]


def test_unrelated_key_cancels_without_retriggering_held_chord():
    observer = EvdevHotkeyObserver()
    fired = []
    observer.register_hotkey(
        "transcribe", "cmd+alt",
        lambda: fired.append("press"),
        lambda: fired.append("release"),
    )
    for code, value in (
        (ecodes.KEY_LEFTMETA, 1),
        (ecodes.KEY_LEFTALT, 1),
        (ecodes.KEY_A, 1),
        (ecodes.KEY_A, 0),
        (ecodes.KEY_LEFTALT, 0),
        (ecodes.KEY_LEFTMETA, 0),
    ):
        observer._handle_key(_event(code, value))
    _drain(observer)

    assert fired == ["press", "release"]


def test_longest_matching_binding_wins_and_transitions_safely():
    observer = EvdevHotkeyObserver()
    fired = []
    observer.register_hotkey(
        "short", "cmd+alt",
        lambda: fired.append("short-press"),
        lambda: fired.append("short-release"),
    )
    observer.register_hotkey(
        "long", "cmd+alt+shift",
        lambda: fired.append("long-press"),
        lambda: fired.append("long-release"),
    )
    for code in (ecodes.KEY_LEFTMETA, ecodes.KEY_LEFTALT,
                 ecodes.KEY_LEFTSHIFT):
        observer._handle_key(_event(code, 1))
    for code in (ecodes.KEY_LEFTSHIFT, ecodes.KEY_LEFTALT,
                 ecodes.KEY_LEFTMETA):
        observer._handle_key(_event(code, 0))
    _drain(observer)

    assert fired == [
        "short-press", "short-release", "long-press", "long-release",
    ]


def test_unknown_or_empty_binding_can_never_fire():
    observer = EvdevHotkeyObserver()
    fired = []
    observer.register_hotkey("empty", "not-a-linux-key", fired.append)
    observer._handle_key(_event(ecodes.KEY_A, 1))
    observer._handle_key(_event(ecodes.KEY_A, 0))
    _drain(observer)

    assert fired == []


def test_malformed_key_value_is_ignored_without_changing_state():
    observer = EvdevHotkeyObserver()
    observer._key_state = {ecodes.KEY_LEFTMETA}

    observer._handle_key(_event(ecodes.KEY_LEFTALT, -1))
    observer._handle_key(_event(ecodes.KEY_LEFTALT, 3))

    assert observer._key_state == {ecodes.KEY_LEFTMETA}


def test_device_discovery_is_readonly_and_excludes_virtual_devices(monkeypatch):
    class DiscoveryDevice:
        def __init__(self, path, bustype, key_count):
            self.path = path
            self.info = Mock(bustype=bustype)
            self._key_count = key_count
            self.closed = False

        def capabilities(self):
            return {ecodes.EV_KEY: list(range(self._key_count))}

        def close(self):
            self.closed = True

    devices = {
        "/dev/input/event1": DiscoveryDevice(
            "/dev/input/event1", hotkey_evdev_observer.BUS_VIRTUAL, 100),
        "/dev/input/event2": DiscoveryDevice(
            "/dev/input/event2", 0x03, 20),
        "/dev/input/event3": DiscoveryDevice(
            "/dev/input/event3", 0x03, 100),
    }
    monkeypatch.setattr(evdev, "list_devices", lambda: list(devices))

    def open_readonly(path, *, readonly=False):
        assert readonly is True
        return devices[path]

    monkeypatch.setattr(evdev, "InputDevice", open_readonly)
    observer = EvdevHotkeyObserver()

    assert observer._find_keyboard_devices() == ["/dev/input/event3"]
    assert all(device.closed for device in devices.values())


def test_opening_keyboards_never_grabs_them(monkeypatch):
    observer = EvdevHotkeyObserver()
    devices = {
        "/dev/input/event3": FakeDevice("/dev/input/event3"),
        "/dev/input/event7": FakeDevice("/dev/input/event7"),
    }
    monkeypatch.setattr(observer, "_find_keyboard_devices", lambda: list(devices))
    def open_readonly(path, *, readonly=False):
        assert readonly is True
        return devices[path]

    monkeypatch.setattr(
        "whisper_flow.hotkey_evdev_observer.evdev.InputDevice", open_readonly)

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


def test_kernel_reconciliation_is_idempotent_after_missed_release():
    observer = EvdevHotkeyObserver()
    fired = []
    observer.register_hotkey(
        "transcribe", "cmd+alt",
        lambda: fired.append("press"),
        lambda: fired.append("release"),
    )
    observer._handle_key(_event(ecodes.KEY_LEFTMETA, 1))
    observer._handle_key(_event(ecodes.KEY_LEFTALT, 1))
    observer._kbd_devices = [Mock(active_keys=Mock(return_value=[]))]

    assert observer._reconcile_with_kernel() == 2
    assert observer._reconcile_with_kernel() == 0
    observer._reset_callback_state()
    observer._reset_callback_state()
    _drain(observer)

    assert fired == ["press", "release"]


def test_two_keyboards_do_not_release_a_modifier_still_held_on_one():
    class HeldDevice:
        def __init__(self, keys):
            self.keys = keys

        def active_keys(self):
            return list(self.keys)

    first = HeldDevice({ecodes.KEY_LEFTMETA})
    second = HeldDevice(set())
    third = HeldDevice(set())
    observer = EvdevHotkeyObserver()
    observer._kbd_devices = [first, second, third]
    fired = []
    observer.register_hotkey(
        "transcribe", "cmd+alt",
        lambda: fired.append("press"),
        lambda: fired.append("release"),
    )

    observer._handle_key(_event(ecodes.KEY_LEFTMETA, 1))
    second.keys.add(ecodes.KEY_LEFTALT)
    third.keys.add(ecodes.KEY_RIGHTALT)
    observer._handle_key(_event(ecodes.KEY_LEFTALT, 1))
    second.keys.clear()
    observer._handle_key(_event(ecodes.KEY_LEFTALT, 0))
    _drain(observer)
    assert fired == ["press"]

    third.keys.clear()
    observer._handle_key(_event(ecodes.KEY_RIGHTALT, 0))
    _drain(observer)
    assert fired == ["press", "release"]


def test_failed_kernel_queries_never_invent_or_drop_key_state():
    observer = EvdevHotkeyObserver()
    observer._key_state = {ecodes.KEY_LEFTMETA}
    observer._kbd_devices = [
        Mock(active_keys=Mock(side_effect=OSError("device vanished"))),
    ]

    assert observer._kernel_held_keys() is None
    assert observer._reconcile_with_kernel() == 0
    assert observer._key_state == {ecodes.KEY_LEFTMETA}


def test_callback_exception_does_not_stop_later_callbacks():
    observer = EvdevHotkeyObserver()
    completed = threading.Event()

    def broken():
        raise RuntimeError("callback exploded")

    thread = threading.Thread(target=observer._dispatch_loop)
    thread.start()
    observer._callbacks.put(("broken", "press", broken))
    observer._callbacks.put(("working", "press", completed.set))
    observer._callbacks.put(None)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert completed.is_set()


def test_stop_while_hotkey_held_dispatches_one_release_and_is_idempotent():
    observer = EvdevHotkeyObserver()
    fired = []
    observer.register_hotkey(
        "transcribe", "cmd+alt",
        lambda: fired.append("press"),
        lambda: fired.append("release"),
    )
    observer._dispatch_thread = threading.Thread(target=observer._dispatch_loop)
    observer._dispatch_thread.start()
    observer._handle_key(_event(ecodes.KEY_LEFTMETA, 1))
    observer._handle_key(_event(ecodes.KEY_LEFTALT, 1))

    observer.stop()
    observer.stop()

    assert fired == ["press", "release"]
    assert observer._key_state == set()
    assert observer._press_triggered == set()


def test_partial_device_open_failure_keeps_available_keyboard(monkeypatch):
    observer = EvdevHotkeyObserver()
    working = FakeDevice("/dev/input/event3")
    monkeypatch.setattr(
        observer, "_find_keyboard_devices",
        lambda: ["/dev/input/event3", "/dev/input/event7"],
    )

    def open_device(path, *, readonly=False):
        assert readonly is True
        if path.endswith("7"):
            raise OSError("permission denied")
        return working

    monkeypatch.setattr(
        "whisper_flow.hotkey_evdev_observer.evdev.InputDevice", open_device)

    assert observer._open_devices() is True
    assert observer._kbd_devices == [working]
    assert observer._open_failures[0][0] == "/dev/input/event7"
    assert working.grab_calls == 0


def test_start_with_no_readable_keyboard_fails_without_background_threads(
        monkeypatch):
    observer = EvdevHotkeyObserver()
    monkeypatch.setattr(observer, "_open_devices", lambda: False)

    with pytest.raises(RuntimeError, match="Cannot open any keyboard"):
        observer.start()

    assert observer._running is False
    assert observer._thread is None
    assert observer._dispatch_thread is None


def test_select_failure_exits_pump_without_touching_devices(monkeypatch):
    observer = EvdevHotkeyObserver()
    device = FakeDevice("/dev/input/event3")
    observer._kbd_devices = [device]
    observer._running = True
    monkeypatch.setattr(
        "whisper_flow.hotkey_evdev_observer.select.select",
        Mock(side_effect=OSError("fd disappeared")),
    )

    observer._pump_current_devices()

    assert device.closed is False
    assert device.grab_calls == 0


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
