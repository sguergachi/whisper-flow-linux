"""Guards on the evdev listener.

This listener sits between the physical keyboard and the compositor: it grabs
the real devices and re-emits everything through a uinput proxy. If it ever
drops or invents a key event, the user's keyboard breaks system-wide - which
has happened, by holding combination keys back and then flushing a synthetic
press that never got its release. The first test below exists so that cannot
return unnoticed.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

evdev = pytest.importorskip("evdev")
from evdev import ecodes  # noqa: E402

from whisper_flow.hotkey_evdev import EvdevHotkeyListener  # noqa: E402


class FakeEvent:
    type = ecodes.EV_KEY

    def __init__(self, code, value):
        self.code = code
        self.value = value


@pytest.fixture
def listener():
    lis = EvdevHotkeyListener()
    lis.forwarded = []
    lis._uinput = type(
        "FakeUInput", (),
        {
            "write_event": lambda _s, ev: lis.forwarded.append((ev.code, ev.value)),
            "write": lambda _s, t, c, v: lis.forwarded.append(("synthetic", c, v)),
            "syn": lambda _s: None,
        },
    )()
    return lis


def _press_release(lis, *codes):
    for c in codes:
        lis._handle_key(FakeEvent(c, 1))
    for c in reversed(codes):
        lis._handle_key(FakeEvent(c, 0))


def test_every_key_event_is_forwarded_exactly_once(listener):
    """The listener must never withhold a real key event or invent a press."""
    fired = []
    listener.register_hotkey(
        "transcribe", "cmd+alt",
        lambda: fired.append("press"), lambda: fired.append("release"),
    )

    sent = [
        (ecodes.KEY_LEFTMETA, 1), (ecodes.KEY_LEFTALT, 1),   # hotkey held
        (ecodes.KEY_LEFTMETA, 2), (ecodes.KEY_LEFTALT, 2),   # auto-repeat
        (ecodes.KEY_LEFTALT, 0), (ecodes.KEY_LEFTMETA, 0),   # released
        (ecodes.KEY_SPACE, 1), (ecodes.KEY_SPACE, 0),        # plain typing
        (ecodes.KEY_A, 1), (ecodes.KEY_A, 0),
    ]
    for code, value in sent:
        listener._handle_key(FakeEvent(code, value))

    assert listener.forwarded == sent
    assert not any(len(f) == 3 for f in listener.forwarded), "invented an event"


def test_space_still_types_while_a_hotkey_binding_exists(listener):
    """Regression: a stuck modifier made every space a shortcut instead."""
    listener.register_hotkey("transcribe", "cmd+alt", lambda: None, lambda: None)
    listener.register_hotkey("auto", "ctrl+alt+space", lambda: None, None)

    _press_release(listener, ecodes.KEY_LEFTMETA, ecodes.KEY_LEFTALT)
    listener.forwarded.clear()
    _press_release(listener, ecodes.KEY_SPACE)

    assert listener.forwarded == [(ecodes.KEY_SPACE, 1), (ecodes.KEY_SPACE, 0)]


def test_no_key_state_survives_a_press_release_cycle(listener):
    """Leftover state is what turns into a stuck modifier."""
    listener.register_hotkey("transcribe", "cmd+alt", lambda: None, lambda: None)
    for _ in range(5):
        _press_release(listener, ecodes.KEY_LEFTMETA, ecodes.KEY_LEFTALT)
    assert listener._key_state == set()
    assert listener._press_triggered == set()


def test_autorepeat_does_not_end_a_push_to_talk(listener):
    """Holding the combination must not look like a release."""
    fired = []
    listener.register_hotkey(
        "transcribe", "cmd+alt",
        lambda: fired.append("press"), lambda: fired.append("release"),
    )

    def drain():
        while not listener._callbacks.empty():
            _n, _k, cb = listener._callbacks.get()
            cb()

    listener._handle_key(FakeEvent(ecodes.KEY_LEFTMETA, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 1))
    drain()
    assert fired == ["press"]

    for _ in range(10):
        listener._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 2))
    drain()
    assert fired == ["press"], "auto-repeat ended the recording"

    listener._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 0))
    drain()
    assert fired == ["press", "release"]


def test_no_key_down_is_ever_synthesised(listener):
    """A synthetic press without a matching release strands a modifier."""
    listener.register_hotkey(
        "transcribe", "cmd+alt", lambda: None, lambda: None,
        release_modifiers=True,
    )
    for code in (ecodes.KEY_LEFTMETA, ecodes.KEY_LEFTALT):
        listener._handle_key(FakeEvent(code, 1))
    for _ in range(20):
        listener._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 2))
    for code in (ecodes.KEY_LEFTALT, ecodes.KEY_LEFTMETA):
        listener._handle_key(FakeEvent(code, 0))

    synthetic = [f for f in listener.forwarded if len(f) == 3 and f[0] == "synthetic"]
    assert synthetic, "expected synthetic releases"
    assert all(f[2] == 0 for f in synthetic), "synthesised a key press"


def test_modifiers_are_released_to_the_compositor_while_held(listener):
    """Otherwise dictated text arrives as global shortcuts, not text."""
    listener.register_hotkey(
        "transcribe", "cmd+alt", lambda: None, lambda: None,
        release_modifiers=True,
    )
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTMETA, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 1))

    released = {
        f[1] for f in listener.forwarded
        if len(f) == 3 and f[0] == "synthetic" and f[2] == 0
    }
    # Both physical sides of each modifier: hardware may have sent the right
    # key while the binding only names the left one.
    assert {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA,
            ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT} <= released

    # A second key goes down before the synthetic releases, so the compositor
    # sees Super+Alt rather than a bare Super tap opening the launcher.
    real = [f for f in listener.forwarded if len(f) == 2]
    assert real[:2] == [(ecodes.KEY_LEFTMETA, 1), (ecodes.KEY_LEFTALT, 1)]

    # Auto-repeat must not re-assert them as held.
    listener.forwarded.clear()
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTMETA, 2))
    assert listener.forwarded == []


def test_right_meta_is_cleared_when_the_binding_names_super(listener):
    """KN85 / many boards send RIGHTMETA; releasing only LEFTMETA stuck Super.

    KDE then treats every typed letter as Meta+letter (power profile,
    Overview, desktop peek, …) for the rest of the hold.
    """
    listener.register_hotkey(
        "transcribe", "cmd+alt", lambda: None, lambda: None,
        release_modifiers=True,
    )
    # Hardware sends the right-side Super; binding still matches via alias.
    listener._handle_key(FakeEvent(ecodes.KEY_RIGHTMETA, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 1))

    synthetic_ups = [
        f[1] for f in listener.forwarded
        if len(f) == 3 and f[0] == "synthetic" and f[2] == 0
    ]
    assert ecodes.KEY_RIGHTMETA in synthetic_ups
    assert ecodes.KEY_LEFTMETA in synthetic_ups

    # Auto-repeat on the right key must stay muted too.
    listener.forwarded.clear()
    listener._handle_key(FakeEvent(ecodes.KEY_RIGHTMETA, 2))
    assert listener.forwarded == []


def test_real_releases_still_reach_the_compositor(listener):
    """A duplicate key-up is harmless; a missing one is not."""
    listener.register_hotkey(
        "transcribe", "cmd+alt", lambda: None, lambda: None,
        release_modifiers=True,
    )
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTMETA, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 1))
    listener.forwarded.clear()
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 0))
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTMETA, 0))

    assert (ecodes.KEY_LEFTALT, 0) in listener.forwarded
    assert (ecodes.KEY_LEFTMETA, 0) in listener.forwarded
    assert listener._muted == set()


# --------------------------------------------------- keyboards coming and going
class _FakeDevice:
    def __init__(self, path, fd):
        self.path, self.fd = path, fd
        self.grabbed = False
        self.closed = False

    def grab(self):
        self.grabbed = True

    def ungrab(self):
        self.grabbed = False

    def close(self):
        self.closed = True


def _listener_with(monkeypatch, paths):
    """A listener wired to a controllable set of keyboard paths."""
    from whisper_flow import hotkey_evdev

    listener = hotkey_evdev.EvdevHotkeyListener()
    listener._uinput = Mock()
    made = {}

    def fake_find():
        return [(p, {}) for p in paths]

    def fake_device(path):
        made.setdefault(path, _FakeDevice(path, 100 + len(made)))
        return made[path]

    monkeypatch.setattr(listener, "_find_keyboard_devices", fake_find)
    monkeypatch.setattr(hotkey_evdev.evdev, "InputDevice", fake_device)
    return listener, made


def test_a_keyboard_plugged_in_later_gets_grabbed(monkeypatch):
    paths = ["/dev/input/event1"]
    listener, made = _listener_with(monkeypatch, paths)
    assert listener._open_devices()
    assert listener._grabbed_paths() == {"/dev/input/event1"}

    paths.append("/dev/input/event2")            # a keyboard appears
    assert listener._grabbed_paths() != listener._keyboard_paths()

    listener._abandon_devices()
    listener._open_devices()
    assert listener._grabbed_paths() == {"/dev/input/event1", "/dev/input/event2"}


def test_rebuilding_never_drops_the_uinput_proxy(monkeypatch):
    """Closing the proxy while a device is grabbed swallows the user's typing."""
    listener, _ = _listener_with(monkeypatch, ["/dev/input/event1"])
    listener._open_devices()
    proxy = listener._uinput

    listener._abandon_devices()
    assert listener._uinput is proxy
    proxy.close.assert_not_called()


def test_rebuilding_ungrabs_every_device(monkeypatch):
    """A grabbed device nobody is reading is a dead keyboard."""
    listener, made = _listener_with(monkeypatch, ["/dev/input/event1",
                                                  "/dev/input/event2"])
    listener._open_devices()
    listener._abandon_devices()
    assert all(not d.grabbed for d in made.values())
    assert listener._kbd_devices == []


def test_a_rebuild_ends_an_active_push_to_talk(monkeypatch):
    """Otherwise the recording it started never stops."""
    listener, _ = _listener_with(monkeypatch, ["/dev/input/event1"])
    listener._open_devices()

    released = []
    listener.register_hotkey("transcribe", "super+alt",
                             lambda: None, lambda: released.append(True))
    listener._press_triggered.add("transcribe")
    listener._active_hotkey = "transcribe"
    listener._key_state.add(ecodes.KEY_LEFTMETA)

    listener._abandon_devices()

    name, kind, cb = listener._callbacks.get_nowait()
    cb()
    assert released == [True]
    assert listener._key_state == set()
    assert listener._active_hotkey is None


def test_full_teardown_does_drop_the_proxy(monkeypatch):
    listener, _ = _listener_with(monkeypatch, ["/dev/input/event1"])
    listener._open_devices()
    proxy = listener._uinput
    listener._release_devices()
    proxy.close.assert_called_once()
    assert listener._uinput is None


def test_debug_logging_only_ever_names_hotkey_keys(monkeypatch):
    """It must not become a keylogger writing into the journal."""
    from whisper_flow import hotkey_evdev

    listener = hotkey_evdev.EvdevHotkeyListener()
    listener._uinput = Mock()
    listener.register_hotkey("transcribe", "super+alt", lambda: None)

    codes = listener._binding_codes
    assert codes == {ecodes.KEY_LEFTMETA, ecodes.KEY_LEFTALT}
    for ordinary in (ecodes.KEY_A, ecodes.KEY_P, ecodes.KEY_1, ecodes.KEY_ENTER):
        assert ordinary not in codes


def test_debug_logging_is_off_unless_asked_for():
    from whisper_flow import hotkey_evdev

    assert hotkey_evdev.DEBUG_KEYS is False


def test_a_busy_keyboard_names_the_other_grabber():
    """EBUSY is the second-daemon case; the tray used to just say 'cannot grab'."""
    import errno

    from whisper_flow.hotkey_evdev import _grab_error

    err = OSError(errno.EBUSY, "Device or resource busy")
    msg = _grab_error([("/dev/input/event4", err)])
    assert "another program already has it" in msg
    assert "whisper-flow" in msg


def test_a_permission_error_is_quoted_rather_than_called_busy():
    import errno

    from whisper_flow.hotkey_evdev import _grab_error

    err = OSError(errno.EACCES, "Permission denied")
    msg = _grab_error([("/dev/input/event4", err)])
    assert "Permission denied" in msg
    assert "another program already has it" not in msg


def test_open_devices_records_why_grab_failed(monkeypatch):
    import errno

    from whisper_flow import hotkey_evdev

    listener = hotkey_evdev.EvdevHotkeyListener()

    class Busy:
        def __init__(self, path):
            self.path = path

        def grab(self):
            raise OSError(errno.EBUSY, "Device or resource busy")

        def close(self):
            pass

    monkeypatch.setattr(listener, "_find_keyboard_devices",
                        lambda: [("/dev/input/event4", {})])
    monkeypatch.setattr(hotkey_evdev.evdev, "InputDevice", Busy)
    assert listener._open_devices() is False
    assert listener._grab_failures
    assert listener._grab_failures[0][0] == "/dev/input/event4"
    assert listener._grab_failures[0][1].errno == errno.EBUSY


def test_start_refuses_when_every_keyboard_is_busy(monkeypatch):
    import errno

    from whisper_flow import hotkey_evdev

    listener = hotkey_evdev.EvdevHotkeyListener()
    monkeypatch.setattr(listener, "_find_keyboard_devices",
                        lambda: [("/dev/input/event4", {ecodes.EV_KEY: [1]})])

    def fake_open():
        listener._grab_failures = [
            ("/dev/input/event4", OSError(errno.EBUSY, "Device or resource busy")),
        ]
        return False

    monkeypatch.setattr(listener, "_open_devices", fake_open)
    monkeypatch.setattr(hotkey_evdev.evdev.UInput, "from_device",
                        lambda *a, **k: Mock(close=Mock()))
    with pytest.raises(RuntimeError, match="another program already has it"):
        listener.start()


def test_escape_fires_even_during_a_held_combination(listener):
    """Escape cancels a recording, so it cannot be subject to
    most-specific-wins: held push-to-talk keys would always outrank it."""
    listener.escape_callback = lambda: None
    listener.register_hotkey("transcribe", "cmd+alt", lambda: None)

    listener._handle_key(FakeEvent(ecodes.KEY_LEFTMETA, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_ESC, 1))

    first = listener._callbacks.get_nowait()
    second = listener._callbacks.get_nowait()
    assert [first[0], second[0]] == ["transcribe", "escape"]
    # The real key event is still forwarded, like any other.
    assert (ecodes.KEY_ESC, 1) in listener.forwarded


def test_auto_chord_fires_when_kernel_state_spans_two_devices(listener):
    """Dual-HID boards put modifiers on one node and Space on another.

    Event-edge tracking alone missed the full chord in production; the
    listener must merge EVIOCGKEY across grabbed devices.
    """
    fired = []
    listener.register_hotkey(
        "auto_transcribe", "ctrl+alt+space",
        lambda: fired.append("press"), None,
        release_modifiers=True,
    )

    class Dev:
        def __init__(self, keys):
            self._keys = keys
        def active_keys(self):
            return list(self._keys)

    # input0: modifiers only; input1: space only (SONiX KN85 shape).
    mods = Dev({ecodes.KEY_LEFTCTRL, ecodes.KEY_LEFTALT})
    space = Dev({ecodes.KEY_SPACE})
    listener._kbd_devices = [mods, space]

    # Space edge arrives on the space-only interface; sync should see all three.
    listener._handle_key(FakeEvent(ecodes.KEY_SPACE, 1))
    while not listener._callbacks.empty():
        _n, _k, cb = listener._callbacks.get()
        cb()

    assert fired == ["press"], f"auto did not fire; state={listener._key_state}"
    assert listener._key_state == {
        ecodes.KEY_LEFTCTRL, ecodes.KEY_LEFTALT, ecodes.KEY_SPACE,
    }


def test_near_miss_logs_when_one_key_short(listener, monkeypatch):
    """A chord missing one key must not stay silent in the journal."""
    logs = []
    monkeypatch.setattr(
        "whisper_flow.logging.log",
        lambda *a, **k: logs.append(" ".join(str(x) for x in a)),
    )
    listener.register_hotkey(
        "auto_transcribe", "ctrl+alt+space", lambda: None, None,
    )
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTCTRL, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTALT, 1))
    # Space never pressed: one short of the chord.
    assert any("almost auto_transcribe" in line and "missing" in line
               for line in logs), logs


# ------------------------------------------------- input safety net
class _RaisingUInput:
    def write_event(self, event):
        raise OSError("proxy gone")

    def write(self, *a):
        raise OSError("proxy gone")

    def syn(self):
        raise OSError("proxy gone")

    def close(self):
        pass


def _alive_thread():
    """A stand-in reader thread so recovery never spawns a real one."""
    thread = Mock()
    thread.is_alive.return_value = True
    return thread


def _grabbed(listener, monkeypatch):
    """One fake grabbed keyboard on the listener."""
    dev = _FakeDevice("/dev/input/event9", 99)
    dev.grab()
    listener._kbd_devices = [dev]
    listener._running = True
    return dev


def test_stalled_pump_frees_the_keyboard(listener):
    """Grabs held + no pump = wedged reader: ungrab first, ask later."""
    import time

    dev = _grabbed(listener, None)
    listener._thread = _alive_thread()
    listener._last_pump_at = time.monotonic() - 5.0
    notified = []
    listener.on_emergency = notified.append

    listener._supervise_once()

    assert dev.grabbed is False
    assert listener._kbd_devices == []
    assert listener._key_state == set()
    assert len(notified) == 1 and "stalled" in notified[0]


def test_healthy_pump_stays_quiet(listener):
    """A turning loop with grabs held is normal operation, not an emergency."""
    import time

    dev = _grabbed(listener, None)
    listener._thread = _alive_thread()
    listener._last_pump_at = time.monotonic()
    notified = []
    listener.on_emergency = notified.append

    listener._supervise_once()

    assert dev.grabbed is True
    assert notified == []


def test_supervisor_ignores_an_unheld_keyboard():
    """Nothing grabbed means nothing to save, however quiet the loop."""
    import time

    from whisper_flow.hotkey_evdev import EvdevHotkeyListener

    lis = EvdevHotkeyListener()
    lis._running = True
    lis._thread = _alive_thread()
    lis._last_pump_at = time.monotonic() - 60.0
    notified = []
    lis.on_emergency = notified.append

    lis._supervise_once()

    assert notified == []


def test_repeated_recoveries_stand_down_but_leave_keys_free(listener):
    """Desktop usable always beats working hotkeys."""
    _grabbed(listener, None)
    listener._thread = _alive_thread()
    listener._running = True
    notified = []
    listener.on_emergency = notified.append

    for _ in range(4):
        listener._emergency_recover("x")
    assert listener._listener_disabled is True
    assert len(notified) == 4
    assert any("disabled" in m for m in notified)

    # Stood down: even a stale pump fires nothing more.
    import time
    listener._last_pump_at = time.monotonic() - 60.0
    listener._supervise_once()
    assert len(notified) == 4


def test_persistent_forward_failures_trigger_recovery(listener):
    """Dozens of lost writes with events flowing means the proxy is dead."""
    dev = _grabbed(listener, None)
    listener._thread = _alive_thread()
    listener._uinput = _RaisingUInput()
    notified = []
    listener.on_emergency = notified.append

    for _ in range(29):
        listener._forward(FakeEvent(ecodes.KEY_A, 1))
    assert notified == []
    listener._forward(FakeEvent(ecodes.KEY_A, 1))

    assert dev.grabbed is False
    assert len(notified) == 1 and "uinput" in notified[0]


def test_single_forward_glitch_does_not_reset_input(listener):
    """One lost write is noise; the counter resets on the next success."""
    _grabbed(listener, None)
    listener._thread = _alive_thread()
    notified = []
    listener.on_emergency = notified.append

    calls = {"n": 0}

    class Flaky:
        def write_event(self, event):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("blip")

        def syn(self):
            pass

        def close(self):
            pass

    listener._uinput = Flaky()
    listener._forward(FakeEvent(ecodes.KEY_A, 1))
    listener._forward(FakeEvent(ecodes.KEY_A, 1))
    assert notified == []
    assert listener._forward_failures == 0


def test_triple_esc_resets_all_input(listener):
    """Three quick Escapes free a confused desktop even mid-chord."""
    dev = _grabbed(listener, None)
    listener._thread = _alive_thread()
    cancelled = []
    listener.escape_callback = lambda: cancelled.append(True)
    notified = []
    listener.on_emergency = notified.append
    listener._key_state.add(ecodes.KEY_LEFTMETA)
    listener._muted.add(ecodes.KEY_LEFTMETA)

    for _ in range(3):
        listener._handle_key(FakeEvent(ecodes.KEY_ESC, 1))
    while not listener._callbacks.empty():
        _n, _k, cb = listener._callbacks.get()
        cb()

    # Every Escape still cancels as usual...
    assert len(cancelled) == 3
    # ...and the third additionally frees everything unconditionally.
    assert len(notified) == 1 and "triple-Esc" in notified[0]
    assert dev.grabbed is False
    assert listener._key_state == set()
    assert listener._muted == set()
    synthetics = [f for f in listener.forwarded if len(f) == 3]
    released = {(c, v) for _, c, v in synthetics}
    for code in (ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA,
                 ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL,
                 ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT,
                 ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT):
        assert (code, 0) in released


def test_single_esc_only_cancels(listener):
    """Ordinary cancel must not trip the emergency path."""
    dev = _grabbed(listener, None)
    listener._thread = _alive_thread()
    cancelled = []
    listener.escape_callback = lambda: cancelled.append(True)
    notified = []
    listener.on_emergency = notified.append

    listener._handle_key(FakeEvent(ecodes.KEY_ESC, 1))
    while not listener._callbacks.empty():
        _n, _k, cb = listener._callbacks.get()
        cb()

    assert cancelled == [True]
    assert notified == []
    assert dev.grabbed is True


def test_esc_counter_cleared_by_other_keys(listener):
    """Esc Esc A Esc Esc is typing, not a reset request."""
    listener._thread = _alive_thread()
    listener.escape_callback = lambda: None
    notified = []
    listener.on_emergency = notified.append

    listener._handle_key(FakeEvent(ecodes.KEY_ESC, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_ESC, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_A, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_ESC, 1))
    listener._handle_key(FakeEvent(ecodes.KEY_ESC, 1))

    assert notified == []


def test_sweep_releases_only_what_we_do_not_hold(listener):
    """A shortcut the user is holding must never be broken by hygiene."""
    listener._running = True
    listener.forwarded.clear()
    listener._key_state.add(ecodes.KEY_LEFTMETA)
    listener._muted.add(ecodes.KEY_LEFTALT)

    assert listener.sweep_unheld_modifiers() == 6
    synthetics = {(c, v) for _, c, v in listener.forwarded}
    assert (ecodes.KEY_LEFTMETA, 0) not in synthetics
    assert (ecodes.KEY_RIGHTMETA, 0) not in synthetics
    assert (ecodes.KEY_LEFTALT, 0) in synthetics
    assert (ecodes.KEY_LEFTCTRL, 0) in synthetics
    assert (ecodes.KEY_RIGHTSHIFT, 0) in synthetics


def test_sweep_is_a_noop_without_proxy_or_when_stopped(listener):
    listener._uinput = None
    assert listener.sweep_unheld_modifiers() == 0
    listener._running = False
    assert listener.sweep_unheld_modifiers() == 0


def test_maybe_sweep_policy_bounds_idle_hygiene(listener):
    """At most one pass a minute, only shortly after hotkey activity."""
    import time

    listener._running = True
    # Never touched hotkeys: silent, no writes.
    assert listener.maybe_sweep_unheld_modifiers() == 0
    assert listener.forwarded == []

    # Recent activity: one bounded pass...
    listener._last_input_risk_at = time.monotonic()
    assert listener.maybe_sweep_unheld_modifiers() == 8
    # ...and the next minute stays quiet.
    assert listener.maybe_sweep_unheld_modifiers() == 0

    # Stale risk, never swept: quiet again.
    listener._last_sweep_at = 0.0
    listener._last_input_risk_at = time.monotonic() - 700.0
    listener.forwarded.clear()
    assert listener.maybe_sweep_unheld_modifiers() == 0
    assert listener.forwarded == []


def test_status_snapshot_counts_never_names(listener):
    """Diagnostics must not become a keylog: counts only."""
    import time

    listener._kbd_devices = [Mock(path="/dev/input/event1")]
    listener._thread = _alive_thread()
    listener._running = True
    listener._last_pump_at = time.monotonic() - 0.5
    listener._key_state.add(42)
    listener._muted.add(43)

    snap = listener.status_snapshot()
    assert snap["backend"] == "evdev"
    assert snap["alive"] is True
    assert snap["grabbed"] == 1
    assert snap["held"] == 1
    assert snap["muted"] == 1
    assert snap["disabled"] is False
    assert snap["recoveries_60s"] == 0
    assert 0.0 <= snap["pump_age_s"] < 5.0
    assert "42" not in str(snap.values())


def test_pump_age_none_before_first_pump():
    from whisper_flow.hotkey_evdev import EvdevHotkeyListener

    assert EvdevHotkeyListener().pump_age_seconds() is None


def test_rebuild_marks_input_risk(listener, monkeypatch):
    """A device-set rebuild with keys held arms the idle sweeper."""
    listener, _ = _listener_with(monkeypatch, ["/dev/input/event1"])
    listener._open_devices()
    released = []
    listener.register_hotkey("transcribe", "super+alt",
                             lambda: None, lambda: released.append(True))
    listener._press_triggered.add("transcribe")

    listener._abandon_devices()

    assert listener._last_input_risk_at > 0.0
    name, kind, cb = listener._callbacks.get_nowait()
    cb()
    assert released == [True]
