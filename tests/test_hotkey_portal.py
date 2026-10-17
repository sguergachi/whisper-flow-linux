"""Portal global-shortcut backend: mapping, bind flow, and fallback."""

import threading
import time

import pytest

from whisper_flow.hotkey_portal import (
    PortalBindError,
    PortalBindPlan,
    PortalHotkeyListener,
    PortalUnavailable,
    config_to_trigger,
)


# ------------------------------------------------------------- trigger mapping
def test_default_chord_maps_to_a_shortcuts_spec_trigger():
    assert config_to_trigger("super+alt") == "LOGO+Alt_L"


def test_regular_key_chords_map_modifiers_and_key():
    assert config_to_trigger("ctrl+alt+space") == "CTRL+ALT+space"
    assert config_to_trigger("ctrl+super+alt") == "CTRL+LOGO+Alt_L"


def test_single_keys_and_sides_map():
    assert config_to_trigger("space") == "space"
    assert config_to_trigger("right_ctrl") == "Control_R"
    assert config_to_trigger("F9") == "F9"


def test_unknown_parts_leave_no_preferred_trigger():
    assert config_to_trigger("") is None
    assert config_to_trigger("super+banana") is None
    assert config_to_trigger("hyper+alt") is None


# ------------------------------------------------------------------ bind plan
def _plan():
    return PortalBindPlan([
        ("transcribe", "LOGO+Alt_L", "Push-to-talk transcription"),
        ("auto_transcribe", "CTRL+ALT+space", "Auto-stop transcription"),
    ])


def test_create_response_extracts_the_session():
    assert _plan().create_response_ok(
        {"session_handle": "/org/freedesktop/portal/session/1_1/x"}) == \
        "/org/freedesktop/portal/session/1_1/x"


def test_create_response_without_session_fails():
    with pytest.raises(PortalBindError):
        _plan().create_response_ok({})


def test_bind_response_accepts_full_triggers():
    bound = _plan().bind_response_ok(0, {"shortcuts": [
        ("transcribe", {"trigger_description": "Meta+Alt"}),
        ("auto_transcribe", {"trigger_description": "Ctrl+Alt+Space"}),
    ]})
    assert bound["transcribe"] == "Meta+Alt"


def test_bind_response_rejects_denial_and_partial_binds():
    with pytest.raises(PortalBindError):
        _plan().bind_response_ok(1, {"shortcuts": []})
    with pytest.raises(PortalBindError):
        _plan().bind_response_ok(0, {"shortcuts": [
            ("transcribe", {"trigger_description": "Meta+Alt"}),
        ]})
    with pytest.raises(PortalBindError):
        _plan().bind_response_ok(0, {"shortcuts": [
            ("transcribe", {"trigger_description": "Meta+Alt"}),
            ("auto_transcribe", {"trigger_description": ""}),
        ]})


# ------------------------------------------------------------------ fake transport
class _FakeParams:
    def __init__(self, payload):
        self._payload = payload

    def unpack(self):
        return self._payload


class FakePortalIO:
    """Drives the listener without D-Bus: tests fire signals by hand."""

    def __init__(self, quick_error=None):
        self.quick_error = quick_error
        self.created = []
        self.bound_calls = []
        self.closed = []
        self.response_subs = {}
        self.signal_subs = {}
        self._sub_id = 0
        self._quit = threading.Event()
        self.loops_run = 0

    def quick_check(self):
        if self.quick_error is not None:
            raise self.quick_error

    def connect(self):
        return self

    def unique_suffix(self, conn):
        return "9_99"

    def request_path(self, suffix, token):
        return f"/org/freedesktop/portal/desktop/request/{suffix}/{token}"

    def subscribe_response(self, conn, path, callback):
        self._sub_id += 1
        self.response_subs[path] = callback
        return self._sub_id

    def subscribe_signal(self, conn, member, arg0, callback):
        self._sub_id += 1
        self.signal_subs[member] = callback
        return self._sub_id

    def unsubscribe(self, conn, sub_id):
        pass

    def call_create_session(self, conn, request_path, handle_token,
                            session_token):
        self.created.append((request_path, handle_token, session_token))

    def call_bind_shortcuts(self, conn, session_handle, shortcuts,
                            request_token):
        self.bound_calls.append(
            (session_handle, shortcuts, request_token))

    def call_session_close(self, conn, session_handle):
        self.closed.append(session_handle)

    def run_loop(self):
        self.loops_run += 1
        self._quit.wait(timeout=10)

    def quit_loop(self):
        self._quit.set()

    def tick_each_second(self, callback):
        self.tick = callback

    def untick(self):
        pass

    # test drivers -------------------------------------------------
    def emit_response(self, path, code, results):
        self.response_subs[path](
            self, "sender", path, "iface", "Response",
            _FakeParams((code, results)), None)

    def emit_shortcut(self, member, session, shortcut_id):
        self.signal_subs[member](
            self, "sender", "/portal", "iface", member,
            _FakeParams((session, shortcut_id, 0, {})), None)


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _bind(listener, io, session="/session/1"):
    assert _wait_for(lambda: len(io.created) == 1)
    create_path = io.created[0][0]
    io.emit_response(create_path, 0, {"session_handle": session})
    assert _wait_for(lambda: len(io.bound_calls) == 1)
    return session


# ------------------------------------------------------------------ listener flow
def test_portal_streams_press_and_release():
    io = FakePortalIO()
    fired = []
    listener = PortalHotkeyListener(io_factory=lambda: io)
    listener.register_hotkey(
        "transcribe", "super+alt",
        lambda: fired.append("press"), lambda: fired.append("release"))
    listener.start()
    try:
        session = _bind(listener, io)
        triggers = dict(
            (sid, "Meta+Alt") for sid, _entry in io.bound_calls[0][1])
        io.emit_response(
            [p for p in io.response_subs if p != io.created[0][0]][0],
            0, {"shortcuts": [
                (sid, {"trigger_description": "Meta+Alt"}) for sid in triggers]})
        assert _wait_for(lambda: listener.is_bound)
        assert listener.status_snapshot()["backend"] == "portal"

        io.emit_shortcut("Activated", session, "transcribe")
        io.emit_shortcut("Deactivated", session, "transcribe")
        assert _wait_for(lambda: fired == ["press", "release"])
    finally:
        listener.stop()
    assert io.closed == [session]


def test_portal_prefers_the_config_trigger():
    io = FakePortalIO()
    listener = PortalHotkeyListener(io_factory=lambda: io)
    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener.register_hotkey(
        "auto_transcribe", "ctrl+alt+space", lambda: None)
    listener.start()
    try:
        _bind(listener, io)
        entries = dict(io.bound_calls[0][1])
        assert entries["transcribe"]["preferred_trigger"] == "LOGO+Alt_L"
        assert entries["auto_transcribe"]["preferred_trigger"] == \
            "CTRL+ALT+space"
    finally:
        listener.stop()


def test_portal_denial_fails_the_backend_once():
    io = FakePortalIO()
    outcomes = []
    listener = PortalHotkeyListener(
        on_bind_complete=lambda ok, detail: outcomes.append((ok, detail)),
        io_factory=lambda: io)
    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    listener.start()
    try:
        _bind(listener, io)
        bind_path = [p for p in io.response_subs if p != io.created[0][0]][0]
        io.emit_response(bind_path, 1, {"shortcuts": []})
        assert _wait_for(lambda: listener.bind_failed)
        assert listener.is_alive() is False
        assert outcomes and outcomes[-1][0] is False
        assert listener.status_snapshot()["failed"]
    finally:
        listener.stop()


def test_portal_ignores_unknown_and_early_signals():
    io = FakePortalIO()
    fired = []
    listener = PortalHotkeyListener(io_factory=lambda: io)
    listener.register_hotkey(
        "transcribe", "super+alt", lambda: fired.append("press"))
    listener.start()
    try:
        session = _bind(listener, io)
        # Before the bind completes, activations must not dispatch.
        io.emit_shortcut("Activated", session, "transcribe")
        io.emit_shortcut("Activated", session, "nope")
        time.sleep(0.2)
        assert fired == []
    finally:
        listener.stop()


def test_portal_fast_failure_raises_for_fallback():
    io = FakePortalIO(quick_error=PortalUnavailable("no bus"))
    listener = PortalHotkeyListener(io_factory=lambda: io)
    listener.register_hotkey("transcribe", "super+alt", lambda: None)
    with pytest.raises(PortalUnavailable):
        listener.start()


def test_portal_needs_shortcuts_to_start():
    listener = PortalHotkeyListener(io_factory=FakePortalIO)
    with pytest.raises(PortalUnavailable):
        listener.start()


# ------------------------------------------------------- manager integration
def _wayland(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


class _FakePortalListener:
    instances = []

    def __init__(self, *args, **kwargs):
        _FakePortalListener.instances.append(self)
        self.on_bind_complete = kwargs.get("on_bind_complete")
        self.registered = []
        self.escape_callback = None
        self.on_emergency = None
        self.backend_name = "portal"
        self.shortcuts_consumed = True
        self.stopped = False

    def register_hotkey(self, name, key_string, press, release=None,
                        release_modifiers=False):
        self.registered.append(name)

    def start(self):
        from whisper_flow.hotkey_portal import PortalUnavailable
        raise PortalUnavailable("no portal here")

    def stop(self):
        self.stopped = True

    def is_alive(self):
        return False

    def status_snapshot(self):
        return {"backend": "portal"}


def test_manager_prefers_portal_then_falls_back_on_fast_failure(
        monkeypatch):
    _wayland(monkeypatch)
    from whisper_flow import hotkey_manager as manager_module
    from whisper_flow.hotkey_manager import HotkeyManager, HotkeyMode

    built_observer = []
    real_observer = manager_module.__dict__.get("HotkeyManager")

    class FakeObserver:
        def __init__(self):
            built_observer.append(self)
            self.escape_callback = None
            self.on_emergency = None
            self.backend_name = "observer"

        def register_hotkey(self, *a, **k):
            pass

        def start(self):
            pass

        def is_alive(self):
            return True

        def stop(self):
            pass

        def status_snapshot(self):
            return {"backend": "evdev-observer"}

    monkeypatch.setattr(
        "whisper_flow.hotkey_portal.PortalHotkeyListener", _FakePortalListener)
    monkeypatch.setattr(
        "whisper_flow.hotkey_evdev_observer.EvdevHotkeyObserver", FakeObserver)

    manager = HotkeyManager()
    statuses = []
    manager.status_callback = lambda state, detail: statuses.append(state)
    manager.register_hotkey(
        "transcribe", "super+alt", HotkeyMode.PUSH_TO_TALK,
        lambda: None, lambda: None)
    manager._start_wayland_locked()

    assert isinstance(manager._evdev_listener, FakeObserver)
    assert "fallback" in statuses


def test_manager_falls_back_once_after_async_denial(monkeypatch):
    _wayland(monkeypatch)
    from whisper_flow.hotkey_manager import HotkeyManager, HotkeyMode

    class PendingPortal:
        backend_name = "portal"

        def __init__(self, *a, **k):
            self.on_bind_complete = k.get("on_bind_complete")
            self.escape_callback = None
            self.stopped = False
            self._failed = False

        def register_hotkey(self, *a, **k):
            pass

        def start(self):
            pass

        def stop(self):
            self.stopped = True

        def is_alive(self):
            return not self._failed

        @property
        def bind_failed(self):
            return self._failed

        def status_snapshot(self):
            return {"backend": "portal"}

    class FakeObserver:
        backend_name = "observer"

        def __init__(self):
            self.escape_callback = None

        def register_hotkey(self, *a, **k):
            pass

        def start(self):
            pass

        def is_alive(self):
            return True

        def stop(self):
            pass

        def status_snapshot(self):
            return {"backend": "evdev-observer"}

    monkeypatch.setattr(
        "whisper_flow.hotkey_portal.PortalHotkeyListener", PendingPortal)
    monkeypatch.setattr(
        "whisper_flow.hotkey_evdev_observer.EvdevHotkeyObserver", FakeObserver)

    manager = HotkeyManager()
    statuses = []
    manager.status_callback = lambda state, detail: statuses.append(state)
    manager.register_hotkey(
        "transcribe", "super+alt", HotkeyMode.PUSH_TO_TALK,
        lambda: None, lambda: None)
    manager._start_wayland_locked()
    portal = manager._evdev_listener
    assert isinstance(portal, PendingPortal)

    # Async denial arrives on the worker thread: exactly one fallback.
    portal._failed = True
    manager._on_portal_bind_complete(False, "denied")
    assert isinstance(manager._evdev_listener, FakeObserver)
    assert manager._portal_failed is True
    assert statuses.count("fallback") == 1

    # A second report (or heartbeat) must not swap again or re-notify.
    manager._heartbeat_check()
    assert isinstance(manager._evdev_listener, FakeObserver)
    assert statuses.count("fallback") == 1


def test_consumed_shortcuts_report_through_the_manager(monkeypatch):
    from whisper_flow.hotkey_manager import HotkeyManager

    manager = HotkeyManager()
    assert manager.hotkeys_consumed() is False

    class Bound:
        backend_name = "portal"
        shortcuts_consumed = True
        is_bound = True

    manager._evdev_listener = Bound()
    assert manager.hotkeys_consumed() is True
