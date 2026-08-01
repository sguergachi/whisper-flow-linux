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
    assert released == {ecodes.KEY_LEFTMETA, ecodes.KEY_LEFTALT}

    # A second key goes down before the synthetic releases, so the compositor
    # sees Super+Alt rather than a bare Super tap opening the launcher.
    real = [f for f in listener.forwarded if len(f) == 2]
    assert real[:2] == [(ecodes.KEY_LEFTMETA, 1), (ecodes.KEY_LEFTALT, 1)]

    # Auto-repeat must not re-assert them as held.
    listener.forwarded.clear()
    listener._handle_key(FakeEvent(ecodes.KEY_LEFTMETA, 2))
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
