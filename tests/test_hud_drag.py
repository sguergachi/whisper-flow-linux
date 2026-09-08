"""The pill must track the cursor while dragging, not trail it.

Each motion event used to re-assert all three layer-shell anchors before
moving the margins. Every anchor call costs a configure round-trip with the
compositor, so a drag paid three round-trips per motion event and the pill
lagged visibly behind the pointer. A move is just margins; anchors are set
once, on the transition between docked and dragged.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

cairo = pytest.importorskip("cairo")
gi = pytest.importorskip("gi")

import whisper_flow.hud_app as hud_app_module


def _drag_window(pos):
    win = hud_app_module.HudWindow.__new__(hud_app_module.HudWindow)
    win._pos = pos
    win._layer_top_left = False
    win._monitor = None
    return win


def _fake_layer_shell():
    calls = {"anchor": [], "margin": []}
    fake = Mock()
    fake.Edge.LEFT = "L"
    fake.Edge.TOP = "T"
    fake.Edge.BOTTOM = "B"
    fake.set_anchor.side_effect = lambda w, e, v: calls["anchor"].append((e, v))
    fake.set_margin.side_effect = lambda w, e, v: calls["margin"].append((e, v))
    return fake, calls


def test_repeated_moves_set_anchors_once(monkeypatch):
    fake, calls = _fake_layer_shell()
    monkeypatch.setattr(hud_app_module, "LayerShell", fake)
    win = _drag_window((100, 200))
    for x in range(100, 110):
        win._pos = (x, 200)
        win._apply_position()
    anchors = [c for c in calls["anchor"] if c[1] is True]
    # One transition to top-left anchoring, then margins only.
    assert anchors.count(("L", True)) == 1
    assert anchors.count(("T", True)) == 1
    assert len([m for m in calls["margin"] if m[0] == "L"]) == 10


def test_docking_resets_the_anchor_state(monkeypatch):
    fake, calls = _fake_layer_shell()
    monkeypatch.setattr(hud_app_module, "LayerShell", fake)
    win = _drag_window((100, 200))
    win._apply_position()
    assert win._layer_top_left is True
    win._pos = None
    win._apply_position()
    assert win._layer_top_left is False
    # Next drag re-asserts anchors exactly once more.
    win._pos = (50, 60)
    win._apply_position()
    assert win._layer_top_left is True
    assert calls["anchor"].count(("L", True)) == 2


# ------------------------------------------------- coalesced drag applies
def _dragging_window(monkeypatch, pos=(100, 200)):
    """A window mid-drag, with position writes recorded, never executed."""
    fake, calls = _fake_layer_shell()
    monkeypatch.setattr(hud_app_module, "LayerShell", fake)
    monkeypatch.setattr(hud_app_module.GLib, "idle_add",
                        lambda *a, **k: calls.setdefault("idle", []).append(a) or 1)
    monkeypatch.setattr(hud_app_module, "_save_position",
                        lambda *a: calls.setdefault("saved", []).append(a))
    win = _drag_window(pos)
    win._connector = "test"
    win._dragging = False
    win._pos_flush_queued = False
    win._drag_origin = None
    return win, calls


def test_updates_retarget_without_applying(monkeypatch):
    """Ten motion events: one queued flush, zero compositor commits."""
    win, calls = _dragging_window(monkeypatch)
    win._on_drag_begin(None, 10, 10)
    for dx in range(1, 11):
        win._on_drag_update(None, dx * 5, 0)
    assert win._pos == (100 + 50, 200)
    assert len(calls.get("idle", [])) == 1
    assert calls["margin"] == []
    # The flush lands the latest target, anchors still set only once.
    win._flush_pos()
    left = [m for m in calls["margin"] if m[0] == "L"]
    assert left == [("L", 150)]
    assert calls["anchor"].count(("L", True)) == 1


def test_stale_flush_after_show_never_moves(monkeypatch):
    """A drag ended by a new recording must not move the fresh pill."""
    win, calls = _dragging_window(monkeypatch)
    win._on_drag_begin(None, 10, 10)
    win._on_drag_update(None, 50, 0)
    # begin_show's part: the recording owns the pill now.
    win._dragging = False
    win._drag_origin = None
    assert win._flush_pos() is False
    assert calls["margin"] == []


def test_drag_end_lands_the_final_target(monkeypatch):
    win, calls = _dragging_window(monkeypatch)
    win._on_drag_begin(None, 10, 10)
    win._on_drag_update(None, 50, 30)
    win._on_drag_end(None, 50, 30)
    assert win._dragging is False
    left = [m for m in calls["margin"] if m[0] == "L"]
    top = [m for m in calls["margin"] if m[0] == "T"]
    assert left == [("L", 150)]
    assert top == [("T", 230)]
    assert ("test", 150, 230) in calls["saved"]


def test_dragged_branch_clears_the_bottom_margin(monkeypatch):
    fake, calls = _fake_layer_shell()
    monkeypatch.setattr(hud_app_module, "LayerShell", fake)
    win = _drag_window((100, 200))
    win._apply_position()
    bottoms = [m for m in calls["margin"] if m[0] == "B"]
    assert bottoms == [("B", 0)]


def test_win32_pin_recorded_before_any_window(monkeypatch):
    """A target recorded with no HWND yet still governs the next move."""
    win = hud_app_module.HudWindow.__new__(hud_app_module.HudWindow)
    win._hwnd = None
    win._pin = None
    win._apply_position_win32(300, 400)
    assert win._pin == (300, 400)
