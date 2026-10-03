"""The pill must track the cursor while dragging, not trail it.

Each motion event used to re-assert all three layer-shell anchors before
moving the margins. Every anchor call costs a configure round-trip with the
compositor, so a drag paid three round-trips per motion event and the pill
lagged visibly behind the pointer. A move is just margins; anchors are set
once, on the transition between docked and dragged.
"""

import sys
import threading
from collections import deque
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


# ------------------------------------------- still content while dragging
def _frame_window():
    """A window with enough state to run one frame, paint recorded."""
    win = hud_app_module.HudWindow.__new__(hud_app_module.HudWindow)
    win._resident = True
    win.get_visible = lambda: True
    win._placed = True
    win._fade_in_t0 = 0.0
    win._fade_out_t0 = None
    win.alpha = 1.0
    win._blur = None
    win.want_hover = False
    win.hover = 0.0
    win.stop_hover = 0.0
    win.stop_hover_target = 0.0
    win.processing = False
    win.toast_text = None
    win.targets = deque([0.5] * hud_app_module.BARS)
    win.shown = [0.0] * hud_app_module.BARS
    win._levels_lock = threading.Lock()
    win.noise_risk = 0.0
    win.shown_risk = 0.0
    win._dragging = False
    win.area = Mock()
    return win


def test_drag_holds_paint_while_the_picture_is_steady():
    """Moving the window needs no repaint; the bars ease underneath."""
    win = _frame_window()
    win._dragging = True
    assert win._frame() is True
    win.area.queue_draw.assert_not_called()
    # Levels still follow the audio, so release paints them current.
    assert win.shown[0] > 0.0


def test_release_resumes_paint_on_the_next_frame():
    win = _frame_window()
    win._dragging = True
    win._frame()
    win._dragging = False
    win._frame()
    win.area.queue_draw.assert_called_once()


def test_animated_states_keep_painting_through_a_drag():
    """A fade, the processing sweep or a toast is watched; keep it live."""
    win = _frame_window()
    win._dragging = True
    win.processing = True
    win._processing_t0 = 0.0
    assert win._drag_holds_paint() is False
    win.processing = False
    win.toast_text = "mic switched"
    assert win._drag_holds_paint() is False
    win.toast_text = None
    win.alpha = 0.5
    assert win._drag_holds_paint() is False
    win.alpha = 1.0
    assert win._drag_holds_paint() is True


def test_chrome_is_built_at_startup_not_on_the_first_frame():
    """The resident warmup blits on first show instead of building."""
    win = hud_app_module.HudWindow.__new__(hud_app_module.HudWindow)
    win._chrome = None
    win._chrome_size = None
    win._style = None
    win.stop_button = False
    win.processing = False
    assert win._warm_chrome() is False
    assert win._chrome is not None
