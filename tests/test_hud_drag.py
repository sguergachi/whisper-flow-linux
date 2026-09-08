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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
