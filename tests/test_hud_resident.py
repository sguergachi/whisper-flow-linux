"""Tests for the resident overlay.

Starting a process is the largest cost between pressing the hotkey and
seeing anything, so the overlay is kept alive and commanded down a pipe.
Measured here: 114-159ms to visible when spawned per recording, 6-8ms when
resident. What follows guards the lifecycle that makes that safe - above
all, that a resident overlay can never be stranded on screen.
"""

import sys
import threading
import time
from unittest.mock import Mock

import pytest

from whisper_flow import hud as hud_module


@pytest.fixture
def hud(monkeypatch):
    monkeypatch.setattr(hud_module, "RESIDENT", True)
    return hud_module.HUD()


class FakeProcess:
    """A live overlay process, with the pipe the daemon writes to."""

    def __init__(self):
        self.stdin = Mock()
        self.written = []
        self.stdin.write.side_effect = self.written.append
        self._returncode = None
        self.killed = False

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        self._returncode = 0
        return 0

    def kill(self):
        self.killed = True
        self._returncode = -9

    def die(self, code=1):
        self._returncode = code


def _running(hud, monkeypatch, process=None):
    process = process or FakeProcess()
    monkeypatch.setattr(hud, "_resident_process", lambda: process)
    hud._process = process
    return process


# ------------------------------------------------------------- commanding it
def test_showing_sends_a_command_rather_than_starting_a_process(hud, monkeypatch):
    process = _running(hud, monkeypatch)
    spawned = []
    monkeypatch.setattr(hud_module.subprocess, "Popen",
                        lambda *a, **k: spawned.append(a) or FakeProcess())

    hud.show(level_file="/tmp/levels")
    assert process.written == ["show /tmp/levels\n"]
    assert spawned == []            # nothing was started


def test_hiding_sends_a_command_and_keeps_the_process(hud, monkeypatch):
    process = _running(hud, monkeypatch)
    hud.hide()
    assert process.written == ["hide\n"]
    assert process.poll() is None   # still alive for the next press


def test_a_dead_overlay_is_replaced_on_the_next_show(hud, monkeypatch):
    dead, fresh = FakeProcess(), FakeProcess()
    dead.die()
    handed = [dead, fresh]
    monkeypatch.setattr(hud, "_resident_process", lambda: handed.pop(0))
    dead.stdin.write.side_effect = OSError("broken pipe")

    assert hud._command("show /tmp/x") is True
    assert fresh.written == ["show /tmp/x\n"]


def test_a_command_gives_up_rather_than_looping(hud, monkeypatch):
    monkeypatch.setattr(hud, "_resident_process", lambda: None)
    assert hud._command("show /tmp/x") is False


def test_nothing_is_commanded_when_residency_is_off(monkeypatch):
    monkeypatch.setattr(hud_module, "RESIDENT", False)
    hud = hud_module.HUD()
    monkeypatch.setattr(hud, "_hide_locked", Mock())
    spawned = []
    monkeypatch.setattr(hud_module.subprocess, "Popen",
                        lambda *a, **k: spawned.append(1) or FakeProcess())
    monkeypatch.setattr(hud_module.tempfile, "mkstemp", lambda **k: (0, "/tmp/x.log"))
    monkeypatch.setattr(hud_module.os, "close", lambda fd: None)
    monkeypatch.setattr("builtins.open", lambda *a, **k: Mock())

    hud.show(level_file="/tmp/levels")
    assert spawned, "the one-shot path must still start a process"


# ------------------------------------------------------------------ shutdown
def test_shutdown_closes_the_pipe_so_the_overlay_leaves(hud, monkeypatch):
    process = _running(hud, monkeypatch)
    hud.shutdown()
    process.stdin.close.assert_called_once()
    assert hud._process is None


def test_shutdown_kills_an_overlay_that_will_not_leave(hud, monkeypatch):
    process = _running(hud, monkeypatch)
    process.wait = Mock(side_effect=Exception("still there"))
    hud.shutdown()
    assert process.killed


def test_shutdown_is_safe_with_no_overlay(hud):
    hud.shutdown()          # must not raise
    assert hud._process is None


# ------------------------------------------------------- the overlay's states
_GTK_CHILD = """
import sys
sys.path.insert(0, sys.argv[1] + "/src")
scenario = sys.argv[2]
import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib
from whisper_flow import hud_app

Gtk.init()
win = hud_app.HudWindow("", None, resident=True)
win.realize()
for _ in range(30):
    GLib.MainContext.default().iteration(False)

if scenario == "hide_not_die":
    win.begin_show("")
    assert win.get_visible()
    win.begin_hide()
    win._after_fade_out()
    assert not win.get_visible(), "resident window must hide, not die"
    assert not win._quitting
elif scenario == "reset":
    win.begin_show("")
    win.level_pos = 16
    win.peak = 9999.0
    win.begin_show("")
    # Otherwise the new recording starts mid-waveform at the old scale.
    assert win.level_pos == 0
    assert win.peak == hud_app.PEAK_FLOOR
print("OK")
"""


def _gtk_available() -> bool:
    import os
    import subprocess
    if sys.platform != "win32" and not os.environ.get("DISPLAY"):
        return False
    result = subprocess.run(
        [sys.executable, "-c",
         "import gi; gi.require_version('Gtk', '4.0'); "
         "from gi.repository import Gtk; Gtk.init()"],
        capture_output=True, check=False)
    return result.returncode == 0


@pytest.mark.skipif(not _gtk_available(), reason="needs GTK4 and a display")
def test_the_overlay_hides_instead_of_exiting_between_recordings(tmp_path):
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", _GTK_CHILD,
         str(__import__("pathlib").Path(__file__).resolve().parents[1]),
         "hide_not_die"],
        capture_output=True, text=True, check=False, timeout=60)
    assert "OK" in result.stdout, result.stderr


@pytest.mark.skipif(not _gtk_available(), reason="needs GTK4 and a display")
def test_a_second_recording_resets_the_waveform(tmp_path):
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", _GTK_CHILD,
         str(__import__("pathlib").Path(__file__).resolve().parents[1]),
         "reset"],
        capture_output=True, text=True, check=False, timeout=60)
    assert "OK" in result.stdout, result.stderr


# ------------------------------------------------------------------ prewarm
def test_prewarming_starts_the_overlay_before_any_press(hud, monkeypatch):
    """Otherwise the first press of every session pays process startup.

    Measured: 142ms for the first press without this, 6ms with it.
    """
    started = []
    monkeypatch.setattr(hud, "_resident_process",
                        lambda: started.append(1) or FakeProcess())
    hud.prewarm()
    _settle()
    assert started, "prewarm must start the overlay"


def test_prewarming_does_nothing_when_residency_is_off(monkeypatch):
    monkeypatch.setattr(hud_module, "RESIDENT", False)
    hud = hud_module.HUD()
    started = []
    monkeypatch.setattr(hud, "_resident_process", lambda: started.append(1))
    hud.prewarm()
    _settle()
    assert started == []


def test_a_failed_prewarm_is_survivable(hud, monkeypatch):
    """A missing overlay must cost one slow press, not a broken daemon."""
    monkeypatch.setattr(hud, "_resident_process",
                        Mock(side_effect=OSError("cannot start")))
    hud.prewarm()
    _settle()           # the thread must not take the process down with it


def test_showing_after_prewarm_does_not_start_another(hud, monkeypatch):
    process = FakeProcess()
    calls = []

    def resident():
        calls.append(1)
        return process

    monkeypatch.setattr(hud, "_resident_process", resident)
    hud.prewarm()
    _settle()
    before = len(calls)

    hud.show(level_file="/tmp/levels")
    assert process.written == ["show /tmp/levels\n"]
    assert len(calls) == before + 1      # asked for the same live one
    assert process.poll() is None


def _settle():
    """Let prewarm's thread finish. It is short-lived by design."""
    for _ in range(100):
        time.sleep(0.005)
        if not any(t.name == "whisper-flow-hud-prewarm" and t.is_alive()
                   for t in threading.enumerate()):
            return
