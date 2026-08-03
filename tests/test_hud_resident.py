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


# ------------------------------------------------------------ opening at all
def test_the_overlay_opens_the_display_itself():
    """Nothing else does it, and GTK 4 does not check.

    The overlay runs a bare GLib.MainLoop rather than a GtkApplication, to
    skip the D-Bus round trip an application id costs on the path the user
    waits on. GtkApplication is also what would otherwise call gtk_init(),
    so leaving it out means calling it here. It was not called: on Windows
    the process built a window with no display and walked off a null pointer
    inside GTK's style machinery, dying with an access violation before its
    first frame - no traceback, no window, and a daemon that reported an
    overlay it had opened.

    Read from the source rather than by running it, because every test that
    drives HudWindow calls Gtk.init() first - which is exactly why this went
    unnoticed for as long as it did.
    """
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1]
              / "src/whisper_flow/hud_app.py").read_text(encoding="utf-8")
    body = source.split("def main(")[-1]
    assert "Gtk.init" in body, (
        "hud_app.main() must initialise GTK; nothing else in the overlay "
        "process does, and constructing a window without it is a crash")
    assert body.index("Gtk.init") < body.index("HudWindow("), (
        "GTK has to be initialised before the window is constructed")


def _hud_app_source() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parents[1]
            / "src/whisper_flow/hud_app.py").read_text(encoding="utf-8")


def test_the_overlay_is_placed_in_physical_pixels():
    """GDK measures in logical units; SetWindowPos takes physical pixels.

    Nothing converted between them, so on a 2x display the pill went to half
    its intended offset - the upper-left quadrant rather than bottom-centre.
    It was on screen the whole time, nowhere near where anyone was looking.
    """
    source = _hud_app_source()
    assert "_monitor_scale" in source, (
        "the overlay needs the monitor's logical-to-physical factor")
    branch = source.split("def _apply_position(", 1)[1].split("\n    def ", 1)[0]
    assert "_monitor_scale()" in branch, (
        "the Win32 branch of _apply_position must scale GDK's logical "
        "coordinates to physical pixels before calling SetWindowPos")


def test_the_overlay_is_placed_after_it_is_mapped():
    """GTK places the window itself when it maps it, and maps after realize.

    Positioning at realize alone was silently undone every time.
    """
    source = _hud_app_source()
    assert '"map"' in source and "_on_map" in source, (
        "the overlay must re-apply its position on map; a position set at "
        "realize is overwritten by GTK's own placement")


def test_a_parked_overlay_keeps_following_the_level_file():
    """The timer must survive the recording that ends, or later ones are dead.

    _read_levels returned False when the daemon deleted the level file, which
    removes the GLib source for good. A resident overlay parks and is shown
    again, and every later recording drew a waveform frozen where the last
    one ended.
    """
    source = _hud_app_source()
    body = source.split("def _read_levels(", 1)[1].split("\n    def ", 1)[0]
    assert "return self._resident" in body, (
        "a resident overlay must keep its level timer when the file goes "
        "away; returning False removes the source permanently")


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
