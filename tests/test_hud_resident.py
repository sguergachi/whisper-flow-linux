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
from pathlib import Path
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
    assert process.written == ["show - /tmp/levels\n"]
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


def test_the_pill_keeps_its_proportions_at_any_height():
    """Windows draws a smaller pill, and it must be the same pill.

    Its outline there is whatever DWM rounds the window to - a region will
    not clip the acrylic behind it - and DWM has one radius. Making that
    read as round means making the pill smaller, which is only acceptable
    if everything scales together rather than the contents being squashed
    into a shorter box.
    """
    source = _hud_app_source()
    assert "SIZE_SCALE" in source, (
        "the pill's measurements must derive from one scale factor")
    for name in ("WIDTH = ", "DOT_X", "WAVE_L", "BAR_MAX", "BAR_W"):
        line = next((line for line in source.splitlines()
                     if line.startswith(name)), "")
        assert "SIZE_SCALE" in line, (
            f"{name.strip()} does not scale with the pill; a shorter pill "
            f"would keep this measurement and lose the proportions")


def test_the_window_is_cut_down_to_the_pill():
    """A toplevel is a rectangle, and DWM paints its backdrop across all of it.

    The pill drew transparent corners and they went nowhere: the acrylic
    filled the whole rect, so the capsule sat on an opaque slab. A window
    region is what actually removes those pixels.
    """
    source = _hud_app_source()
    assert "_apply_shape_win32" in source, (
        "the overlay must clip its window to the pill, or the corners stay "
        "opaque whatever cairo draws")
    assert "_squircle_points" in source, (
        "the region must be cut from the same geometry the pill is drawn "
        "with, or the two drift apart")
    blur = (Path(__file__).resolve().parents[1]
            / "src/whisper_flow/blur_win.py").read_text(encoding="utf-8")
    assert "SetWindowRgn" in blur and "CreatePolygonRgn" in blur
    assert "restype = ctypes.c_void_p" in blur, (
        "a region handle is pointer-sized; untyped ctypes truncates it")


def test_the_overlay_reclaims_topmost_every_time_it_is_shown():
    """The topmost band is ordered by whoever claimed it last.

    Set once at realize, the pill ended up behind the taskbar and behind any
    other topmost window that appeared later.
    """
    source = _hud_app_source()
    assert "_raise_win32" in source, "the overlay needs an explicit raise"
    body = source.split("def _reposition(", 1)[1].split("\n    def ", 1)[0]
    assert "_raise_win32()" in body, (
        "topmost must be re-asserted when the pill is shown, not only when "
        "the window is first realized")


def test_the_levels_are_read_off_the_render_thread():
    """Disk I/O between frames is disk I/O inside a frame.

    Reading the level file was an open, a seek and a read on the main loop,
    the same thread GTK draws on and the one with 16ms to do everything. Any
    stall on the filesystem showed up as a stutter in the waveform.

    It also fixes what the old GLib timer got wrong: that timer returned
    False when the daemon deleted the level file, removing the source for
    good, so every later recording on a parked overlay drew a waveform
    frozen where the last one ended. A thread that loops has no such edge.
    """
    source = _hud_app_source()
    assert "_levels_loop" in source and "_levels_thread" in source, (
        "the level file must be followed on its own thread")
    body = source.split("def _levels_loop(", 1)[1].split("\n    def ", 1)[0]
    assert "while not self._levels_done" in body, (
        "the reader must loop rather than being a one-shot, or a parked "
        "overlay stops following levels after the first recording")
    assert "_levels_lock" in source, (
        "targets are written by the reader and read by the render loop")
    frame = source.split("def _frame(", 1)[1].split("\n    def ", 1)[0]
    assert "_levels_lock" in frame, (
        "the render loop must take the lock to copy the levels out")


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
    assert process.written == ["show - /tmp/levels\n"]
    assert len(calls) == before + 1      # asked for the same live one
    assert process.poll() is None


def _settle():
    """Let prewarm's thread finish. It is short-lived by design."""
    for _ in range(100):
        time.sleep(0.005)
        if not any(t.name == "whisper-flow-hud-prewarm" and t.is_alive()
                   for t in threading.enumerate()):
            return


# ------------------------------------------- which screen it appears on
def test_the_active_windows_screen_goes_with_every_show(hud, monkeypatch):
    """A resident overlay outlives the recording that decided the screen.

    It picks its output once, at startup, and at startup there is no
    recording and so no window to point at - which pinned the pill to
    whichever monitor happened to be first for the whole session, wherever
    the user was actually typing. The point has to travel per recording,
    because only the environment of a per-recording process could carry it.
    """
    process = _running(hud, monkeypatch)
    hud.show(level_file="/tmp/levels", point=(2560, 400))
    assert process.written == ["show 2560,400 /tmp/levels\n"]


def test_a_second_recording_can_land_on_a_different_screen(hud, monkeypatch):
    process = _running(hud, monkeypatch)
    hud.show(level_file="/tmp/a", point=(100, 100))
    hud.show(level_file="/tmp/b", point=(3000, 100))
    assert process.written == ["show 100,100 /tmp/a\n", "show 3000,100 /tmp/b\n"]


def test_a_point_that_makes_no_sense_does_not_stop_the_overlay(hud, monkeypatch):
    """Placement is a hint. A bad one falls back; it never costs the pill."""
    process = _running(hud, monkeypatch)
    for bad in (None, (), ("left", "top"), (1,)):
        process.written.clear()
        hud.show(level_file="/tmp/levels", point=bad)
        assert process.written == ["show - /tmp/levels\n"]


def test_nothing_is_drawn_before_the_pill_has_been_placed():
    """GTK maps the window where it likes; we move it a frame later.

    The correction arrives on an idle callback, so the pill used to fade up
    at GTK's spot and then jump to ours. Read from the source, like the rest
    of the overlay's layout checks - it needs GTK 4 and a display to run.
    """
    source = _hud_app_source()
    assert "_placed" in source, (
        "the overlay needs to know whether it has been moved yet")

    frame = source.split("def _frame(", 1)[1].split("\n    def ", 1)[0]
    assert "_placed" in frame, (
        "_frame must not advance the fade before the window has been moved; "
        "those are the frames drawn at the wrong position")

    reposition = source.split("def _reposition(", 1)[1].split("\n    def ", 1)[0]
    assert "_fade_in_t0" in reposition, (
        "the fade has to start from the move, or it is already part-way "
        "through by the time the pill is where it belongs")


def test_the_hold_cannot_strand_the_pill_off_screen():
    """Every platform and path has to end with something visible.

    Only Windows places the window after mapping it, and only a window that
    is not already up gets a map event - so both of those must start placed
    or the pill waits for a correction that is never coming.
    """
    source = _hud_app_source()
    assert "self._placed = not IS_WINDOWS" in source, (
        "off Windows the compositor places the surface; nothing holds it")
    assert "self._placed = not IS_WINDOWS or self.get_mapped()" in source, (
        "a window already on screen gets no map event and no correction")
