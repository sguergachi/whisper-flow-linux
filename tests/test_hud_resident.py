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
    assert process.written == ["show - 0 /tmp/levels\n"]
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


def test_a_per_recording_overlay_is_told_about_the_stop_button(monkeypatch):
    """Off Windows the overlay is spawned per recording, so everything it
    needs to know must arrive in the environment it is born with."""
    monkeypatch.setattr(hud_module, "RESIDENT", False)
    hud = hud_module.HUD()
    monkeypatch.setattr(hud, "_hide_locked", Mock())
    seen = {}

    def fake_popen(*args, **kwargs):
        seen["env"] = kwargs.get("env", {})
        return FakeProcess()

    monkeypatch.setattr(hud_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(hud_module.tempfile, "mkstemp", lambda **k: (0, "/tmp/x.log"))
    monkeypatch.setattr(hud_module.os, "close", lambda fd: None)
    monkeypatch.setattr("builtins.open", lambda *a, **k: Mock())

    hud.show(level_file="/tmp/levels", stop_button=True)
    assert seen["env"].get("WHISPER_FLOW_HUD_STOP_BUTTON") == "1"

    seen.clear()
    hud.show(level_file="/tmp/levels")
    assert "WHISPER_FLOW_HUD_STOP_BUTTON" not in seen["env"]


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


def test_the_overlay_is_never_shown_anywhere_but_its_own_position():
    """Re-placing it after the map is a correction the user watches happen.

    GDK keeps its own record of where the surface is, and SetWindowPos
    behind its back does not update it - so every show put the pill at GDK's
    remembered spot and our correction arrived an idle callback later.
    Holding the fade until then was supposed to cover it and cannot: with
    acrylic the pill's whole silhouette is drawn by DWM across the window
    rectangle, not by us, so it is on screen the moment the window is mapped
    whatever our alpha says. It appeared in the corner and slid down.

    WM_WINDOWPOSCHANGING is the move while it is still a proposal, and a
    proposal can be rewritten. Then there is no wrong position to hide.
    """
    source = _hud_app_source()
    assert "WM_WINDOWPOSCHANGING" in source, (
        "nothing refuses GDK's placement, so the pill is shown at it first")
    realize = source.split("def _realize_win32(", 1)[1].split(
        "\n    def ", 1)[0]
    assert "_pin_position_win32" in realize, (
        "the pin must be installed at realize; the map that follows is the "
        "move it exists to overrule")
    assert realize.index("_pin_position_win32") < realize.index(
        "_apply_position"), (
        "pinning after the first position leaves that position unguarded")
    pin = source.split("def _pin_position_win32(", 1)[1].split(
        "\n    def ", 1)[0]
    assert "self._pin_proc = _WNDPROC(hook)" in pin, (
        "ctypes keeps no reference to a callback; a collected trampoline "
        "leaves the window's procedure pointing at freed memory")
    assert "SetWindowLongPtrW" in pin, (
        "a window procedure is pointer-sized, and the 32-bit call hands "
        "back a truncated one to chain to")


def test_the_pill_follows_the_window_across_screens_on_windows():
    """The point comes from GetWindowRect and is physical; GDK's is logical.

    Compared untranslated, a point on a second screen falls inside the
    first one's logical rectangle - so the pill went to the wrong monitor,
    which is the one failure the point exists to prevent.
    """
    source = _hud_app_source()
    pick = source.split("def _pick_monitor(", 1)[1].split("\n    def ", 1)[0]
    assert "_monitor_bounds" in pick, (
        "the point must be tested against bounds in its own units")
    bounds = source.split("def _monitor_bounds(", 1)[1].split(
        "\n    def ", 1)[0]
    assert "IS_WINDOWS" in bounds and "_monitor_scale" in bounds, (
        "on Windows the monitor's logical geometry has to be scaled to "
        "physical pixels before a physical point is tested against it")


def test_windows_knows_where_the_dictation_is_happening():
    """The overlay can only follow the window if the daemon tells it where.

    active_window_center went through kdotool, which does not exist on
    Windows, so it answered None on every recording and the overlay fell
    back to the first monitor GDK listed.
    """
    source = (Path(__file__).resolve().parents[1]
              / "src/whisper_flow/system.py").read_text(encoding="utf-8")
    body = source.split("def active_window_center(", 1)[1].split(
        "\n    def ", 1)[0]
    assert "IS_WINDOWS" in body and "window_center" in body, (
        "the Windows path must not fall through to kdotool, which is not "
        "there - it returns None and the pill picks a screen at random")
    win = (Path(__file__).resolve().parents[1]
           / "src/whisper_flow/system_win.py").read_text(encoding="utf-8")
    assert "def window_center(" in win and "GetWindowRect" in win


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
elif scenario == "stop_button":
    # The auto-transcribe modes grow a centred stop notch from the pill:
    # same piece of glass, not a full-width slab and not a floating tab.
    win.begin_show("", "", stop_button=True)
    assert win.stop_button
    assert win._window_height() > hud_app.HEIGHT
    mid = hud_app.WIDTH / 2
    assert win._in_stop_button(mid, hud_app.HEIGHT + 5)
    # The open corners beside the notch are not the button.
    assert not win._in_stop_button(10, hud_app.HEIGHT + 5)
    assert not win._in_stop_button(hud_app.WIDTH - 10, hud_app.HEIGHT + 5)
    assert not win._in_stop_button(10, 5)   # the pill itself is not the button
    # The notch is gone once there is nothing left to stop.
    win.begin_processing()
    assert win._window_height() == hud_app.HEIGHT
elif scenario == "stop_request":
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".levels")
    os.close(fd)
    win.level_file = path
    win._request_stop()
    assert os.path.exists(path + hud_app.STOP_SUFFIX)
    os.unlink(path)
    os.unlink(path + hud_app.STOP_SUFFIX)
elif scenario == "close_aborts":
    # Dismissing the HUD while recording must ask the daemon to stop —
    # otherwise the mic keeps listening with nothing on screen.
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".levels")
    os.close(fd)
    win.begin_show(path, "", stop_button=True)
    win.level_file = path
    # Tap the close X (right side of the pill, not the notch).
    win._drag_origin = (0, 0)
    win._drag_start_xy = (hud_app.WIDTH - 10, hud_app.HEIGHT / 2)
    win._on_drag_end(None, 0, 0)
    assert os.path.exists(path + hud_app.STOP_SUFFIX), (
        "closing the HUD must request a stop so transcription aborts")
    os.unlink(path)
    os.unlink(path + hud_app.STOP_SUFFIX)
print("OK")
"""


def _gtk_available() -> bool:
    """Everything the child imports, not just GTK.

    The overlay requires Gtk4LayerShell before Gdk on Linux - without it the
    pill becomes an ordinary decorated toplevel, so it is not optional and
    hud_app does not treat it as such. Probing for GTK alone said yes on a
    machine that has GTK 4 and no layer-shell typelib, which is every Ubuntu
    runner: Ubuntu packages gtk-layer-shell for GTK 3 and nothing for GTK 4.
    The tests then ran and failed on an import, rather than skipping.
    """
    import os
    import subprocess
    if sys.platform != "win32" and not os.environ.get("DISPLAY"):
        return False
    probe = "import gi; gi.require_version('Gtk', '4.0'); "
    if sys.platform != "win32":
        probe += "gi.require_version('Gtk4LayerShell', '1.0'); "
    probe += "from gi.repository import Gtk; Gtk.init()"
    result = subprocess.run([sys.executable, "-c", probe],
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


@pytest.mark.skipif(not _gtk_available(), reason="needs GTK4 and a display")
def test_the_stop_button_grows_the_pill_and_shrinks_back(tmp_path):
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", _GTK_CHILD,
         str(__import__("pathlib").Path(__file__).resolve().parents[1]),
         "stop_button"],
        capture_output=True, text=True, check=False, timeout=60)
    assert "OK" in result.stdout, result.stderr


@pytest.mark.skipif(not _gtk_available(), reason="needs GTK4 and a display")
def test_pressing_the_stop_button_asks_the_daemon(tmp_path):
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", _GTK_CHILD,
         str(__import__("pathlib").Path(__file__).resolve().parents[1]),
         "stop_request"],
        capture_output=True, text=True, check=False, timeout=60)
    assert "OK" in result.stdout, result.stderr


@pytest.mark.skipif(not _gtk_available(), reason="needs GTK4 and a display")
def test_closing_the_hud_aborts_transcription(tmp_path):
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", _GTK_CHILD,
         str(__import__("pathlib").Path(__file__).resolve().parents[1]),
         "close_aborts"],
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


def test_overlay_spawn_hides_the_console_on_windows(hud, monkeypatch):
    """Source runs launch python.exe; without CREATE_NO_WINDOW it flashes."""
    monkeypatch.setattr(hud_module, "IS_WINDOWS", True)
    seen = {}

    def fake_popen(*args, **kwargs):
        seen.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(hud_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(hud_module.tempfile, "mkstemp",
                        lambda **k: (-1, "/tmp/hud.log"))
    monkeypatch.setattr(hud_module.os, "close", lambda fd: None)
    monkeypatch.setattr("builtins.open", lambda *a, **k: Mock())

    assert hud._resident_process() is not None
    flags = seen.get("creationflags", 0)
    assert flags & hud_module.CREATE_NO_WINDOW
    assert flags & hud_module.CREATE_NEW_PROCESS_GROUP


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
    assert process.written == ["show - 0 /tmp/levels\n"]
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
    assert process.written == ["show 2560,400 0 /tmp/levels\n"]


def test_a_second_recording_can_land_on_a_different_screen(hud, monkeypatch):
    process = _running(hud, monkeypatch)
    hud.show(level_file="/tmp/a", point=(100, 100))
    hud.show(level_file="/tmp/b", point=(3000, 100))
    assert process.written == ["show 100,100 0 /tmp/a\n", "show 3000,100 0 /tmp/b\n"]


def test_a_point_that_makes_no_sense_does_not_stop_the_overlay(hud, monkeypatch):
    """Placement is a hint. A bad one falls back; it never costs the pill."""
    process = _running(hud, monkeypatch)
    for bad in (None, (), ("left", "top"), (1,)):
        process.written.clear()
        hud.show(level_file="/tmp/levels", point=bad)
        assert process.written == ["show - 0 /tmp/levels\n"]


def test_auto_modes_ask_for_the_stop_button(hud, monkeypatch):
    """The single-press modes get the button; the held one does not."""
    process = _running(hud, monkeypatch)
    hud.show(level_file="/tmp/levels", stop_button=True)
    hud.show(level_file="/tmp/levels")
    assert process.written == [
        "show - 1 /tmp/levels\n",
        "show - 0 /tmp/levels\n",
    ]


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
    is not already up gets a map event - so anything else has to start
    placed, or the pill waits for a correction that is never coming. A
    pinned position is a third way out: the map cannot move the window off
    it, so there is nothing to wait for.

    Written as the conditions rather than as one line of source, because the
    ways out are a list and a list grows.
    """
    source = _hud_app_source()
    assert "self._placed = not IS_WINDOWS" in source, (
        "off Windows the compositor places the surface; nothing holds it")
    show = source.split("def begin_show(", 1)[1].split("\n    def ", 1)[0]
    assert "self._placed" in show, (
        "showing again must decide afresh whether a correction is coming")
    for way_out in ("not IS_WINDOWS", "self.get_mapped()",
                    "self._pin_proc is not None"):
        assert way_out in show, (
            f"{way_out} no longer clears the hold, so a pill shown that way "
            f"waits for a correction that is never coming")
