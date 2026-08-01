"""Tests for the resident overlay.

Starting a process is the largest cost between pressing the hotkey and
seeing anything, so the overlay is kept alive and commanded down a pipe.
Measured here: 114-159ms to visible when spawned per recording, 6-8ms when
resident. What follows guards the lifecycle that makes that safe - above
all, that a resident overlay can never be stranded on screen.
"""

import sys
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
@pytest.mark.skipif(not __import__("os").environ.get("DISPLAY")
                    and sys.platform != "win32", reason="needs a display")
def test_the_overlay_hides_instead_of_exiting_between_recordings(tmp_path):
    import tkinter as tk

    from whisper_flow import hud_win

    try:
        window = hud_win.HudWindow("", resident=True)
    except tk.TclError as e:
        pytest.skip(f"no usable display: {e}")
    try:
        assert window._resident
        assert not window._visible          # built hidden

        levels = tmp_path / "a.levels"
        levels.write_bytes(b"\x00" * 16)
        window._begin(str(levels))
        assert window._visible
        assert window.level_file == str(levels)

        window._end()
        assert window._quitting              # fading out, not gone
        window._retire()
        assert not window._visible
        assert window.level_file == ""
        assert window._level_handle is None  # the file can now be deleted
        assert window.root.winfo_exists()    # process and window survive
    finally:
        try:
            window.root.destroy()
        except tk.TclError:
            pass


@pytest.mark.skipif(not __import__("os").environ.get("DISPLAY")
                    and sys.platform != "win32", reason="needs a display")
def test_a_second_recording_resets_the_waveform(tmp_path):
    import tkinter as tk

    from whisper_flow import hud_win

    try:
        window = hud_win.HudWindow("", resident=True)
    except tk.TclError as e:
        pytest.skip(f"no usable display: {e}")
    try:
        first = tmp_path / "a.levels"
        first.write_bytes(b"\x01" * 64)
        window._begin(str(first))
        window.level_pos = 16
        window.peak = 9999.0

        second = tmp_path / "b.levels"
        second.write_bytes(b"\x02" * 64)
        window._begin(str(second))
        # Otherwise the new recording starts mid-waveform at the old scale.
        assert window.level_pos == 0
        assert window.peak == hud_win.PEAK_FLOOR
    finally:
        try:
            window.root.destroy()
        except tk.TclError:
            pass
