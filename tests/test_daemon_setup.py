"""Tests for how the daemon offers and adopts the speech-model setup.

The rule these pin down: the app never downloads gigabytes without being
asked, and never asks twice about the same thing.
"""

import subprocess
import sys
import time
from unittest.mock import Mock, patch

import pytest

from whisper_flow.daemon import WhisperFlowDaemon


@pytest.fixture
def daemon(temp_config_dir):
    with (
        patch("whisper_flow.daemon.Config") as config_class,
        patch("whisper_flow.daemon.WhisperFlow"),
        patch("whisper_flow.daemon.HotkeyManager"),
        patch("whisper_flow.daemon.HUD"),
        patch("whisper_flow.daemon.LocalBackend") as backend_class,
    ):
        config = Mock()
        config.hotkey_transcribe = "ctrl+cmd"
        config.hotkey_auto_transcribe = "ctrl+cmd+space"
        config.hotkey_command = "ctrl+cmd+alt"
        config.manage_local_server = True
        config.local_whisper_url = ""
        config_class.return_value = config
        backend_class.return_value = Mock()
        yield WhisperFlowDaemon(temp_config_dir)


# ------------------------------------------------------------ launching it
def test_no_settings_window_on_a_headless_machine(daemon, monkeypatch):
    """A daemon on a server has no screen to put this on."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert daemon._open_settings_window() is False


def test_the_window_opens_on_a_linux_desktop(daemon, monkeypatch):
    """Linux laptops open the same window; install() works there now."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        assert daemon._open_settings_window() is True
    assert popen.call_args[0][0][1:] == ["-m", "whisper_flow.settings_gtk"]


def test_the_frozen_build_relaunches_itself_with_settings(daemon, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\app\whisper-flow.exe")
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        assert daemon._open_settings_window() is True
    assert popen.call_args[0][0] == [r"C:\app\whisper-flow.exe", "--settings"]


def test_a_second_request_does_not_stack_windows(daemon, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    running = Mock()
    running.poll.return_value = None            # still open
    daemon._setup_process = running
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        assert daemon._open_settings_window() is True
        popen.assert_not_called()


def test_a_closed_window_can_be_reopened(daemon, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    finished = Mock()
    finished.poll.return_value = 0              # already exited
    daemon._setup_process = finished
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        assert daemon._open_settings_window() is True
        popen.assert_called_once()


def test_a_launch_failure_reports_false_so_the_caller_can_fall_back(
        daemon, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with patch("whisper_flow.daemon.subprocess.Popen",
               side_effect=OSError("no such file")):
        assert daemon._open_settings_window() is False


# --------------------------------------------------------- when it is shown
def test_a_missing_model_opens_settings_instead_of_a_notification(daemon):
    daemon.backend.working_model.return_value = None
    with patch.object(daemon, "_open_settings_window", return_value=True) as opened:
        with patch.object(daemon, "notify") as notify:
            daemon._start_managed_backend()
    opened.assert_called_once()
    notify.assert_not_called()
    daemon.backend.start.assert_not_called()


def test_a_missing_model_still_notifies_where_there_is_no_window(daemon):
    daemon.backend.working_model.return_value = None
    with patch.object(daemon, "_open_settings_window", return_value=False):
        with patch.object(daemon, "notify") as notify:
            daemon._start_managed_backend()
    assert "speech model" in notify.call_args[0][0].lower()


def test_a_gpu_offer_is_made_only_after_the_app_already_works(daemon):
    """And as a notification: the app works, so it must not seize the screen."""
    daemon.backend.working_model.return_value = "ggml-small.en"
    daemon.backend.start.return_value = "http://127.0.0.1:18080"
    daemon.backend.setup_reason.return_value = "gpu"
    with patch.object(daemon, "_use_backend_url") as used:
        with patch.object(daemon, "_open_settings_window") as opened:
            with patch.object(daemon, "notify") as notify:
                daemon._start_managed_backend()
    used.assert_called_once()                   # working first
    opened.assert_not_called()                  # never an uninvited window
    assert "gpu" in notify.call_args[0][0].lower()


def test_a_dead_server_is_not_followed_by_a_gpu_offer(daemon):
    """Nothing works yet, so an upsell would be the wrong conversation."""
    daemon.backend.working_model.return_value = "ggml-small.en"
    daemon.backend.start.return_value = None
    daemon.backend.setup_reason.return_value = "gpu"
    with patch.object(daemon, "_open_settings_window") as opened:
        with patch.object(daemon, "notify") as notify:
            daemon._start_managed_backend()
    opened.assert_not_called()
    # Startup failure is now surfaced (was silent "No whisper server
    # configured" only on first dictation); the GPU upsell is still suppressed.
    assert notify.call_count == 1
    assert "failed to start" in notify.call_args[0][0].lower()
    assert "gpu" not in notify.call_args[0][0].lower()


def test_nothing_is_offered_when_the_model_is_already_right(daemon):
    daemon.backend.working_model.return_value = "ggml-large-v3-turbo"
    daemon.backend.start.return_value = "http://127.0.0.1:18080"
    daemon.backend.setup_reason.return_value = None
    with patch.object(daemon, "_use_backend_url"):
        with patch.object(daemon, "_open_settings_window") as opened:
            with patch.object(daemon, "notify") as notify:
                daemon._start_managed_backend()
    opened.assert_not_called()
    notify.assert_not_called()


def test_an_external_server_is_left_alone(daemon):
    daemon.config.local_whisper_url = "http://192.168.1.5:8080"
    daemon._start_managed_backend()
    daemon.backend.start.assert_not_called()


# ------------------------------------------------------- adopting the result
def test_a_new_model_is_picked_up_when_the_window_closes(daemon):
    daemon._setup_process = Mock(returncode=0)
    daemon._backend_model = "ggml-small.en"
    daemon.backend.working_model.return_value = "ggml-large-v3-turbo"
    daemon.backend.start.return_value = "http://127.0.0.1:18080"
    with patch.object(daemon, "_use_backend_url") as used:
        with patch.object(daemon, "notify"):
            daemon._after_tool_window()
    daemon.backend.stop.assert_called_once()
    used.assert_called_once_with("http://127.0.0.1:18080")
    assert daemon._backend_model == "ggml-large-v3-turbo"


def test_a_dismissed_window_does_not_restart_a_working_server(daemon):
    daemon._setup_process = Mock(returncode=0)
    daemon._backend_model = "ggml-small.en"
    daemon._backend_engine = "cpu"
    daemon.backend.working_model.return_value = "ggml-small.en"
    daemon.backend.installed_engine.return_value = "cpu"
    with patch.object(daemon, "_use_backend_url") as used:
        daemon._after_tool_window()
    daemon.backend.stop.assert_not_called()
    used.assert_not_called()


def test_a_new_engine_restarts_the_server_even_on_the_same_model(daemon):
    """Installing the GPU engine leaves the model exactly as it was.

    Restarting only on a changed model meant the server carried on running
    against the CPU engine it was started with, so the upgrade appeared to do
    nothing whatever until the next sign-in.
    """
    daemon._setup_process = Mock(returncode=0)
    daemon._backend_model = "ggml-large-v3-turbo"
    daemon._backend_engine = "cpu"
    daemon.backend.working_model.return_value = "ggml-large-v3-turbo"
    daemon.backend.installed_engine.return_value = "cuda12"
    daemon.backend.start.return_value = "http://127.0.0.1:18080"
    with patch.object(daemon, "_use_backend_url") as used:
        with patch.object(daemon, "notify"):
            daemon._after_tool_window()
    daemon.backend.stop.assert_called_once()
    used.assert_called_once_with("http://127.0.0.1:18080")
    assert daemon._backend_engine == "cuda12"


# ------------------------------------------- what the window has to say
def test_the_windows_output_reaches_the_log(daemon):
    """The frozen build is windowed, so the child has no console of its own.

    Everything it prints - the startup timings, a traceback, the reason no
    window appeared - went to a handle nobody was reading. It has to land in
    the log the tray menu offers or it may as well not be written.
    """
    import io

    from whisper_flow import logging as log_module

    process = Mock()
    process.stdout = io.StringIO(
        "[SETTINGS] +   12ms gtk imported\n"
        "\n"                                   # blank lines are not events
        "Traceback (most recent call last):\n")

    log_module.clear_log()
    daemon._drain_tool_output(process)
    recorded = log_module.recent_log()

    assert "+   12ms gtk imported" in recorded
    assert "[TOOL] Traceback (most recent call last):" in recorded


def test_a_window_with_no_pipe_is_not_an_error(daemon):
    """Popen can be mocked, or the process replaced; neither may raise here."""
    daemon._drain_tool_output(Mock(stdout=None))


def test_the_output_is_drained_before_anything_waits_on_the_process(daemon):
    """A full pipe blocks the child, so wait() before a read is a deadlock."""
    order = []
    process = Mock(returncode=0)
    process.stdout = None
    process.wait.side_effect = lambda *a, **k: order.append("wait")

    with patch.object(WhisperFlowDaemon, "_drain_tool_output",
                      side_effect=lambda p: order.append("drain")):
        daemon.backend.working_model.return_value = None
        daemon._after_tool_window(process)

    assert order == ["drain", "wait"]


def test_the_click_is_stamped_for_the_window_to_measure_from(daemon, monkeypatch):
    """Process start is most of the wait and is invisible from inside."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        assert daemon._open_settings_window() is True

    stamped = popen.call_args.kwargs["env"]["WHISPER_FLOW_TOOL_T0"]
    assert float(stamped) > 0


# ------------------------------------------------- the window built in advance
#
# Measured on Windows, on the shipped build: 3.1s from the click to a window,
# and 15.5s the first time after a boot, when the GTK runtime is still coming
# off the disk. Almost none of that is this app's own code and none of it
# depends on anything the user does, so it happens at login instead and the
# click gets a window that is already built.


class WaitingWindow:
    """A settings process built ahead of the click, with the pipe to it."""

    def __init__(self):
        self.stdin = Mock()
        self.written = []
        self.stdin.write.side_effect = self.written.append
        self.stdout = None
        self.returncode = None
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        self.returncode = self.returncode or 0
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def _waiting(daemon, window=None):
    window = window or WaitingWindow()
    daemon._setup_process = window
    daemon._setup_waiting = True
    daemon._setup_shown = False
    return window


def test_prewarming_builds_a_window_nobody_has_asked_for_yet(daemon, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        assert daemon._start_tool_window(
            "--settings", "whisper_flow.settings_gtk", resident=True) is True

    assert popen.call_args.kwargs["env"]["WHISPER_FLOW_SETTINGS_RESIDENT"] == "1"
    # Without a pipe there is no way to tell it the click has happened.
    assert popen.call_args.kwargs["stdin"] is not None
    assert daemon._setup_waiting is True
    assert daemon._setup_shown is False
    # Source runs launch python.exe (console); hide its prompt at login.
    expected = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    assert popen.call_args.kwargs.get("creationflags") == expected


def test_the_click_shows_the_waiting_window_rather_than_starting_a_process(
        daemon, monkeypatch):
    """The whole point: a map, not a process start and a GTK import."""
    monkeypatch.setattr(sys, "platform", "win32")
    window = _waiting(daemon)
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        assert daemon._open_settings_window() is True
        popen.assert_not_called()

    assert window.written == ["show\n"]
    assert daemon._setup_waiting is False
    assert daemon._setup_shown is True


def test_a_second_click_raises_the_window_rather_than_doing_nothing(
        daemon, monkeypatch):
    """A window buried behind other windows read as a click that missed."""
    monkeypatch.setattr(sys, "platform", "win32")
    window = _waiting(daemon)
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        daemon._open_settings_window()
        daemon._open_settings_window()
        popen.assert_not_called()       # and still only ever one process
    assert window.written == ["show\n", "show\n"]


def test_a_window_that_cannot_be_told_is_replaced_by_one_that_can(
        daemon, monkeypatch):
    """It died between the poll and the write; the user still gets a window."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    window = _waiting(daemon)
    window.stdin.write.side_effect = OSError("the pipe is closed")

    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        assert daemon._open_settings_window() is True
        popen.assert_called_once()
    # And the replacement is one that shows itself, not another that waits.
    assert "WHISPER_FLOW_SETTINGS_RESIDENT" not in popen.call_args.kwargs["env"]
    # The broken process must be killed or two windows stack on screen.
    assert window.killed is True


def test_a_dead_pipe_kills_the_old_process_before_starting_another(
        daemon, monkeypatch):
    """Regression: second Settings click left the first process running."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    window = _waiting(daemon)
    window.returncode = None  # still "running" as far as poll() is concerned
    window.stdin.write.side_effect = OSError("broken pipe")

    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        daemon._open_settings_window()

    assert window.killed is True
    popen.assert_called_once()


def test_a_window_lost_before_anyone_wanted_it_is_not_announced(daemon):
    """Nobody asked for it, so a notification would explain nothing."""
    process = Mock(returncode=3221225477)
    process.stdout = None
    daemon._setup_shown = False
    daemon.backend.working_model.return_value = None
    with patch.object(daemon, "notify") as notify:
        with patch.object(daemon, "prewarm_settings"):
            daemon._after_tool_window(process)
    notify.assert_not_called()


def test_a_window_that_dies_after_the_click_is_still_announced(daemon):
    process = Mock(returncode=3221225477)
    process.stdout = None
    daemon._setup_shown = True
    daemon.backend.working_model.return_value = None
    with patch.object(daemon, "notify") as notify:
        with patch.object(daemon, "prewarm_settings"):
            daemon._after_tool_window(process)
    assert "could not open" in notify.call_args[0][0]


def test_closing_the_window_builds_the_next_one(daemon):
    """It is a process and it exits, so only the first click would be fast."""
    process = Mock(returncode=0)
    process.stdout = None
    daemon.is_running = True
    daemon._setup_shown = True
    daemon.backend.working_model.return_value = None
    with patch.object(daemon, "prewarm_settings") as prewarm:
        daemon._after_tool_window(process)
    prewarm.assert_called_once()


def test_a_window_that_cannot_start_is_not_started_again_and_again(daemon):
    """Otherwise a machine that cannot open it spawns a process a second."""
    process = Mock(returncode=1)
    process.stdout = None
    daemon.is_running = True
    daemon._setup_shown = False
    daemon.backend.working_model.return_value = None
    with patch.object(daemon, "notify"):
        with patch.object(daemon, "prewarm_settings") as prewarm:
            daemon._after_tool_window(process)
    prewarm.assert_not_called()


def test_one_that_leaves_at_once_without_being_seen_is_not_replaced(daemon):
    """A clean exit nobody asked for is the same loop by a quieter route."""
    process = Mock(returncode=0)
    process.stdout = None
    daemon.is_running = True
    daemon._setup_shown = False
    daemon._setup_started = time.time()      # started a moment ago
    daemon.backend.working_model.return_value = None
    with patch.object(daemon, "prewarm_settings") as prewarm:
        daemon._after_tool_window(process)
    prewarm.assert_not_called()


def test_the_daemon_takes_the_waiting_window_with_it(daemon):
    """An invisible process with no tray icon must not outlive the app."""
    window = _waiting(daemon)
    daemon.shutdown_settings()
    window.stdin.close.assert_called_once()
    # Restart path must not wait on or kill settings - Save still owns it.
    assert window.wait_calls == 0
    assert not window.killed
    assert daemon._setup_process is None


def test_tray_exit_tells_settings_to_quit(daemon):
    """A visible settings window must not outlive tray Exit.

    EOF alone keeps an open window (Save restarts the daemon from it).
    Exit is intentional app shutdown, so the child gets an explicit quit.
    """
    window = _waiting(daemon)
    daemon.shutdown_settings(quit=True)
    assert any("quit" in str(chunk) for chunk in window.written), window.written
    window.stdin.close.assert_called_once()
    assert window.wait_calls == 1
    assert daemon._setup_process is None


def test_a_headless_machine_builds_nothing_in_advance(daemon, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with patch.object(daemon, "_start_tool_window") as start:
        daemon.prewarm_settings()
    start.assert_not_called()
