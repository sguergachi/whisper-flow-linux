"""Tests for how the daemon offers and adopts the speech-model setup.

The rule these pin down: the app never downloads gigabytes without being
asked, and never asks twice about the same thing.
"""

import sys
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
def test_no_setup_window_on_a_headless_machine(daemon, monkeypatch):
    """A daemon on a server has no screen to put this on."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert daemon._open_setup_window() is False


def test_the_window_opens_on_a_linux_desktop(daemon, monkeypatch):
    """Linux laptops need the same one button; install() works there now."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        assert daemon._open_setup_window() is True
    assert popen.call_args[0][0][1:] == ["-m", "whisper_flow.setup_gtk"]


def test_the_frozen_build_relaunches_itself_with_setup(daemon, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\app\whisper-flow.exe")
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        assert daemon._open_setup_window() is True
    assert popen.call_args[0][0] == [r"C:\app\whisper-flow.exe", "--setup"]


def test_a_second_request_does_not_stack_windows(daemon, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    running = Mock()
    running.poll.return_value = None            # still open
    daemon._setup_process = running
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        assert daemon._open_setup_window() is True
        popen.assert_not_called()


def test_a_closed_window_can_be_reopened(daemon, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    finished = Mock()
    finished.poll.return_value = 0              # already exited
    daemon._setup_process = finished
    with patch("whisper_flow.daemon.subprocess.Popen") as popen:
        popen.return_value.poll.return_value = None
        assert daemon._open_setup_window() is True
        popen.assert_called_once()


def test_a_launch_failure_reports_false_so_the_caller_can_fall_back(
        daemon, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with patch("whisper_flow.daemon.subprocess.Popen",
               side_effect=OSError("no such file")):
        assert daemon._open_setup_window() is False


# --------------------------------------------------------- when it is shown
def test_a_missing_model_opens_setup_instead_of_a_notification(daemon):
    daemon.backend.working_model.return_value = None
    with patch.object(daemon, "_open_setup_window", return_value=True) as opened:
        with patch.object(daemon, "notify") as notify:
            daemon._start_managed_backend()
    opened.assert_called_once()
    notify.assert_not_called()
    daemon.backend.start.assert_not_called()


def test_a_missing_model_still_notifies_where_there_is_no_window(daemon):
    daemon.backend.working_model.return_value = None
    with patch.object(daemon, "_open_setup_window", return_value=False):
        with patch.object(daemon, "notify") as notify:
            daemon._start_managed_backend()
    assert "speech model" in notify.call_args[0][0].lower()


def test_a_gpu_offer_is_made_only_after_the_app_already_works(daemon):
    daemon.backend.working_model.return_value = "ggml-small.en"
    daemon.backend.start.return_value = "http://127.0.0.1:18080"
    daemon.backend.setup_reason.return_value = "gpu"
    with patch.object(daemon, "_use_backend_url") as used:
        with patch.object(daemon, "_open_setup_window") as opened:
            daemon._start_managed_backend()
    used.assert_called_once()                   # working first
    opened.assert_called_once()                 # then offered


def test_a_dead_server_is_not_followed_by_a_gpu_offer(daemon):
    """Nothing works yet, so an upsell would be the wrong conversation."""
    daemon.backend.working_model.return_value = "ggml-small.en"
    daemon.backend.start.return_value = None
    daemon.backend.setup_reason.return_value = "gpu"
    with patch.object(daemon, "_open_setup_window") as opened:
        daemon._start_managed_backend()
    opened.assert_not_called()


def test_nothing_is_offered_when_the_setup_is_already_right(daemon):
    daemon.backend.working_model.return_value = "ggml-large-v3-turbo"
    daemon.backend.start.return_value = "http://127.0.0.1:18080"
    daemon.backend.setup_reason.return_value = None
    with patch.object(daemon, "_use_backend_url"):
        with patch.object(daemon, "_open_setup_window") as opened:
            daemon._start_managed_backend()
    opened.assert_not_called()


def test_an_external_server_is_left_alone(daemon):
    daemon.config.local_whisper_url = "http://192.168.1.5:8080"
    daemon._start_managed_backend()
    daemon.backend.start.assert_not_called()


# ------------------------------------------------------- adopting the result
def test_a_new_model_is_picked_up_when_the_window_closes(daemon):
    daemon._setup_process = Mock()
    daemon._backend_model = "ggml-small.en"
    daemon.backend.working_model.return_value = "ggml-large-v3-turbo"
    daemon.backend.start.return_value = "http://127.0.0.1:18080"
    with patch.object(daemon, "_use_backend_url") as used:
        with patch.object(daemon, "notify"):
            daemon._after_setup_window()
    daemon.backend.stop.assert_called_once()
    used.assert_called_once_with("http://127.0.0.1:18080")
    assert daemon._backend_model == "ggml-large-v3-turbo"


def test_a_dismissed_window_does_not_restart_a_working_server(daemon):
    daemon._setup_process = Mock()
    daemon._backend_model = "ggml-small.en"
    daemon.backend.working_model.return_value = "ggml-small.en"
    with patch.object(daemon, "_use_backend_url") as used:
        daemon._after_setup_window()
    daemon.backend.stop.assert_not_called()
    used.assert_not_called()
