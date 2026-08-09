"""The device commands must number microphones the way the recorder opens them."""

import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typer.testing import CliRunner

from whisper_flow import cli


def _flow(config_dir: Path) -> Mock:
    flow = Mock()
    flow.config.config_dir = config_dir
    flow.config.mic_device_index = None
    return flow


def _stub_pyaudio(monkeypatch, device_count: int) -> Mock:
    pa = Mock()
    pa.get_device_count.return_value = device_count
    pa.get_device_info_by_index.return_value = {"maxInputChannels": 2}
    monkeypatch.setitem(sys.modules, "pyaudio", Mock(PyAudio=lambda: pa))
    return pa


def test_set_device_stores_the_global_portaudio_index(monkeypatch, tmp_path):
    """The recorder passes the index straight to pa.open(), so the CLI must
    validate and store global indices - host-API-relative ones name a
    different device on any machine with more than one audio API."""
    pa = _stub_pyaudio(monkeypatch, device_count=5)
    monkeypatch.setattr(cli, "WhisperFlow", lambda config_dir: _flow(tmp_path))

    result = CliRunner().invoke(cli.app, ["set-device", "--device-index", "3"])

    assert result.exit_code == 0
    pa.get_device_info_by_index.assert_called_once_with(3)
    assert "WHISPER_FLOW_MIC_DEVICE_INDEX=3" in (tmp_path / ".env").read_text()


def test_set_device_rejects_an_out_of_range_index(monkeypatch, tmp_path):
    _stub_pyaudio(monkeypatch, device_count=2)
    monkeypatch.setattr(cli, "WhisperFlow", lambda config_dir: _flow(tmp_path))

    result = CliRunner().invoke(cli.app, ["set-device", "--device-index", "4"])

    assert result.exit_code == 1
    assert not (tmp_path / ".env").exists()


# ---------------------------------------------------------------- capture-test
def test_capture_test_forces_the_samples_library(monkeypatch, tmp_path):
    """The point of the command is the corpus: it must keep the clip even if
    the settings toggle is off, and it must not leave a server it started."""
    import threading

    flow = Mock()
    flow.config.config_dir = tmp_path
    flow.transcription_service.local_url = "http://127.0.0.1:9999"
    flow.run_voice_flow_push_to_talk_daemon.return_value = True

    monkeypatch.setattr(cli, "WhisperFlow", lambda config_dir: flow)
    monkeypatch.setattr(threading, "Timer", lambda t, fn: Mock())

    result = CliRunner().invoke(cli.app, ["capture-test", "--seconds", "3"])

    assert result.exit_code == 0
    assert flow.config.keep_all_captures is True
    assert flow.run_voice_flow_push_to_talk_daemon.called
    args, _ = flow.run_voice_flow_push_to_talk_daemon.call_args
    assert isinstance(args[1], threading.Event)


def test_capture_test_without_a_server_says_so(monkeypatch, tmp_path):
    """No daemon, no reachable URL, and nothing to start: tell the user."""
    import sys

    flow = Mock()
    flow.config.config_dir = tmp_path
    flow.config.manage_local_server = True
    flow.transcription_service.local_url = None

    backend = Mock()
    backend.working_model.return_value = None
    backend.start.return_value = None

    monkeypatch.setattr(cli, "WhisperFlow", lambda config_dir: flow)
    fake_backend = Mock(LocalBackend=lambda config: backend)
    monkeypatch.setitem(sys.modules, "whisper_flow.backend", fake_backend)

    result = CliRunner().invoke(cli.app, ["capture-test"])

    assert result.exit_code == 1
    assert "No whisper server" in result.output
    assert not flow.run_voice_flow_push_to_talk_daemon.called


def test_set_device_rejects_an_output_only_device(monkeypatch, tmp_path):
    pa = _stub_pyaudio(monkeypatch, device_count=3)
    pa.get_device_info_by_index.return_value = {"maxInputChannels": 0}
    monkeypatch.setattr(cli, "WhisperFlow", lambda config_dir: _flow(tmp_path))

    result = CliRunner().invoke(cli.app, ["set-device", "--device-index", "1"])

    assert result.exit_code == 1
    assert not (tmp_path / ".env").exists()


def test_version_reports_the_build_rather_than_a_literal():
    """`--version` printed a hardcoded 0.1.0 that nothing ever updated, so
    the one command whose whole job is to say which version this is had been
    wrong since 0.2. It reads the same source as the tray and the report."""
    import whisper_flow

    result = CliRunner().invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert whisper_flow.__version__ in result.output, result.output


def test_version_prefers_the_stamp_a_packaged_build_leaves(monkeypatch,
                                                           tmp_path):
    exe = tmp_path / "whisper-flow.exe"
    exe.write_bytes(b"")
    (tmp_path / "BUILD.txt").write_text("0.4.171", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(exe))

    result = CliRunner().invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert "0.4.171" in result.output, result.output
