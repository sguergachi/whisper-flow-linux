"""Exercise real process pipes and shutdown; no model needed for IPC tests."""
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from whisper_flow import backend as bm
from whisper_flow.backend import LocalBackend
from whisper_flow.resident import ResidentWorker


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "speech.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.array([0, -32768, 32767] * 400, dtype="<i2").tobytes())
    return path


@pytest.fixture
def config():
    return SimpleNamespace(language="en", beam_size=5, best_of=3,
                           no_speech_thold=0.25, suppress_nst=True,
                           fast_encoder=False)


def child(tmp_path, behavior="normal"):
    script = tmp_path / "fake-worker.py"
    script.write_text('''
import os, struct, sys, time
def exact(n):
    parts = []
    while n:
        data = sys.stdin.buffer.read(n)
        if not data: sys.exit(0)
        parts.append(data)
        n -= len(data)
    return b"".join(parts)
def reply(text):
    data = text.encode()
    sys.stdout.buffer.write(struct.pack("<4sII", b"WFR1", len(data), 0) + data)
    sys.stdout.buffer.flush()
reply("READY CPU")
count = 0
while True:
    h = struct.unpack("<4s7I2f", exact(40))
    language = exact(h[6]).decode()
    prompt = exact(h[7]).decode()
    pcm = exact(h[1] * 4)
    count += 1
''' + {
        "normal": '    reply(f"{os.getpid()}:{count}:{language}:{prompt}:{h[2]}:{h[3]}")\n',
        "blank": '    reply("")\n',
        "stall": '    time.sleep(60)\n',
        "exit": '    sys.exit(9)\n',
        "bad": '    sys.stdout.buffer.write(struct.pack("<4sII", b"WFR1", 999999999, 0))\n    sys.stdout.buffer.flush()\n',
    }[behavior])
    return ResidentWorker(Path(sys.executable), script, 4, False, tmp_path / "worker.log")


def test_resident_requests_reuse_process_and_forward_decode_options(tmp_path, audio, config):
    worker = child(tmp_path)
    try:
        pid = worker.process.pid
        assert worker.transcribe(audio, config, prompt="Echo", temperature=0.4) == f"{pid}:1:en:Echo:5:3"
        assert worker.transcribe(audio, config) == f"{pid}:2:en::5:3"
        assert worker.alive
    finally:
        worker.stop()
    assert worker.process.poll() is not None


def test_blank_audio_result_is_success_and_keeps_worker_alive(tmp_path, audio, config):
    worker = child(tmp_path, "blank")
    try:
        assert worker.transcribe(audio, config) == ""
        assert worker.alive
    finally:
        worker.stop()


@pytest.mark.parametrize("behavior", ["stall", "exit", "bad"])
def test_failed_exchange_reaps_worker_and_cannot_reuse_late_reply(tmp_path, audio, config, behavior):
    worker = child(tmp_path, behavior)
    with pytest.raises((RuntimeError, TimeoutError)):
        worker.transcribe(audio, config, timeout=0.2)
    assert not worker.alive
    assert worker.process.poll() is not None


def test_request_pcm_scale_language_and_prompt(audio, config):
    config.language = "auto"
    request = ResidentWorker._request(audio, config, "café", 0.4)
    header = struct.unpack("<4s7I2f", request[:40])
    assert header[:8] == (b"WFW1", 1200, 5, 3, 1, 0, 4, 5)
    assert header[8:] == pytest.approx((0.4, 0.25))
    assert request[40:49] == "autocafé".encode()
    pcm = np.frombuffer(request[49:], dtype="<f4")
    assert pcm[:3] == pytest.approx([0, -1, 32767 / 32768])


def test_worker_refuses_wrong_audio_format_before_sending(tmp_path, config):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(b"\0" * 16)
    with pytest.raises(ValueError, match="mono 16 kHz"):
        ResidentWorker._request(path, config)


@pytest.fixture
def windows_backend(tmp_path, monkeypatch):
    cfg = SimpleNamespace(config_dir=tmp_path, model_name="ggml-large-v3-turbo", local_server_port=8082)
    backend = LocalBackend(cfg)
    bundle = tmp_path / "bundle"
    exe = bundle / "worker" / "whisper-flow-worker.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    model = tmp_path / "models" / f"{cfg.model_name}.bin"
    model.parent.mkdir()
    model.touch()
    monkeypatch.setattr(bm.sys, "platform", "win32")
    monkeypatch.setattr(bm, "bundled_dir", lambda: bundle)
    monkeypatch.setattr(bm, "detect_accelerator", lambda: "cuda12")
    return backend


def test_windows_worker_ignores_legacy_cli_marker_and_skips_http(windows_backend, monkeypatch):
    backend = windows_backend
    backend.set_cli_mode(True)
    worker = Mock(alive=True)
    worker.process.poll.return_value = None
    factory = Mock(return_value=worker)
    monkeypatch.setattr("whisper_flow.resident.ResidentWorker", factory)
    monkeypatch.setattr(bm, "stop_managed_strays", Mock(side_effect=AssertionError("HTTP cleanup called")))
    assert backend.start_with_fallback("ggml-large-v3-turbo", allow_download=False) == "pipe://whisper-flow"
    assert backend.is_ready
    assert backend.cli_mode() is False
    assert factory.call_args.args[3] is True
    assert not backend._cli_mode_marker.exists()
    backend.stop()
    worker.stop.assert_called_once()


def test_failed_worker_is_not_relaunched_or_redownloaded(windows_backend, monkeypatch):
    factory = Mock(side_effect=RuntimeError("not permitted"))
    monkeypatch.setattr("whisper_flow.resident.ResidentWorker", factory)
    assert windows_backend.start("ggml-large-v3-turbo") is None
    assert windows_backend.cli_mode() is True
    assert windows_backend.start("ggml-large-v3-turbo") is None
    factory.assert_called_once()


def test_windows_worker_install_downloads_model_without_cuda(windows_backend, monkeypatch):
    model = windows_backend.model_path("ggml-large-v3-turbo")
    model.unlink()
    calls = []
    def download(url, dest, *args, **kwargs):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"model")
    monkeypatch.setattr(bm, "_download", download)
    assert windows_backend.install("ggml-large-v3-turbo") is True
    assert len(calls) == 1 and "ggml-large-v3-turbo.bin" in calls[0]
    assert windows_backend.engine_is_gpu()


def test_explicit_http_transport_keeps_windows_server(windows_backend):
    windows_backend.config.local_engine_transport = "server"
    assert windows_backend.resident_available() is False
    assert windows_backend.server_exe.name == "whisper-server.exe"
