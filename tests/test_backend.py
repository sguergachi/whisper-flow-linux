"""Tests for the managed speech engine and the first-run setup decision."""

import sys
import types
from pathlib import Path

import pytest

from whisper_flow import backend as backend_module
from whisper_flow.backend import LocalBackend, _stage, recommended_model


class FakeConfig:
    def __init__(self, config_dir, model_name="ggml-small.en"):
        self.config_dir = config_dir
        self.model_name = model_name
        self.local_server_port = 18080


@pytest.fixture
def config(tmp_path):
    return FakeConfig(tmp_path)


@pytest.fixture
def local_backend(config):
    return LocalBackend(config)


def _install_engine(backend, config):
    """Pretend a downloaded engine and model exist."""
    exe = Path(config.config_dir) / "runtime" / backend._exe_name
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("")
    model = Path(config.config_dir) / "models" / f"{config.model_name}.bin"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("")


# --------------------------------------------------------------- model choice
def test_gpu_gets_the_large_model_and_cpu_does_not():
    assert recommended_model("cuda12") == "ggml-large-v3-turbo"
    assert recommended_model("cuda11") == "ggml-large-v3-turbo"
    # A large model on a CPU transcribes slower than speech, which is useless.
    assert recommended_model("cpu") == "ggml-small.en"


# ------------------------------------------------------------------- progress
def test_stage_binds_a_name_and_passes_the_fraction():
    seen = []
    _stage(lambda name, fraction: seen.append((name, fraction)), "model")(0.5)
    assert seen == [("model", 0.5)]


def test_stage_is_none_without_a_callback():
    assert _stage(None, "model") is None


def test_download_reports_progress_and_never_leaves_a_partial_file(
        tmp_path, monkeypatch):
    """A failed download must not leave something that looks installed."""
    class Response:
        headers = {"Content-Length": "3"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size):
            raise OSError("connection reset")

    monkeypatch.setattr(backend_module.urllib.request, "urlopen",
                        lambda *a, **k: Response())
    dest = tmp_path / "model.bin"
    with pytest.raises(OSError):
        backend_module._download("http://example.invalid/m", dest)
    assert not dest.exists()


# ---------------------------------------------------------------- setup gate
def test_setup_is_needed_when_nothing_is_installed(local_backend):
    assert local_backend.setup_reason() == "missing"


def test_no_setup_prompt_when_installed_without_a_gpu(
        local_backend, config, monkeypatch):
    _install_engine(local_backend, config)
    monkeypatch.setattr(backend_module, "detect_accelerator", lambda: "cpu")
    assert local_backend.setup_reason() is None


def test_gpu_upgrade_is_offered_once_then_never_again(
        local_backend, config, monkeypatch):
    """Downloading 1.6GB unasked is not on; nagging every sign-in is not either."""
    monkeypatch.setattr(sys, "platform", "win32")   # before _exe_name is read
    bundled = Path(config.config_dir) / "bundled"
    (bundled / "models").mkdir(parents=True)
    (bundled / "models" / "ggml-small.en.bin").write_text("")
    (bundled / "engine").mkdir()
    (bundled / "engine" / local_backend._exe_name).write_text("")
    monkeypatch.setattr(backend_module, "bundled_dir", lambda: bundled)
    monkeypatch.setattr(backend_module, "detect_accelerator", lambda: "cuda12")

    assert local_backend.setup_reason() == "gpu"
    local_backend.mark_setup_seen()
    assert local_backend.setup_reason() is None


def test_downloaded_engine_wins_over_the_bundled_one(
        local_backend, config, monkeypatch):
    """The downloaded engine only exists because it is the better one."""
    bundled = Path(config.config_dir) / "bundled"
    (bundled / "engine").mkdir(parents=True)
    (bundled / "engine" / local_backend._exe_name).write_text("bundled")
    monkeypatch.setattr(backend_module, "bundled_dir", lambda: bundled)
    assert local_backend.server_exe.parent == bundled / "engine"

    _install_engine(local_backend, config)
    assert local_backend.server_exe.parent == Path(config.config_dir) / "runtime"


def test_a_downloaded_model_beats_the_bundled_one_without_a_config_reload(
        local_backend, config, monkeypatch):
    """pydantic resolves .env once at import, so the choice cannot live there."""
    bundled = Path(config.config_dir) / "bundled"
    (bundled / "models").mkdir(parents=True)
    (bundled / "models" / "ggml-small.en.bin").write_text("")
    monkeypatch.setattr(backend_module, "bundled_dir", lambda: bundled)
    assert local_backend.working_model() == "ggml-small.en"

    downloaded = Path(config.config_dir) / "models"
    downloaded.mkdir(parents=True)
    (downloaded / "ggml-large-v3-turbo.bin").write_text("")
    # config.model_name still says small.en, and is deliberately ignored.
    assert local_backend.working_model() == "ggml-large-v3-turbo"


def test_the_configured_model_wins_among_several_downloads(local_backend, config):
    downloaded = Path(config.config_dir) / "models"
    downloaded.mkdir(parents=True)
    for name in ("ggml-large-v3-turbo", "ggml-base.en", "ggml-small.en"):
        (downloaded / f"{name}.bin").write_text("")
    config.model_name = "ggml-base.en"
    assert local_backend.working_model() == "ggml-base.en"


def test_install_is_a_no_op_off_windows(local_backend, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert local_backend.install("ggml-small.en") is False


def test_detect_accelerator_falls_back_to_cpu_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(backend_module.shutil, "which", lambda _: None)
    assert backend_module.detect_accelerator() == "cpu"


def test_old_drivers_get_the_cuda_11_build(monkeypatch):
    monkeypatch.setattr(backend_module.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        backend_module.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="470.86\n"),
    )
    assert backend_module.detect_accelerator() == "cuda11"
