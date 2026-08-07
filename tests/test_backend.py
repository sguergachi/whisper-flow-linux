"""Tests for the managed speech engine and the first-run setup decision."""

import sys
import types
from pathlib import Path

import pytest

from whisper_flow import backend as backend_module
from whisper_flow.backend import MODELS, LocalBackend, _stage, recommended_model


class FakeConfig:
    def __init__(self, config_dir, model_name="ggml-small.en-q8_0"):
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
def _machine(monkeypatch, cores, ram):
    monkeypatch.setattr(backend_module, "usable_cores", lambda: cores)
    monkeypatch.setattr(backend_module, "total_ram_gb", lambda: ram)


def test_a_cuda_machine_gets_the_large_model(monkeypatch):
    _machine(monkeypatch, 2, 4)             # ignored entirely on a GPU
    assert recommended_model("cuda12") == "ggml-large-v3-turbo"
    assert recommended_model("cuda11") == "ggml-large-v3-turbo"


def test_a_workstation_cpu_gets_small(monkeypatch):
    _machine(monkeypatch, 8, 32)
    assert recommended_model("cpu") == "ggml-small.en-q8_0"


def test_a_typical_laptop_gets_base(monkeypatch):
    """The common case: four real cores, 16GB. small.en would fall behind."""
    _machine(monkeypatch, 4, 16)
    assert recommended_model("cpu") == "ggml-base.en-q8_0"


def test_a_thin_laptop_gets_tiny(monkeypatch):
    _machine(monkeypatch, 2, 8)
    assert recommended_model("cpu") == "ggml-tiny.en-q8_0"


def test_an_8gb_laptop_is_not_asked_to_hold_small_en(monkeypatch):
    """350MB resident for as long as the daemon runs is a lot on 8GB."""
    _machine(monkeypatch, 7, 8)
    assert recommended_model("cpu") == "ggml-small.en-q8_0"
    _machine(monkeypatch, 7, 6)
    assert recommended_model("cpu") == "ggml-base.en-q8_0"


def test_a_memory_starved_machine_gets_tiny_however_many_cores(monkeypatch):
    _machine(monkeypatch, 16, 3)
    assert recommended_model("cpu") == "ggml-tiny.en-q8_0"


def test_the_bundled_model_runs_on_the_weakest_machine_we_target(monkeypatch):
    """Whatever CI bundles has to be right everywhere; base.en is that model."""
    from whisper_flow.config import Config

    bundled = Config().model_name
    assert bundled == "ggml-base.en-q8_0"
    # Never larger than what the tiers would pick for a modest laptop.
    _machine(monkeypatch, 4, 8)
    assert MODELS[bundled][0] <= MODELS[recommended_model("cpu")][0]


def test_unknown_memory_does_not_force_the_smallest_model(monkeypatch):
    """A platform that will not report RAM should not be punished for it."""
    _machine(monkeypatch, 8, 0.0)
    assert recommended_model("cpu") == "ggml-small.en-q8_0"


def test_every_recommendation_is_a_model_we_can_actually_fetch(monkeypatch):
    for cores, ram in ((16, 32), (8, 16), (4, 16), (2, 8), (1, 2), (32, 4)):
        _machine(monkeypatch, cores, ram)
        assert recommended_model("cpu") in backend_module.MODELS
    assert recommended_model("cuda12") in backend_module.MODELS


# ------------------------------------------------------------------- threads
def _cpus(monkeypatch, logical, physical):
    monkeypatch.setattr(backend_module.os, "cpu_count", lambda: logical)
    monkeypatch.delattr(backend_module.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(backend_module, "_physical_cores", lambda _: physical)


def test_threads_match_the_physical_cores(monkeypatch):
    """One thread per real core is where the encoder measured fastest."""
    _cpus(monkeypatch, 16, 8)
    assert backend_module.usable_cores() == 8

    _cpus(monkeypatch, 12, 6)
    assert backend_module.usable_cores() == 6

    _cpus(monkeypatch, 4, 2)
    assert backend_module.usable_cores() == 2

    _cpus(monkeypatch, 2, 1)
    assert backend_module.usable_cores() == 2        # never below 2


def test_threads_never_run_away_on_a_large_machine(monkeypatch):
    """Oversubscription does not degrade gently: 12 threads on 6 cores was 20x."""
    _cpus(monkeypatch, 128, 64)
    assert backend_module.usable_cores() == 8


def test_threads_respect_a_narrowed_affinity(monkeypatch):
    """A cgroup or taskset limit outranks however many cores the box has."""
    # raising=: sched_getaffinity does not exist off Linux to be patched.
    monkeypatch.setattr(backend_module.os, "sched_getaffinity",
                        lambda _: {0, 1, 2}, raising=False)
    monkeypatch.setattr(backend_module, "_physical_cores", lambda _: 8)
    assert backend_module.usable_cores() == 3


def test_smt_siblings_are_counted_out_where_the_topology_is_unreadable(monkeypatch):
    """Off Linux there is no topology to read, so halving is the assumption."""
    monkeypatch.setattr(backend_module.sys, "platform", "win32")
    assert backend_module._physical_cores(16) == 8
    assert backend_module._physical_cores(1) == 1    # never zero


def test_threads_survive_a_platform_that_will_not_say(monkeypatch):
    monkeypatch.setattr(backend_module.os, "cpu_count", lambda: None)
    monkeypatch.delattr(backend_module.os, "sched_getaffinity", raising=False)
    assert backend_module.usable_cores() == 4


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
    (bundled / "models" / "ggml-small.en-q8_0.bin").write_text("")
    monkeypatch.setattr(backend_module, "bundled_dir", lambda: bundled)
    assert local_backend.working_model() == "ggml-small.en-q8_0"

    downloaded = Path(config.config_dir) / "models"
    downloaded.mkdir(parents=True)
    (downloaded / "ggml-large-v3-turbo.bin").write_text("")
    # config.model_name still says small.en, and is deliberately ignored.
    assert local_backend.working_model() == "ggml-large-v3-turbo"


def test_the_configured_model_wins_among_several_downloads(local_backend, config):
    downloaded = Path(config.config_dir) / "models"
    downloaded.mkdir(parents=True)
    for name in ("ggml-large-v3-turbo", "ggml-base.en-q8_0", "ggml-small.en-q8_0"):
        (downloaded / f"{name}.bin").write_text("")
    config.model_name = "ggml-base.en-q8_0"
    assert local_backend.working_model() == "ggml-base.en-q8_0"


def test_linux_installs_the_cpu_tarball(local_backend, monkeypatch):
    """Laptops without CUDA are the point; Linux must not be a dead end."""
    monkeypatch.setattr(sys, "platform", "linux")
    assert backend_module._server_archive("cpu") == "whisper-bin-ubuntu-x64.tar.gz"
    # An NVIDIA card changes nothing: upstream ships no Linux CUDA asset.
    assert backend_module._server_archive("cuda12") == "whisper-bin-ubuntu-x64.tar.gz"


def test_windows_picks_the_engine_matching_the_driver(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert "cublas-12.4" in backend_module._server_archive("cuda12")
    assert "cublas-11.8" in backend_module._server_archive("cuda11")
    assert backend_module._server_archive("cpu") == "whisper-blas-bin-x64.zip"


def test_no_gpu_upgrade_is_offered_on_linux(local_backend, config, monkeypatch):
    """There is no Linux CUDA engine to move up to, so offering one would lie."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(backend_module, "detect_accelerator", lambda: "cuda12")
    assert local_backend.needs_gpu_upgrade() is False


def test_a_tarball_that_escapes_its_directory_is_refused(tmp_path):
    """Extraction must not be able to write outside the runtime directory."""
    import tarfile as tf

    archive = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("owned")
    with tf.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="../../escaped")

    into = tmp_path / "into"
    into.mkdir()
    with pytest.raises(Exception):
        backend_module._extract(archive, into)
    assert not (tmp_path.parent / "escaped").exists()


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


def test_nvidia_smi_is_spawned_without_a_console_window(monkeypatch):
    """The tray app is windowed; a console child flashes a black prompt.

    CREATE_NO_WINDOW is what stops that. Without it every Windows login
    shows a brief command-line window while the GPU is probed.
    """
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(returncode=0, stdout="560.01\n")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(backend_module.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(backend_module.subprocess, "run", fake_run)
    assert backend_module.detect_accelerator() == "cuda12"
    expected = getattr(backend_module.subprocess, "CREATE_NO_WINDOW", 0x08000000)
    assert seen.get("creationflags") == expected


# -------------------------------------------------------------- engine choice
def _bundled_cpu_engine(local_backend, config, monkeypatch):
    """A frozen build's payload: the CPU engine and a small model."""
    bundled = Path(config.config_dir) / "bundled"
    (bundled / "engine").mkdir(parents=True)
    (bundled / "engine" / local_backend._exe_name).write_text("cpu engine")
    (bundled / "models").mkdir(parents=True)
    (bundled / "models" / "ggml-base.en-q8_0.bin").write_text("")
    monkeypatch.setattr(backend_module, "bundled_dir", lambda: bundled)
    return bundled


def test_a_gpu_machine_fetches_the_cuda_engine_over_the_bundled_cpu_one(
        local_backend, config, monkeypatch):
    """The bug that made an NVIDIA desktop slower than a laptop.

    install() asked "is there an engine", a frozen build always bundles one,
    and so the cuBLAS engine was never fetched. Downloading large-v3-turbo
    from the settings window then ran a GPU-sized model on the CPU engine:
    sixteen to twenty-five seconds per utterance, every press during which was
    dropped as busy, and the transcript arriving wherever the focus had gone.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    _bundled_cpu_engine(local_backend, config, monkeypatch)
    monkeypatch.setattr(backend_module, "detect_accelerator", lambda: "cuda12")

    fetched = []

    def fake_download(url, dest, progress=None):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("")

    def fake_extract(archive, into):
        """Unpack what the real cuBLAS zip holds: the binary and cuBLAS."""
        (into / local_backend._exe_name).write_text("cuda engine")
        (into / "cublas64_12.dll").write_text("")

    monkeypatch.setattr(backend_module, "_download", fake_download)
    monkeypatch.setattr(backend_module, "_extract", fake_extract)

    assert local_backend.install("ggml-large-v3-turbo") is True
    assert any("cublas-12.4" in url for url in fetched), (
        f"the GPU engine was never fetched; only {fetched}")
    assert local_backend.installed_engine() == "cuda12"
    assert local_backend.engine_is_gpu()


def test_a_cpu_machine_keeps_the_bundled_engine(
        local_backend, config, monkeypatch):
    """The whole point of bundling one: no download on the common machine."""
    monkeypatch.setattr(sys, "platform", "win32")
    _bundled_cpu_engine(local_backend, config, monkeypatch)
    monkeypatch.setattr(backend_module, "detect_accelerator", lambda: "cpu")

    fetched = []
    monkeypatch.setattr(
        backend_module, "_download",
        lambda url, dest, progress=None: fetched.append(url))

    local_backend.install("ggml-base.en-q8_0")
    assert fetched == [], f"downloaded {fetched} with everything already here"


def test_an_engine_installed_before_the_marker_is_read_from_its_libraries(
        local_backend, config, monkeypatch):
    """Upgrades must not have to re-download to know what they already have."""
    monkeypatch.setattr(sys, "platform", "win32")
    runtime = Path(config.config_dir) / "runtime"
    runtime.mkdir(parents=True)
    (runtime / local_backend._exe_name).write_text("")
    assert local_backend.installed_engine() == "cpu"

    (runtime / "cublas64_12.dll").write_text("")
    assert local_backend.installed_engine() == "cuda12"


def test_the_gpu_offer_survives_downloading_a_model(
        local_backend, config, monkeypatch):
    """Fetching a model creates the runtime directory too.

    needs_gpu_upgrade() used to ask whether anything had been downloaded, so
    accepting a model from the settings window silenced the offer - at exactly
    the moment the machine most needed the engine to go with it.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    _bundled_cpu_engine(local_backend, config, monkeypatch)
    monkeypatch.setattr(backend_module, "detect_accelerator", lambda: "cuda12")
    assert local_backend.needs_gpu_upgrade() is True

    models = Path(config.config_dir) / "models"
    models.mkdir(parents=True)
    (models / "ggml-large-v3-turbo.bin").write_text("")
    assert local_backend.needs_gpu_upgrade() is True, (
        "a downloaded model was mistaken for a downloaded engine")

    runtime = Path(config.config_dir) / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / local_backend._exe_name).write_text("")
    local_backend._record_engine("cuda12")
    assert local_backend.needs_gpu_upgrade() is False


# ------------------------------------------------------- the model chosen
def test_a_saved_model_choice_is_read_from_disk_not_from_the_stale_config(
        local_backend, config):
    """Picking a model in the settings window has to actually change the model.

    The window writes .env and offers a restart; the daemon that comes back
    resolves .env at import, so working_model() reading config.model_name saw
    the value from before. It reported the old model as in use and went on
    running it - the "In Use" label never moved however many times the choice
    was saved and the daemon restarted.
    """
    models = Path(config.config_dir) / "models"
    models.mkdir(parents=True)
    for name in ("ggml-large-v3-turbo", "ggml-base.en-q8_0"):
        (models / f"{name}.bin").write_text("")
    assert local_backend.working_model() == "ggml-large-v3-turbo"

    env = Path(config.config_dir) / ".env"
    env.write_text("WHISPER_FLOW_MODEL_NAME=ggml-base.en-q8_0\n",
                   encoding="utf-8")
    assert local_backend.working_model() == "ggml-base.en-q8_0"


def test_a_saved_choice_of_a_bundled_model_is_honoured_too(
        local_backend, config, monkeypatch):
    """The chosen model need not be one that was downloaded."""
    bundled = _bundled_cpu_engine(local_backend, config, monkeypatch)
    assert (bundled / "models" / "ggml-base.en-q8_0.bin").exists()
    models = Path(config.config_dir) / "models"
    models.mkdir(parents=True)
    (models / "ggml-large-v3-turbo.bin").write_text("")

    (Path(config.config_dir) / ".env").write_text(
        "WHISPER_FLOW_MODEL_NAME=ggml-base.en-q8_0\n", encoding="utf-8")
    assert local_backend.working_model() == "ggml-base.en-q8_0"


def test_a_saved_choice_that_is_not_on_this_machine_is_ignored(
        local_backend, config):
    """A name with no file behind it must not stop transcription dead."""
    models = Path(config.config_dir) / "models"
    models.mkdir(parents=True)
    (models / "ggml-large-v3-turbo.bin").write_text("")
    (Path(config.config_dir) / ".env").write_text(
        "WHISPER_FLOW_MODEL_NAME=ggml-medium.en-q8_0\n", encoding="utf-8")
    assert local_backend.working_model() == "ggml-large-v3-turbo"


# ------------------------------------------------------------- accelerator
def test_the_accelerator_is_detected_once_and_remembered(monkeypatch):
    """nvidia-smi loads a driver before it answers; it is not a cheap call."""
    calls = []

    def counted(_name):
        calls.append(_name)
        return None

    monkeypatch.setattr(backend_module.shutil, "which", counted)
    assert backend_module.detect_accelerator() == "cpu"
    assert backend_module.detect_accelerator() == "cpu"
    assert len(calls) == 1


def test_the_accelerator_can_be_inherited_from_the_environment(monkeypatch):
    """So the windows the daemon launches do not each pay for the probe."""
    monkeypatch.setenv(backend_module.ACCELERATOR_ENV, "cuda12")
    monkeypatch.setattr(
        backend_module.shutil, "which",
        lambda _: pytest.fail("probed despite being told the answer"))
    assert backend_module.detect_accelerator() == "cuda12"


def test_a_nonsense_value_in_the_environment_is_probed_past(monkeypatch):
    monkeypatch.setenv(backend_module.ACCELERATOR_ENV, "quantum")
    monkeypatch.setattr(backend_module.shutil, "which", lambda _: None)
    assert backend_module.detect_accelerator() == "cpu"


# ------------------------------------------------------------------- version
def test_the_two_recorded_versions_agree():
    """pyproject names the installer; __version__ had drifted two releases."""
    import tomllib
    from pathlib import Path as P

    import whisper_flow

    root = P(__file__).resolve().parent.parent
    declared = tomllib.load(open(root / "pyproject.toml", "rb"))["project"]["version"]
    assert whisper_flow.__version__ == declared


# ------------------------------------------------------------ model inventory
def test_inventory_marks_installed_current_and_recommended(
        config, local_backend, monkeypatch):
    _install_engine(local_backend, config)   # writes ggml-small.en (the fixture model)
    monkeypatch.setattr(backend_module, "recommended_model",
                        lambda accelerator: "ggml-base.en-q8_0")
    monkeypatch.setattr(backend_module, "detect_accelerator", lambda: "cpu")

    rows = {row["name"]: row for row in local_backend.model_inventory()}

    assert set(rows) == set(MODELS)
    small = rows["ggml-small.en-q8_0"]
    assert small["installed"] and small["current"] and not small["recommended"]
    base = rows["ggml-base.en-q8_0"]
    assert not base["installed"] and not base["current"] and base["recommended"]


def test_a_model_in_use_that_is_not_in_the_catalogue_still_gets_a_row(
        config, local_backend, monkeypatch):
    """Otherwise the settings window cannot say what it is running.

    MODELS lists the q8_0 builds, but the model in use is whatever is on
    disk - a bundled file, or one put there by hand, under its own name.
    Without a row for it every radio in the window is blank while the app
    transcribes perfectly well, which is the one question that page exists
    to answer.
    """
    config.model_name = "ggml-small.en"          # no q8_0: not in MODELS
    _install_engine(local_backend, config)
    monkeypatch.setattr(backend_module, "recommended_model",
                        lambda accelerator: "ggml-base.en-q8_0")
    monkeypatch.setattr(backend_module, "detect_accelerator", lambda: "cpu")

    rows = local_backend.model_inventory()

    assert rows[0]["name"] == "ggml-small.en"    # first: it is what is true now
    assert rows[0]["installed"] and rows[0]["current"]
    assert [row["name"] for row in rows].count("ggml-small.en") == 1
    assert sum(row["current"] for row in rows) == 1
    # The catalogue is still offered, so a better model is a click away.
    assert set(MODELS) <= {row["name"] for row in rows}
