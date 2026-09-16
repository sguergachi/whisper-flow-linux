"""Guards on the engine doctor.

The doctor exists because a machine where every whisper.cpp build dies with
0xC0000005 cannot be debugged from a support thread. What is tested here is
that it measures rather than guesses: probes every engine, keeps a heal only
when a probe proves it, restores what did not help, and always leaves a
verdict in the log - the log is the deliverable.
"""

import os
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_flow import engine_doctor as doctor_module  # noqa: E402
from whisper_flow.engine_doctor import EngineDoctor, arch_kernels  # noqa: E402
from whisper_flow.logging import clear_log, recent_log  # noqa: E402


class FakeConfig:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.model_name = "ggml-base.en-q8_0"
        self.local_server_port = 18099


class FakeBackend:
    _exe_name = "whisper-server.exe"

    def __init__(self, config_dir):
        self.config = FakeConfig(config_dir)
        self._exe = Path(config_dir) / "runtime" / self._exe_name
        self._exe.parent.mkdir(parents=True, exist_ok=True)
        self._exe.write_text("")
        for name in ("ggml-cpu-alderlake.dll", "ggml-cpu-haswell.dll",
                     "ggml-cpu-x64.dll"):
            (self._exe.parent / name).write_text("")
        models = Path(config_dir) / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "ggml-base.en-q8_0.bin").write_text("model")

    def model_path(self, name=None):
        name = name or self.config.model_name
        return self.config.config_dir / "models" / f"{name}.bin"


@pytest.fixture
def doctor(tmp_path, monkeypatch):
    doc = EngineDoctor(FakeBackend(tmp_path))
    # The facts gatherers shell out / read the registry; the probes are what
    # these tests are about.
    monkeypatch.setattr(doc, "_facts_machine", lambda: None)
    monkeypatch.setattr(doc, "_facts_crash_report", lambda: None)
    monkeypatch.setattr(doc, "_facts_gpu", lambda: None)
    monkeypatch.setattr(doc, "_facts_security", lambda: None)
    monkeypatch.setattr(doc, "_cli_probe", lambda *a, **k: None)
    clear_log()
    return doc


def _probe_decider(ready_for):
    """A stand-in _probe: ready when ready_for(label, extra_args) says so."""
    def fake(exe, model, label, extra_args=(), timeout=0.0):
        return {"ready": bool(ready_for(label, tuple(extra_args))),
                "exit": 0xC0000005, "seconds": 1.0}
    return fake


# ------------------------------------------------------------------- kernels
def test_arch_kernels_excludes_the_always_valid_baseline(tmp_path):
    for name in ("ggml-cpu-alderlake.dll", "ggml-cpu-x64.dll",
                 "ggml-cpu-haswell.dll"):
        (tmp_path / name).write_text("")
    names = [path.name for path in arch_kernels(tmp_path)]
    assert names == ["ggml-cpu-alderlake.dll", "ggml-cpu-haswell.dll"]


# --------------------------------------------------------------------- entry
def test_start_is_windows_only(doctor, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert doctor.start("cpu died", "cpu") is False
    assert "Windows-only" in recent_log(50)


def test_start_runs_at_most_once_per_engine_family(doctor, monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, target=None, args=(), **kwargs):
            self.args = args

        def start(self):
            started.append(self.args)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(doctor_module.threading, "Thread", FakeThread)

    assert doctor.start("cuda engine died", "cuda") is True
    assert doctor.start("cuda engine died again", "cuda") is False
    assert doctor.start("no-BLAS engine died too", "cpu") is True
    assert started == [("cuda engine died",), ("no-BLAS engine died too",)]


# --------------------------------------------------------------------- heals
def test_hiding_arch_kernels_is_kept_when_it_heals(doctor, monkeypatch):
    monkeypatch.setattr(doctor, "_probe",
                        _probe_decider(lambda label, extra:
                                       label.startswith("no-arch-kernels")))

    doctor.run("cpu died")

    runtime = doctor.backend._exe.parent
    assert not (runtime / "ggml-cpu-alderlake.dll").exists()
    assert (runtime / "ggml-cpu-alderlake.dll.off").exists()
    assert (runtime / "ggml-cpu-haswell.dll.off").exists()
    log = recent_log(200)
    assert "HEALED" in log
    assert "hid 2 per-arch CPU kernel" in log


def test_kernels_are_restored_when_hiding_does_not_help(doctor, monkeypatch):
    monkeypatch.setattr(doctor, "_probe", _probe_decider(lambda l, e: False))

    doctor.run("cpu died")

    runtime = doctor.backend._exe.parent
    assert (runtime / "ggml-cpu-alderlake.dll").exists()
    assert not (runtime / "ggml-cpu-alderlake.dll.off").exists()
    assert "restored" in recent_log(200)


def test_single_thread_heal_is_pinned_for_later_starts(doctor, monkeypatch):
    monkeypatch.setattr(doctor, "_probe",
                        _probe_decider(lambda label, extra:
                                       label.startswith("single-thread")))

    doctor.run("cpu died")

    pinned = doctor.config_dir / "runtime" / "engine-threads.txt"
    assert pinned.read_text(encoding="utf-8") == "1"
    assert "one thread" in recent_log(200)


def test_verdict_reports_what_was_found(doctor, monkeypatch):
    monkeypatch.setattr(doctor, "_probe", _probe_decider(lambda l, e: False))
    doctor._findings["no_crash_report"] = True
    doctor._findings["security_products"] = "CrowdStrike Falcon"

    doctor.run("cpu died")

    log = recent_log(300)
    assert "verdict" in log
    assert "no crash report was written" in log
    assert "CrowdStrike Falcon" in log


# --------------------------------------------------------------- diagnostics
def test_the_real_probe_spawns_and_reports_a_dead_engine(doctor):
    """_probe itself, not a stand-in: spawn, wait, report, clean up.

    The running interpreter plays the engine so the test works everywhere -
    the server arguments make it exit at once, which is the shape of a
    crashing engine.
    """
    result = doctor._probe(Path(sys.executable), Path("/tmp/none.bin"),
                           "fake", timeout=5.0)

    assert result["ready"] is False
    assert "fake" in recent_log(50)


def test_probe_wav_is_a_real_second_of_16k_mono():
    path = EngineDoctor._write_probe_wav()
    try:
        with wave.open(path) as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            assert wav.getnframes() == 16000
    finally:
        os.unlink(path)


def test_the_backend_thread_count_honours_the_pin(tmp_path, monkeypatch):
    from whisper_flow import backend as backend_module
    from whisper_flow.backend import LocalBackend

    local = LocalBackend(FakeConfig(tmp_path))
    pinned = tmp_path / "runtime" / "engine-threads.txt"
    pinned.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(backend_module, "usable_cores", lambda: 8)
    assert local._thread_count() == 8          # nothing pinned
    pinned.write_text("1", encoding="utf-8")
    assert local._thread_count() == 1
    pinned.write_text("99", encoding="utf-8")   # nonsense cannot oversubscribe
    assert local._thread_count() == 8
    pinned.write_text("banana", encoding="utf-8")
    assert local._thread_count() == 8


def test_the_backend_routes_families_to_one_doctor(tmp_path, monkeypatch):
    """The backend reuses one doctor and forwards each family once.

    The once-per-family guard lives in EngineDoctor (tested above); what
    matters here is that the crash path reaches it at all, with the right
    family, and does not build a new doctor per crash.
    """
    from whisper_flow import backend as backend_module

    calls, built = [], []

    class FakeDoctor:
        def __init__(self, backend):
            built.append(backend)

        def start(self, reason, family):
            calls.append((reason, family))
            return True

    monkeypatch.setattr(doctor_module, "EngineDoctor", FakeDoctor)
    local = backend_module.LocalBackend(FakeConfig(tmp_path))

    local._start_engine_doctor("cuda engine died", "cuda")
    local._start_engine_doctor("no-BLAS engine died too", "cpu")
    assert calls == [("cuda engine died", "cuda"),
                     ("no-BLAS engine died too", "cpu")]
    assert len(built) == 1
