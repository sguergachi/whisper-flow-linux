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
        self.cli_mode_enabled = False
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

    def cli_mode(self):
        return self.cli_mode_enabled

    def set_cli_mode(self, enabled):
        self.cli_mode_enabled = bool(enabled)


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


def test_cli_mode_turns_on_when_only_the_cli_can_decode(doctor, monkeypatch):
    """No server survives, whisper-cli does: that is a working configuration.

    The machine in the 0.4.356 report is killed at startup by Cortex XDR for
    every server build while whisper-cli decodes fine. Refusing to switch
    would leave it permanently unable to transcribe for a fixable reason.
    """
    cli = doctor.backend._exe.parent / "whisper-cli.exe"
    cli.write_text("")
    monkeypatch.setattr(doctor, "_probe", _probe_decider(lambda l, e: False))
    monkeypatch.setattr(doctor, "_cli_probe", lambda engines, model: cli)

    doctor.run("no-BLAS engine died too")

    assert doctor.backend.cli_mode_enabled is True
    log = recent_log(300)
    assert "HEALED" in log
    assert "without a server" in log


def test_a_server_that_starts_clears_a_stale_cli_mode(doctor, monkeypatch):
    """CLI mode is a diagnosis, not a setting: retract it when wrong."""
    doctor.backend.cli_mode_enabled = True
    monkeypatch.setattr(doctor, "_probe",
                        _probe_decider(lambda l, e: l.startswith("baseline")))

    doctor.run("cpu died")

    assert doctor.backend.cli_mode_enabled is False


def test_verdict_reports_what_was_found(doctor, monkeypatch):
    monkeypatch.setattr(doctor, "_probe", _probe_decider(lambda l, e: False))
    doctor._findings["no_crash_report"] = True
    doctor._findings["security_products"] = "CrowdStrike Falcon"

    doctor.run("cpu died")

    log = recent_log(300)
    assert "verdict" in log
    assert "no crash report was written" in log
    assert "CrowdStrike Falcon" in log


# ------------------------------------------------- trigger isolation
def _help_run(monkeypatch, behavior):
    """Stand in subprocess.run for the --help probe only."""
    import subprocess as _sp

    def fake(cmd, **kwargs):
        assert cmd[-1] == "--help"
        assert kwargs.get("cwd") == str(Path(cmd[0]).parent)
        if behavior == "lives":
            return _ns(0)
        if behavior == "flagged":
            return _ns(0xC0000005)
        if behavior == "hung":
            raise _sp.TimeoutExpired(cmd, 20)
        raise OSError("exec format")

    monkeypatch.setattr(doctor_module.subprocess, "run", fake)


class _ns:
    def __init__(self, returncode):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


def test_help_probe_lives_means_serve_is_the_trigger(doctor, monkeypatch):
    """--help exits, serving dies: the listener/load is the trigger."""
    _help_run(monkeypatch, "lives")
    clear_log()

    doctor._probe_help(doctor.backend._exe)

    assert doctor._findings["help_lives"] == ["runtime"]
    assert "the image runs" in recent_log(50)


def test_help_probe_dies_means_the_image_is_flagged(doctor, monkeypatch):
    """Even --help dies: no binary from that directory will ever run."""
    _help_run(monkeypatch, "flagged")
    clear_log()

    doctor._probe_help(doctor.backend._exe)

    assert doctor._findings["help_dies"] == "runtime"
    assert "itself is flagged" in recent_log(50)


def test_help_probe_hung_and_unrunnable(doctor, monkeypatch):
    _help_run(monkeypatch, "hung")
    clear_log()
    assert doctor._probe_help(doctor.backend._exe) == {"help": "hung"}
    assert "hung or suspended" in recent_log(50)
    _help_run(monkeypatch, "missing")
    assert doctor._probe_help(doctor.backend._exe) == {"help": "unrunnable"}


def test_bundled_engine_is_probed_but_not_for_gpu_models(
        doctor, monkeypatch, tmp_path):
    """The install-dir copy is the safe-path experiment, except for large."""
    from whisper_flow import backend as backend_module

    bundle = tmp_path / "bundle"
    engine = bundle / "engine" / "whisper-server.exe"
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_text("bundled")
    monkeypatch.setattr(backend_module, "bundled_dir", lambda: bundle)
    clear_log()

    binaries = doctor._engine_binaries()
    assert engine in binaries
    # CUDA first, bundled last.
    assert binaries[-1] == engine

    doctor.backend.config.model_name = "ggml-large-v3-turbo"
    clear_log()
    assert engine not in doctor._engine_binaries()
    assert "CPU-only and cannot serve it" in recent_log(50)


def test_verdict_prints_an_it_exclusion_request(doctor):
    """The verdict hands IT the exact paths to allowlist."""
    doctor._findings["no_crash_report"] = True
    doctor._findings["security_products"] = "Cortex XDR"
    clear_log()

    doctor._verdict([doctor.backend._exe], doctor.backend.model_path())

    log = recent_log(50)
    assert "exclusion request for IT" in log
    assert "runtime\\whisper-server.exe" in log
    assert "%PROGRAMDATA%\\whisper-flow\\" in log


def test_defender_log_hit_is_quoted(doctor, monkeypatch):
    """A Defender ASR block naming whisper is evidence, not a guess."""
    import types

    monkeypatch.setattr(sys, "platform", "win32")
    event = types.SimpleNamespace(EventID=1121,
                                  StringInserts=("whisper-server.exe blocked",))
    fake_log = types.SimpleNamespace(
        OpenEventLog=lambda *a: "hand",
        CloseEventLog=lambda *a: None,
        ReadEventLog=lambda hand, flags, n: [event],
        EVENTLOG_BACKWARDS_READ=1,
        EVENTLOG_SEQUENTIAL_READ=2,
    )
    monkeypatch.setitem(sys.modules, "win32evtlog", fake_log)
    clear_log()

    doctor._facts_defender_log()

    assert "1121" in doctor._findings["defender_hit"]
    assert "Defender log names whisper" in recent_log(50)


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


def test_findings_survive_the_log_ring(doctor, monkeypatch):
    """The crash loop scrolled the first doctor run out of the tray report.

    Everything the doctor says is also appended to engine-doctor.log, which
    the report attaches separately.
    """
    from whisper_flow.engine_doctor import recent_findings

    monkeypatch.setattr(doctor, "_probe", _probe_decider(lambda l, e: False))
    doctor.run("cpu died")

    findings = recent_findings(80, doctor.config_dir)
    assert "starting engine diagnosis" in findings
    assert "verdict" in findings
    assert (doctor.config_dir / "engine-doctor.log").exists()


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
