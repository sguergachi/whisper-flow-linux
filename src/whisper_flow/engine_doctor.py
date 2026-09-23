"""Native-crash forensics and self-healing for the speech engine.

Windows, in practice. When whisper-server dies with 0xC0000005 and the engine
ladder has run out - the no-BLAS plain build crashing the same way, or the
CUDA build dying before it prints a single line - the app used to stop with
one message and no way forward. Everything needed to go further is measurable
from here: which CPU kernel the loader dispatched to, whether whisper can run
at all outside the server path, whether Windows wrote a crash report at all (a
process killed from outside leaves none), what the GPU and the MSVC runtime
look like, and whether an EDR product is installed on the machine.

So this measures them: one probe per hypothesis, every result logged with the
same shape, and whatever actually heals the engine is kept. Two heals are
real: hiding the per-architecture ggml CPU kernels (a broken variant dies
after the model loads, exactly where this class of crash lands) and pinning a
single thread. Both are written to disk so they survive restarts.

It runs at most twice per process - once for a CUDA failure, once for a CPU
failure - on its own daemon thread, after a native crash. Nothing here blocks
dictation, the tray, or an engine download.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .logging import log

# The baseline ggml CPU kernel every x86-64 machine can run. The loader scores
# the per-architecture kernels beside it and picks the best match for the CPU -
# and a variant that faults is a known way to get "model loads, then dies".
# x64 is never hidden, so hiding the rest always leaves a working loader path.
_BASELINE_KERNEL = "ggml-cpu-x64.dll"

# A server that is going to start binds its port once the model is in memory;
# one that is going to crash is usually gone in a few seconds. These are caps,
# not waits: readiness ends every probe early.
_PROBE_TIMEOUT = 45.0
_CUDA_PROBE_TIMEOUT = 90.0        # its DLL set alone is >1GB to map
_CLI_TIMEOUT = 180.0


def _free_port() -> int:
    """A localhost port nobody holds right now."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _port_open(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.25)
    try:
        sock.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def arch_kernels(engine_dir: Path) -> list[Path]:
    """The per-architecture ggml kernels beside an engine, baseline excluded."""
    try:
        found = sorted(Path(engine_dir).glob("ggml-cpu-*.dll"))
    except OSError:
        return []
    return [path for path in found if path.name.lower() != _BASELINE_KERNEL]


def recent_findings(limit: int = 80, config_dir: Path | None = None) -> str:
    """The doctor's durable findings, for the failure report.

    The tray report carries the last 200 log lines, and a crash loop is loud
    enough to push a diagnosis made early in a session out of that window -
    which is exactly what happened to the first report with the doctor in it.
    The file outlives the ring, so the report appends it explicitly.
    """
    try:
        path = Path(config_dir) / "engine-doctor.log"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


class EngineDoctor:
    """Measures why whisper-server dies, heals what it can, logs everything."""

    def __init__(self, backend):
        self.backend = backend
        self.config_dir = Path(backend.config.config_dir)
        self._log_path = self.config_dir / "engine-doctor.log"
        self._lock = threading.Lock()
        self._families_done: set[str] = set()
        self._findings: dict = {}
        self._trim_findings_file()

    # ------------------------------------------------------------- logging
    def _log(self, message: str) -> None:
        """Record a finding in the ring buffer and in a durable file.

        The ring holds 300 lines and one crash writes ten; a diagnosis from
        the start of the session is long gone by the time a person reads the
        report. The file is the artifact that survives.
        """
        log(message)
        try:
            from datetime import datetime

            with open(self._log_path, "a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}\n")
        except Exception:
            pass                    # a diagnosis is never worth failing over

    def _trim_findings_file(self, keep_lines: int = 500) -> None:
        """Bound the file's growth without losing the recent history."""
        try:
            if self._log_path.stat().st_size < 200_000:
                return
            lines = self._log_path.read_text(
                encoding="utf-8", errors="replace").splitlines()
            trimmed = "\n".join(lines[-keep_lines:]) + "\n"
            self._log_path.write_text(trimmed, encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------- entry
    def start(self, reason: str, family: str) -> bool:
        """Diagnose once per engine family (cuda, cpu). True if started."""
        with self._lock:
            if family in self._families_done:
                return False
            self._families_done.add(family)
        if sys.platform != "win32":
            self._log(f"[DOCTOR] engine diagnosis is Windows-only; skipping ({reason})")
            return False
        thread = threading.Thread(
            target=self._run_guarded, args=(reason,),
            daemon=True, name="whisper-flow-engine-doctor")
        thread.start()
        return True

    def _run_guarded(self, reason: str) -> None:
        try:
            self.run(reason)
        except Exception as e:
            self._log(f"[DOCTOR] engine diagnosis failed: {e}")

    # --------------------------------------------------------------- run
    def run(self, reason: str) -> None:
        self._log(f"[DOCTOR] starting engine diagnosis ({reason})")
        self._facts_machine()
        self._facts_crash_report()
        self._facts_gpu()
        self._facts_security()
        self._facts_defender_log()

        model = Path(self.backend.model_path())
        engines = self._engine_binaries()
        if not engines:
            self._log("[DOCTOR] no engine binary on disk to probe")
            return
        self._log(f"[DOCTOR] model {model.name} present={model.exists()}")
        if not model.exists():
            self._log("[DOCTOR] the model file is missing - that is the problem, "
                "not the engine")
            return

        # Trigger isolation first: --help loads no model and binds no
        # socket. If even that dies, the image itself is flagged (path /
        # reputation / userland-execution rule) and no engine from the same
        # directory will ever run. If --help lives but serving dies, the
        # trigger is the listener or the model load - which is what decides
        # between "run it from somewhere safer" and "stop binding a port".
        self._probe_help_all(engines)
        started = self._probe_engines(engines, model, label="baseline")
        if started:
            self._log(f"[DOCTOR] {started.name} starts on its own - the crash "
                      f"is intermittent or specific to the model that was "
                      f"loading")
            # A server starts here, so any CLI-mode marker from an earlier
            # diagnosis is stale; clear it before the app is told anything.
            self.backend.set_cli_mode(False)
            self._cli_probe(engines, model)
            return

        if self._heal_without_arch_kernels(engines, model):
            self._findings["healed"] = "hid per-arch CPU kernels"
            return
        if self._heal_single_thread(engines, model):
            self._findings["healed"] = "pinned one thread"
            return

        # Last rung: no server survives, but the CLI beside it may decode
        # without ever touching a socket. That is the difference between a
        # machine that cannot transcribe and one that transcribes slower -
        # the usual cause is endpoint security killing the listener.
        cli = self._cli_probe(engines, model)
        if cli is not None:
            self.backend.set_cli_mode(True)
            self._findings["healed"] = f"CLI transcription via {cli.parent.name}"
            self._log(
                "[DOCTOR] HEALED: every server binary is killed at startup "
                "here, but whisper-cli decodes with the same engine and "
                "model - switching to transcription without a server. "
                "Dictation works; there is no live preview and each "
                "utterance pays the model load.")
            return

        self._verdict(engines, model)

    def _engine_binaries(self) -> list[Path]:
        """Every engine this install could run, CUDA first, deduplicated.

        The bundled install-dir copy is last: same bytes as a fresh
        download, but admin-written rather than app-downloaded into a
        user-writable path - which is exactly the distinction reputation
        and userland-execution rules care about. When the AppData copies
        are killed at startup and this one binds, the fix is to run from
        here, not to download again. Skipped for GPU-only models, which
        the CPU-only bundle cannot serve (probing it would just time out
        mid-load and muddy the result).
        """
        from .backend import (_cuda_dir, _plain_dir, _runtime_dir,
                              GPU_MODELS, bundled_dir)

        candidates = [
            _cuda_dir(self.config_dir) / self.backend._exe_name,
            _plain_dir(self.config_dir) / self.backend._exe_name,
            _runtime_dir(self.config_dir) / self.backend._exe_name,
        ]
        try:
            model_name = Path(self.backend.model_path()).stem
        except Exception:
            model_name = ""
        bundled = bundled_dir()
        if bundled is not None and model_name not in GPU_MODELS:
            candidates.append(bundled / "engine" / self.backend._exe_name)
        elif bundled is not None:
            self._log(f"[DOCTOR] install-dir engine skipped for {model_name}: "
                      f"the bundle is CPU-only and cannot serve it")
        seen: set[str] = set()
        present: list[Path] = []
        for path in candidates:
            key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                present.append(path)
        return present

    # ------------------------------------------------------------- probes
    @staticmethod
    def _no_console() -> int:
        """No console window flashes for each probed engine."""
        from .backend import no_console_flags

        return no_console_flags()

    def _probe_engines(self, engines, model, label: str,
                       extra_args=()) -> Path | None:
        """Probe every engine; log each; return the first that bound a port."""
        started: Path | None = None
        for exe in engines:
            timeout = (_CUDA_PROBE_TIMEOUT
                       if "cuda" in str(exe.parent).lower()
                       else _PROBE_TIMEOUT)
            result = self._probe(exe, model, f"{label} {exe.parent.name}",
                                 extra_args=extra_args, timeout=timeout)
            if result.get("ready") and started is None:
                started = exe
        return started

    def _probe(self, exe: Path, model: Path, label: str,
               extra_args=(), timeout: float = _PROBE_TIMEOUT) -> dict:
        """Run the server against a free port and report what it did.

        Readiness is the port accepting a connection: it is the first moment
        the server is actually usable, and a model-load-then-die crash never
        reaches it.
        """
        port = _free_port()
        cmd = [str(exe), "-m", str(model), "-l", "en",
               "--host", "127.0.0.1", "--port", str(port), *extra_args]
        fd, out_path = tempfile.mkstemp(prefix="doctor-", suffix=".log")
        os.close(fd)
        started_at = time.monotonic()
        ready = False
        proc = None
        try:
            with open(out_path, "wb") as out:
                proc = subprocess.Popen(
                    cmd, stdout=out, stderr=subprocess.STDOUT,
                    cwd=str(Path(exe).parent),
                    creationflags=self._no_console(),
                )
            while time.monotonic() - started_at < timeout:
                if proc.poll() is not None:
                    break
                if _port_open(port):
                    ready = True
                    break
                time.sleep(0.2)
        except Exception as e:
            self._log(f"[DOCTOR] {label}: could not run ({e})")
            return {"ready": False}
        finally:
            if proc is not None:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                except Exception:
                    pass

        seconds = time.monotonic() - started_at
        code = proc.poll() if proc is not None else None
        output = self._tail(out_path)
        if ready:
            self._log(f"[DOCTOR] {label}: started and bound its port in {seconds:.1f}s")
        else:
            shown = "none" if code is None else f"0x{(code | 0) & 0xFFFFFFFF:08X}"
            self._log(f"[DOCTOR] {label}: died after {seconds:.1f}s (exit={shown})")
            if output:
                self._log(f"[DOCTOR] {label} last output: {output}")
        try:
            os.unlink(out_path)
        except OSError:
            pass
        return {"ready": ready, "exit": code, "seconds": seconds}

    def _probe_help_all(self, engines) -> None:
        """Run each engine's --help: no model, no socket, just the image.

        Cheap (seconds) and decisive next to the serve probes: it separates
        "this binary may not execute here" from "this binary may not serve
        here". Results feed the verdict; nothing is healed or hidden.
        """
        for exe in engines:
            self._probe_help(exe)

    def _probe_help(self, exe: Path) -> dict:
        """What --help does: exit code and how long the image survived."""
        label = f"help {exe.parent.name}"
        started_at = time.monotonic()
        try:
            result = subprocess.run(
                [str(exe), "--help"],
                capture_output=True, text=True, timeout=20,
                cwd=str(Path(exe).parent),
                creationflags=self._no_console(),
            )
        except subprocess.TimeoutExpired:
            self._log(f"[DOCTOR] {label}: still alive after 20s on --help "
                      f"(hung or suspended, not crashed)")
            return {"help": "hung"}
        except Exception as e:
            self._log(f"[DOCTOR] {label}: could not run ({e})")
            return {"help": "unrunnable"}
        seconds = time.monotonic() - started_at
        code = result.returncode
        if (code & 0xFFFFFFFF) == 0xC0000005:
            self._log(f"[DOCTOR] {label}: died even for --help "
                      f"(exit=0xC0000005 after {seconds:.1f}s) - the image "
                      f"itself is flagged here, not the listener or the model")
            self._findings["help_dies"] = exe.parent.name
        else:
            self._log(f"[DOCTOR] {label}: exited {code} in {seconds:.1f}s - "
                      f"the image runs, the trigger is serving/loading")
            self._findings.setdefault("help_lives", []).append(exe.parent.name)
        return {"help": code, "seconds": seconds}

    @staticmethod
    def _tail(path: str, lines: int = 8) -> str:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        stripped = text.strip()
        if not stripped:
            return "(nothing printed before the crash)"
        return " | ".join(stripped.splitlines()[-lines:])

    # -------------------------------------------------------------- heals
    def _heal_without_arch_kernels(self, engines, model) -> bool:
        """Hide the per-arch CPU kernels, keep the hiding only if it helps."""
        hidden: list[tuple[Path, Path]] = []
        for exe in engines:
            for kernel in arch_kernels(exe.parent):
                disabled = kernel.with_suffix(kernel.suffix + ".off")
                try:
                    kernel.rename(disabled)
                    hidden.append((kernel, disabled))
                except OSError as e:
                    self._log(f"[DOCTOR] could not hide {kernel.name}: {e}")
        if not hidden:
            self._log("[DOCTOR] no per-arch CPU kernels to hide (baseline x64 only)")
            return False
        names = ", ".join(kernel.name for kernel, _ in hidden)
        self._log(f"[DOCTOR] hid {len(hidden)} per-arch CPU kernel(s): {names}")
        started = self._probe_engines(engines, model, label="no-arch-kernels")
        if started:
            self._log(f"[DOCTOR] HEALED: {started.name} starts with the per-arch "
                f"kernels hidden - leaving them disabled. That kernel was the "
                f"fault; report this so it can be pinned upstream.")
            return True
        for kernel, disabled in hidden:
            try:
                disabled.rename(kernel)
            except OSError as e:
                self._log(f"[DOCTOR] could not restore {kernel.name}: {e}")
        self._log("[DOCTOR] not the per-arch kernels; restored them")
        return False

    def _heal_single_thread(self, engines, model) -> bool:
        """One thread: a dodge for thread-pool/affinity faults at init."""
        started = self._probe_engines(engines, model, label="single-thread",
                                      extra_args=("-t", "1"))
        if not started:
            return False
        try:
            from .backend import _runtime_dir

            pinned = _runtime_dir(self.config_dir) / "engine-threads.txt"
            pinned.parent.mkdir(parents=True, exist_ok=True)
            pinned.write_text("1", encoding="utf-8")
            self._log(f"[DOCTOR] HEALED: {started.name} starts with one thread; "
                f"pinned -t 1 in {pinned.name}")
        except OSError as e:
            self._log(f"[DOCTOR] one thread works but pinning it failed: {e}")
        return True

    # ---------------------------------------------------------- diagnostics
    def _cli_probe(self, engines, model) -> Path | None:
        """Does whisper work at all outside the server path?

        The server wrapper, its flags and its port handling are extra moving
        parts; whisper-cli loads the same model and runs the same graph with
        none of them. A CLI that works while the server dies is both the
        diagnosis (the engine is fine) and the cure (transcribe with it).
        Returns the first CLI that decoded, for the caller to switch to.
        """
        candidates = []
        for exe in engines:
            candidate = exe.parent / "whisper-cli.exe"
            if candidate.exists():
                candidates.append(candidate)
        if not candidates:
            self._log("[DOCTOR] no whisper-cli shipped beside the engines")
            return None
        wav = self._write_probe_wav()
        if wav is None:
            self._log("[DOCTOR] could not write a probe wav for the CLI check")
            return None
        try:
            for cli in candidates:
                started_at = time.monotonic()
                try:
                    result = subprocess.run(
                        [str(cli), "-m", str(model), "-f", str(wav),
                         "-l", "en", "-nt"],
                        capture_output=True, text=True, timeout=_CLI_TIMEOUT,
                        cwd=str(cli.parent), creationflags=self._no_console(),
                    )
                except Exception as e:
                    self._log(f"[DOCTOR] {cli.parent.name} whisper-cli probe "
                              f"failed to run: {e}")
                    continue
                seconds = time.monotonic() - started_at
                if result.returncode == 0:
                    self._log(
                        f"[DOCTOR] {cli.parent.name} whisper-cli loaded the "
                        f"model and ran in {seconds:.1f}s - the engine works "
                        f"outside the server")
                    return cli
                tail = " | ".join((result.stdout or "").strip().splitlines()[-4:])
                self._log(f"[DOCTOR] {cli.parent.name} whisper-cli also failed: "
                          f"exit=0x{(result.returncode | 0) & 0xFFFFFFFF:08X} "
                          f"after {seconds:.1f}s; output: {tail or '(none)'}")
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass
        return None

    @staticmethod
    def _write_probe_wav() -> str | None:
        """One second of quiet tone at 16kHz: enough to run the graph."""
        try:
            import math
            import struct
            import wave

            fd, path = tempfile.mkstemp(prefix="doctor-", suffix=".wav")
            os.close(fd)
            frames = b"".join(
                struct.pack("<h", int(1200 * math.sin(2 * math.pi * 440 * i / 16000)))
                for i in range(16000))
            with wave.open(path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(frames)
            return path
        except Exception:
            return None

    def _facts_machine(self) -> None:
        try:
            from .backend import machine_facts

            self._log(f"[DOCTOR] machine: {machine_facts()}")
        except Exception as e:
            self._log(f"[DOCTOR] machine facts failed: {e}")

    def _facts_crash_report(self) -> None:
        """Was a crash report written at all, and does it name a module?

        A genuine access violation produces Application Error 1000 naming the
        faulting DLL. Nothing at all means the process was ended from outside
        (security software) or crash reporting is switched off - and that is
        the difference between a bug we can fix and a machine policy.
        """
        try:
            from .backend import event_log_reader, faulting_module

            module = faulting_module("whisper-server.exe")
            if module:
                self._findings["faulting_module"] = module
                self._log(f"[DOCTOR] crash report names the faulting module: {module}")
            else:
                unreadable = event_log_reader()
                if unreadable:
                    self._log(f"[DOCTOR] Application log unreadable ({unreadable}); "
                        f"no crash-report conclusion can be drawn")
                else:
                    self._findings["no_crash_report"] = True
                    self._log("[DOCTOR] no Application Error event was written: the "
                        "process was killed from outside (EDR/AV) or Windows "
                        "Error Reporting is disabled")
        except Exception as e:
            self._log(f"[DOCTOR] crash-report facts failed: {e}")
        self._registry_dword(
            "Windows Error Reporting disabled",
            r"SOFTWARE\Microsoft\Windows\Windows Error Reporting", "Disabled",
            note="1 means no crash reports are written on this machine")

    def _facts_gpu(self) -> None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15,
                creationflags=self._no_console(),
            )
            info = (result.stdout or "").strip()
            if info:
                self._log(f"[DOCTOR] nvidia-smi: {info}")
            else:
                self._log(f"[DOCTOR] nvidia-smi printed nothing (exit={result.returncode})")
        except Exception as e:
            self._log(f"[DOCTOR] nvidia-smi unavailable: {e}")
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        for name in ("msvcp140.dll", "vcruntime140_1.dll"):
            present = (system32 / name).exists()
            self._log(f"[DOCTOR] {name}: {'present' if present else 'MISSING'}"
                + ("" if present else " - install the MSVC 2015-2022 x64 "
                                      "redistributable"))

    def _facts_security(self) -> None:
        """Name the security software, if any: it is a suspect, not a bug."""
        self._registry_dword(
            "Smart App Control",
            r"SYSTEM\CurrentControlSet\Control\CI\Policy",
            "VerifiedAndReputablePolicyState",
            note="1 blocks untrusted binaries; 0 is off")
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-CimInstance -Namespace root/SecurityCenter2 "
                 "-ClassName AntiVirusProduct).displayName -join ', '"],
                capture_output=True, text=True, timeout=30,
                creationflags=self._no_console(),
            )
            names = (result.stdout or "").strip()
            if names:
                self._findings["security_products"] = names
                self._log(f"[DOCTOR] security products on this machine: {names}")
            else:
                self._log("[DOCTOR] no security product registered in SecurityCenter2")
        except Exception as e:
            self._log(f"[DOCTOR] security-product query failed: {e}")

    def _facts_defender_log(self) -> None:
        """Did Defender itself record blocking whisper? Best effort.

        Cortex kills leave no local trace, but Windows Defender's own
        operational channel logs its ASR-rule blocks and detections
        (ids 1006-1015, 1116/1117, 1121/1122). A whisper hit there names
        the exact rule - e.g. blocking low-prevalence executables in
        user-writable paths - which is both the diagnosis and the text
        of the exclusion to ask IT for.
        """
        if sys.platform != "win32":
            return
        try:
            import win32evtlog
        except ImportError as e:
            self._log(f"[DOCTOR] Defender log unreadable ({e})")
            return
        channel = "Microsoft-Windows-Windows Defender/Operational"
        try:
            hand = win32evtlog.OpenEventLog(None, channel)
        except Exception as e:
            self._log(f"[DOCTOR] Defender log unavailable ({e})")
            return
        try:
            flags = (win32evtlog.EVENTLOG_BACKWARDS_READ
                     | win32evtlog.EVENTLOG_SEQUENTIAL_READ)
            seen = 0
            for _ in range(6):
                try:
                    events = win32evtlog.ReadEventLog(hand, flags, 0)
                except Exception:
                    break
                if not events:
                    break
                for ev in events:
                    seen += 1
                    if seen > 60:
                        break
                    try:
                        data = getattr(ev, "StringInserts", None) or ()
                        blob = "\n".join(str(part) for part in data)
                        if "whisper" not in blob.lower():
                            continue
                        eid = getattr(ev, "EventID", 0) & 0xFFFF
                        first = next((line.strip() for line in blob.splitlines()
                                      if line.strip()), "(no detail)")
                        self._findings["defender_hit"] = (
                            f"event {eid}: {first[:200]}")
                        self._log(f"[DOCTOR] Defender log names whisper: "
                                  f"event {eid}: {first[:200]}")
                    except Exception:
                        continue
                if seen > 60:
                    break
            if "defender_hit" not in self._findings:
                self._log("[DOCTOR] Defender log: no whisper block in the "
                          "newest events")
        finally:
            try:
                win32evtlog.CloseEventLog(hand)
            except Exception:
                pass

    def _registry_dword(self, label: str, key: str, value: str,
                        note: str = "") -> None:
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                data, _ = winreg.QueryValueEx(handle, value)
            self._log(f"[DOCTOR] {label}: {data}" + (f" ({note})" if note else ""))
        except FileNotFoundError:
            self._log(f"[DOCTOR] {label}: not set" + (f" ({note})" if note else ""))
        except Exception as e:
            self._log(f"[DOCTOR] {label} unreadable: {e}")

    # ------------------------------------------------------------- verdict
    def _verdict(self, engines, model) -> None:
        """One paragraph an IT desk or a bug report can act on."""
        names = ", ".join(exe.parent.name for exe in engines)
        module = self._findings.get("faulting_module")
        parts = [f"every engine ({names}) dies the same way"]
        if module:
            parts.append(f"and Windows names {module} as the faulting module")
        elif self._findings.get("no_crash_report"):
            parts.append("and no crash report was written, which points at "
                         "something outside the process (security software) or "
                         "disabled crash reporting")
        products = self._findings.get("security_products")
        if products:
            parts.append(f"on a machine running {products}")
        self._log("[DOCTOR] verdict: " + "; ".join(parts) + ". Send this log with "
            "the tray's Copy log before changing anything else.")
        if self._findings.get("help_dies"):
            self._log("[DOCTOR] even --help dies, so the binary itself may not "
                      "execute from the user profile here - ask IT to allow it "
                      "by path (below), not by hash: hashes change per release")
        defender_hit = self._findings.get("defender_hit")
        if defender_hit:
            self._log(f"[DOCTOR] Defender recorded this itself: {defender_hit} "
                      f"- quote that event id to IT")
        if self._findings.get("no_crash_report") and products:
            self._log("[DOCTOR] exclusion request for IT: allow "
                      f"{self.config_dir}\\runtime\\whisper-server.exe plus "
                      f"{self.config_dir}\\runtime\\cuda\\ and "
                      f"{self.config_dir}\\runtime\\plain\\ (behavioral "
                      f"protection / userland execution), or seed "
                      f"%PROGRAMDATA%\\whisper-flow\\ with the engine and "
                      f"models - the app runs those read-only with no "
                      f"downloads")
