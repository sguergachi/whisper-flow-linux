"""Fetch and run a local whisper.cpp server, so the app works out of the box.

Without this, a fresh install does nothing useful: it records, finds no
transcription backend, and gives up. Asking someone to build whisper.cpp,
pick a CUDA variant and download a multi-gigabyte model by hand is not an
install experience.

So this detects what the machine can run, downloads the matching whisper.cpp
release and a model to match, and supervises the server. Everything lands
under the user's data directory and can be deleted without trace.

Model choice follows the hardware rather than defaulting to the largest:
large-v3-turbo is excellent on a GPU and unusably slow on a CPU, so a machine
without one gets a small model instead. A fast wrong-sized default is worse
than a slower right-sized one.
"""

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from . import envfile
from .logging import log

WHISPER_CPP_RELEASE = "v1.9.1"
RELEASE_URL = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/" + WHISPER_CPP_RELEASE
)
# Used to build the Linux CUDA engine, which upstream ships no binary for.
SOURCE_URL = (
    "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/"
    + WHISPER_CPP_RELEASE + ".tar.gz"
)
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# name -> (approximate download size MB, VRAM/RAM it wants)
#
# The CPU tiers are 8-bit quantised. Measured on a six-core desktop, q8_0
# transcribes 20-25% faster than the same model in f16 and downloads at less
# than half the size, for no difference in the transcript. The 5-bit formats
# are the trap here: q5_1 is smaller again but *slower* than f16, because
# unpacking five-bit weights costs more than the memory traffic it saves.
#
# large-v3-turbo stays f16 because it is the GPU tier, where the arithmetic
# is free and the quantisation would only add unpacking work.
MODELS = {
    "ggml-large-v3-turbo": (1624, "~2GB VRAM"),
    "ggml-medium.en-q8_0": (785, "~950MB"),
    "ggml-small.en-q8_0": (252, "~350MB"),
    "ggml-base.en-q8_0": (78, "~150MB"),
    "ggml-tiny.en-q8_0": (42, "~80MB"),
}

# Models that only make sense on the GPU engine. Not "runs faster on a GPU" -
# every model does - but "unusable without one": large-v3-turbo on a CPU takes
# fifteen to twenty-five seconds for a two second clip, which is not a slower
# dictation but a broken one, and it is exactly what shipped to someone who
# downloaded it from the settings window on a machine whose CUDA engine had
# never been fetched.
GPU_MODELS = frozenset({"ggml-large-v3-turbo"})


def bundled_dir() -> Path | None:
    """Where the installer put the engine and model, if this is a real build.

    A frozen build ships a working engine and a small model so the app
    transcribes the moment it is installed, with no network. The GPU engine
    and the large model are far too big to ship the same way - together they
    exceed the 2GB ceiling on a release asset - so those are fetched later,
    in the background, only on a machine that can use them.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        if (base / "engine").exists():
            return base
    return None


def _runtime_dir(config_dir: Path) -> Path:
    return Path(config_dir) / "runtime"


def _model_dir(config_dir: Path) -> Path:
    return Path(config_dir) / "models"


def _cuda_dir(config_dir: Path) -> Path:
    """Where the source-built Linux CUDA engine lives (runtime/cuda/).

    Separate from the flat runtime directory because the Linux CUDA build
    cannot sit beside the CPU build there: the flat layout expects one
    whisper-server and one set of libraries, and the two builds would
    clobber each other whenever either was refreshed. On Windows the
    cuBLAS engine unpacks flat like the CPU one and this is never used.
    """
    return _runtime_dir(config_dir) / "cuda"


def _cuda_toolkit() -> str | None:
    """The nvcc compiler path, or None when there is no CUDA toolkit.

    The Linux GPU engine is built from source, and nvcc is the whole
    toolkit's anchor: its parent's parent is the toolkit root, which is
    where the runtime libraries live. Looked up on PATH first (the
    toolkit's own bin directory is usually there), then the conventional
    fixed locations, because /opt/cuda is commonly installed but never
    exported.
    """
    found = shutil.which("nvcc")
    if found:
        return found
    for root in ("/opt/cuda", "/usr/local/cuda", "/usr/cuda"):
        candidate = Path(root) / "bin" / "nvcc"
        if candidate.exists():
            return str(candidate)
    return None


def _cuda_lib_dirs(nvcc: str) -> list[Path]:
    """The toolkit's runtime-library directories, given its nvcc path."""
    root = Path(nvcc).resolve().parent.parent
    return [p for name in ("lib64", "lib") if (p := root / name).is_dir()]


# Answered once per process. nvidia-smi is a real program that has to load a
# driver before it prints a version, and on Windows that is most of a second;
# this used to be asked afresh six times while the settings window was being
# built, which is most of the wait before it appeared.
ACCELERATOR_ENV = "WHISPER_FLOW_ACCELERATOR"
_accelerator: str | None = None


def detect_accelerator() -> str:
    """What this machine can run whisper.cpp on: 'cuda12', 'cuda11' or 'cpu'.

    There is deliberately no integrated-GPU option. whisper.cpp publishes no
    prebuilt Vulkan, SYCL or OpenVINO binaries for any platform - only CPU
    and cuBLAS - so on an Intel or AMD laptop there is simply no GPU engine
    to point at, and pretending otherwise would mean shipping a build of our
    own that nobody here can test against those drivers. Those machines get
    the CPU engine, sized to the CPU they actually have.

    Cached, and seedable through the environment so the windows the daemon
    launches inherit the answer instead of paying for it again.
    """
    global _accelerator
    if _accelerator is not None:
        return _accelerator
    inherited = os.environ.get(ACCELERATOR_ENV, "").strip()
    if inherited in ("cpu", "cuda11", "cuda12"):
        _accelerator = inherited
        return _accelerator
    _accelerator = _probe_accelerator()
    return _accelerator


def no_console_flags() -> int:
    """creationflags so a console-subsystem child does not flash a prompt.

    The tray app is a windowed executable (console=False). Spawning a
    console tool without CREATE_NO_WINDOW allocates a new console for it -
    a black command-prompt window that appears and vanishes. nvidia-smi
    and whisper-server are both console apps; this is what hides them.
    Harmless on other platforms (returns 0).
    """
    if sys.platform != "win32":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def _probe_accelerator() -> str:
    try:
        if not shutil.which("nvidia-smi"):
            return "cpu"
        # CREATE_NO_WINDOW: without it this flashes a console at every
        # Windows login, because the tray process is windowed and
        # nvidia-smi is a console app.
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
            creationflags=no_console_flags(),
        )
        if out.returncode != 0 or not out.stdout.strip():
            return "cpu"
        major = int(out.stdout.strip().split(".")[0])
        # CUDA 12 needs a 525+ driver on Windows; below that use the 11.8 build.
        return "cuda12" if major >= 527 else "cuda11"
    except Exception:
        return "cpu"


def _logical_cpus() -> int:
    """Logical CPUs this process may run on, or 0 if the OS will not say."""
    try:
        if hasattr(os, "sched_getaffinity"):     # respects cgroup/taskset limits
            return len(os.sched_getaffinity(0))
        return os.cpu_count() or 0
    except Exception:
        return 0


def _physical_cores(logical: int) -> int:
    """Real cores behind `logical`, measured where the OS will say.

    Linux publishes the topology, so SMT siblings can be counted out exactly
    rather than guessed at. Everywhere else, halving is right on every
    desktop CPU that has SMT and merely conservative on the ones that do not.
    """
    if sys.platform == "linux":
        try:
            cores = set()
            for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
                topology = cpu / "topology"
                try:
                    cores.add((
                        (topology / "physical_package_id").read_text(
                            encoding="utf-8").strip(),
                        (topology / "core_id").read_text(encoding="utf-8").strip(),
                    ))
                except OSError:
                    continue
            if cores:
                return len(cores)
        except Exception:
            pass
    return max(1, logical // 2)


def usable_cores() -> int:
    """Cores whisper.cpp should actually use: one thread per physical core.

    Not the logical count. whisper.cpp is compute-bound, so SMT siblings add
    contention rather than throughput, and oversubscribing does not degrade
    gently - measured on a six-core desktop, base.en took 0.42s at six
    threads, 0.48s at eight, and thirteen seconds at twelve. The cap below
    is what stands between a busy machine and that cliff.

    One per core rather than one fewer: the same measurement put the optimum
    exactly at the core count, and holding a core back cost about 11% for a
    responsiveness gain nobody could feel.
    """
    logical = _logical_cpus()
    if not logical:
        return 4
    return max(2, min(8, min(_physical_cores(logical), logical)))


def total_ram_gb() -> float:
    """Roughly how much memory this machine has, or 0.0 if it will not say."""
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except (OSError, ValueError, AttributeError):
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            import ctypes.wintypes

            class Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.wintypes.DWORD),
                    ("dwMemoryLoad", ctypes.wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = Status()
            status.dwLength = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / (1024 ** 3)
        except Exception:
            pass
    return 0.0


def recommended_model(accelerator: str) -> str:
    """The largest model this machine can run without falling behind speech.

    Sizing matters far more on a CPU than on a GPU. large-v3-turbo on an
    NVIDIA card transcribes faster than anyone talks; the same model on a
    laptop CPU takes minutes per sentence, which is not a slower feature but
    a broken one. So the CPU tiers are set by what keeps up in real time:

      small.en   a full desktop or workstation CPU
      base.en    a typical laptop - clearly the common case
      tiny.en    a thin, old or memory-starved machine

    A too-small model is merely less accurate. A too-large one is unusable,
    so every boundary here rounds down.
    """
    if accelerator.startswith("cuda"):
        return "ggml-large-v3-turbo"

    cores = usable_cores()
    ram = total_ram_gb()
    if ram and ram < 4:
        return "ggml-tiny.en-q8_0"
    # small.en holds ~350MB of weights resident for as long as the daemon
    # runs, which is a lot to ask of an 8GB machine that is also running a
    # browser. base.en asks ~150MB and is the safe middle.
    #
    # The thresholds are seven and four physical cores, which is the same
    # hardware these tiers have always meant. usable_cores() used to hold a
    # core back and so reported one fewer than it does now; sizing the model
    # off the new number without moving the thresholds would have quietly
    # promoted every six-core desktop to a model it was never offered.
    if cores >= 7 and (not ram or ram >= 8):
        return "ggml-small.en-q8_0"
    if cores >= 4:
        return "ggml-base.en-q8_0"
    return "ggml-tiny.en-q8_0"


def wanted_engine(accelerator: str) -> str:
    """The engine kind that should be installed for this machine.

    Windows fetches a prebuilt cuBLAS engine for a CUDA card. Linux has no
    upstream CUDA binary - the Linux release asset is CPU-only - so there
    the GPU engine is built from source, which needs the CUDA toolkit on
    the host. A Linux machine without nvcc gets the CPU engine whatever
    card is fitted.
    """
    if not accelerator.startswith("cuda"):
        return "cpu"
    if sys.platform != "win32" and not _cuda_toolkit():
        return "cpu"
    return accelerator


def _server_archive(accelerator: str) -> str:
    """The whisper.cpp release asset to fetch for this machine.

    The Linux build is CPU-only because that is all upstream publishes for
    it; it does ship per-microarchitecture ggml backends and loads the best
    one for the host at runtime, so a laptop still gets AVX2 rather than a
    lowest-common-denominator binary.
    """
    if sys.platform != "win32":
        return "whisper-bin-ubuntu-x64.tar.gz"
    return {
        "cuda12": "whisper-cublas-12.4.0-bin-x64.zip",
        "cuda11": "whisper-cublas-11.8.0-bin-x64.zip",
    }.get(accelerator, "whisper-blas-bin-x64.zip")


def _extract(archive: Path, into: Path) -> None:
    """Unpack a release asset, whichever form this platform's comes in."""
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive) as tar:
            # filter= is required from 3.14 and refuses paths escaping `into`.
            try:
                tar.extractall(into, filter="data")
            except TypeError:                   # older Python without filter=
                tar.extractall(into)
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)


def _run_build_command(cmd: list[str], message: str) -> None:
    """Run a compiler step, raising RuntimeError with its tail on failure.

    The CUDA build is the one thing this module runs that is not ours: a
    cmake configure or compile can fail on the host's toolkit, compilers,
    or kernel headers in ways a download never can. The tail of the
    output is the whole diagnosis, and swallowing it would leave the
    settings window saying "could not install" with nothing to act on.
    """
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"{message}: {e}") from e
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(f"{message}:\n" + "\n".join(tail))


def _stage(progress, name: str):
    """Bind a stage name to a (stage, fraction) callback, or None."""
    if not progress:
        return None
    return lambda fraction: progress(name, fraction)


def _download(url: str, dest: Path, progress=None) -> None:
    """Download to a temporary name, then move: a partial file is never used."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        last_report = 0.0
        with open(tmp, "wb") as f:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                # Often enough for a progress bar to look continuous;
                # callers that only want notifications throttle further.
                if progress and total and time.monotonic() - last_report > 0.25:
                    last_report = time.monotonic()
                    progress(done / total)
        if progress and total:
            progress(1.0)
    tmp.replace(dest)


def _stop_strays_win(exe_path, keep_pid: int | None = None) -> int:
    """Windows: kill leftover servers whose image path matches `exe_path`."""
    stopped = 0
    try:
        import ctypes
        from ctypes import wintypes

        class _ENTRY(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.OpenProcess.restype = wintypes.HANDLE
        wanted = os.path.normcase(os.path.abspath(str(exe_path)))
        name = os.path.basename(wanted)

        snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)   # PROCESSES
        entry = _ENTRY()
        entry.dwSize = ctypes.sizeof(_ENTRY)
        more = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            pid = entry.th32ProcessID
            if (os.path.normcase(entry.szExeFile) == name
                    and pid != keep_pid and pid != os.getpid()):
                # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE
                handle = kernel32.OpenProcess(0x1000 | 0x1, False, pid)
                if handle:
                    try:
                        buffer = ctypes.create_unicode_buffer(32768)
                        size = wintypes.DWORD(32768)
                        if kernel32.QueryFullProcessImageNameW(
                                handle, 0, buffer, ctypes.byref(size)):
                            if os.path.normcase(buffer.value) == wanted:
                                kernel32.TerminateProcess(handle, 1)
                                stopped += 1
                                log(f"[BACKEND] stopped a stray server, pid {pid}")
                    finally:
                        kernel32.CloseHandle(handle)
            more = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        kernel32.CloseHandle(snapshot)
    except Exception as e:
        log(f"[BACKEND] could not look for stray servers: {e}")
    return stopped


def _linux_exe_path(pid: int) -> str | None:
    """Resolved path of `/proc/<pid>/exe`, or None if unreadable."""
    try:
        return str(Path(f"/proc/{pid}/exe").resolve())
    except (OSError, ValueError):
        return None


def _linux_cmdline0(pid: int) -> str | None:
    """argv[0] for a process, used when /proc/pid/exe is gone (deleted)."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    return raw.split(b"\0", 1)[0].decode(errors="replace") or None


def _kill_pid(pid: int, label: str) -> bool:
    """SIGTERM then SIGKILL a process. Returns True if we sent a signal."""
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError:
        log(f"[BACKEND] no permission to stop {label} pid {pid}")
        return False
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log(f"[BACKEND] stopped a stray server, pid {pid}")
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
        log(f"[BACKEND] force-stopped a stray server, pid {pid}")
        return True
    except ProcessLookupError:
        log(f"[BACKEND] stopped a stray server, pid {pid}")
        return True
    except PermissionError:
        return False


def _path_matches_server(path: str | None, wanted: str) -> bool:
    """True when `path` is our server binary (including a deleted copy)."""
    if not path:
        return False
    if path == wanted or path == wanted + " (deleted)":
        return True
    # argv[0] sometimes keeps the pre-delete path without the suffix.
    try:
        return str(Path(path).resolve()) == wanted
    except OSError:
        return False


def _stop_strays_linux(exe_path, keep_pid: int | None = None) -> int:
    """Linux: kill leftover servers whose executable path matches `exe_path`.

    Matches on /proc/pid/exe (and argv[0] when the binary was replaced),
    never on the bare name: a hand-built whisper-server elsewhere on the
    machine is not ours to kill.
    """
    try:
        wanted = str(Path(exe_path).resolve())
    except OSError:
        wanted = os.path.abspath(str(exe_path))
    stopped = 0
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in (keep_pid, os.getpid()):
                continue
            # Prefer the kernel's view of the binary; fall back to argv[0]
            # when the link is gone (binary replaced after the process started).
            if not (_path_matches_server(_linux_exe_path(pid), wanted)
                    or _path_matches_server(_linux_cmdline0(pid), wanted)):
                continue
            if _kill_pid(pid, "server"):
                stopped += 1
    except Exception as e:
        log(f"[BACKEND] could not look for stray servers: {e}")
    return stopped


def stop_strays(exe_path, keep_pid: int | None = None) -> int:
    """Kill any copy of our server left behind by an earlier run.

    Matched by executable path, never by name: another application ships a
    whisper-server too, and killing that one would be inexcusable.

    Without this a second server is started while the first still holds the
    port. The new one cannot bind, exits, and the readiness check connects
    to the *orphan* and reports success - so the daemon believes it owns a
    server it never started and cannot stop, and two models sit in memory
    competing for the same cores (and on a GPU, for the same VRAM).

    Linux used to be a no-op here. Four orphan servers on one port were
    measured turning a 0.7s large-v3-turbo pass into a 25s one.
    """
    if sys.platform == "win32":
        return _stop_strays_win(exe_path, keep_pid=keep_pid)
    if sys.platform.startswith("linux"):
        return _stop_strays_linux(exe_path, keep_pid=keep_pid)
    return 0


def stop_managed_strays(config_dir, keep_pid: int | None = None) -> int:
    """Kill orphans of every engine we install under runtime/, not only one.

    The CUDA binary lives in runtime/cuda/; the CPU one in runtime/. A
    CPU→GPU upgrade (or a daemon restart that picks a different engine)
    leaves the other path still listening on the same port. stop_strays of
    only the engine about to start would miss it.
    """
    runtime = _runtime_dir(Path(config_dir))
    name = "whisper-server.exe" if sys.platform == "win32" else "whisper-server"
    candidates = [
        runtime / name,
        runtime / "cuda" / name,
    ]
    stopped = 0
    seen: set[str] = set()
    for path in candidates:
        key = os.path.normcase(os.path.abspath(str(path)))
        if key in seen:
            continue
        seen.add(key)
        stopped += stop_strays(path, keep_pid=keep_pid)
    return stopped


def _linux_cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    return [p.decode(errors="replace") for p in raw.split(b"\0") if p]


def _cmdline_listens_on_port(args: list[str], port: int) -> bool:
    """True when argv looks like whisper-server --port <port>."""
    token = str(port)
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args) and args[i + 1] == token:
            return True
        if arg == f"--port={token}":
            return True
    return False


def stop_port_whisper_servers(port: int, keep_pid: int | None = None) -> int:
    """Kill whisper-server processes still bound to our listen port.

    Path-matched stop_strays misses a hand-built binary (e.g. under
    ~/Dev/whisper.cpp) that was started on the same port. Leaving it up
    means our child fails to bind, the readiness check answers on the
    foreign process, and two CUDA models thrash the same GPU.

    Only processes whose cmdline contains whisper-server *and* our --port
    are touched. An unrelated service on the same port is left alone.
    Linux only: Windows already reaps by image path via stop_strays.
    """
    if not sys.platform.startswith("linux"):
        return 0
    stopped = 0
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in (keep_pid, os.getpid()):
                continue
            args = _linux_cmdline(pid)
            if not args:
                continue
            if not any("whisper-server" in a for a in args):
                continue
            if not _cmdline_listens_on_port(args, port):
                continue
            if _kill_pid(pid, "port-holder"):
                stopped += 1
    except Exception as e:
        log(f"[BACKEND] could not free port {port}: {e}")
    return stopped


def _kill_on_exit_job():
    """A Windows job object that kills its members when this process dies.

    stop() only runs when the daemon shuts down cleanly. Killed, crashed, or
    killed by an installer that wants its files back, it never runs and the
    server is orphaned - still holding the port, still holding a model in
    memory, still charging for threads. Two of them were found running at
    once, one with five minutes of CPU behind it, which is most of what
    "inference got slow" was.

    A job with KILL_ON_JOB_CLOSE closes the gap: the handle is held by this
    process, and Windows closes every handle a process owns when it dies,
    however it dies. There is no path out of here that leaves the server
    behind.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount",
                "OtherOperationCount", "ReadTransferCount",
                "WriteTransferCount", "OtherTransferCount")]

        class _EXTENDED(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC),
                ("IoInfo", _IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        limits = _EXTENDED()
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        limits.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        JobObjectExtendedLimitInformation = 9
        if not kernel32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(limits), ctypes.sizeof(limits)):
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception as e:
        log(f"[BACKEND] no job object, the server may outlive us: {e}")
        return None


def _adopt_into_job(job, process) -> bool:
    """Put a started process into the job, so it cannot outlive us."""
    if not job or sys.platform != "win32":
        return False
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        return bool(kernel32.AssignProcessToJobObject(
            job, int(process._handle)))
    except Exception as e:
        log(f"[BACKEND] could not put the server in the job: {e}")
        return False


class LocalBackend:
    """Downloads, starts and supervises a local whisper.cpp server."""

    def __init__(self, config, notify=None):
        self.config = config
        self._notify = notify or (lambda msg: None)
        self._process = None
        self._lock = threading.Lock()
        # Held for the life of this process; closing it kills the server.
        self._job = _kill_on_exit_job()

    # ---------------------------------------------------------------- paths
    @property
    def _exe_name(self) -> str:
        return "whisper-server.exe" if sys.platform == "win32" else "whisper-server"

    @property
    def server_exe(self) -> Path:
        """The engine binary that will actually run.

        Downloaded engines win over the bundled one - a downloaded engine
        only exists because this machine has a GPU and the better engine
        was fetched for it. On Linux the source-built CUDA engine lives in
        its own directory (runtime/cuda), so it is the one that wins there.
        """
        if self.installed_engine().startswith("cuda"):
            cuda = _cuda_dir(self.config.config_dir) / self._exe_name
            if cuda.exists():
                return cuda
        downloaded = _runtime_dir(self.config.config_dir) / self._exe_name
        if downloaded.exists():
            return downloaded
        bundled = bundled_dir()
        if bundled:
            return bundled / "engine" / self._exe_name
        return downloaded

    def _engine_exe(self, kind: str) -> Path:
        """Where a given engine kind's binary lives."""
        if sys.platform != "win32" and kind.startswith("cuda"):
            return _cuda_dir(self.config.config_dir) / self._exe_name
        return _runtime_dir(self.config.config_dir) / self._exe_name

    @property
    def _engine_marker(self) -> Path:
        """Which engine the downloaded one is, written when it is unpacked."""
        return _runtime_dir(self.config.config_dir) / "engine.kind"

    def installed_engine(self) -> str:
        """The kind of engine that will actually run: 'cuda12', 'cuda11', 'cpu'.

        This is the question nothing used to ask. `install()` skipped the
        download whenever *any* engine was present, and a frozen build always
        bundles one - the CPU build - so a machine with an NVIDIA card
        downloaded the 1.6GB GPU model, ran it on the CPU engine that shipped
        beside it, and took twenty seconds a sentence. The engine and the
        model were each fine; nothing compared them.

        The marker records the kind when an engine is fetched or built, and
        is checked against what is actually on disk: a marker left pointing
        at an engine that has since been deleted must not win.
        """
        flat = _runtime_dir(self.config.config_dir) / self._exe_name
        cuda = _cuda_dir(self.config.config_dir) / self._exe_name
        if not flat.exists() and not cuda.exists():
            return "cpu"        # bundled, or nothing: the shipped build is CPU
        try:
            recorded = self._engine_marker.read_text(encoding="utf-8").strip()
        except OSError:
            recorded = ""
        if recorded in ("cpu", "cuda11", "cuda12"):
            if self._engine_exe(recorded).exists():
                return recorded
        # Installed before the marker existed. cuBLAS ships its own libraries
        # beside the binary and the CPU build has none, so the directory says
        # what it is without needing to have been told.
        names = {path.name.lower() for path in flat.parent.glob("*")}
        if any(name.startswith("cublas64_12") for name in names):
            return "cuda12"
        if any(name.startswith("cublas64_11") for name in names):
            return "cuda11"
        if any(name.startswith(("ggml-cuda", "cudart64_")) for name in names):
            return "cuda12"     # a CUDA build of unknown vintage; not the CPU one
        if sys.platform != "win32" and cuda.exists():
            return "cuda12"     # the source-built Linux engine, pre-marker
        return "cpu"

    def engine_is_gpu(self) -> bool:
        return self.installed_engine().startswith("cuda")

    def model_path(self, name: str | None = None) -> Path:
        name = name or self.config.model_name
        downloaded = _model_dir(self.config.config_dir) / f"{name}.bin"
        if downloaded.exists():
            return downloaded
        bundled = bundled_dir()
        if bundled and (bundled / "models" / f"{name}.bin").exists():
            return bundled / "models" / f"{name}.bin"
        return downloaded

    def bundled_model(self) -> str | None:
        """Whichever model shipped with the build, if any."""
        base = bundled_dir()
        if not base:
            return None
        for found in sorted((base / "models").glob("*.bin")):
            return found.stem
        return None

    def is_installed(self, model: str | None = None) -> bool:
        return self.server_exe.exists() and self.model_path(model).exists()

    def model_inventory(self) -> list[dict]:
        """Every known model with its status, for the settings window.

        One row per model: size, whether it is on disk (downloaded or
        bundled), whether it is the one in use, and whether it is the one
        this machine would be recommended.
        """
        recommended = recommended_model(detect_accelerator())
        current = self.working_model()
        on_gpu = self.engine_is_gpu()
        rows = [
            {
                "name": name,
                "size_mb": size_mb,
                "wants": wants,
                "installed": self.model_path(name).exists(),
                "current": name == current,
                "recommended": name == recommended,
                # Whether it needs a GPU to keep up, and whether it would get
                # one here. A row that says both is a row that cannot be
                # chosen by mistake.
                "gpu_only": name in GPU_MODELS,
                "accelerated": on_gpu,
            }
            for name, (size_mb, wants) in MODELS.items()
        ]

        # The model in use need not be one of the five above: a bundled or
        # hand-placed file keeps whatever name it has, and MODELS lists the
        # q8_0 builds. Leaving it out meant a window whose every radio was
        # blank while the app was transcribing perfectly well - the one
        # question the page exists to answer, and it had no row to answer it
        # with. It goes first, because it is the one that is true now.
        if current and not any(row["name"] == current for row in rows):
            path = self.model_path(current)
            size_mb = int(path.stat().st_size / 1_000_000) if path.exists() else 0
            rows.insert(0, {
                "name": current,
                "size_mb": size_mb,
                "wants": "already on this machine",
                "installed": True,
                "current": True,
                "recommended": current == recommended,
                "gpu_only": current in GPU_MODELS,
                "accelerated": on_gpu,
            })
        return rows

    def chosen_model(self) -> str | None:
        """The model last asked for, read from disk rather than from memory.

        Not `config.model_name`: pydantic resolves the .env path once at
        import, so what the settings window wrote a moment ago is not in the
        running config and will not be until the next launch. Reading the file
        is what lets a saved choice take effect on the restart the window
        offers - without it the daemon restarted, re-read nothing, and the
        page went on reporting the old model as in use.
        """
        # The environment first, because that is the order Config resolves
        # them in: pydantic-settings lets a real environment variable beat
        # the .env file. Nothing in this app sets this one - it is there for
        # someone who exports it in their shell - but reading the file first
        # would quietly invert that precedence for this one setting.
        from_env = os.environ.get("WHISPER_FLOW_MODEL_NAME", "").strip()
        if from_env:
            return from_env
        saved = envfile.get(Path(self.config.config_dir) / ".env",
                            "WHISPER_FLOW_MODEL_NAME")
        return saved.strip() if saved else None

    def working_model(self) -> str | None:
        """The best model actually present.

        A model the user has explicitly chosen wins, provided it is on this
        machine at all - that choice is the whole content of the Speech page,
        and ignoring it is why picking a smaller model there changed nothing.

        Failing that a downloaded model outranks the bundled one: a model is
        only ever downloaded because someone pressed the button asking for it,
        and that has to take effect the moment the window closes, before
        anything has been saved or restarted.
        """
        chosen = self.chosen_model()
        if chosen and self.model_path(chosen).exists():
            return chosen

        downloaded = sorted(
            p.stem for p in _model_dir(self.config.config_dir).glob("*.bin"))
        if downloaded:
            if self.config.model_name in downloaded:
                return self.config.model_name
            # Otherwise the largest one that was fetched.
            return max(downloaded, key=lambda name: MODELS.get(name, (0, ""))[0])
        if self.model_path(self.config.model_name).exists():
            return self.config.model_name
        return self.bundled_model()

    # ------------------------------------------------------------- install
    def install(self, model: str | None = None, force_download: bool = False,
                progress=None) -> bool:
        """Fetch the engine and model into the user's data directory.

        `force_download` ignores anything bundled, which is how the GPU
        upgrade replaces the shipped CPU engine.

        `progress` is called with (stage, fraction) as bytes arrive, so a
        setup window can draw a bar. Without it the only feedback is the
        occasional notification, which is enough for a background upgrade
        and not enough for someone watching a download they just started.
        """
        accelerator = detect_accelerator()
        model = model or recommended_model(accelerator)
        log(f"[BACKEND] accelerator={accelerator} model={model}")

        wanted = wanted_engine(accelerator)
        downloaded_exe = _runtime_dir(self.config.config_dir) / self._exe_name
        downloaded_model = _model_dir(self.config.config_dir) / f"{model}.bin"
        # The engine this machine wants does not always live in the flat
        # runtime directory: on Linux the source-built CUDA engine has its
        # own directory, because it cannot sit beside the CPU build without
        # the two clobbering each other whenever either is refreshed.
        engine_exe = self._engine_exe(wanted)
        have_exe = engine_exe.exists() if force_download else self.server_exe.exists()
        # An engine built for the wrong thing is not an engine we have.
        #
        # This is the whole of the bug that made a GPU machine slower than a
        # laptop: the check was "is there an engine", the bundled CPU build
        # always answered yes, and so the cuBLAS engine was never fetched no
        # matter which model was chosen. Downloading large-v3-turbo then put a
        # GPU-sized model on a CPU engine - fifteen to twenty-five seconds per
        # utterance, during which every further press was dropped as busy and
        # the text landed wherever the focus had wandered to by the time it
        # arrived.
        if have_exe and self.installed_engine() != wanted:
            log(f"[BACKEND] the installed {self.installed_engine()} engine is "
                f"not the {wanted} one this machine wants; fetching it")
            have_exe = False

        have_model = (downloaded_model.exists() if force_download
                      else self.model_path(model).exists())

        try:
            if not have_exe:
                if sys.platform != "win32" and wanted.startswith("cuda"):
                    # Upstream ships no Linux CUDA binary; build one. Raises
                    # on a missing toolkit or a failed compile, and the
                    # failure lands in the notification below.
                    self._build_cuda_engine(progress)
                else:
                    archive = _server_archive(accelerator)
                    # Name what is actually being fetched. On Linux that is the
                    # CPU build even on a CUDA machine, and claiming otherwise
                    # would make a slow transcription look like a broken GPU.
                    kind = "GPU" if "cublas" in archive else "CPU"
                    self._notify(f"Downloading the {kind} speech engine...")
                    runtime = _runtime_dir(self.config.config_dir)
                    archive_path = runtime / archive
                    _download(f"{RELEASE_URL}/{archive}", archive_path,
                              _stage(progress, "engine"))
                    # Extract into a staging directory so the flattening
                    # below cannot reach engines that already live in the
                    # runtime directory (the Linux CUDA build in runtime/cuda).
                    staging = runtime / ".extract"
                    shutil.rmtree(staging, ignore_errors=True)
                    _extract(archive_path, staging)
                    archive_path.unlink(missing_ok=True)
                    # The archives nest the binaries a directory deep, and the
                    # shared libraries beside them have to stay beside them:
                    # the Linux build finds them through RUNPATH=$ORIGIN.
                    found = next(staging.rglob(self._exe_name), None)
                    if found is None:
                        raise RuntimeError(
                            f"{self._exe_name} not found in {archive}")
                    for item in found.parent.iterdir():
                        shutil.move(str(item), runtime)
                    shutil.rmtree(staging, ignore_errors=True)
                    if sys.platform != "win32":
                        # A tarball's mode bits do not survive every extraction path.
                        downloaded_exe.chmod(0o755)
                # Last, and only once the binary is in place: the marker is
                # read as "this engine is that kind", so it must never be
                # there describing an engine that failed to unpack.
                self._record_engine(wanted)

            if not have_model:
                size_mb = MODELS.get(model, (0, ""))[0]
                self._notify(f"Downloading speech model {model} ({size_mb}MB)...")

                staged = _stage(progress, "model")
                last_notice = [0.0]

                def report(fraction):
                    if staged:
                        staged(fraction)
                    # A notification every quarter second would be spam.
                    if time.monotonic() - last_notice[0] > 15.0:
                        last_notice[0] = time.monotonic()
                        self._notify(f"Speech model {int(fraction * 100)}%")

                _download(f"{MODEL_URL}/{model}.bin", downloaded_model, report)

            self._notify("Speech model ready")
            return True
        except Exception as e:
            log(f"[BACKEND] install failed: {e}")
            self._notify(f"Could not set up the speech model: {e}")
            return False

    def _build_cuda_engine(self, progress=None) -> None:
        """Compile whisper.cpp with the CUDA backend into runtime/cuda/.

        Upstream publishes no Linux CUDA binary - the release assets are
        the CPU tarball and Windows cuBLAS zips - so the only way a Linux
        machine gets a GPU engine is to build one, which is what this does:
        fetch the release sources, configure with GGML_CUDA, and compile
        whisper-server. It takes a few minutes once; the result is a plain
        binary in the runtime directory like any other engine.

        Requires the CUDA toolkit (nvcc) and cmake on the host. Raises
        RuntimeError when either is missing or the build fails, which the
        caller reports as an install failure.
        """
        runtime = _runtime_dir(self.config.config_dir)
        src_dir = runtime / "whisper.cpp-src"
        build_dir = runtime / "whisper.cpp-build"
        out_dir = _cuda_dir(self.config.config_dir)
        stage = _stage(progress, "engine")

        nvcc = _cuda_toolkit()
        if not nvcc:
            raise RuntimeError("the CUDA toolkit (nvcc) is not installed")
        cmake = shutil.which("cmake")
        if not cmake:
            raise RuntimeError("cmake is not installed")

        # The sources are tagged with the version they were built from, so
        # a release bump re-extracts rather than quietly compiling a stale
        # checkout against a build directory that still thinks otherwise.
        version_file = src_dir / ".whisper-cpp-version"
        fresh = (version_file.exists()
                 and version_file.read_text(encoding="utf-8").strip()
                 == WHISPER_CPP_RELEASE)
        if not fresh:
            self._notify("Downloading the whisper.cpp sources...")
            shutil.rmtree(src_dir, ignore_errors=True)
            shutil.rmtree(build_dir, ignore_errors=True)
            archive = runtime / f"whisper.cpp-{WHISPER_CPP_RELEASE}-src.tar.gz"
            _download(SOURCE_URL, archive, stage)
            _extract(archive, runtime)
            archive.unlink(missing_ok=True)
            # The source archive nests one directory deep: whisper.cpp-v1.9.1/.
            top = next(p for p in runtime.iterdir()
                       if (p / "CMakeLists.txt").exists())
            top.rename(src_dir)
            version_file.write_text(WHISPER_CPP_RELEASE, encoding="utf-8")

        self._notify("Building the GPU speech engine (one-time compile)...")
        configure = [cmake, "-S", str(src_dir), "-B", str(build_dir),
                     "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON",
                     f"-DCMAKE_CUDA_COMPILER={nvcc}"]
        _run_build_command(configure, "the CUDA engine would not configure")
        if stage:
            stage(0.4)
        build = [cmake, "--build", str(build_dir), "--target", "whisper-server",
                 "-j", str(usable_cores())]
        _run_build_command(build, "the CUDA engine would not compile")
        if stage:
            stage(0.9)

        built = build_dir / "bin" / "whisper-server"
        if not built.exists():
            raise RuntimeError("the CUDA build produced no whisper-server")
        out_dir.mkdir(parents=True, exist_ok=True)
        # The build tree is disposable; the engine is not. whisper.cpp puts
        # an absolute build-directory RUNPATH on its binaries, so the
        # shared libraries have to come along, and start() points
        # LD_LIBRARY_PATH - which outranks RUNPATH - at this directory.
        for item in (build_dir / "bin").iterdir():
            if item.is_file() and item.name.startswith("lib"):
                shutil.copy2(item, out_dir / item.name)
        # Installed under a temporary name and renamed, so a failed copy can
        # never leave a half-written binary that looks like an engine.
        staged = out_dir / (self._exe_name + ".new")
        shutil.copy2(built, staged)
        staged.chmod(0o755)
        staged.rename(out_dir / self._exe_name)

    def _record_engine(self, kind: str) -> None:
        try:
            self._engine_marker.parent.mkdir(parents=True, exist_ok=True)
            self._engine_marker.write_text(kind, encoding="utf-8")
        except OSError as e:
            log(f"[BACKEND] could not record the engine kind: {e}")

    # --------------------------------------------------------------- serve
    def start(self, model: str | None = None) -> str | None:
        """Start the server if it is not already up. Returns its URL."""
        with self._lock:
            if self._process and self._process.poll() is None:
                return self.url

            if not self.is_installed(model):
                return None

            # Clear anything a previous run left holding the port, or the
            # server started below cannot bind and the readiness check ends
            # up connecting to the orphan instead. Every managed engine path
            # is cleared - not only the one about to start - so a CPU→GPU
            # upgrade cannot leave the old binary listening and thrashing
            # VRAM / RAM alongside the new one. Then free the port of any
            # other whisper-server still bound there (hand-built binaries
            # under a different path).
            stop_managed_strays(self.config.config_dir)
            stop_port_whisper_servers(self.config.local_server_port)

            cmd = [
                str(self.server_exe),
                "-m", str(self.model_path(model)),
                "-l", "en",
                "--host", "127.0.0.1",
                "--port", str(self.config.local_server_port),
            ]
            env = None
            if detect_accelerator() == "cpu":
                # Left to itself whisper.cpp takes four threads whatever the
                # machine has, which is short of the optimum on a desktop and
                # past it on a two-core laptop. See usable_cores().
                cmd += ["-t", str(usable_cores())]
            elif self.installed_engine().startswith("cuda") \
                    and sys.platform != "win32":
                # The source-built engine links the ggml libraries it ships
                # with (in runtime/cuda) and the CUDA runtime out of the
                # toolkit. LD_LIBRARY_PATH outranks the binary's build-dir
                # RUNPATH, so listing the engine's own directory first is
                # what keeps it working after the build tree is gone; the
                # toolkit dirs are there because /opt/cuda is installed on
                # many distros and never added to ldconfig.
                lib_dirs = [str(_cuda_dir(self.config.config_dir))]
                nvcc = _cuda_toolkit()
                if nvcc:
                    lib_dirs += [str(p) for p in _cuda_lib_dirs(nvcc)]
                if lib_dirs:
                    env = dict(os.environ)
                    existing = env.get("LD_LIBRARY_PATH", "")
                    env["LD_LIBRARY_PATH"] = ":".join(
                        lib_dirs + ([existing] if existing else []))
            # No console window for a background helper.
            try:
                self._process = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=no_console_flags(), env=env,
                )
            except Exception as e:
                log(f"[BACKEND] could not start the server: {e}")
                return None

            # Before it has done anything: from here on it dies with us.
            _adopt_into_job(self._job, self._process)

            if self._wait_until_ready():
                # Ready means the port accepts connections. If our child
                # already exited, that is an orphan (or a foreign server)
                # answering - the exact failure stop_strays exists to prevent.
                # Refuse rather than pretend we own a process we do not.
                if self._process is not None and self._process.poll() is not None:
                    log("[BACKEND] server exited after start; port is held by "
                        "something else - not claiming it as ours")
                    self._process = None
                    return None
                log(f"[BACKEND] server ready on {self.url}")
                return self.url
            log("[BACKEND] server did not become ready")
            return None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.config.local_server_port}"

    def _wait_until_ready(self, timeout: float = 120.0) -> bool:
        """Loading a multi-gigabyte model into VRAM is not instant."""
        import socket

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                return False        # it exited
            try:
                with socket.create_connection(
                        ("127.0.0.1", self.config.local_server_port), timeout=2):
                    return True
            except OSError:
                time.sleep(0.5)
        return False

    def stop(self) -> None:
        with self._lock:
            if not self._process:
                return
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    @property
    def _setup_marker(self) -> Path:
        return Path(self.config.config_dir) / "setup-seen"

    def setup_seen(self) -> bool:
        return self._setup_marker.exists()

    def mark_setup_seen(self) -> None:
        """Remember that the setup window has been through once.

        Without this the window would reappear at every sign-in for anyone
        who looked at the GPU offer and decided against it, which is nagging
        rather than helping.
        """
        try:
            self._setup_marker.parent.mkdir(parents=True, exist_ok=True)
            self._setup_marker.write_text(
                time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        except Exception as e:
            log(f"[BACKEND] could not write the setup marker: {e}")

    def setup_reason(self) -> str | None:
        """Why the setup window should open, or None to stay quiet.

        'missing' means nothing can transcribe yet, so setup is the whole
        difference between a working app and a dead one. 'gpu' means it
        works but could work far better - worth offering once, never worth
        downloading unasked on a connection that might be metered.
        """
        if not self.working_model() or not self.server_exe.exists():
            return "missing"
        if self.needs_gpu_upgrade() and not self.setup_seen():
            return "gpu"
        return None

    def needs_gpu_upgrade(self) -> bool:
        """Whether this machine could do better than what is installed.

        True when there is an NVIDIA GPU but the engine that will run is a
        CPU one. Asked of the engine rather than of whether anything was ever
        downloaded: downloading a *model* also creates the runtime directory,
        so the old test went quiet the moment someone accepted a model from
        the settings window - which is precisely when the machine was left
        running a GPU model on the CPU engine with nothing saying so.

        On Linux the GPU engine is built from source, so the offer exists
        only where the toolkit and cmake that a build needs are present;
        without them there is no upgrade to offer.
        """
        if not detect_accelerator().startswith("cuda"):
            return False
        if self.engine_is_gpu():
            return False
        if sys.platform != "win32":
            return bool(_cuda_toolkit()) and bool(shutil.which("cmake"))
        return True

    def engine_summary(self) -> str:
        """What the speech engine runs on, in words, for the settings window."""
        engine = self.installed_engine()
        if engine.startswith("cuda"):
            return f"NVIDIA GPU ({engine})"
        if detect_accelerator().startswith("cuda"):
            # A card is fitted but the engine is not. Worth saying - but only
            # where it can be changed: without the toolkit there is nothing
            # on Linux to build the engine with, and that is a dead end, not
            # a button waiting to be pressed.
            if sys.platform == "win32" or _cuda_toolkit():
                return f"CPU, {usable_cores()} threads - GPU engine not installed"
        return f"CPU, {usable_cores()} threads"

    def gpu_upgrade_note(self) -> str:
        """What the GPU upgrade means in words, for the Install button."""
        if sys.platform == "win32":
            return "1.6GB, and makes the larger models usable"
        return ("compiles the CUDA engine (needs the CUDA toolkit), and "
                "makes the larger models usable")

    def describe(self) -> str:
        """One line for the diagnostics report."""
        accelerator = detect_accelerator()
        return (f"{platform.system()} / {accelerator} / "
                f"engine {self.installed_engine()} / "
                f"server {'present' if self.server_exe.exists() else 'missing'} / "
                f"model {'present' if self.model_path().exists() else 'missing'}")
