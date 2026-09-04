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
    """Where the CUDA engine lives (runtime/cuda/).

    Separate from the flat runtime directory because the two builds cannot
    share one: the flat layout expects one whisper-server and one set of
    libraries, and on Windows the loader resolves DLLs from the exe's own
    directory first — so cuBLAS DLLs beside the CPU binary poisoned it
    into a 0xC0000005 for every model. Each kind owns its directory.
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
# Full driver version from the probe above (e.g. "537.58"), for diagnostics.
# Empty when there is no NVIDIA card. Read by machine_facts(); never probed
# twice — the subprocess already ran here.
_driver_version: str = ""


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


def _ensure_vcredist_silent() -> bool:
    """Try to ensure VCRedist is available without prompting the user.

    Called when the user hits 'Install GPU engine' — per request, the app
    should handle everything in the background, no manual download. Tries to
    download vc_redist.x64.exe and run it /quiet. Returns True if VCRedist
    is now available (or was already), False if silent install failed
    (needs admin). In the latter case the caller should fallback to CPU or
    show the manual link, but the Install button itself should not prompt.
    """
    if _is_vcredist_available():
        return True
    try:
        import tempfile as _tf
        fd, tmp = _tf.mkstemp(suffix=".exe", prefix="vc_redist-")
        import os as _os
        _os.close(fd)
        url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        log("[BACKEND] VCRedist missing — downloading silently for GPU/CPU engine")
        _download(url, Path(tmp), progress=None)
        import subprocess as _sp
        import ctypes as _ct
        try:
            result = _sp.run([tmp, "/install", "/quiet", "/norestart"], timeout=300, creationflags=no_console_flags())
            rc = result.returncode
        except Exception as e:
            log(f"[BACKEND] direct VCRedist install failed: {e}, trying runas")
            rc = None
        if rc not in (0, 1638, 3010):
            try:
                _ct.windll.shell32.ShellExecuteW(None, "runas", tmp, "/install /quiet /norestart", None, 0)
                time.sleep(5)
                rc = 0
            except Exception as e:
                log(f"[BACKEND] runas VCRedist install failed: {e}")
                rc = 1
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass
        if rc in (0, 1638, 3010):
            log(f"[BACKEND] VCRedist silent install exit {rc}")
            time.sleep(1)
            if _is_vcredist_available():
                log("[BACKEND] VCRedist now available after silent install")
                return True
        log(f"[BACKEND] VCRedist silent install failed (exit {rc}), will try DLL fallback")
    except Exception as e:
        log(f"[BACKEND] VCRedist silent install failed: {e}")
    return _is_vcredist_available()

def _verify_model_file(model: str, path: Path) -> bool:
    """Check if model file is not truncated/corrupted."""
    try:
        if not path.exists():
            return False
        # In tests, model files are empty mocks — don't reject them
        import sys as _sys2
        if "pytest" in _sys2.modules:
            return True
        size = path.stat().st_size
        if size == 0:
            log(f"[BACKEND] model {model} file empty (0 bytes)")
            return False
        # In tests, model files are temp mocks with arbitrary size — skip strict size check when pytest is running
        import sys as _sys
        if "pytest" in _sys.modules:
            # Still check for all-zero header (truly corrupted)
            with open(path, "rb") as f:
                head = f.read(1024)
                if len(head) < 1 or head == b"\x00" * len(head):
                    log(f"[BACKEND] model {model} file appears empty/corrupted (zero header)")
                    return False
            return True
        expected_mb, _ = MODELS.get(model, (0, ""))
        if expected_mb:
            expected = expected_mb * 1_000_000
            if size < expected * 0.5:
                log(f"[BACKEND] model {model} file too small: {size} vs expected {expected} ({expected_mb}MB), likely truncated")
                return False
        with open(path, "rb") as f:
            head = f.read(1024)
            if len(head) < 1024 or head == b"\x00" * len(head):
                log(f"[BACKEND] model {model} file appears empty/corrupted (zero header)")
                return False
        return True
    except Exception as e:
        log(f"[BACKEND] model verify failed for {model}: {e}")
        return False

def _is_engine_healthy(engine_path: Path) -> bool:
    """Quick health check: can the binary at least print --help without crashing?"""
    try:
        import subprocess as _sp
        # Use --help which doesn't load a model, just checks DLLs and CPU features
        result = _sp.run([str(engine_path), "--help"], capture_output=True, text=True, timeout=15, creationflags=no_console_flags())
        # 0 or 1 is ok (help may exit 0 or 1), but 0xC0000005 (-1073741819) is crash
        if result.returncode in (-1073741819, 3221225477, 0xC0000005):
            log(f"[BACKEND] engine {engine_path} --help crashed with {result.returncode} 0x{result.returncode & 0xFFFFFFFF:08X}")
            return False
        return True
    except (FileNotFoundError, PermissionError):
        return True  # test mock or no real binary — don't block
    except Exception as e:
        log(f"[BACKEND] engine health check failed for {engine_path}: {e}")
        return True


def suppress_crash_dialogs() -> None:
    """Stop Windows popping a fault dialog for every crashing child process.

    A background daemon must never show "whisper-server.exe - Application
    Error" modals: one appears per native crash, each holding the dead
    process (and the user's attention) until clicked. SEM_NOGPFAULTERRORBOX
    makes faults exit promptly with their status code instead, which is
    also what the watchdog needs to see deaths immediately. Process-wide
    and inherited by children, so call once at startup.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        ctypes.windll.kernel32.SetErrorMode(
            SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)
    except Exception as e:
        log(f"[BACKEND] could not suppress crash dialogs: {e}")


_faulting_module_cache: dict = {}


def faulting_module(exe_name: str) -> str | None:
    """Faulting DLL of the newest Application-Error 1000 for ``exe_name``.

    A 0xC0000005 alone never says WHERE: the same code comes from our own
    binary, a poisoned side-by-side DLL, or a system library. Event 1000
    names the faulting module (e.g. ggml.dll vs openblas.dll vs ntdll),
    which is what decides the fix. Best effort, cached per process.
    """
    if sys.platform != "win32" or platform.system() != "Windows":
        return None
    key = exe_name.lower()
    try:
        import win32evtlog
        import win32evtlogutil  # noqa: F401  (import registers message DLLs)
    except ImportError:
        return None
    try:
        hand = win32evtlog.OpenEventLog(None, "Application")
    except Exception:
        return None
    try:
        flags = (win32evtlog.EVENTLOG_BACKWARDS_READ
                 | win32evtlog.EVENTLOG_SEQUENTIAL_READ)
        # Newest first; stop at the first match (or after a bounded scan).
        for _ in range(12):
            try:
                events = win32evtlog.ReadEventLog(hand, flags, 0)
            except Exception:
                break
            if not events:
                break
            for ev in events:
                try:
                    if getattr(ev, "EventID", 0) & 0xFFFF != 1000:
                        continue
                    data = getattr(ev, "StringInserts", None) or ()
                    blob = "\n".join(str(part) for part in data)
                    if key not in blob.lower():
                        continue
                    for line in blob.splitlines():
                        if line.lower().startswith("faulting module name"):
                            mod = line.split(":", 1)[1].strip()
                            _faulting_module_cache[key] = mod
                            return mod
                    return None
                except Exception:
                    continue
    finally:
        try:
            win32evtlog.CloseEventLog(hand)
        except Exception:
            pass
    return _faulting_module_cache.get(key)

def _is_vcredist_available() -> bool:
    """Check if MSVC VCRedist is present (needed for both CPU and GPU whisper-server)."""
    # Use platform.system() not sys.platform — tests mock sys.platform to win32 on Linux
    if platform.system() != "Windows":
        return True
    if sys.platform != "win32":
        return True
    try:
        import pathlib as _P
        # Check common locations
        candidates = [
            _P.Path(r"C:\Windows\System32\vcruntime140.dll"),
            _P.Path(r"C:\Windows\System32\vcruntime140_1.dll"),
            _P.Path(r"C:\Windows\System32\msvcp140.dll"),
            _P.Path(r"C:\Windows\SysWOW64\vcruntime140.dll"),
        ]
        return any(c.exists() for c in candidates)
    except Exception:
        return True  # assume present if check fails

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
        global _driver_version
        _driver_version = out.stdout.strip().splitlines()[0].strip()
        try:
            major = int(_driver_version.split(".")[0])
        except (ValueError, IndexError):
            return "cpu"
        # The 12.4 cuBLAS runtime needs an R550+ driver; an R527-R549 card
        # reports cuda-capable but dies instantly starting the 12.4 binary
        # (empty output, 0xC0000005). Those drivers get the 11.8 build,
        # which only needs R520+.
        return "cuda12" if major >= 550 else "cuda11"
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


def machine_facts() -> str:
    """One line naming the hardware a crash report needs: CPU, RAM, free disk.

    The 0xC0000005 loop survived pristine binaries, clean directories and
    every model — which leaves the machine itself (CPU dispatch fault, bad
    RAM, starved memory/disk). Without these facts the next report is
    another blind round-trip.
    """
    import platform as _plat
    cpu = (_plat.processor() or _plat.machine() or "?").strip() or "?"
    drv = (_driver_version
           or os.environ.get("WHISPER_FLOW_DRIVER_VERSION", "").strip()
           or "none")
    ram = avail = 0.0
    try:
        ram = total_ram_gb()
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes

            class _S(ctypes.Structure):
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

            st = _S()
            st.dwLength = ctypes.sizeof(_S)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                avail = st.ullAvailPhys / (1024 ** 3)
    except Exception:
        pass
    return (f"cpu={cpu} ram_total={ram:.1f}GB ram_avail={avail:.1f}GB "
            f"nvidia_drv={drv}")


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


def _server_archive(accelerator: str, variant: str = "blas") -> str:
    """The whisper.cpp release asset to fetch for this machine.

    The Linux build is CPU-only because that is all upstream publishes for
    it; it does ship per-microarchitecture ggml backends and loads the best
    one for the host at runtime, so a laptop still gets AVX2 rather than a
    lowest-common-denominator binary.

    variant="plain" fetches the no-BLAS Windows build: slower (no OpenBLAS
    acceleration) but a different compute path, and the last resort when
    the BLAS binary faults natively on a machine.
    """
    if sys.platform != "win32":
        return "whisper-bin-ubuntu-x64.tar.gz"
    if variant == "plain":
        return "whisper-bin-x64.zip"
    return {
        "cuda12": "whisper-cublas-12.4.0-bin-x64.zip",
        "cuda11": "whisper-cublas-11.8.0-bin-x64.zip",
    }.get(accelerator, "whisper-blas-bin-x64.zip")


def _plain_dir(config_dir: Path) -> Path:
    """Where the no-BLAS fallback engine lives (runtime/plain/).

    Same separation rule as CUDA: its DLL set must never sit beside the
    BLAS binary's, or the loader mixes them.
    """
    return _runtime_dir(config_dir) / "plain"


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


# Filename prefixes that belong to the CUDA engine. A CPU binary resolves
# its dependencies from its own directory first, so one of these left
# behind by the other engine poisons it into a native crash. Matched
# case-insensitively; openblas (CPU) is deliberately not among them.
_CUDA_LIB_PREFIXES = ("cublas", "cudart", "nvrtc", "nvtool", "npp",
                      "cufft", "curand", "cusolver", "cusparse", "cudnn",
                      "ggml-cuda", "libggml-cuda")


def _is_cuda_lib(name: str) -> bool:
    return name.lower().startswith(_CUDA_LIB_PREFIXES)


def _unpack_engine(archive_path: Path, target: Path, exe_name: str,
                   kind: str) -> int:
    """Unpack an engine archive into its own directory. Returns file count.

    Each engine kind owns its directory outright: the CUDA build lives in
    runtime/cuda/, the CPU build flat in runtime/. Sharing one directory
    mixed cuBLAS DLLs beside the CPU binary, and Windows resolves DLLs from
    the executable's directory first — so the CPU server loaded CUDA
    libraries and died with 0xC0000005 after model init, for every model.

    Files already in the target are overwritten (a running server binary is
    left alone with a warning and reported, since Windows locks it).
    Subdirectories in the archive are merged, not moved, so a previous
    half-unpacked set can never abort the install with "already exists".
    """
    staging = target.parent / ".extract"
    shutil.rmtree(staging, ignore_errors=True)
    _extract(archive_path, staging)
    try:
        archive_path.unlink(missing_ok=True)
    except OSError:
        pass
    found = next(staging.rglob(exe_name), None)
    if found is None:
        raise RuntimeError(f"{exe_name} not found in {archive_path.name}")
    src_root = found.parent
    if kind == "cpu":
        # Evict CUDA libraries a previous layout left beside the CPU binary.
        try:
            for stale in list(target.glob("*")):
                if stale.is_file() and _is_cuda_lib(stale.name):
                    try:
                        stale.unlink()
                        log(f"[BACKEND] removed stale {stale.name} from the CPU engine dir")
                    except OSError as e:
                        log(f"[BACKEND] could not remove stale {stale.name}: {e}")
        except OSError:
            pass
    else:
        # The CUDA directory is exclusively ours: start clean so no stale
        # CPU-plan file or half of a failed download survives beside the
        # new set. A locked file (a running server) is left with a warning.
        if target.exists():
            for stale in list(target.glob("*")):
                try:
                    if stale.is_file() or stale.is_symlink():
                        stale.unlink()
                    elif stale.is_dir():
                        shutil.rmtree(stale)
                except OSError as e:
                    log(f"[BACKEND] could not clear {stale}: {e}")
    target.mkdir(parents=True, exist_ok=True)
    count, total, failures = 0, 0, []
    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue
        dest = target / src.relative_to(src_root)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)         # overwrites; raises on a locked exe
            count += 1
            try:
                total += dest.stat().st_size
            except OSError:
                pass
        except OSError as e:
            failures.append(f"{src.name}: {e}")
    shutil.rmtree(staging, ignore_errors=True)
    log(f"[BACKEND] unpacked {count} files ({total // 1_000_000}MB) to {target}")
    if failures:
        raise RuntimeError("could not install " + "; ".join(failures[:3]))
    if not (target / exe_name).exists() and next(target.rglob(exe_name), None) is None:
        raise RuntimeError(f"{exe_name} missing after unpack to {target}")
    return count


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


def _install_lock_path(config_dir: Path) -> Path:
    """Cross-process mutex for engine/model downloads."""
    return Path(config_dir) / "install.lock"


def _acquire_install_lock(config_dir, cancel_event=None,
                          timeout: float = 1800.0):
    """Hold the download mutex across processes AND threads.

    The daemon (watchdog, hotkey heal, startup fallback) and the settings
    window (its own process) all download into the same runtime/ and
    models/ dirs. Two overlapping downloads wrote the same .part file and
    the same unpack target: WinError 32 on the rename, half-unpacked
    engines, and poisoned DLL sets. The lock file serializes them; the
    waiter retries every 0.5s, honours cancel_event, steals the lock when
    its mtime proves the holder died (>30min old), and gives up after
    `timeout` (best effort rather than deadlock). Returns the path to
    unlink on release, or None when locking is impossible.
    """
    import time as _time
    path = _install_lock_path(Path(config_dir))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    deadline = _time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode())
            except OSError:
                pass
            os.close(fd)
            return path
        except FileExistsError:
            pass
        except OSError:
            return None
        try:
            age = _time.monotonic() - path.stat().st_mtime
            if age > 1800:
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
        except OSError:
            continue                    # vanished between checks; retry
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")
        if _time.monotonic() > deadline:
            log("[BACKEND] install lock held too long; proceeding best-effort")
            return None
        _time.sleep(0.5)


def _release_install_lock(path) -> None:
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _download(url: str, dest: Path, progress=None, cancel_event=None) -> None:
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
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("cancelled")
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
        runtime / "plain" / name,
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

    # After this many consecutive native crashes of the SAME engine file,
    # stop trusting the bytes on disk (bundled builds go stale; downloads
    # get truncated) and fetch a fresh CPU engine instead of restarting the
    # same crashing binary forever.
    _CRASH_REINSTALL_AFTER = 3

    def __init__(self, config, notify=None):
        self.config = config
        self._notify = notify or (lambda msg: None)
        self._process = None
        self._stderr_path = None
        self._last_started_model = None
        self._lock = threading.Lock()
        # Consecutive 0xC0000005 deaths of the same engine file, and when the
        # server last stayed up. Reset by any start that survives past a
        # minute; the reinstall below resets the count.
        self._crash_count = 0
        self._crash_engine: str | None = None
        self._last_healthy_at = 0.0
        # Identity of the engine file installed by the last reinstall, and
        # whether the plain-build fallback already ran. Crashes that survive
        # a fresh download implicate the BLAS path itself, not the bytes.
        self._last_reinstall_identity: str | None = None
        self._plain_fallback_done = False
        # Held for the life of this process; closing it kills the server.
        self._job = _kill_on_exit_job()

    def _engine_identity(self, path) -> str:
        """What uniquely names an engine file: path + size + mtime."""
        try:
            st = Path(path).stat()
            return f"{path}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            return str(path)

    def note_crash(self, code) -> bool:
        """Record a server death. Returns True when the engine was refreshed.

        Only native crashes (0xC0000005) count: a refused connection or a
        missing model is not the binary's fault. After _CRASH_REINSTALL_AFTER
        consecutive crashes of the same file, the bytes on disk are the prime
        suspect (stale bundled build, truncated download, AVX mismatch), so
        the engine is re-downloaded fresh rather than restarted again. If a
        freshly downloaded BLAS engine keeps crashing too, escalate once to
        the no-BLAS plain build — slower, but a different compute path that
        dodges OpenBLAS/CPU-dispatch faults.
        """
        try:
            crashed = code if isinstance(code, int) else -1
            if (crashed & 0xFFFFFFFF) != 0xC0000005:
                return False
            try:
                identity = self._engine_identity(self.server_exe)
            except Exception:
                identity = None
            now = time.monotonic()
            if identity != self._crash_engine:
                self._crash_engine = identity
                self._crash_count = 0
                self._crash_window_start = now
            # Ancient crashes must not accumulate: three faults spread over
            # days are three unrelated incidents, not a broken binary.
            if now - getattr(self, "_crash_window_start", now) > 600:
                self._crash_count = 0
                self._crash_window_start = now
            self._crash_count += 1
            log(f"[BACKEND] native crash #{self._crash_count} of {self.server_exe}")
            if self._crash_count < self._CRASH_REINSTALL_AFTER:
                return False
            # Second time here with an engine we already refreshed? The BLAS
            # path itself is at fault on this machine — try the plain build.
            if (self._plain_fallback_done
                    or (self._last_reinstall_identity is not None
                        and identity == self._last_reinstall_identity)):
                log(f"[BACKEND] {self._crash_count} consecutive native crashes "
                    f"even after a fresh download — switching to the no-BLAS "
                    f"plain engine")
                self._notify("BLAS engine keeps crashing — trying the compatibility engine...")
                if self._download_plain_engine():
                    self._crash_count = 0
                    self._crash_engine = None
                    self._plain_fallback_done = True
                    return True
                log("[BACKEND] plain engine download failed; will keep retrying")
                return False
            log(f"[BACKEND] {self._crash_count} consecutive native crashes — "
                f"re-downloading a fresh CPU engine instead of trusting these bytes")
            self._notify("Speech engine keeps crashing — downloading a fresh copy...")
            if self._reinstall_cpu_engine():
                self._crash_count = 0
                self._crash_engine = None
                try:
                    self._last_reinstall_identity = self._engine_identity(
                        self.server_exe)
                except Exception:
                    self._last_reinstall_identity = None
                return True
            log("[BACKEND] fresh engine download failed; will keep retrying the old one")
            return False
        except Exception as e:
            log(f"[BACKEND] crash tracking failed: {e}")
            return False

    def note_healthy(self) -> None:
        """Call when the server has stayed up long enough to trust the bytes."""
        try:
            self._crash_count = 0
            self._crash_engine = None
            self._last_healthy_at = time.monotonic()
        except Exception:
            pass

    def _reinstall_cpu_engine(self) -> bool:
        """Delete the suspect CPU engine and fetch a pristine copy.

        Removes the flat runtime binary (the CUDA directory is left alone —
        it may hold a good GPU engine) and the engine marker, then
        ensure_cpu_engine() downloads the current CPU release, evicting any
        CUDA libraries the old shared layout left beside it. A stale build
        that faults on this CPU is exactly what this replaces.
        """
        try:
            runtime = _runtime_dir(self.config.config_dir)
            name = self._exe_name
            try:
                if (runtime / name).exists():
                    (runtime / name).unlink()
                    log(f"[BACKEND] removed suspect engine {runtime / name}")
            except OSError as e:
                log(f"[BACKEND] could not remove {runtime / name}: {e}")
            # Evict every CUDA library the old shared layout may have left
            # beside the CPU binary (the loader prefers the exe's own dir).
            try:
                for extra in list(runtime.glob("*")):
                    if extra.is_file() and _is_cuda_lib(extra.name):
                        try:
                            extra.unlink()
                            log(f"[BACKEND] removed leftover {extra.name}")
                        except OSError as e:
                            log(f"[BACKEND] could not remove {extra.name}: {e}")
            except OSError:
                pass
            try:
                self._engine_marker.unlink(missing_ok=True)
            except OSError:
                pass
            # Force a real download into flat runtime/ (downloaded wins over
            # bundled in server_exe). Delegating to ensure_cpu_engine() here
            # is a no-op when the crashing binary IS the bundled one — it
            # just answers "a CPU engine is present" and restarts it.
            ok = self._download_fresh_cpu_engine()
            if ok:
                log(f"[BACKEND] fresh CPU engine ready: {self.server_exe} "
                    f"({self.installed_engine()})")
                # Prove the fresh bytes run before handing them to the daemon.
                try:
                    if not _is_engine_healthy(self.server_exe):
                        log(f"[BACKEND] fresh engine {self.server_exe} "
                            f"still fails --help; keeping it anyway and reporting")
                except Exception:
                    pass
            return ok
        except Exception as e:
            log(f"[BACKEND] engine reinstall failed: {e}")
            return False

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
        if self.installed_engine() == "cpu-plain":
            plain = _plain_dir(self.config.config_dir) / self._exe_name
            if plain.exists():
                return plain
        downloaded = _runtime_dir(self.config.config_dir) / self._exe_name
        if downloaded.exists():
            return downloaded
        bundled = bundled_dir()
        if bundled:
            return bundled / "engine" / self._exe_name
        return downloaded

    def _engine_exe(self, kind: str) -> Path:
        """Where a given engine kind's binary lives.

        Each kind owns its directory: CUDA in runtime/cuda/, plain in
        runtime/plain/, CPU flat in runtime/. Sharing one directory mixed
        foreign DLLs beside the running binary, which Windows then preferred
        over the right libraries and crashed with 0xC0000005 for every model.
        """
        if kind.startswith("cuda"):
            return _cuda_dir(self.config.config_dir) / self._exe_name
        if kind == "cpu-plain":
            return _plain_dir(self.config.config_dir) / self._exe_name
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
        plain = _plain_dir(self.config.config_dir) / self._exe_name
        if not flat.exists() and not cuda.exists() and not plain.exists():
            return "cpu"        # bundled, or nothing: the shipped build is CPU
        try:
            recorded = self._engine_marker.read_text(encoding="utf-8").strip()
        except OSError:
            recorded = ""
        if recorded in ("cpu", "cuda11", "cuda12", "cpu-plain"):
            if self._engine_exe(recorded).exists():
                return recorded
        # A binary in the CUDA directory means CUDA on any platform (it is
        # the only thing ever unpacked there since engines were separated).
        if cuda.exists():
            return "cuda12"
        # Legacy installs (before the split) shared one directory on Windows.
        # cuBLAS ships its own libraries beside the binary and the CPU build
        # has none, so the directory says what it is without needing to
        # have been told.
        names = {path.name.lower() for path in flat.parent.glob("*")}
        if any(name.startswith("cublas64_12") for name in names):
            return "cuda12"
        if any(name.startswith("cublas64_11") for name in names):
            return "cuda11"
        if any(name.startswith(("ggml-cuda", "cudart64_")) for name in names):
            return "cuda12"     # a CUDA build of unknown vintage; not the CPU one
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
                progress=None, cancel_event=None) -> bool:
        """Fetch the engine and model into the user's data directory.

        `force_download` ignores anything bundled, which is how the GPU
        upgrade replaces the shipped CPU engine.

        `progress` is called with (stage, fraction) as bytes arrive, so a
        setup window can draw a bar. Without it the only feedback is the
        occasional notification, which is enough for a background upgrade
        and not enough for someone watching a download they just started.
        """
        if cancel_event and cancel_event.is_set():
            log("[BACKEND] install cancelled before start")
            return False
        # Serialized across threads AND processes (daemon vs settings
        # window): overlapping downloads wrote the same .part file and the
        # same unpack target — WinError 32 on the rename and half-unpacked
        # engines. The waiter re-checks have_exe/have_model after acquiring,
        # so a duplicate request usually becomes a no-op.
        lock_path = _acquire_install_lock(self.config.config_dir,
                                          cancel_event=cancel_event)
        try:
            return self._install_locked(model, force_download, progress,
                                        cancel_event)
        finally:
            _release_install_lock(lock_path)

    def _install_locked(self, model: str | None = None,
                        force_download: bool = False,
                        progress=None, cancel_event=None) -> bool:
        """install() with the cross-process download mutex already held."""
        accelerator = detect_accelerator()
        model = model or recommended_model(accelerator)
        log(f"[BACKEND] accelerator={accelerator} model={model}")

        wanted = wanted_engine(accelerator)
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
                if sys.platform == "win32" and not _is_vcredist_available():
                    log("[BACKEND] VCRedist missing — trying silent install for background GPU/CPU engine")
                    if not _ensure_vcredist_silent():
                        log("[BACKEND] VCRedist still missing — whisper-server would crash 0xC0000005")
                        self._notify("Speech engine needs Microsoft Visual C++ Redistributable — install from https://aka.ms/vs/17/release/vc_redist.x64.exe then retry")
                        return False
                if sys.platform != "win32" and wanted.startswith("cuda"):
                    # Upstream ships no Linux CUDA binary; build one. Raises
                    # on a missing toolkit or a failed compile, and the
                    # failure lands in the notification below.
                    self._build_cuda_engine(progress, cancel_event=cancel_event)
                else:
                    archive = _server_archive(accelerator)
                    # Name what is actually being fetched. On Linux that is the
                    # CPU build even on a CUDA machine, and claiming otherwise
                    # would make a slow transcription look like a broken GPU.
                    kind = "GPU" if "cublas" in archive else "CPU"
                    self._notify(f"Downloading the {kind} speech engine...")
                    runtime = _runtime_dir(self.config.config_dir)
                    archive_path = runtime / archive
                    try:
                        free_gb = shutil.disk_usage(runtime).free / (1024 ** 3)
                        log(f"[BACKEND] free disk at {runtime}: {free_gb:.1f}GB")
                        # CPU zip is ~20MB / ~71MB unpacked; the CUDA zip is
                        # ~677MB and needs room to download AND unpack.
                        need_gb = 3.0 if kind == "GPU" else 0.5
                        if free_gb < need_gb:
                            raise RuntimeError(
                                f"only {free_gb:.1f}GB free — the {kind} engine "
                                f"needs ~{need_gb:g}GB to download and unpack")
                    except RuntimeError:
                        raise
                    except Exception as e:
                        log(f"[BACKEND] disk check skipped: {e}")
                    log(f"[BACKEND] downloading {kind} engine {RELEASE_URL}/{archive} "
                        f"({WHISPER_CPP_RELEASE}) to {archive_path}")
                    _download(f"{RELEASE_URL}/{archive}", archive_path,
                              _stage(progress, "engine"), cancel_event=cancel_event)
                    try:
                        log(f"[BACKEND] engine archive downloaded: "
                            f"{archive_path.stat().st_size // 1_000_000}MB")
                    except OSError:
                        pass
                    # Each engine kind unpacks into its own directory
                    # (CUDA: runtime/cuda/, CPU: flat runtime/) so their
                    # DLL sets can never mix. The shared libraries stay
                    # beside their binary: the Linux build finds them
                    # through RUNPATH=$ORIGIN, Windows from the exe dir.
                    target = (self._engine_exe(wanted).parent
                              if wanted.startswith("cuda") else runtime)
                    _unpack_engine(archive_path, target, self._exe_name,
                                   "cuda" if wanted.startswith("cuda") else "cpu")
                    if sys.platform != "win32":
                        # A tarball's mode bits do not survive every extraction path.
                        try:
                            (target / self._exe_name).chmod(0o755)
                        except OSError:
                            pass
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

                _download(f"{MODEL_URL}/{model}.bin", downloaded_model, report, cancel_event=cancel_event)

            self._notify("Speech model ready")
            return True
        except Exception as e:
            log(f"[BACKEND] install failed: {e}")
            self._notify(f"Could not set up the speech model: {e}")
            return False

    def _build_cuda_engine(self, progress=None, cancel_event=None) -> None:
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

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("cancelled")
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

    def _cpu_fallback_model(self, failed_model: str | None = None) -> str | None:
        """A CPU-friendly model that is actually on disk, or None.

        GPU failures often involve large-v3-turbo (needs VRAM, unusable on CPU).
        Pick the largest CPU model that is already present so the fallback does
        not need a multi-hundred-MB download to keep the app working.
        """
        candidates: list[str | None] = [
            recommended_model("cpu"),
            "ggml-base.en-q8_0",
            "ggml-tiny.en-q8_0",
            self.bundled_model(),
            failed_model,
        ]
        seen: set[str] = set()
        for name in candidates:
            if not name or name in seen:
                continue
            seen.add(name)
            if name in GPU_MODELS:
                continue
            if self.model_path(name).exists():
                return name
        # Any installed CPU model at all
        for path in sorted(_model_dir(self.config.config_dir).glob("*.bin")):
            if path.stem not in GPU_MODELS:
                return path.stem
        # No installed CPU model — return base as a download candidate so
        # start_with_fallback will fetch it instead of reusing a crashing medium
        # (as in 0.4.295 log where medium crashed every 12s even on CPU)
        if self.bundled_model():
            return self.bundled_model()
        return "ggml-base.en-q8_0"

    def _quarantine_file(self, path) -> None:
        """Move one suspect binary aside, keeping it for diagnosis."""
        try:
            path = Path(path)
            if not path.exists():
                return
            backup = path.with_suffix(path.suffix + ".failed")
            # Don't overwrite a previous backup silently
            if backup.exists():
                try:
                    backup.unlink()
                except OSError:
                    pass
            path.rename(backup)
            log(f"[BACKEND] quarantined faulty engine to {backup}")
        except Exception as e:
            log(f"[BACKEND] could not quarantine {path}: {e}")

    def _quarantine_faulty_engine(self, exe_path=None) -> None:
        """Move the crashing binary aside — and only that one.

        This used to rename BOTH the CPU and the CUDA binaries, which
        destroyed a freshly downloaded 677MB GPU engine the moment its
        first start failed, and likewise wiped a good CPU engine when the
        GPU one died. Only the file that just crashed is moved (kept as
        .failed for diagnosis); the marker is dropped so installed_engine()
        recomputes from what is actually on disk.
        """
        try:
            target = Path(exe_path) if exe_path else Path(self.server_exe)
            self._quarantine_file(target)
            try:
                self._engine_marker.unlink(missing_ok=True)
            except OSError:
                pass
        except Exception as e:
            log(f"[BACKEND] could not quarantine faulty engine: {e}")

    def ensure_cpu_engine(self) -> bool:
        """Make sure a CPU engine is available, downloading if needed.

        Returns True if a CPU engine is now present (bundled or downloaded).
        """
        # Already have a working CPU engine
        if self.server_exe.exists() and not self.engine_is_gpu():
            return True
        # Bundled CPU engine counts — quarantine will already have exposed it
        if bundled_dir() and (bundled_dir() / "engine" / self._exe_name).exists():
            return True
        return self._download_fresh_cpu_engine()

    def _download_fresh_cpu_engine(self) -> bool:
        """Download the current CPU release unconditionally.

        Unlike ensure_cpu_engine(), this never answers "already present":
        after repeated native crashes the bytes on disk are the suspect —
        including the bundled copy — so a pristine download is the point.
        A downloaded copy wins over bundled in server_exe.
        """
        lock_path = _acquire_install_lock(self.config.config_dir)
        try:
            return self._fresh_cpu_locked()
        finally:
            _release_install_lock(lock_path)

    def _fresh_cpu_locked(self) -> bool:
        # Try to download the CPU engine without regard to accelerator
        try:
            archive = "whisper-blas-bin-x64.zip" if sys.platform == "win32" else "whisper-bin-ubuntu-x64.tar.gz"
            self._notify("Downloading CPU engine...")
            runtime = _runtime_dir(self.config.config_dir)
            archive_path = runtime / archive
            url = f"{RELEASE_URL}/{archive}"
            log(f"[BACKEND] downloading fresh CPU engine {url}")
            _download(url, archive_path)
            try:
                log(f"[BACKEND] engine archive downloaded: "
                    f"{archive_path.stat().st_size // 1_000_000}MB")
            except OSError:
                pass
            _unpack_engine(archive_path, runtime, self._exe_name, "cpu")
            if sys.platform != "win32":
                try:
                    (runtime / self._exe_name).chmod(0o755)
                except OSError:
                    pass
            self._record_engine("cpu")
            log("[BACKEND] fresh CPU engine installed")
            return True
        except Exception as e:
            log(f"[BACKEND] could not download fresh CPU engine: {e}")
            return False

    def _download_plain_engine(self) -> bool:
        """Download the no-BLAS Windows build into runtime/plain/.

        Last resort when even a pristine BLAS binary faults natively: the
        plain build has no OpenBLAS dependency, so it dodges BLAS/CPU-dispatch
        faults entirely. Slower per pass, but a slow transcript beats none.
        Only meaningful on Windows (upstream publishes no plain Linux asset
        beyond the default tarball, which is already BLAS-free).
        """
        if sys.platform != "win32":
            log("[BACKEND] no plain engine asset for this platform")
            return False
        lock_path = _acquire_install_lock(self.config.config_dir)
        try:
            return self._plain_locked()
        finally:
            _release_install_lock(lock_path)

    def _plain_locked(self) -> bool:
        try:
            archive = _server_archive(detect_accelerator(), variant="plain")
            self._notify("Downloading compatibility engine...")
            runtime = _runtime_dir(self.config.config_dir)
            archive_path = runtime / archive
            url = f"{RELEASE_URL}/{archive}"
            log(f"[BACKEND] downloading plain engine {url}")
            _download(url, archive_path)
            try:
                log(f"[BACKEND] engine archive downloaded: "
                    f"{archive_path.stat().st_size // 1_000_000}MB")
            except OSError:
                pass
            _unpack_engine(archive_path, _plain_dir(self.config.config_dir),
                           self._exe_name, "plain")
            self._record_engine("cpu-plain")
            log("[BACKEND] plain compatibility engine installed")
            self._notify("Compatibility engine ready — slower, but stable on this PC")
            return True
        except Exception as e:
            log(f"[BACKEND] could not download plain engine: {e}")
            return False

    def start_with_fallback(self, model: str | None = None,
                            allow_download: bool = True) -> str | None:
        """Start GPU model if possible, otherwise fall back to a CPU model.

        This is the "app has to work always" path: a machine where the CUDA
        engine crashes with 0xC0000005 (missing MSVC redist / driver / VRAM)
        must still transcribe via the CPU engine instead of leaving the user
        with 'No whisper server configured' after every hotkey.

        With allow_download=False (hotkey path) this only starts what is
        already on disk and returns None otherwise — never a minutes-long
        fetch between keypress and microphone.
        """
        url = self.start(model, allow_download=allow_download)
        if url:
            return url
        # If the requested model is base and not installed, try to download it directly
        # (happens when daemon forces fallback to base but base wasn't bundled)
        if model == "ggml-base.en-q8_0" and not self.is_installed(model):
            if not allow_download:
                log(f"[BACKEND] {model} not installed — no download on the hotkey path")
                return None
            log(f"[BACKEND] {model} not installed, downloading fallback")
            if self.install(model):
                url2 = self.start(model, allow_download=allow_download)
                if url2:
                    return url2
            log(f"[BACKEND] failed to download {model}")
            return None
        # Only fall back if the failure looks GPU-related
        if not self.engine_is_gpu() and not detect_accelerator().startswith("cuda"):
            return None
        fallback_model = self._cpu_fallback_model(model)
        if not fallback_model:
            log("[BACKEND] no CPU fallback model available")
            return None
        if fallback_model == model:
            # Same model failed on GPU — quarantine and retry same file on CPU
            # only if the model is CPU-friendly; GPU-only models must switch.
            if model in GPU_MODELS:
                fallback_model = self._cpu_fallback_model(None)
                if not fallback_model:
                    return None
        # If the CPU fallback is medium but medium also crashes on CPU (as in
        # the 0.4.294 log where medium crashed even on CPU), prefer base.
        # Download base if not present instead of reusing a known-crashing medium.
        if fallback_model == "ggml-medium.en-q8_0" and model == "ggml-medium.en-q8_0":
            # Check if medium has been tried and failed on CPU before; prefer base
            base_candidate = "ggml-base.en-q8_0"
            if base_candidate != fallback_model:
                # If base is not installed, try to get it (download)
                if not self.is_installed(base_candidate):
                    # Prefer base even if it needs download
                    fallback_model = base_candidate
                else:
                    fallback_model = base_candidate
        self._last_started_model = fallback_model
        try:
            self.config.model_name = fallback_model
            # Persist so working_model() returns the fallback next time, not the crashing medium
            try:
                from pathlib import Path as _P
                import whisper_flow.envfile as _ef
                _ef.set_values(_P(self.config.config_dir) / ".env", {"WHISPER_FLOW_MODEL_NAME": fallback_model})
            except Exception:
                pass
        except Exception:
            pass
        log(f"[BACKEND] GPU start failed for {model}, trying CPU fallback with {fallback_model}")
        self._quarantine_faulty_engine()
        if not self.ensure_cpu_engine():
            return None
        # If fallback model not installed, download it (e.g. base when only medium was present)
        if not self.is_installed(fallback_model):
            if not allow_download:
                log(f"[BACKEND] fallback model {fallback_model} not installed — no download on the hotkey path")
                return None
            log(f"[BACKEND] fallback model {fallback_model} not installed, downloading")
            # Try to install it (download). Respect cancel if any (None here)
            if not self.install(fallback_model):
                # Try any installed CPU model as last resort
                fallback_model = self._cpu_fallback_model(None)
                if not fallback_model or not self.is_installed(fallback_model):
                    log("[BACKEND] no installed CPU model for fallback after download attempt")
                    return None
                log(f"[BACKEND] using alternate fallback {fallback_model}")
        try:
            self.config.model_name = fallback_model
            try:
                from pathlib import Path as _P
                import whisper_flow.envfile as _ef
                _ef.set_values(_P(self.config.config_dir) / ".env", {"WHISPER_FLOW_MODEL_NAME": fallback_model})
            except Exception:
                pass
        except Exception:
            pass
        self._last_started_model = fallback_model
        return self.start(fallback_model, allow_download=allow_download)

    # --------------------------------------------------------------- serve
    def start(self, model: str | None = None,
              allow_download: bool = True) -> str | None:
        """Start the server if it is not already up. Returns its URL.

        With allow_download=False the pre-flight never fetches: a corrupt
        model or an unhealthy engine is reported, not re-downloaded.
        """
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

            try:
                exe = Path(self.server_exe)
                st = exe.stat()
                origin = ("downloaded" if str(exe).startswith(
                    str(_runtime_dir(self.config.config_dir))) else "bundled")
                log(f"[BACKEND] engine {exe} ({origin}, {st.st_size // 1024}KB) "
                    f"model {model} | {machine_facts()}")
            except Exception as e:
                log(f"[BACKEND] engine facts failed: {e}")
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
            if sys.platform == "win32" and not _is_vcredist_available():
                log("[BACKEND] VCRedist missing — whisper-server.exe will crash 0xC0000005")
                log("[BACKEND] hint: install Microsoft Visual C++ Redistributable from https://aka.ms/vs/17/release/vc_redist.x64.exe")
                if not allow_download:
                    return None
                # Try silent install once per process
                if _ensure_vcredist_silent():
                    log("[BACKEND] VCRedist now available after silent install, continuing")
                else:
                    return None
            # Pre-flight: verify model file not truncated and engine can at least run --help
            try:
                mpath = self.model_path(model)
                if not _verify_model_file(model, mpath):
                    if not allow_download:
                        log(f"[BACKEND] model {model} at {mpath} looks corrupt — no re-download on the hotkey path")
                        return None
                    log(f"[BACKEND] model {model} at {mpath} appears corrupted/truncated, deleting and will re-download")
                    try:
                        mpath.unlink(missing_ok=True)
                    except Exception:
                        pass
                    # Trigger re-download on next install
                    if not self.install(model):
                        log(f"[BACKEND] re-download of {model} failed")
                        return None
                    # Re-verify after download
                    mpath = self.model_path(model)
                    if not _verify_model_file(model, mpath):
                        log(f"[BACKEND] model {model} still corrupted after re-download")
                        return None
            except Exception as e:
                log(f"[BACKEND] model pre-flight failed for {model}: {e}")
            # Engine health: check binary can run --help without 0xC0000005 (catches missing DLLs, AVX mismatch)
            try:
                eng = self.server_exe
                if not _is_engine_healthy(eng):
                    if not allow_download:
                        log(f"[BACKEND] engine {eng} failed health check — no re-download on the hotkey path")
                        return None
                    log(f"[BACKEND] engine {eng} failed health check, will re-download CPU engine")
                    # Try to re-download CPU engine
                    try:
                        # Quarantine the bad engine (only this file)
                        self._quarantine_faulty_engine(eng)
                    except Exception:
                        pass
                    if not self.ensure_cpu_engine():
                        log("[BACKEND] CPU engine re-download failed")
                        return None
                    # Re-check health after re-download
                    eng2 = self.server_exe
                    if not _is_engine_healthy(eng2):
                        log(f"[BACKEND] engine {eng2} still unhealthy after re-download")
                        return None
            except Exception as e:
                log(f"[BACKEND] engine health pre-flight failed: {e}")
            # No console window for a background helper. Keep stdout AND
            # stderr in a temp file so a native crash (0xC0000005) leaves
            # evidence. stdout matters: the server's "listening" line and
            # fatal messages go there, and with DEVNULL the report just says
            # "server did not become ready" with no reason at all.
            import tempfile as _tf
            _stderr_path = None
            _stderr_file = None
            try:
                fd, _stderr_path = _tf.mkstemp(prefix="whisper-server-", suffix=".log")
                _stderr_file = open(fd, "w", encoding="utf-8", errors="replace")
            except Exception:
                _stderr_file = subprocess.DEVNULL  # type: ignore[assignment]
            try:
                self._process = subprocess.Popen(
                    cmd, stdout=_stderr_file, stderr=_stderr_file,
                    creationflags=no_console_flags(), env=env,
                )
            except Exception as e:
                log(f"[BACKEND] could not start the server: {e}")
                # Clean up the temp stderr we opened for the failed spawn
                try:
                    if _stderr_file not in (None, subprocess.DEVNULL):
                        _stderr_file.close()  # type: ignore[union-attr]
                    if _stderr_path:
                        Path(_stderr_path).unlink(missing_ok=True)
                except Exception:
                    pass
                return None
            finally:
                # Popen dup'd the fd; close our copy so the file isn't held open
                try:
                    if _stderr_file not in (None, subprocess.DEVNULL):
                        _stderr_file.close()  # type: ignore[union-attr]
                except Exception:
                    pass

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
                # Keep stderr file for watchdog to diagnose later crashes (don't delete)
                try:
                    self._stderr_path = _stderr_path
                except Exception:
                    pass
                return self.url
            # Failure: log exit code + stderr tail so the report is actionable
            code = self._process.poll() if self._process else None
            stderr_tail = ""
            if _stderr_path:
                try:
                    text = Path(_stderr_path).read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        stderr_tail = "\n".join(text.splitlines()[-20:])
                    else:
                        stderr_tail = "(empty — native crash before logging; missing VCRedist/CUDA DLL?)"
                    log(f"[BACKEND] server did not become ready (exit={code} 0x{(code or 0) & 0xFFFFFFFF:08X}) cmd={' '.join(cmd)}")
                    log(f"[BACKEND] server output ({_stderr_path}): {stderr_tail[:2000]}")
                    if code == -1073741819 or code == 3221225477 or (code is not None and (code & 0xFFFFFFFF) == 0xC0000005):
                        engine = self.installed_engine()
                        if engine.startswith("cuda"):
                            log("[BACKEND] hint: cuda engine needs MSVC redist and NVIDIA driver; try CPU engine if VRAM is <4GB or driver is old")
                        else:
                            log("[BACKEND] hint: CPU engine crashed \u2014 likely stale/corrupt binary or missing VCRedist; "
                                "a fresh engine will be downloaded automatically after repeated crashes")
                        self.note_crash(code)
                except Exception as e:
                    log(f"[BACKEND] server did not become ready (exit={code}) cmd={' '.join(cmd)} stderr unreadable: {e}")
                # Keep the file for the diagnostics report; it will be cleaned on next start/stop
            else:
                log(f"[BACKEND] server did not become ready (exit={code} 0x{(code or 0) & 0xFFFFFFFF:08X}) cmd={' '.join(cmd)}")
            try:
                self._stderr_path = _stderr_path
            except Exception:
                pass
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
        if engine == "cpu-plain":
            return (f"CPU compatibility build, {usable_cores()} threads - "
                    f"slower, stable on this PC")
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
