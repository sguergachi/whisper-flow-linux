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

import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from .logging import log

WHISPER_CPP_RELEASE = "v1.9.1"
RELEASE_URL = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/" + WHISPER_CPP_RELEASE
)
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# name -> (approximate download size MB, VRAM/RAM it wants)
MODELS = {
    "ggml-large-v3-turbo": (1624, "~2GB VRAM"),
    "ggml-medium.en": (1530, "~1.5GB"),
    "ggml-small.en": (466, "~600MB"),
    "ggml-base.en": (142, "~250MB"),
}


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


def detect_accelerator() -> str:
    """What this machine can run whisper.cpp on: 'cuda12', 'cuda11' or 'cpu'."""
    if not shutil.which("nvidia-smi"):
        return "cpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return "cpu"
        major = int(out.stdout.strip().split(".")[0])
        # CUDA 12 needs a 525+ driver on Windows; below that use the 11.8 build.
        return "cuda12" if major >= 527 else "cuda11"
    except Exception:
        return "cpu"


def recommended_model(accelerator: str) -> str:
    """Largest model the machine can run at a sensible speed."""
    if accelerator.startswith("cuda"):
        return "ggml-large-v3-turbo"
    # On CPU, anything larger than this transcribes slower than speech.
    return "ggml-small.en"


def _server_archive(accelerator: str) -> str:
    return {
        "cuda12": "whisper-cublas-12.4.0-bin-x64.zip",
        "cuda11": "whisper-cublas-11.8.0-bin-x64.zip",
    }.get(accelerator, "whisper-blas-bin-x64.zip")


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


class LocalBackend:
    """Downloads, starts and supervises a local whisper.cpp server."""

    def __init__(self, config, notify=None):
        self.config = config
        self._notify = notify or (lambda msg: None)
        self._process = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- paths
    @property
    def _exe_name(self) -> str:
        return "whisper-server.exe" if sys.platform == "win32" else "whisper-server"

    @property
    def server_exe(self) -> Path:
        """The downloaded engine if there is one, else the bundled engine.

        Downloaded wins: it is only ever present because this machine has a
        GPU and the better engine was fetched for it.
        """
        downloaded = _runtime_dir(self.config.config_dir) / self._exe_name
        if downloaded.exists():
            return downloaded
        bundled = bundled_dir()
        if bundled:
            return bundled / "engine" / self._exe_name
        return downloaded

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

    def working_model(self) -> str | None:
        """The best model actually present.

        A downloaded model outranks the bundled one, and outranks the
        configured name too: a model is only ever downloaded because someone
        pressed the button asking for it. Relying on the config file instead
        would lose that choice, because pydantic resolves the .env path once
        at import - so a file the setup window writes afterwards is not read
        back until the next launch.
        """
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
        if sys.platform != "win32":
            log("[BACKEND] Automatic install is Windows-only for now")
            return False

        accelerator = detect_accelerator()
        model = model or recommended_model(accelerator)
        log(f"[BACKEND] accelerator={accelerator} model={model}")

        downloaded_exe = _runtime_dir(self.config.config_dir) / self._exe_name
        downloaded_model = _model_dir(self.config.config_dir) / f"{model}.bin"
        have_exe = downloaded_exe.exists() if force_download else self.server_exe.exists()
        have_model = (downloaded_model.exists() if force_download
                      else self.model_path(model).exists())

        try:
            if not have_exe:
                archive = _server_archive(accelerator)
                self._notify(f"Downloading speech engine ({accelerator})...")
                zip_path = _runtime_dir(self.config.config_dir) / archive
                _download(f"{RELEASE_URL}/{archive}", zip_path,
                          _stage(progress, "engine"))
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(_runtime_dir(self.config.config_dir))
                zip_path.unlink(missing_ok=True)
                # The archives nest the binaries a directory deep.
                if not downloaded_exe.exists():
                    for found in _runtime_dir(self.config.config_dir).rglob(
                            self.server_exe.name):
                        for item in found.parent.iterdir():
                            shutil.move(str(item), _runtime_dir(self.config.config_dir))
                        break
                if not downloaded_exe.exists():
                    raise RuntimeError(f"{self._exe_name} not found in {archive}")

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

    # --------------------------------------------------------------- serve
    def start(self, model: str | None = None) -> str | None:
        """Start the server if it is not already up. Returns its URL."""
        with self._lock:
            if self._process and self._process.poll() is None:
                return self.url

            if not self.is_installed(model):
                return None

            cmd = [
                str(self.server_exe),
                "-m", str(self.model_path(model)),
                "-l", "en",
                "--host", "127.0.0.1",
                "--port", str(self.config.local_server_port),
            ]
            # No console window for a background helper.
            flags = 0x08000000 if sys.platform == "win32" else 0
            try:
                self._process = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=flags,
                )
            except Exception as e:
                log(f"[BACKEND] could not start the server: {e}")
                return None

            if self._wait_until_ready():
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
            self._setup_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
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

        True when there is an NVIDIA GPU but the app is still running the
        bundled CPU engine and small model.
        """
        if sys.platform != "win32":
            return False
        if not detect_accelerator().startswith("cuda"):
            return False
        downloaded = _runtime_dir(self.config.config_dir) / self._exe_name
        return not downloaded.exists()

    def describe(self) -> str:
        """One line for the diagnostics report."""
        accelerator = detect_accelerator()
        return (f"{platform.system()} / {accelerator} / "
                f"server {'present' if self.server_exe.exists() else 'missing'} / "
                f"model {'present' if self.model_path().exists() else 'missing'}")
