"""Checking for and applying updates.

Windows (Velopack) and the Linux AppImage both self-update from the rolling
``latest`` GitHub release. A source checkout updates with git, so it reports
unavailable rather than pretending.

On Windows, updates are delta by default: most of what ships is a speech
model that does not change between versions, so a code-only release is a
few megabytes rather than the ~160MB the full installer weighs. On Linux
the AppImage is one file (~120MB) replaced in place, so the desktop entry
and login autostart keep pointing at the same path.

The flow is check -> download in the background -> apply on click. The
download never blocks a hotkey and the apply never interrupts a dictation:
both wait their turn, and every step retries instead of failing loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .logging import log
from .version import build_version

# Where the release files live. The rolling "latest" release keeps a stable
# download URL, so it works as a static update feed with no server.
UPDATE_URL = ("https://github.com/sguergachi/whisper-flow-linux/"
              "releases/download/latest")
GITHUB_API_RELEASE = (
    "https://api.github.com/repos/sguergachi/whisper-flow-linux/"
    "releases/tags/latest"
)
LINUX_FEED = UPDATE_URL + "/releases.linux.json"

_APPIMAGE_NAME = re.compile(
    r"WhisperFlow-(\d+(?:\.\d+)+)-x86_64\.AppImage$")


def available() -> bool:
    """Whether this build can update itself at all."""
    if _is_linux_appimage():
        return True
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return False
    try:
        import velopack           # noqa: F401
    except ImportError as e:
        # Once, not per call: on a frozen build this means the module did
        # not make it into the bundle, and every update feature is quietly
        # dead until it does.
        global _availability_logged
        if not _availability_logged:
            _availability_logged = True
            log(f"[UPDATE] updater unavailable: velopack not importable ({e})")
        return False
    return True


def _manager():
    import velopack

    return velopack.UpdateManager(UPDATE_URL)


def check(notify=None) -> str | None:
    """Look for a newer version. Returns its version, or None.

    Never raises: an update check failing is not a reason for anything else
    to stop, and the machine may simply be offline.
    """
    if not available():
        return None
    try:
        update = _discover_update()
    except Exception as e:
        log(f"[UPDATE] check failed: {e}")
        if notify:
            notify("Could not check for updates")
        return None

    if not update:
        log("[UPDATE] already current")
        return None
    version = _version_of(update)
    log(f"[UPDATE] {version} is available")
    return version


def apply_now(notify=None) -> bool:
    """Download the update and restart into it.

    Returns False if there was nothing to do or it did not work. On success
    this does not return - the process is replaced.
    """
    if not available():
        return False
    try:
        update = _discover_update()
        if not update:
            if notify:
                notify("whisper-flow is up to date")
            return False

        version = _version_of(update)
        if notify:
            notify(f"Downloading {version}...")
        # One attempt: the user clicked, and a 50s retry loop would look
        # like a hang. Background fetches are the ones that retry.
        if isinstance(update, _LinuxUpdate):
            dest = _linux_dest(update)
            _http_download(update.url, dest, update.sha256)
            update.path = dest
        else:
            _manager().download_updates(update)
        if notify:
            notify(f"Restarting into {version}")
        return _apply_update(update, notify)
    except Exception as e:
        log(f"[UPDATE] could not apply the update: {e}")
        if notify:
            notify(f"Update failed: {e}")
        return False


def _version_of(update) -> str:
    """The version out of whatever shape the update object has."""
    if isinstance(update, _LinuxUpdate):
        return update.version
    for attribute in ("target_full_release", "TargetFullRelease"):
        release = getattr(update, attribute, None)
        if release is not None:
            return str(getattr(release, "version", None)
                       or getattr(release, "Version", release))
    return str(getattr(update, "version", "a new version"))


def check_in_background(notify=None) -> None:
    """Look for an update without holding up startup.

    Only reports when there is something to say. A notification on every
    launch saying nothing has changed is noise.
    """
    if not available():
        return

    def work():
        version = check(notify=None)     # quiet: startup is not the moment
        if version and notify:
            notify(f"whisper-flow {version} is available - "
                   f"use 'Check for updates' to install it")

    threading.Thread(target=work, daemon=True,
                     name="whisper-flow-update-check").start()


# ------------------------------------------------------------------ state
# Everything below is one shared state machine so the periodic checker, a
# tray click and the apply step cannot trip over each other. All of it is a
# no-op where updates are unavailable (source checkouts, non-AppImage Linux).

_lock = threading.Lock()
_availability_logged = False
_checked_version: str | None = None   # newest version seen (downloaded or not)
_pending_update = None                # velopack object or _LinuxUpdate
_pending_version: str | None = None   # its version
_downloading = False                  # a fetch is in flight right now
_notified_version: str | None = None  # last version the user was told about
_auto_started = False
# Outcome of the most recent background round: True = current or fetched,
# False = the check or the download failed, None = no round ran yet. Lets
# the loop tell "offline" apart from "up to date" without extra checks.
_last_check_ok: bool | None = None
_consecutive_failures = 0

# Download attempts per version before giving up until the next check.
_DOWNLOAD_RETRIES = 3
_DOWNLOAD_BACKOFF = (5.0, 15.0, 30.0)


def pending_version() -> str | None:
    """Version of the fully-downloaded update waiting for a restart, if any."""
    with _lock:
        return _pending_version


def is_downloading() -> bool:
    """Whether a background fetch is currently in flight."""
    with _lock:
        return _downloading


def last_check_failed() -> bool:
    """Whether the most recent check or download did not complete."""
    with _lock:
        return _last_check_ok is False


def _remember_checked(version: str | None) -> bool:
    """Record a sighting. True when this version is new to us."""
    global _checked_version
    with _lock:
        if not version or version == _checked_version:
            return False
        _checked_version = version
        return True


def download_in_background(notify=None, on_ready=None) -> str | None:
    """Check once and fetch what is new, without blocking the caller.

    Returns the downloaded version (or the already-pending one). Retries a
    flaky download instead of failing loudly; quiet when already current.
    Never raises.
    """
    global _downloading, _pending_update, _pending_version, _last_check_ok
    if not available():
        return pending_version()
    with _lock:
        if _downloading:
            return _pending_version     # a fetch is already doing the work
    try:
        update = _discover_update()
    except Exception as e:
        log(f"[UPDATE] check failed: {e}")
        with _lock:
            _last_check_ok = False
        return pending_version()
    if not update:
        log("[UPDATE] already current")
        with _lock:
            _last_check_ok = True
        return pending_version()
    version = _version_of(update)
    if not _remember_checked(version):
        with _lock:
            _last_check_ok = True
        return pending_version()        # seen before; nothing new to fetch
    with _lock:
        if _pending_version == version:
            return version              # downloaded while we were checking
        _downloading = True
    try:
        if _fetch_with_retries(version, update, notify):
            with _lock:
                _pending_update = update
                _pending_version = version
                _last_check_ok = True
            log(f"[UPDATE] {version} downloaded in the background")
            if on_ready:
                try:
                    on_ready(version)
                except Exception as e:
                    log(f"[UPDATE] ready callback failed: {e}")
            return version
        with _lock:
            _last_check_ok = False
        return pending_version()
    finally:
        with _lock:
            _downloading = False


def _fetch_with_retries(version, update, notify) -> bool:
    """Download, retrying a flaky connection. True when it landed."""
    if isinstance(update, _LinuxUpdate):
        return _linux_fetch_with_retries(version, update, notify)
    try:
        manager = _manager()
    except Exception as e:
        log(f"[UPDATE] cannot build update manager: {e}")
        return False
    for attempt in range(_DOWNLOAD_RETRIES):
        try:
            if notify and attempt == 0:
                try:
                    notify(f"Downloading whisper-flow {version} in the background...")
                except Exception:
                    pass
            manager.download_updates(update)
            return True
        except Exception as e:
            log(f"[UPDATE] download attempt {attempt + 1} for {version} failed: {e}")
            if attempt + 1 < _DOWNLOAD_RETRIES:
                time.sleep(_DOWNLOAD_BACKOFF[
                    min(attempt, len(_DOWNLOAD_BACKOFF) - 1)])
    log(f"[UPDATE] giving up on {version} until the next check")
    if notify:
        try:
            notify(f"Could not download whisper-flow {version} - will retry later")
        except Exception:
            pass
    return False


def apply_pending(notify=None) -> bool:
    """Restart into the downloaded update. Returns False if there is none.

    Uses the stored update object (no re-check, no re-download). On any
    failure falls back to one fresh check-download-apply round before
    giving up. Never raises.
    """
    global _pending_update, _pending_version
    if not available():
        return False
    with _lock:
        update = _pending_update
        version = _pending_version
    if update is None:
        return False
    try:
        if notify:
            try:
                notify(f"Restarting into whisper-flow {version or ''}".rstrip())
            except Exception:
                pass
        return _apply_update(update, notify)
    except Exception as e:
        log(f"[UPDATE] apply of {version} failed: {e}, trying one fresh round")
    # The stored object went stale (or the restart was refused): one fresh
    # round, then report honestly.
    with _lock:
        _pending_update = None
        _pending_version = None
    try:
        fresh = _discover_update()
        if not fresh:
            return False
        if not _fetch_with_retries(_version_of(fresh), fresh, notify=None):
            raise RuntimeError("download failed")
        return _apply_update(fresh, notify)
    except Exception as e:
        log(f"[UPDATE] fresh apply round failed: {e}")
        if notify:
            try:
                notify(f"Update failed: {e}")
            except Exception:
                pass
        return False


def start_auto_update(notify=None, on_ready=None,
                      first_delay: float = 60.0,
                      interval: float = 6 * 3600.0) -> None:
    """Download new releases in the background, forever, on one thread.

    First check `first_delay` after startup (let the boot settle), then
    every `interval`. Each new version is announced once via notify; the
    on_ready callback fires when its download lands (so the tray can offer
    "Update to X"). Never raises, never busy-loops, never runs twice.
    """
    global _auto_started
    if not available():
        log("[UPDATE] background updater off: not an installed build")
        return
    with _lock:
        if _auto_started:
            return
        _auto_started = True

    cleanup_replaced_appimage()

    def announce(version: str):
        global _notified_version
        with _lock:
            if version == _notified_version:
                return
            _notified_version = version
        if notify:
            try:
                notify(f"whisper-flow {version} downloaded — "
                       f"right-click the tray and pick Update to restart into it")
            except Exception as e:
                log(f"[UPDATE] notify failed: {e}")

    def loop():
        log("[UPDATE] background updater started")
        try:
            time.sleep(first_delay)
            while True:
                try:
                    _auto_update_round(
                        notify,
                        lambda v: (announce(v),
                                   on_ready(v) if on_ready else None),
                    )
                except Exception as e:
                    log(f"[UPDATE] background round failed: {e}")
                time.sleep(interval)
        except Exception as e:
            log(f"[UPDATE] background updater stopped: {e}")

    threading.Thread(target=loop, daemon=True,
                     name="whisper-flow-update-loop").start()


def _auto_update_round(notify=None, on_ready=None) -> None:
    """One background check-download cycle, with an offline tripwire.

    Three failed rounds in a row earn exactly one toast ("couldn't reach
    the update server"), then the counter re-arms. Silent failures look
    identical to a dead updater from the tray, so without this there is no
    telling broken-network apart from broken-code. Never raises.
    """
    global _consecutive_failures
    try:
        download_in_background(notify=None, on_ready=on_ready)
    except Exception as e:
        log(f"[UPDATE] background round failed: {e}")
        with _lock:
            global _last_check_ok
            _last_check_ok = False
    with _lock:
        failed = _last_check_ok is False
        if failed:
            _consecutive_failures += 1
        else:
            _consecutive_failures = 0
        streak = _consecutive_failures
        if streak >= 3:
            _consecutive_failures = 0   # re-arm for the next streak
    if streak >= 3 and notify:
        try:
            notify("Couldn't reach the update server — will keep trying "
                   "in the background")
        except Exception as e:
            log(f"[UPDATE] notify failed: {e}")


# ------------------------------------------------------------- discovery

def _discover_update():
    """Newest remote build if it is newer than us, else None.

    Linux reads releases.linux.json (GitHub API as fallback). Windows asks
    Velopack. Raises on network failure so the caller can log it.
    """
    if _is_linux_appimage():
        return _linux_newer()
    return _manager().check_for_updates()


def _apply_update(update, notify=None) -> bool:
    """Restart into `update`. Does not return on success for either platform."""
    if isinstance(update, _LinuxUpdate):
        return _apply_linux(update, notify)
    _manager().apply_updates_and_restart(update)
    return True


def _current_version() -> str:
    return build_version()


def _version_tuple(text: str) -> tuple[int, int, int]:
    """Numeric (major, minor, patch) from a version string.

    ``0.4.336``, ``0.4.0 (source)``, leftover words after the number: all
    fine. Missing parts pad with zeros so ``0.4`` and ``0.4.0`` compare equal.
    """
    core = (text or "").strip().split()[0]
    nums: list[int] = []
    for part in core.split("."):
        if part.isdigit():
            nums.append(int(part))
        else:
            break
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def _is_newer(remote: str, local: str) -> bool:
    return _version_tuple(remote) > _version_tuple(local)


# -------------------------------------------------------- Linux AppImage

class _LinuxUpdate:
    """A remote AppImage waiting to be fetched, or already on disk."""

    __slots__ = ("version", "url", "sha256", "path")

    def __init__(self, version: str, url: str, sha256: str | None = None,
                 path: Path | None = None):
        self.version = version
        self.url = url
        self.sha256 = sha256
        self.path = path


def _appimage_path() -> str | None:
    """Path of the running AppImage, or None."""
    try:
        from .desktop_install import appimage_path
        return appimage_path()
    except Exception:
        return None


def _is_linux_appimage() -> bool:
    """Frozen Linux running from an AppImage file we can replace."""
    if sys.platform == "win32" or not getattr(sys, "frozen", False):
        return False
    return bool(_appimage_path())


def _user_agent() -> str:
    return (f"whisper-flow/{_current_version()} "
            "(+https://github.com/sguergachi/whisper-flow-linux)")


def _http_get(url: str, timeout: float = 20.0) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _http_download(url: str, dest: Path, expected_sha: str | None = None,
                   timeout: float = 60.0) -> None:
    """Stream `url` to `dest`. Raises on HTTP/IO/hash failure."""
    request = urllib.request.Request(
        url, headers={"User-Agent": _user_agent()})
    hasher = hashlib.sha256()
    partial = dest.with_name(dest.name + ".partial")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(partial, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    hasher.update(chunk)
        digest = hasher.hexdigest()
        if expected_sha and digest.lower() != expected_sha.lower():
            raise ValueError(
                f"download hash mismatch (got {digest}, expected {expected_sha})")
        if partial.stat().st_size < 64:
            raise ValueError("download too small to be an AppImage")
        with open(partial, "rb") as handle:
            magic = handle.read(4)
        if magic != b"\x7fELF":
            raise ValueError("download is not an ELF AppImage")
        os.chmod(partial, 0o755)
        os.replace(partial, dest)
    except Exception:
        try:
            partial.unlink()
        except OSError:
            pass
        raise


def _linux_update_from_feed(data: dict) -> _LinuxUpdate | None:
    version = str(data.get("version") or data.get("Version") or "").strip()
    filename = str(data.get("file") or data.get("FileName") or "").strip()
    sha = data.get("sha256") or data.get("SHA256")
    sha = str(sha).strip() if sha else None
    if not version or not filename:
        return None
    url = filename if filename.startswith("http") else f"{UPDATE_URL}/{filename}"
    return _LinuxUpdate(version, url, sha)


def _version_from_filename(name: str) -> str | None:
    match = _APPIMAGE_NAME.search(name or "")
    return match.group(1) if match else None


def _linux_update_from_github(data: dict) -> _LinuxUpdate | None:
    for asset in data.get("assets") or []:
        name = asset.get("name") or ""
        if not _APPIMAGE_NAME.search(name):
            continue
        version = _version_from_filename(name)
        if not version:
            continue
        url = asset.get("browser_download_url")
        if not url:
            continue
        sha = None
        digest = asset.get("digest") or ""
        if isinstance(digest, str) and digest.lower().startswith("sha256:"):
            sha = digest.split(":", 1)[1]
        return _LinuxUpdate(version, url, sha)
    return None


def _linux_remote() -> _LinuxUpdate | None:
    """The newest AppImage on the rolling feed.

    Prefers releases.linux.json (tiny, no API quota). Falls back to the
    GitHub release API so a missing feed file is not a dead updater.
    Raises on total network failure.
    """
    feed_error: Exception | None = None
    try:
        data = json.loads(_http_get(LINUX_FEED))
        update = _linux_update_from_feed(data)
        if update:
            return update
        feed_error = ValueError("releases.linux.json had no AppImage")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, ValueError, OSError) as e:
        feed_error = e
        log(f"[UPDATE] linux feed missed ({e}); trying GitHub API")
    data = json.loads(_http_get(GITHUB_API_RELEASE))
    update = _linux_update_from_github(data)
    if update:
        return update
    if feed_error:
        raise feed_error
    return None


def _linux_newer() -> _LinuxUpdate | None:
    """Remote AppImage if it is newer than this build, else None."""
    remote = _linux_remote()
    if not remote:
        return None
    if not _is_newer(remote.version, _current_version()):
        return None
    return remote


def _linux_dest(update: _LinuxUpdate) -> Path:
    current = _appimage_path()
    if not current:
        raise RuntimeError("not running from an AppImage")
    return Path(current).with_name(Path(current).name + ".new")


def _linux_fetch_with_retries(version, update: _LinuxUpdate, notify) -> bool:
    try:
        dest = _linux_dest(update)
    except Exception as e:
        log(f"[UPDATE] cannot choose AppImage destination: {e}")
        return False
    if dest.is_file() and _file_sha256(dest) == (update.sha256 or "").lower():
        update.path = dest
        return True
    for attempt in range(_DOWNLOAD_RETRIES):
        try:
            if notify and attempt == 0:
                try:
                    notify(f"Downloading whisper-flow {version} in the background...")
                except Exception:
                    pass
            _http_download(update.url, dest, update.sha256)
            update.path = dest
            return True
        except Exception as e:
            log(f"[UPDATE] download attempt {attempt + 1} for {version} failed: {e}")
            if attempt + 1 < _DOWNLOAD_RETRIES:
                time.sleep(_DOWNLOAD_BACKOFF[
                    min(attempt, len(_DOWNLOAD_BACKOFF) - 1)])
    log(f"[UPDATE] giving up on {version} until the next check")
    if notify:
        try:
            notify(f"Could not download whisper-flow {version} - will retry later")
        except Exception:
            pass
    return False


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _apply_linux(update: _LinuxUpdate, notify=None) -> bool:
    """Replace the running AppImage and restart into it.

    Rename-aside so the FUSE mount of the running file stays valid until
    this process exits: current -> current.old, then .new -> current.
    A one-second delayed exec starts the new file after the instance lock
    is released by this process dying. On success this does not return.
    """
    current = _appimage_path()
    if not current:
        raise RuntimeError("not running from an AppImage")
    new_path = Path(update.path) if update.path else _linux_dest(update)
    if not new_path.is_file():
        raise RuntimeError(f"downloaded AppImage missing: {new_path}")
    current_path = Path(current)
    old_path = current_path.with_name(current_path.name + ".old")
    os.replace(current_path, old_path)
    try:
        os.replace(new_path, current_path)
        os.chmod(current_path, 0o755)
    except Exception:
        try:
            if old_path.is_file() and not current_path.exists():
                os.replace(old_path, current_path)
        except OSError:
            pass
        raise
    _linux_respawn(str(current_path))
    _exit_after_apply()
    return True                     # tests mock _exit_after_apply


def _linux_respawn(path: str) -> None:
    """Start `path` after this process has had a moment to die.

    The daemon holds an flock; a replacement started too soon exits with
    "already running" and the tray never comes back. sleep-then-exec in a
    new session is the same shape restart.py uses from the settings window.
    """
    subprocess.Popen(
        ["sh", "-c", f"sleep 1; exec {shlex.quote(path)}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _exit_after_apply() -> None:
    """Leave so the replacement can take the instance lock. Tests mock this."""
    os._exit(0)


def cleanup_replaced_appimage() -> None:
    """Delete the previous AppImage left beside us after a successful swap.

    Safe on the next start: the old process (and its FUSE mount) is gone,
    so the leftover ``*.AppImage.old`` is just a file. Never raises.
    """
    current = _appimage_path()
    if not current:
        return
    old = Path(current).with_name(Path(current).name + ".old")
    try:
        if old.is_file():
            old.unlink()
            log("[UPDATE] removed the replaced AppImage")
    except OSError as e:
        log(f"[UPDATE] could not remove the replaced AppImage: {e}")
