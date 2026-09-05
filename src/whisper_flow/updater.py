"""Checking for and applying updates, through Velopack.

Only meaningful in an installed Windows build. A source checkout updates
with git, and the Linux package is installed by its own script, so both
report "not applicable" rather than pretending.

Updates are delta by default: most of what ships is a speech model that does
not change between versions, so a code-only release is a few megabytes
rather than the ~160MB the full installer weighs.

The flow is check -> download in the background -> apply on click. The
download never blocks a hotkey and the apply never interrupts a dictation:
both wait their turn, and every step retries instead of failing loudly.
"""

import sys
import threading
import time

from .logging import log

# Where the release files live. The rolling "latest" release keeps a stable
# download URL, so it works as a static update feed with no server.
UPDATE_URL = ("https://github.com/sguergachi/whisper-flow-linux/"
              "releases/download/latest")


def available() -> bool:
    """Whether this build can update itself at all."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return False
    try:
        import velopack           # noqa: F401
    except ImportError:
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
        update = _manager().check_for_updates()
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
        manager = _manager()
        update = manager.check_for_updates()
        if not update:
            if notify:
                notify("whisper-flow is up to date")
            return False

        version = _version_of(update)
        if notify:
            notify(f"Downloading {version}...")
        manager.download_updates(update)
        if notify:
            notify(f"Restarting into {version}")
        manager.apply_updates_and_restart(update)
        return True
    except Exception as e:
        log(f"[UPDATE] could not apply the update: {e}")
        if notify:
            notify(f"Update failed: {e}")
        return False


def _version_of(update) -> str:
    """The version out of whatever shape the update object has."""
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
# no-op where updates are unavailable (Linux, source checkouts).

_lock = threading.Lock()
_checked_version: str | None = None   # newest version seen (downloaded or not)
_pending_update = None                # velopack update object, downloaded
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
        update = _manager().check_for_updates()
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
        _manager().apply_updates_and_restart(update)
        return True                     # does not return on success
    except Exception as e:
        log(f"[UPDATE] apply of {version} failed: {e}, trying one fresh round")
    # The stored object went stale (or the restart was refused): one fresh
    # round, then report honestly.
    with _lock:
        _pending_update = None
        _pending_version = None
    try:
        fresh = _manager().check_for_updates()
        if not fresh:
            return False
        _manager().download_updates(fresh)
        _manager().apply_updates_and_restart(fresh)
        return True
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
        log("[UPDATE] background updater off: not an installed Windows build")
        return
    with _lock:
        if _auto_started:
            return
        _auto_started = True

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
