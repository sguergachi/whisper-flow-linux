"""Checking for and applying updates, through Velopack.

Only meaningful in an installed Windows build. A source checkout updates
with git, and the Linux package is installed by its own script, so both
report "not applicable" rather than pretending.

Updates are delta by default: most of what ships is a speech model that does
not change between versions, so a code-only release is a few megabytes
rather than the ~160MB the full installer weighs.
"""

import sys
import threading

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
