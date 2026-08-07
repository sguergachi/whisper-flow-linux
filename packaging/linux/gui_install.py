#!/usr/bin/env python3
"""Double-click installer for whisper-flow — no terminal window.

Run this (or the self-extracting setup that execs it) from a file manager.
All progress and errors go through a desktop dialog, never a shell.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTALL_SH = HERE / "install.sh"


def _which(*names: str) -> str | None:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def _notify(kind: str, title: str, text: str) -> None:
    """Show a desktop dialog. kind is info | error | warning."""
    # zenity is on almost every GNOME/Ubuntu box; kdialog for KDE; notify-send
    # is last resort (no buttons, but better than nothing).
    zenity = _which("zenity")
    if zenity:
        flag = {
            "info": "--info",
            "error": "--error",
            "warning": "--warning",
        }.get(kind, "--info")
        subprocess.run(
            [zenity, flag, "--title", title, "--width=420", "--text", text],
            check=False,
        )
        return
    kdialog = _which("kdialog")
    if kdialog:
        flag = {
            "info": "--msgbox",
            "error": "--error",
            "warning": "--sorry",
        }.get(kind, "--msgbox")
        subprocess.run([kdialog, "--title", title, flag, text], check=False)
        return
    notify = _which("notify-send")
    if notify:
        subprocess.run([notify, title, text], check=False)
        return
    # Absolute last resort when double-clicked with no dialog tool: write a
    # note the user will see if they open a terminal later.
    print(f"{title}: {text}", file=sys.stderr)


def _run_install() -> tuple[int, str]:
    """Run install.sh, capture output for the error dialog."""
    if not INSTALL_SH.is_file():
        return 1, f"install.sh not found next to {HERE}"

    env = os.environ.copy()
    # install.sh starts the service itself; keep that. Force past soft
    # dependency checks only when the user already opted in.
    env.setdefault("WHISPER_FLOW_FORCE", env.get("WHISPER_FLOW_FORCE", "0"))

    log_path = Path(tempfile.mkstemp(prefix="whisper-flow-install-", suffix=".log")[1])
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                ["bash", str(INSTALL_SH)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return proc.returncode, text
    finally:
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass


def _progress_context():
    """zenity pulsating progress, or a no-op context if unavailable."""

    class _Null:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def done(self):
            pass

    zenity = _which("zenity")
    if not zenity:
        return _Null()

    class _ZenityProgress:
        def __enter__(self):
            self.proc = subprocess.Popen(
                [
                    zenity,
                    "--progress",
                    "--pulsate",
                    "--no-cancel",
                    "--auto-close",
                    "--title=WhisperFlow",
                    "--text=Installing WhisperFlow…\nThis takes a minute on first run.",
                    "--width=400",
                ],
                stdin=subprocess.PIPE,
                text=True,
            )
            return self

        def __exit__(self, *args):
            self.done()
            return False

        def done(self):
            if self.proc.poll() is None and self.proc.stdin:
                try:
                    self.proc.stdin.write("100\n")
                    self.proc.stdin.close()
                except OSError:
                    pass
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()

    return _ZenityProgress()


def main() -> int:
    if not INSTALL_SH.is_file():
        _notify("error", "WhisperFlow", "Installer is incomplete (install.sh missing).")
        return 1

    with _progress_context():
        code, log = _run_install()

    if code == 0:
        _notify(
            "info",
            "WhisperFlow is ready",
            "Look for the microphone in your notification area.\n\n"
            "Hold Super+Alt and talk — words appear where you type.\n\n"
            "You can also open WhisperFlow from the app menu.",
        )
        return 0

    # Tail the log so the dialog stays readable.
    tail = "\n".join(log.strip().splitlines()[-20:]) if log.strip() else "(no log)"
    _notify(
        "error",
        "WhisperFlow install failed",
        "Something went wrong while installing.\n\n"
        f"{tail}\n\n"
        "If packages are missing, install GTK4, python3-gi, and ydotool\n"
        "from your software centre, then double-click this installer again.",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
