"""Restarting the daemon from outside it.

The settings window owns this because the daemon cannot respawn itself: on
Windows the replacement would trip the single-instance mutex while the old
process still held it.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_flow import restart


def test_systemd_installations_restart_through_systemd(monkeypatch):
    """Kill only the main process - not the settings child in the cgroup.

    ``systemctl restart`` uses KillMode=control-group by default and took
    the settings window down mid-Save, so the toast never appeared.
    """
    monkeypatch.setattr(restart, "systemd_unit_active", lambda: True)
    monkeypatch.setattr(restart, "daemon_pid", lambda: None)
    run = Mock(return_value=Mock(returncode=0, stderr=b""))
    monkeypatch.setattr(restart.subprocess, "run", run)
    kill = Mock()
    monkeypatch.setattr(restart.os, "kill", kill)

    ok, _detail = restart.restart_daemon()

    assert ok
    assert run.call_count == 2
    assert run.call_args_list[0][0][0] == [
        "systemctl", "--user", "kill", "-s", "SIGTERM",
        "--kill-who=main", restart.UNIT]
    assert run.call_args_list[1][0][0] == [
        "systemctl", "--user", "start", restart.UNIT]
    kill.assert_not_called()


def test_a_running_daemon_is_stopped_then_respawned(monkeypatch):
    monkeypatch.setattr(restart, "systemd_unit_active", lambda: False)
    monkeypatch.setattr(restart, "daemon_pid", lambda: 4321)
    monkeypatch.setattr(restart, "_wait_for_exit",
                        lambda pid, timeout=10.0: True)
    kill = Mock()
    monkeypatch.setattr(restart.os, "kill", kill)
    popen = Mock()
    monkeypatch.setattr(restart.subprocess, "Popen", popen)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    # Source path: command[0] is this interpreter, which exists.
    monkeypatch.setattr(restart, "respawn_command", lambda: [
        sys.executable, "-m", "whisper_flow.cli", "daemon", "--foreground"])

    ok, _detail = restart.restart_daemon()

    assert ok
    kill.assert_called_once_with(4321, 15)
    assert popen.call_args[0][0] == [
        sys.executable, "-m", "whisper_flow.cli", "daemon", "--foreground"]


def test_frozen_respawn_prefers_the_appimage_file(monkeypatch, tmp_path):
    """sys.executable under AppImage is the FUSE mount; it dies with the daemon."""
    img = tmp_path / "WhisperFlow.AppImage"
    img.write_bytes(b"ELF")
    mount = tmp_path / ".mount_dead" / "whisper-flow"
    mount.parent.mkdir()
    # Mount path deliberately does NOT exist - that is the Save race.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(mount))
    monkeypatch.setenv("APPIMAGE", str(img))

    assert restart.respawn_command() == [str(img.resolve())]


def test_frozen_respawn_falls_back_to_desktop_entry(monkeypatch, tmp_path):
    img = tmp_path / "WhisperFlow.AppImage"
    img.write_bytes(b"ELF")
    data = tmp_path / "share"
    apps = data / "applications"
    apps.mkdir(parents=True)
    (apps / "whisper-flow.desktop").write_text(
        f'[Desktop Entry]\nExec="{img}"\n', encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/tmp/.mount_gone/whisper-flow")
    monkeypatch.delenv("APPIMAGE", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(data))

    assert restart.respawn_command() == [str(img)]


def test_restart_resolves_binary_before_killing_daemon(monkeypatch, tmp_path):
    """Killing first unmounted the AppImage; resolve while the path still exists."""
    img = tmp_path / "WhisperFlow.AppImage"
    img.write_bytes(b"ELF")
    order = []

    monkeypatch.setattr(restart, "systemd_unit_active", lambda: False)
    monkeypatch.setattr(restart, "daemon_pid", lambda: 99)
    monkeypatch.setattr(restart, "_wait_for_exit", lambda *a, **k: True)
    monkeypatch.setattr(restart, "respawn_command", lambda: order.append("cmd") or [str(img)])
    monkeypatch.setattr(
        restart.os, "kill",
        lambda pid, sig: order.append(("kill", pid)))
    popen = Mock(side_effect=lambda *a, **k: order.append("popen"))
    monkeypatch.setattr(restart.subprocess, "Popen", popen)

    ok, _ = restart.restart_daemon()

    assert ok
    assert order[0] == "cmd"
    assert order[1] == ("kill", 99)
    assert order[2] == "popen"


def test_restart_fails_clearly_when_binary_is_gone(monkeypatch):
    monkeypatch.setattr(restart, "systemd_unit_active", lambda: False)
    monkeypatch.setattr(restart, "daemon_pid", lambda: None)
    monkeypatch.setattr(
        restart, "respawn_command",
        lambda: ["/tmp/.mount_WhispeaEGLho/whisper-flow"])
    kill = Mock()
    monkeypatch.setattr(restart.os, "kill", kill)
    popen = Mock()
    monkeypatch.setattr(restart.subprocess, "Popen", popen)

    ok, detail = restart.restart_daemon()

    assert ok is False
    assert "cannot find the daemon binary" in detail
    kill.assert_not_called()
    popen.assert_not_called()


def test_respawn_hides_the_console_on_windows(monkeypatch):
    """Save restarts the daemon; a console python.exe must not flash a prompt."""
    import subprocess

    monkeypatch.setattr(restart, "systemd_unit_active", lambda: False)
    monkeypatch.setattr(restart, "daemon_pid", lambda: None)
    monkeypatch.setattr(sys, "platform", "win32")
    popen = Mock()
    monkeypatch.setattr(restart.subprocess, "Popen", popen)
    monkeypatch.setattr(restart, "respawn_command", lambda: [sys.executable])

    ok, _detail = restart.restart_daemon()

    assert ok
    expected = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    assert popen.call_args.kwargs.get("creationflags") == expected


def test_no_running_daemon_just_starts_one(monkeypatch):
    monkeypatch.setattr(restart, "systemd_unit_active", lambda: False)
    monkeypatch.setattr(restart, "daemon_pid", lambda: None)
    kill = Mock()
    monkeypatch.setattr(restart.os, "kill", kill)
    popen = Mock()
    monkeypatch.setattr(restart.subprocess, "Popen", popen)
    monkeypatch.setattr(restart, "respawn_command", lambda: [
        sys.executable, "-m", "whisper_flow.cli", "daemon", "--foreground"])

    ok, _detail = restart.restart_daemon()

    assert ok
    kill.assert_not_called()
    popen.assert_called_once()


def test_a_stubborn_daemon_still_gets_a_replacement(monkeypatch):
    """Do not freeze settings for 10s if the old process is slow to exit.

    Save runs restart from the settings window; a long hard wait made the
    UI look locked until the old daemon finally died. Prefer starting the
    replacement after a short wait.
    """
    monkeypatch.setattr(restart, "systemd_unit_active", lambda: False)
    monkeypatch.setattr(restart, "daemon_pid", lambda: 4321)
    monkeypatch.setattr(restart, "_wait_for_exit", lambda pid, timeout=10.0: False)
    monkeypatch.setattr(restart.os, "kill", Mock())
    popen = Mock()
    monkeypatch.setattr(restart.subprocess, "Popen", popen)
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(restart, "respawn_command", lambda: [
        sys.executable, "-m", "whisper_flow.cli", "daemon", "--foreground"])

    ok, _detail = restart.restart_daemon()

    assert ok
    popen.assert_called_once()


def test_restart_daemon_never_raises(monkeypatch):
    """Settings treats every failure as (False, reason), never an exception."""
    monkeypatch.setattr(restart, "systemd_unit_active",
                        Mock(side_effect=RuntimeError("dbus dead")))

    ok, detail = restart.restart_daemon()

    assert ok is False
    assert "dbus dead" in detail


def test_systemctl_os_error_is_reported(monkeypatch):
    monkeypatch.setattr(restart, "systemd_unit_active", lambda: True)
    monkeypatch.setattr(
        restart.subprocess, "run",
        Mock(side_effect=OSError(2, "No such file or directory")),
    )

    ok, detail = restart.restart_daemon()

    assert ok is False
    assert "systemctl" in detail.lower() or "No such file" in detail


def test_wait_for_exit_returns_immediately_when_the_pid_is_gone():
    """A ctypes WaitForSingleObject held the GIL for the whole timeout.

    Sleep-and-probe must notice a dead pid on the first check, not after
    the full wait - Save runs this on a worker and a GIL hold freezes GTK.
    """
    import time

    t0 = time.monotonic()
    assert restart._wait_for_exit(2**30, timeout=5.0) is True
    assert time.monotonic() - t0 < 1.0
