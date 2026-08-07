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

    ok, _detail = restart.restart_daemon()

    assert ok
    kill.assert_called_once_with(4321, 15)
    assert popen.call_args[0][0] == [
        sys.executable, "-m", "whisper_flow.cli", "daemon", "--foreground"]


def test_respawn_hides_the_console_on_windows(monkeypatch):
    """Save restarts the daemon; a console python.exe must not flash a prompt."""
    import subprocess

    monkeypatch.setattr(restart, "systemd_unit_active", lambda: False)
    monkeypatch.setattr(restart, "daemon_pid", lambda: None)
    monkeypatch.setattr(sys, "platform", "win32")
    popen = Mock()
    monkeypatch.setattr(restart.subprocess, "Popen", popen)

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
