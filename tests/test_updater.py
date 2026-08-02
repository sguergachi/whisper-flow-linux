"""Tests for the self-updater.

The rules: it never raises into the caller, it never claims an update it did
not find, and it reports honestly that it is unavailable outside an
installed Windows build.
"""

import sys
from unittest.mock import Mock

import pytest

from whisper_flow import updater


def test_a_source_checkout_cannot_update_itself(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert updater.available() is False


def test_linux_cannot_update_itself(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert updater.available() is False


def test_nothing_is_attempted_when_unavailable(monkeypatch):
    monkeypatch.setattr(updater, "available", lambda: False)
    monkeypatch.setattr(updater, "_manager",
                        Mock(side_effect=AssertionError("must not be built")))
    assert updater.check() is None
    assert updater.apply_now() is False


def test_a_failed_check_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "_manager",
                        Mock(side_effect=OSError("offline")))
    told = []
    assert updater.check(notify=told.append) is None
    assert told and "check" in told[0].lower()


def test_no_update_available_returns_none(monkeypatch):
    manager = Mock()
    manager.check_for_updates.return_value = None
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "_manager", lambda: manager)
    assert updater.check() is None


def test_an_available_update_reports_its_version(monkeypatch):
    manager = Mock()
    manager.check_for_updates.return_value = Mock(
        target_full_release=Mock(version="0.4.0"))
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "_manager", lambda: manager)
    assert updater.check() == "0.4.0"


def test_applying_downloads_then_restarts(monkeypatch):
    manager = Mock()
    update = Mock(target_full_release=Mock(version="0.4.0"))
    manager.check_for_updates.return_value = update
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "_manager", lambda: manager)

    assert updater.apply_now(notify=Mock()) is True
    manager.download_updates.assert_called_once_with(update)
    manager.apply_updates_and_restart.assert_called_once_with(update)


def test_applying_says_so_when_already_current(monkeypatch):
    manager = Mock()
    manager.check_for_updates.return_value = None
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "_manager", lambda: manager)

    told = []
    assert updater.apply_now(notify=told.append) is False
    assert told and "up to date" in told[0]
    manager.download_updates.assert_not_called()


def test_a_failed_download_does_not_restart(monkeypatch):
    manager = Mock()
    manager.check_for_updates.return_value = Mock(
        target_full_release=Mock(version="0.4.0"))
    manager.download_updates.side_effect = OSError("connection reset")
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "_manager", lambda: manager)

    told = []
    assert updater.apply_now(notify=told.append) is False
    manager.apply_updates_and_restart.assert_not_called()
    assert any("failed" in m.lower() for m in told)


def test_the_startup_check_is_quiet_when_current(monkeypatch):
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda notify=None: None)
    told = []
    updater.check_in_background(notify=told.append)
    import time
    time.sleep(0.2)
    assert told == []            # a launch that says nothing changed is noise


def test_the_update_feed_points_at_the_rolling_release():
    assert updater.UPDATE_URL.endswith("/releases/download/latest")


@pytest.mark.parametrize("shape,expected", [
    (Mock(target_full_release=Mock(version="1.2.3")), "1.2.3"),
    (Mock(spec=["version"], version="9.9.9"), "9.9.9"),
])
def test_the_version_is_read_from_whatever_shape_arrives(shape, expected):
    assert updater._version_of(shape) == expected
