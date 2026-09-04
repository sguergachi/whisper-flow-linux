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


# ------------------------------------------------- background auto-update
@pytest.fixture
def _clean_updater_state(monkeypatch):
    """Isolate the updater's global state machine per test."""
    monkeypatch.setattr(updater, "available", lambda: True)
    updater._checked_version = None
    updater._pending_update = None
    updater._pending_version = None
    updater._downloading = False
    updater._notified_version = None
    updater._auto_started = False
    yield
    updater._checked_version = None
    updater._pending_update = None
    updater._pending_version = None
    updater._downloading = False
    updater._notified_version = None
    updater._auto_started = False


def test_background_download_stores_pending_and_fires_ready(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    update = Mock(target_full_release=Mock(version="0.5.0"))
    manager.check_for_updates.return_value = update
    monkeypatch.setattr(updater, "_manager", lambda: manager)

    ready = []
    assert updater.download_in_background(
        notify=Mock(), on_ready=ready.append) == "0.5.0"
    manager.download_updates.assert_called_once_with(update)
    assert updater.pending_version() == "0.5.0"
    assert ready == ["0.5.0"]


def test_second_sighting_of_the_same_version_fetches_nothing(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    update = Mock(target_full_release=Mock(version="0.5.0"))
    manager.check_for_updates.return_value = update
    monkeypatch.setattr(updater, "_manager", lambda: manager)

    assert updater.download_in_background() == "0.5.0"
    assert updater.download_in_background() == "0.5.0"
    manager.download_updates.assert_called_once_with(update)


def test_a_flaky_download_is_retried_not_reported(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    update = Mock(target_full_release=Mock(version="0.5.0"))
    manager.check_for_updates.return_value = update
    manager.download_updates.side_effect = [OSError("reset"), None]
    monkeypatch.setattr(updater, "_manager", lambda: manager)
    monkeypatch.setattr(updater, "_DOWNLOAD_BACKOFF", (0.0, 0.0, 0.0))

    told = []
    assert updater.download_in_background(notify=told.append) == "0.5.0"
    assert manager.download_updates.call_count == 2
    assert updater.pending_version() == "0.5.0"


def test_a_dead_download_reports_once_and_gives_up(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    update = Mock(target_full_release=Mock(version="0.5.0"))
    manager.check_for_updates.return_value = update
    manager.download_updates.side_effect = OSError("offline")
    monkeypatch.setattr(updater, "_manager", lambda: manager)
    monkeypatch.setattr(updater, "_DOWNLOAD_BACKOFF", (0.0, 0.0, 0.0))

    told = []
    assert updater.download_in_background(notify=told.append) is None
    assert manager.download_updates.call_count == 3
    assert updater.pending_version() is None


def test_apply_uses_the_pending_object_without_rechecking(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    update = Mock(target_full_release=Mock(version="0.5.0"))
    manager.check_for_updates.return_value = update
    monkeypatch.setattr(updater, "_manager", lambda: manager)
    updater.download_in_background()

    manager.check_for_updates.reset_mock()
    assert updater.apply_pending(notify=Mock()) is True
    manager.check_for_updates.assert_not_called()
    manager.apply_updates_and_restart.assert_called_once_with(update)


def test_apply_with_nothing_pending_reports_false(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    monkeypatch.setattr(updater, "_manager", lambda: manager)
    assert updater.apply_pending(notify=Mock()) is False
    manager.apply_updates_and_restart.assert_not_called()


def test_a_stale_pending_object_falls_back_to_a_fresh_round(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    stale = Mock(target_full_release=Mock(version="0.5.0"))
    fresh = Mock(target_full_release=Mock(version="0.5.1"))
    manager.check_for_updates.return_value = stale
    monkeypatch.setattr(updater, "_manager", lambda: manager)
    updater.download_in_background()

    manager.apply_updates_and_restart.side_effect = [
        OSError("stale"), None]
    manager.check_for_updates.return_value = fresh
    assert updater.apply_pending(notify=Mock()) is True
    manager.download_updates.assert_called_with(fresh)
    assert updater.pending_version() is None      # cleared, then applied


def test_concurrent_downloads_share_one_fetch(
        monkeypatch, _clean_updater_state):
    import threading
    manager = Mock()
    update = Mock(target_full_release=Mock(version="0.5.0"))
    manager.check_for_updates.return_value = update
    started = threading.Event()
    release = threading.Event()

    def slow_download(u):
        started.set()
        assert release.wait(timeout=5)

    manager.download_updates.side_effect = slow_download
    monkeypatch.setattr(updater, "_manager", lambda: manager)

    first = threading.Thread(
        target=lambda: updater.download_in_background())
    first.start()
    assert started.wait(timeout=5)
    # Second caller arrives mid-fetch: no second download.
    assert updater.download_in_background() is None
    release.set()
    first.join(timeout=5)
    assert manager.download_updates.call_count == 1


def test_auto_update_announces_each_version_once(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    update = Mock(target_full_release=Mock(version="0.5.0"))
    manager.check_for_updates.return_value = update
    monkeypatch.setattr(updater, "_manager", lambda: manager)

    told, ready = [], []
    updater.start_auto_update(notify=told.append, on_ready=ready.append,
                              first_delay=0.0, interval=3600.0)
    import time
    deadline = time.monotonic() + 5.0
    while not ready and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready == ["0.5.0"]
    assert any("0.5.0" in m for m in told)
    # Second loop start is a no-op: still one thread's worth of work.
    updater.start_auto_update(notify=told.append, first_delay=0.0)
    assert updater._auto_started is True


def test_auto_update_is_a_no_op_where_unavailable(monkeypatch):
    monkeypatch.setattr(updater, "available", lambda: False)
    updater.start_auto_update(notify=Mock())
    assert updater._auto_started is False
    assert updater.download_in_background(notify=Mock()) is None
