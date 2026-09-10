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


def test_linux_onedir_without_appimage_cannot_update_itself(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("APPIMAGE", raising=False)
    assert updater.available() is False


def test_linux_appimage_can_update_itself(monkeypatch, tmp_path):
    img = tmp_path / "WhisperFlow-0.4.336-x86_64.AppImage"
    img.write_bytes(b"\x7fELF" + b"\0" * 60)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPIMAGE", str(img))
    assert updater.available() is True


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
    updater._last_check_ok = None
    updater._consecutive_failures = 0
    yield
    updater._checked_version = None
    updater._pending_update = None
    updater._pending_version = None
    updater._downloading = False
    updater._notified_version = None
    updater._auto_started = False
    updater._last_check_ok = None
    updater._consecutive_failures = 0


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


def test_three_failed_rounds_earn_one_offline_toast(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    manager.check_for_updates.side_effect = OSError("dns down")
    monkeypatch.setattr(updater, "_manager", lambda: manager)

    told = []
    updater._auto_update_round(notify=told.append)
    updater._auto_update_round(notify=told.append)
    assert told == []
    updater._auto_update_round(notify=told.append)
    assert len(told) == 1 and "update server" in told[0]
    # Re-armed: the next streak earns its own single toast.
    updater._auto_update_round(notify=told.append)
    updater._auto_update_round(notify=told.append)
    assert len(told) == 1
    updater._auto_update_round(notify=told.append)
    assert len(told) == 2


def test_a_good_round_clears_the_failure_streak(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    manager.check_for_updates.side_effect = OSError("dns down")
    monkeypatch.setattr(updater, "_manager", lambda: manager)

    told = []
    updater._auto_update_round(notify=told.append)
    updater._auto_update_round(notify=told.append)
    manager.check_for_updates.side_effect = None
    manager.check_for_updates.return_value = None   # up to date: fine
    updater._auto_update_round(notify=told.append)
    assert told == []
    assert updater._consecutive_failures == 0


def test_round_outcome_is_visible_for_diagnosis(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    manager.check_for_updates.return_value = None
    monkeypatch.setattr(updater, "_manager", lambda: manager)
    updater._auto_update_round()
    assert updater._last_check_ok is True

    manager.check_for_updates.side_effect = OSError("offline")
    updater._auto_update_round()
    assert updater._last_check_ok is False


def test_a_failed_round_is_visible_as_last_check_failed(
        monkeypatch, _clean_updater_state):
    manager = Mock()
    manager.check_for_updates.side_effect = OSError("offline")
    monkeypatch.setattr(updater, "_manager", lambda: manager)
    updater._auto_update_round()
    assert updater.last_check_failed() is True


def test_version_tuple_pads_and_strips_words():
    assert updater._version_tuple("0.4.336") == (0, 4, 336)
    assert updater._version_tuple("0.4.0 (source)") == (0, 4, 0)
    assert updater._version_tuple("0.4") == (0, 4, 0)
    assert updater._is_newer("0.4.337", "0.4.336") is True
    assert updater._is_newer("0.4.336", "0.4.336") is False
    assert updater._is_newer("0.5.0", "0.4.400") is True
    assert updater._is_newer("0.4.1", "0.4.336") is False


def test_the_release_job_publishes_the_linux_update_feed():
    from pathlib import Path

    workflow = (Path(__file__).resolve().parents[1]
                / ".github/workflows/build.yml").read_text(encoding="utf-8")
    assert "releases.linux.json" in workflow
    assert "Write the Linux update feed" in workflow


def test_linux_feed_json_is_parsed():
    data = {
        "version": "0.4.400",
        "file": "WhisperFlow-0.4.400-x86_64.AppImage",
        "sha256": "abc",
    }
    update = updater._linux_update_from_feed(data)
    assert update.version == "0.4.400"
    assert update.url.endswith("/WhisperFlow-0.4.400-x86_64.AppImage")
    assert update.sha256 == "abc"


def test_github_api_payload_is_parsed():
    data = {
        "assets": [
            {
                "name": "WhisperFlow-win-Setup.exe",
                "browser_download_url": "https://example/setup",
            },
            {
                "name": "WhisperFlow-0.4.401-x86_64.AppImage",
                "browser_download_url": "https://example/appimage",
                "digest": "sha256:deadbeef",
            },
        ]
    }
    update = updater._linux_update_from_github(data)
    assert update.version == "0.4.401"
    assert update.url == "https://example/appimage"
    assert update.sha256 == "deadbeef"


def test_linux_check_reports_a_newer_appimage(monkeypatch, tmp_path,
                                              _clean_updater_state):
    img = tmp_path / "WhisperFlow-0.4.300-x86_64.AppImage"
    img.write_bytes(b"\x7fELF" + b"\0" * 60)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPIMAGE", str(img))
    monkeypatch.setattr(updater, "_current_version", lambda: "0.4.300")
    monkeypatch.setattr(
        updater, "_http_get",
        lambda url, timeout=20.0: (
            b'{"version":"0.4.400","file":"WhisperFlow-0.4.400-x86_64.AppImage",'
            b'"sha256":"abc"}'
            if url.endswith("releases.linux.json") else b"{}"
        ),
    )
    assert updater.check() == "0.4.400"


def test_linux_check_is_quiet_when_current(monkeypatch, tmp_path,
                                           _clean_updater_state):
    img = tmp_path / "WhisperFlow-0.4.400-x86_64.AppImage"
    img.write_bytes(b"\x7fELF" + b"\0" * 60)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPIMAGE", str(img))
    monkeypatch.setattr(updater, "_current_version", lambda: "0.4.400")
    monkeypatch.setattr(
        updater, "_http_get",
        lambda url, timeout=20.0: (
            b'{"version":"0.4.400","file":"WhisperFlow-0.4.400-x86_64.AppImage",'
            b'"sha256":"abc"}'
        ),
    )
    assert updater.check() is None


def test_linux_falls_back_to_github_api_when_feed_is_missing(
        monkeypatch, tmp_path, _clean_updater_state):
    import urllib.error

    img = tmp_path / "WhisperFlow-0.4.300-x86_64.AppImage"
    img.write_bytes(b"\x7fELF" + b"\0" * 60)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPIMAGE", str(img))
    monkeypatch.setattr(updater, "_current_version", lambda: "0.4.300")

    def http_get(url, timeout=20.0):
        if url.endswith("releases.linux.json"):
            raise urllib.error.URLError("not found")
        return (
            b'{"assets":[{"name":"WhisperFlow-0.4.410-x86_64.AppImage",'
            b'"browser_download_url":"https://example/img",'
            b'"digest":"sha256:ff"}]}'
        )

    monkeypatch.setattr(updater, "_http_get", http_get)
    assert updater.check() == "0.4.410"


def test_linux_background_download_stores_pending(
        monkeypatch, tmp_path, _clean_updater_state):
    img = tmp_path / "WhisperFlow-0.4.300-x86_64.AppImage"
    img.write_bytes(b"\x7fELF" + b"\0" * 60)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPIMAGE", str(img))
    monkeypatch.setattr(updater, "_current_version", lambda: "0.4.300")
    monkeypatch.setattr(
        updater, "_http_get",
        lambda url, timeout=20.0: (
            b'{"version":"0.4.400","file":"WhisperFlow-0.4.400-x86_64.AppImage",'
            b'"sha256":"abc"}'
        ),
    )

    def fake_download(url, dest, expected_sha=None, timeout=60.0):
        dest.write_bytes(b"\x7fELF" + b"\0" * 60)
        dest.chmod(0o755)

    monkeypatch.setattr(updater, "_http_download", fake_download)
    ready = []
    assert updater.download_in_background(on_ready=ready.append) == "0.4.400"
    assert updater.pending_version() == "0.4.400"
    assert ready == ["0.4.400"]
    pending = img.with_name(img.name + ".new")
    assert pending.is_file()


def test_linux_apply_swaps_in_place_and_respawns(
        monkeypatch, tmp_path, _clean_updater_state):
    img = tmp_path / "WhisperFlow-0.4.300-x86_64.AppImage"
    img.write_bytes(b"\x7fELFOLD" + b"\0" * 60)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPIMAGE", str(img))
    monkeypatch.setattr(updater, "_current_version", lambda: "0.4.300")
    monkeypatch.setattr(
        updater, "_http_get",
        lambda url, timeout=20.0: (
            b'{"version":"0.4.400","file":"WhisperFlow-0.4.400-x86_64.AppImage",'
            b'"sha256":"abc"}'
        ),
    )

    def fake_download(url, dest, expected_sha=None, timeout=60.0):
        dest.write_bytes(b"\x7fELFNEW" + b"\0" * 60)
        dest.chmod(0o755)

    monkeypatch.setattr(updater, "_http_download", fake_download)
    updater.download_in_background()

    respawned, exited = [], []
    monkeypatch.setattr(updater, "_linux_respawn", respawned.append)
    monkeypatch.setattr(updater, "_exit_after_apply",
                        lambda: exited.append(True))

    assert updater.apply_pending() is True
    assert img.read_bytes().startswith(b"\x7fELFNEW")
    old = img.with_name(img.name + ".old")
    assert old.is_file()
    assert old.read_bytes().startswith(b"\x7fELFOLD")
    assert respawned == [str(img)]
    assert exited == [True]


def test_cleanup_removes_the_replaced_appimage(monkeypatch, tmp_path):
    img = tmp_path / "WhisperFlow-0.4.400-x86_64.AppImage"
    img.write_bytes(b"\x7fELFNEW" + b"\0" * 60)
    old = img.with_name(img.name + ".old")
    old.write_bytes(b"\x7fELFOLD" + b"\0" * 60)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPIMAGE", str(img))
    updater.cleanup_replaced_appimage()
    assert img.is_file()
    assert not old.exists()


def test_http_download_rejects_a_hash_mismatch(monkeypatch, tmp_path):
    dest = tmp_path / "out.AppImage"

    class _Resp:
        def read(self, n=-1):
            data = b"\x7fELF" + b"\0" * 60
            self.read = lambda n=-1: b""
            return data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(updater.urllib.request, "urlopen",
                        lambda *a, **k: _Resp())
    with pytest.raises(ValueError, match="hash mismatch"):
        updater._http_download("https://example/img", dest, expected_sha="nope")
    assert not dest.exists()
    assert not dest.with_name("out.AppImage.partial").exists()


def test_unavailable_updater_names_the_reason_once(monkeypatch):
    """A frozen build with no velopack says why, exactly once."""
    import sys

    import whisper_flow.updater as updater_module
    from whisper_flow.logging import recent_log

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(sys.modules, "velopack", None)
    monkeypatch.setattr(updater_module, "_availability_logged", False)
    assert updater_module.available() is False
    assert "velopack not importable" in recent_log(50)
    before = len(recent_log(200))
    assert updater_module.available() is False
    assert len(recent_log(200)) == before
