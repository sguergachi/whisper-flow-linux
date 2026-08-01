"""Tests for what the user gets when something fails.

A toast holds one line. The rule these pin down is that every failure the
user sees also puts the log that explains it on the clipboard, so it can be
pasted into a message rather than described from memory.
"""

import sys
import tempfile
from unittest.mock import Mock, patch

import pytest

from whisper_flow import logging as wf_logging
from whisper_flow.daemon import WhisperFlowDaemon


@pytest.fixture(autouse=True)
def clean_log():
    wf_logging.clear_log()
    yield
    wf_logging.clear_log()


@pytest.fixture
def daemon():
    with (
        patch("whisper_flow.daemon.Config") as config_class,
        patch("whisper_flow.daemon.WhisperFlow"),
        patch("whisper_flow.daemon.HotkeyManager"),
        patch("whisper_flow.daemon.HUD"),
        patch("whisper_flow.daemon.LocalBackend"),
    ):
        config = Mock()
        config.hotkey_transcribe = "super+alt"
        config.hotkey_command = "cmd+shift+alt"
        config.hotkey_auto_transcribe = "ctrl+alt+space"
        config.local_whisper_url = None
        config.model_name = "ggml-base.en"
        config.live_transcription = True
        config.logging_enabled = False
        config_class.return_value = config
        d = WhisperFlowDaemon(tempfile.mkdtemp())

    d.backend.describe.return_value = "Windows / cpu"
    d.hud.crash_log.return_value = ""
    d.notify = Mock()
    return d


def _clipboard(daemon):
    """Capture what gets copied, and report success.

    The mock is spec'd against the real SystemManager, so if the daemon ever
    calls a clipboard method that does not exist, this fails instead of
    silently reporting "not copied" the way the shipped bug did.
    """
    from whisper_flow.system import SystemManager

    seen = {}
    manager = Mock(spec=SystemManager)
    manager.copy_to_clipboard.side_effect = (
        lambda text: seen.setdefault("text", text) is None or True
    )
    daemon.transcribe_app.system_manager = manager
    return seen


# ------------------------------------------------------------------ the ring
def test_lines_are_kept_even_when_logging_is_off():
    """The frozen Windows build has no console, so printing records nothing."""
    wf_logging.set_logging_enabled(False)
    wf_logging.log("[AUDIO] the microphone gave nothing")
    assert "the microphone gave nothing" in wf_logging.recent_log()


def test_the_ring_does_not_grow_without_bound():
    for i in range(wf_logging._RING_SIZE + 50):
        wf_logging.log(f"line {i}")
    kept = wf_logging.recent_log(10_000).splitlines()
    assert len(kept) == wf_logging._RING_SIZE
    assert "line 0" not in wf_logging.recent_log(10_000)


def test_logging_never_raises_on_an_awkward_argument():
    class Hostile:
        def __str__(self):
            raise RuntimeError("nope")

    wf_logging.log(Hostile())           # must not propagate


# --------------------------------------------------------------- the report
def test_a_failure_copies_the_log_that_explains_it(daemon):
    seen = _clipboard(daemon)
    wf_logging.log("[AUDIO] no frames captured - the microphone gave nothing")
    daemon._report_failure("Recording failed (transcribe)")

    report = seen["text"]
    assert "the microphone gave nothing" in report
    assert "Recording failed (transcribe)" in report
    assert "super+alt" in report                 # the binding actually in use
    assert "ggml-base.en" in report


def test_the_notification_says_the_details_were_copied(daemon):
    _clipboard(daemon)
    daemon._report_failure("Recording failed")
    assert "copied to clipboard" in daemon.notify.call_args[0][0]


def test_a_failure_is_still_announced_when_the_clipboard_refuses(daemon):
    from whisper_flow.system import SystemManager
    manager = Mock(spec=SystemManager)
    manager.copy_to_clipboard.return_value = False
    daemon.transcribe_app.system_manager = manager
    daemon._report_failure("Recording failed")
    message = daemon.notify.call_args[0][0]
    assert "Recording failed" in message
    assert "could not reach the clipboard" in message


def test_a_broken_clipboard_does_not_swallow_the_failure(daemon):
    from whisper_flow.system import SystemManager
    manager = Mock(spec=SystemManager)
    manager.copy_to_clipboard.side_effect = OSError("no clipboard")
    daemon.transcribe_app.system_manager = manager
    daemon._report_failure("Recording failed")
    daemon.notify.assert_called_once()


def test_a_crashing_overlay_is_included(daemon):
    """No HUD plus a failure should say why the HUD never appeared."""
    seen = _clipboard(daemon)
    daemon.hud.crash_log.return_value = "overlay exited 1\nModuleNotFoundError"
    daemon._report_failure("Recording failed")
    assert "ModuleNotFoundError" in seen["text"]


def test_a_traceback_is_included_when_there_is_one(daemon):
    seen = _clipboard(daemon)
    daemon._report_failure("Recording error", "Traceback...\nValueError: bad")
    assert "ValueError: bad" in seen["text"]


def test_an_unreachable_backend_does_not_break_the_report(daemon):
    seen = _clipboard(daemon)
    daemon.backend.describe.side_effect = RuntimeError("engine gone")
    daemon._report_failure("Recording failed")
    assert "engine gone" in seen["text"]


# ----------------------------------------------------------- copy last error
def test_the_last_error_can_be_copied_again_later(daemon):
    seen = _clipboard(daemon)
    daemon._report_failure("Recording failed")
    first = seen["text"]

    seen.clear()
    daemon.copy_last_error()
    assert seen["text"] == first                # same report, copied again


def test_copying_says_so_when_nothing_has_failed(daemon):
    _clipboard(daemon)
    daemon.copy_last_error()
    assert "No errors" in daemon.notify.call_args[0][0]


# ------------------------------------------------- the interface really exists
def test_system_manager_actually_has_the_method_the_daemon_calls():
    """The bug: the daemon called copy_to_clipboard, the method was private.

    Every other test mocked the name it wished for, so nothing noticed until
    a real failure produced an empty clipboard.
    """
    from whisper_flow.system import SystemManager

    assert callable(getattr(SystemManager, "copy_to_clipboard", None))


def test_the_daemon_only_calls_methods_system_manager_defines(daemon):
    """A spec'd mock raises on anything SystemManager does not define."""
    from whisper_flow.system import SystemManager

    manager = Mock(spec=SystemManager)
    daemon.transcribe_app.system_manager = manager
    daemon._report_failure("Recording failed (transcribe)")
    manager.copy_to_clipboard.assert_called_once()


@pytest.mark.skipif(sys.platform == "win32",
                    reason="drives the POSIX helper path; Windows is covered "
                           "for real in test_windows_native.py")
def test_the_report_reaches_a_real_system_manager(tmp_path, monkeypatch):
    """End to end through the actual class, not a mock of it.

    This is the test that would have caught the shipped bug: it exercises
    the real SystemManager, with only the platform clipboard call replaced.
    """
    from whisper_flow.config import Config
    from whisper_flow.system import SystemManager

    monkeypatch.setenv("WHISPER_FLOW_CONFIG_DIR", str(tmp_path))
    manager = SystemManager(Config(config_dir=tmp_path))

    captured = {}
    monkeypatch.setattr(manager, "_is_wayland", lambda: False)
    monkeypatch.setattr("whisper_flow.system.shutil.which",
                        lambda name: "/usr/bin/xclip" if name == "xclip" else None)

    class FakeStdin:
        def write(self, data):
            captured["data"] = data

        def close(self):
            captured["closed"] = True

    class FakeProc:
        stdin = FakeStdin()

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("whisper_flow.system.subprocess.Popen",
                        lambda *a, **k: FakeProc())

    assert manager.copy_to_clipboard("hello report") is True
    assert b"hello report" in captured["data"]


def test_a_clipboard_helper_that_stays_alive_counts_as_success(monkeypatch):
    """wl-copy and xclip keep running to serve the selection; that is normal."""
    import subprocess

    from whisper_flow.system import SystemManager

    class Stdin:
        def write(self, data): pass
        def close(self): pass

    class Lingering:
        stdin = Stdin()

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("wl-copy", timeout)

    monkeypatch.setattr("whisper_flow.system.subprocess.Popen",
                        lambda *a, **k: Lingering())
    assert SystemManager._pipe_to_clipboard(["wl-copy"], "text") is True


def test_a_helper_that_will_not_take_the_text_is_a_failure(monkeypatch):
    from whisper_flow.system import SystemManager

    class Stdin:
        def write(self, data):
            raise BrokenPipeError("gone")

        def close(self): pass

    class Dead:
        stdin = Stdin()

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr("whisper_flow.system.subprocess.Popen",
                        lambda *a, **k: Dead())
    assert SystemManager._pipe_to_clipboard(["wl-copy"], "text") is False
