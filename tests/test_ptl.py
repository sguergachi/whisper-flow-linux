"""PTL - press-to-listen: hotkey press to the microphone capturing.

The one latency the user feels. Everything else optimised so far - the
overlay, the tray icons, the package imports, the audio device - is a
component of it, so it is measured directly and broken into stages, and the
slow part names itself instead of being guessed at.
"""

import time

from whisper_flow import logging as wf_logging
from whisper_flow.daemon import PressToListen


def setup_function():
    wf_logging.clear_log()


def test_the_total_and_every_stage_are_reported():
    ptl = PressToListen()
    ptl.mark("dispatch")
    ptl.mark("save window")
    ptl.mark("mic open")
    ptl.report()

    line = wf_logging.recent_log()
    assert "[PTL]" in line
    assert "press-to-listen" in line
    for stage in ("dispatch", "save window", "mic open"):
        assert stage in line


def test_a_slow_stage_is_visible_in_the_breakdown():
    """The point of stages: the expensive one is named, not inferred."""
    ptl = PressToListen()
    ptl.mark("quick")
    time.sleep(0.05)
    ptl.mark("slow")
    ptl.report()

    import re

    line = wf_logging.recent_log()
    slow_ms = int(re.search(r"slow (\d+)", line).group(1))
    assert slow_ms >= 40


def test_it_reports_once_however_often_it_is_asked():
    """Two presses in quick succession must not double-count one window."""
    ptl = PressToListen()
    ptl.mark("dispatch")
    ptl.report()
    ptl.report()
    assert wf_logging.recent_log().count("[PTL]") == 1


def test_the_total_covers_the_whole_window_not_just_the_stages():
    ptl = PressToListen()
    time.sleep(0.03)                 # time before the first mark still counts
    ptl.mark("late")
    ptl.report()

    total = int(wf_logging.recent_log().split("[PTL]")[1].split("ms")[0])
    assert total >= 30


def test_marking_after_reporting_is_ignored(monkeypatch):
    """A daemon method may fire late; it must not reopen a closed window."""
    from unittest.mock import Mock

    from whisper_flow.daemon import WhisperFlowDaemon

    daemon = WhisperFlowDaemon.__new__(WhisperFlowDaemon)
    daemon._ptl = PressToListen()
    daemon._ptl.report()
    daemon._mark_ptl("too late")     # must not raise or re-report
    assert wf_logging.recent_log().count("[PTL]") == 1

    daemon._ptl = None
    daemon._mark_ptl("no window")    # nothing in flight
    assert isinstance(daemon, WhisperFlowDaemon) or Mock
