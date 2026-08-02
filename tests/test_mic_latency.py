"""Getting the microphone live quickly after the hotkey.

Opening the capture stream is the whole delay between pressing the hotkey
and being able to speak. Two things reduce it: not using PortAudio's default
host API on Windows, which is MME and dates from 1991, and not reopening the
stream for every recording.
"""

import sys
from unittest.mock import Mock

import pytest

from whisper_flow.audio import MIC_KEEP_WARM_SECONDS, AudioRecorder


@pytest.fixture
def recorder():
    rec = AudioRecorder.__new__(AudioRecorder)     # no real audio system
    rec.config = Mock(mic_device_index=None, sample_rate=16000, frame_ms=30)
    rec.pa = Mock()
    rec._warm_stream = None
    rec._warm_chunk = None
    rec._warm_timer = None
    import threading
    rec._stream_lock = threading.Lock()
    return rec


# ------------------------------------------------------------ device choice
def test_an_explicit_device_always_wins(recorder, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    recorder.config.mic_device_index = 7
    assert recorder._input_device_index() == 7


def test_the_default_device_is_used_off_windows(recorder, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert recorder._input_device_index() is None


def test_windows_picks_wasapi_over_the_default_host_api(recorder, monkeypatch):
    """PortAudio lists MME first, so the unqualified default is the slow one."""
    monkeypatch.setattr(sys, "platform", "win32")
    apis = [
        {"name": "MME", "defaultInputDevice": 0},
        {"name": "Windows DirectSound", "defaultInputDevice": 1},
        {"name": "Windows WASAPI", "defaultInputDevice": 4},
        {"name": "Windows WDM-KS", "defaultInputDevice": 5},
    ]
    recorder.pa.get_host_api_count.return_value = len(apis)
    recorder.pa.get_host_api_info_by_index.side_effect = lambda i: apis[i]
    assert recorder._input_device_index() == 4


def test_a_missing_wasapi_falls_back_rather_than_failing(recorder, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    recorder.pa.get_host_api_count.return_value = 1
    recorder.pa.get_host_api_info_by_index.return_value = {
        "name": "MME", "defaultInputDevice": 0}
    assert recorder._input_device_index() is None


def test_a_wasapi_host_with_no_input_device_is_skipped(recorder, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    recorder.pa.get_host_api_count.return_value = 1
    recorder.pa.get_host_api_info_by_index.return_value = {
        "name": "Windows WASAPI", "defaultInputDevice": -1}
    assert recorder._input_device_index() is None


def test_a_broken_host_api_query_does_not_stop_recording(recorder, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    recorder.pa.get_host_api_count.side_effect = OSError("portaudio confused")
    assert recorder._input_device_index() is None


# -------------------------------------------------------------- warm stream
def test_a_warm_stream_is_reused_instead_of_reopened(recorder):
    warm = Mock()
    warm.is_active.return_value = False
    recorder._warm_stream, recorder._warm_chunk = warm, 480

    assert recorder._open_input_stream(480) is warm
    warm.start_stream.assert_called_once()
    recorder.pa.open.assert_not_called()
    assert recorder._warm_stream is None      # handed over, not shared


def test_a_stream_of_the_wrong_shape_is_not_reused(recorder):
    warm = Mock()
    recorder._warm_stream, recorder._warm_chunk = warm, 480
    recorder._open_input_stream(960)
    recorder.pa.open.assert_called_once()


def test_an_unusable_warm_stream_is_replaced(recorder):
    warm = Mock()
    warm.is_active.return_value = False
    warm.start_stream.side_effect = OSError("device gone")
    recorder._warm_stream, recorder._warm_chunk = warm, 480

    recorder._open_input_stream(480)
    warm.close.assert_called_once()
    recorder.pa.open.assert_called_once()


def test_a_finished_stream_is_kept_open_for_the_next_recording(recorder):
    stream = Mock()
    recorder._keep_stream_warm(stream, 480)
    try:
        stream.stop_stream.assert_called_once()
        stream.close.assert_not_called()
        assert recorder._warm_stream is stream
        assert recorder._warm_timer.interval == MIC_KEEP_WARM_SECONDS
    finally:
        recorder._warm_timer.cancel()


def test_the_microphone_is_released_when_it_goes_idle(recorder):
    stream = Mock()
    recorder._keep_stream_warm(stream, 480)
    recorder._warm_timer.cancel()
    recorder._release_warm_stream()
    stream.close.assert_called_once()
    assert recorder._warm_stream is None


def test_only_one_stream_is_ever_held(recorder):
    """Two in a row must not leave the first one holding the microphone."""
    first, second = Mock(), Mock()
    recorder._keep_stream_warm(first, 480)
    recorder._keep_stream_warm(second, 480)
    try:
        first.close.assert_called_once()
        assert recorder._warm_stream is second
    finally:
        recorder._warm_timer.cancel()


def test_a_stream_that_will_not_stop_is_closed_outright(recorder):
    stream = Mock()
    stream.stop_stream.side_effect = OSError("already gone")
    recorder._keep_stream_warm(stream, 480)
    stream.close.assert_called_once()
    assert recorder._warm_stream is None


# ------------------------------------------- the device must never break it
def test_a_wasapi_device_that_cannot_do_our_rate_is_not_used(recorder, monkeypatch):
    """WASAPI shared mode does not resample; MME did. Choosing it blindly
    broke recording outright with "invalid sample rate"."""
    monkeypatch.setattr(sys, "platform", "win32")
    recorder.pa.get_host_api_count.return_value = 1
    recorder.pa.get_host_api_info_by_index.return_value = {
        "name": "Windows WASAPI", "defaultInputDevice": 9}
    recorder.pa.is_format_supported.return_value = False
    assert recorder._input_device_index() is None


def test_a_wasapi_device_that_can_do_our_rate_is_used(recorder, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    recorder.pa.get_host_api_count.return_value = 1
    recorder.pa.get_host_api_info_by_index.return_value = {
        "name": "Windows WASAPI", "defaultInputDevice": 9}
    recorder.pa.is_format_supported.return_value = True
    assert recorder._input_device_index() == 9


def test_an_unsupported_rate_query_that_raises_means_no(recorder, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    recorder.pa.is_format_supported.side_effect = ValueError("invalid rate")
    assert recorder._device_supports_our_rate(9) is False


def test_a_device_that_will_not_open_falls_back_to_the_default(recorder, monkeypatch):
    """A chosen device must never be why a recording does not happen."""
    monkeypatch.setattr(recorder, "_input_device_index", lambda: 9)
    opened = []

    def fake_open(**kw):
        opened.append(kw["input_device_index"])
        if kw["input_device_index"] is not None:
            raise OSError("[Errno -9997] Invalid sample rate")
        return Mock()

    recorder.pa.open.side_effect = lambda **kw: fake_open(**kw)
    assert recorder._open_input_stream(480) is not None
    assert opened == [9, None]


def test_the_default_device_failing_is_still_reported(recorder, monkeypatch):
    monkeypatch.setattr(recorder, "_input_device_index", lambda: None)
    recorder.pa.open.side_effect = OSError("no audio device at all")
    with pytest.raises(OSError):
        recorder._open_input_stream(480)
