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
    rec._devices_logged = True          # not under test here
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
def test_a_wasapi_device_that_cannot_do_our_rate_is_still_used(recorder, monkeypatch):
    """It is kept and converted, rather than handed back for MME to take.

    WASAPI shared mode does not resample, so asking a 48kHz array for 16kHz
    fails - and returning None here meant the platform default through MME,
    which delivers the same microphone several times quieter. Measured on an
    AMD array in one room seconds apart: 429 peak through MME at 16kHz,
    1579 at its native 48kHz. That gap is whisper hearing a sentence or
    returning nothing at all.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    recorder.pa.get_host_api_count.return_value = 1
    recorder.pa.get_host_api_info_by_index.return_value = {
        "name": "Windows WASAPI", "defaultInputDevice": 9}
    recorder.pa.is_format_supported.return_value = False
    recorder.pa.get_device_info_by_index.return_value = {
        "defaultSampleRate": 48000.0}
    assert recorder._input_device_index() == 9


def test_audio_is_converted_to_the_rate_whisper_expects(recorder):
    """Whatever the device runs at, what leaves here is 16kHz.

    The frame length has to be exact as well as the rate: webrtcvad takes
    whole 10, 20 or 30ms frames and rejects anything a sample short, so a
    conversion that rounded would silently disable voice detection.
    """
    import numpy as np

    chunk = int(recorder.config.sample_rate * recorder.config.frame_ms / 1000)
    for source in (48000, 44100, 16000):
        frames = recorder._capture_frames(chunk, source)
        assert frames == (chunk if source == recorder.config.sample_rate
                          else round(chunk * source / recorder.config.sample_rate))
        # A tone in, the same tone out, at the right length.
        moment = np.arange(frames) / source
        tone = (np.sin(2 * np.pi * 440 * moment) * 12000).astype(np.int16)
        converted = recorder._resample(tone.tobytes(), source, chunk)
        assert len(converted) == chunk * 2, (
            f"{source}Hz produced {len(converted) // 2} samples, not {chunk}")
        loudest = int(np.abs(np.frombuffer(converted, dtype=np.int16)).max())
        assert loudest > 8000, (
            f"{source}Hz conversion lost the signal (peak {loudest})")


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


# ------------------------------------------------- did the microphone work?
def test_loudness_tells_speech_from_silence():
    """Whisper reporting blank audio is the same message whether the user
    said nothing or the capture device handed us silence. These are
    different problems: one is nothing to fix, the other is the wrong
    device, and the log has to say which."""
    import numpy as np

    speech = [np.random.randint(-9000, 9000, 480, dtype=np.int16).tobytes()
              for _ in range(30)]
    silence = [np.zeros(480, dtype=np.int16).tobytes() for _ in range(30)]

    peak, mean = AudioRecorder._loudness(speech)
    assert peak > 200 and mean > 0

    peak, mean = AudioRecorder._loudness(silence)
    assert peak == 0 and mean == 0


def test_loudness_survives_no_frames_and_bad_data():
    assert AudioRecorder._loudness([]) == (0, 0)
    assert AudioRecorder._loudness([b"\x01"]) == (0, 0)      # odd byte count


def test_a_barely_audible_recording_still_reads_as_silent():
    """A mic that is muted or on the wrong device gives dither, not zeros."""
    import numpy as np

    faint = [np.random.randint(-40, 40, 480, dtype=np.int16).tobytes()
             for _ in range(30)]
    peak, _ = AudioRecorder._loudness(faint)
    assert peak < 200


# ------------------------------------------------------- device visibility
def test_the_devices_are_listed_with_the_chosen_one_marked(recorder, monkeypatch):
    """Silence is nearly always the wrong device, and the log said nothing
    about which device was in use or what else was available."""
    from whisper_flow import logging as wf

    wf.clear_log()
    monkeypatch.setattr(recorder, "_input_device_index", lambda: 2)
    recorder._devices_logged = False
    recorder.pa.get_default_input_device_info.return_value = {
        "index": 0, "name": "Speakers (loopback)", "defaultSampleRate": 44100.0}
    recorder.pa.get_device_count.return_value = 3
    devices = [
        {"name": "Speakers (loopback)", "maxInputChannels": 2,
         "hostApi": 0, "defaultSampleRate": 44100.0},
        {"name": "Headphones", "maxInputChannels": 0,
         "hostApi": 0, "defaultSampleRate": 44100.0},
        {"name": "Blue Yeti", "maxInputChannels": 1,
         "hostApi": 1, "defaultSampleRate": 48000.0},
    ]
    recorder.pa.get_device_info_by_index.side_effect = lambda i: devices[i]
    recorder.pa.get_host_api_info_by_index.side_effect = (
        lambda i: {"name": ["MME", "Windows WASAPI"][i]})

    recorder.log_input_devices()
    logged = wf.recent_log()

    assert "Blue Yeti" in logged and "<- using" in logged
    assert "Speakers (loopback)" in logged        # the default is named too
    assert "Headphones" not in logged             # outputs are not capture


def test_the_device_list_is_logged_once_not_per_recording(recorder, monkeypatch):
    from whisper_flow import logging as wf

    monkeypatch.setattr(recorder, "_input_device_index", lambda: 0)
    recorder._devices_logged = False
    recorder.pa.get_default_input_device_info.return_value = {
        "index": 0, "name": "Mic", "defaultSampleRate": 48000.0}
    recorder.pa.get_device_count.return_value = 0

    wf.clear_log()
    for _ in range(5):
        recorder.log_input_devices()
    assert wf.recent_log().count("default input") == 1


def test_a_failure_to_enumerate_does_not_stop_a_recording(recorder, monkeypatch):
    recorder._devices_logged = False
    recorder.pa.get_default_input_device_info.side_effect = OSError("no devices")
    recorder.log_input_devices()        # must not raise


# ------------------------------------------- the host API can misreport itself
def _reported_machine(recorder, wasapi_default, default_input_name):
    """The device table from a real report, where WASAPI named a WDM-KS device.

    Refusing the bad answer is not enough on its own: falling through lands
    on the platform default, which is MME - the same microphone several
    times quieter, and the whole reason a WASAPI device is wanted.
    """
    MME, DSOUND, WASAPI, WDMKS = 0, 1, 2, 3
    devices = {
        0:  ("Microsoft Sound Mapper - Input", MME, 2),
        1:  ("Microphone (Logi USB Headset)", MME, 1),
        17: ("Microphone (2- Realtek(R) Audio)", WASAPI, 2),
        18: ("Microphone (Logi USB Headset)", WASAPI, 1),
        21: ("Input (btha2dp.sys, MOMENTUM 3)", WDMKS, 2),
        35: ("Microphone (Logi USB Headset)", WDMKS, 1),
    }
    apis = {
        MME:    {"name": "MME", "defaultInputDevice": 0},
        DSOUND: {"name": "Windows DirectSound", "defaultInputDevice": -1},
        WASAPI: {"name": "Windows WASAPI",
                 "defaultInputDevice": wasapi_default},
        WDMKS:  {"name": "Windows WDM-KS", "defaultInputDevice": 21},
    }

    def info(index):
        if index not in devices:
            raise OSError("no such device")
        name, api, channels = devices[index]
        return {"name": name, "hostApi": api, "maxInputChannels": channels,
                "defaultSampleRate": 48000.0}

    recorder.pa.get_host_api_count.return_value = len(apis)
    recorder.pa.get_host_api_info_by_index.side_effect = lambda i: apis[i]
    recorder.pa.get_device_count.return_value = max(devices) + 1
    recorder.pa.get_device_info_by_index.side_effect = info
    recorder.pa.get_default_input_device_info.return_value = {
        "name": default_input_name}
    recorder.pa.is_format_supported.return_value = False
    return devices


def test_a_host_api_naming_another_apis_device_is_replaced_not_abandoned(
        recorder, monkeypatch):
    """WASAPI named device 21, which is WDM-KS - kernel streaming, exclusive
    to whatever already holds the endpoint. It would not open, capture fell
    back to MME, and the recording came back at a peak of 467 out of 32767.
    The right microphone was on the very API that had just misreported it.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    _reported_machine(recorder, wasapi_default=21,
                      default_input_name="Microphone (Logi USB Headset)")
    assert recorder._input_device_index() == 18


def test_the_replacement_is_the_microphone_windows_was_told_to_use(
        recorder, monkeypatch):
    """Not merely the first input on the API: that is a coin toss between a
    headset and whatever else is plugged in."""
    monkeypatch.setattr(sys, "platform", "win32")
    _reported_machine(recorder, wasapi_default=21,
                      default_input_name="Microphone (2- Realtek(R) Audio)")
    assert recorder._input_device_index() == 17


def test_a_host_api_with_no_default_still_gets_searched(recorder, monkeypatch):
    """-1 is no reason to hand the recording back to MME either."""
    monkeypatch.setattr(sys, "platform", "win32")
    _reported_machine(recorder, wasapi_default=-1,
                      default_input_name="Microphone (Logi USB Headset)")
    assert recorder._input_device_index() == 18


def test_an_unnameable_default_falls_back_to_the_first_input(
        recorder, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    _reported_machine(recorder, wasapi_default=21, default_input_name="")
    recorder.pa.get_default_input_device_info.side_effect = OSError("no default")
    assert recorder._input_device_index() == 17
