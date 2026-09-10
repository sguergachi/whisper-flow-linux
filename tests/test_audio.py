"""Guards on the capture-stream warm pool."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_flow.audio import AudioRecorder, capture_name_listed


def _recorder() -> AudioRecorder:
    """A recorder with no audio system; the warm pool does not need one."""
    config = Mock()
    config.vad_mode = 2
    config.frame_ms = 30
    config.sample_rate = 16000
    with patch("whisper_flow.audio.pyaudio", None):
        return AudioRecorder(config, Mock())


def _cancel_warm_timer(recorder: AudioRecorder) -> None:
    if recorder._warm_timer is not None:
        recorder._warm_timer.cancel()
        recorder._warm_timer = None


def test_warm_stream_is_reused_when_the_chunk_matches():
    recorder = _recorder()
    stream = Mock()
    stream.is_active.return_value = False
    try:
        recorder._keep_stream_warm(stream, 480)

        assert recorder._open_input_stream(480) is stream
        stream.start_stream.assert_called_once()
        assert recorder._warm_stream is None
    finally:
        _cancel_warm_timer(recorder)


def test_stale_warm_timer_cannot_close_the_current_stream():
    """A timer firing for a superseded stream must find nothing to do.

    Cancelling a timer that has already begun firing does nothing, so the
    release runs anyway; without the identity check it closed the stream a
    newer recording had just stored, and the next press paid to reopen the
    microphone the pool was meant to keep warm.
    """
    recorder = _recorder()
    first, second = Mock(), Mock()
    try:
        recorder._keep_stream_warm(first, 480)
        recorder._keep_stream_warm(second, 480)
        first.close.assert_called_once()   # the displaced stream goes at once

        recorder._release_warm_stream(first)

        assert recorder._warm_stream is second
        second.close.assert_not_called()

        _cancel_warm_timer(recorder)
        recorder._release_warm_stream(second)
        assert recorder._warm_stream is None
        second.close.assert_called_once()
    finally:
        _cancel_warm_timer(recorder)


# ------------------------------------------------- the HUD's level file
def _frame_at(rms: int) -> bytes:
    """A 30ms frame whose RMS is exactly `rms`."""
    
    return np.full(480, rms, dtype=np.int16).tobytes()


def test_every_level_is_four_bytes_wide(tmp_path):
    """The overlay unpacks the file as int32s and nothing tells it otherwise.

    os.open defaults to text mode on Windows, where the CRT inserts a \r
    before every \n it is given - and 0x0A is a perfectly ordinary byte of a
    packed level. One of those shifts every later sample by a byte, and from
    there the overlay reads halves of neighbouring values as single ones.
    """
    import struct

    recorder = _recorder()
    path = tmp_path / "levels"
    path.write_bytes(b"")

    # 10, 266 and 2600 each carry an 0x0A; 13 carries the \r itself.
    written = [300, 10, 266, 2600, 13, 500]
    for rms in written:
        recorder._write_level(str(path), _frame_at(rms))

    raw = path.read_bytes()
    assert len(raw) == 4 * len(written), (
        f"{len(written)} levels wrote {len(raw)} bytes, not {4 * len(written)}")
    assert struct.unpack("<%di" % len(written), raw) == tuple(written)


def test_a_level_the_overlay_cannot_use_is_never_written(tmp_path):
    """Nothing above the RMS of full-scale 16-bit audio is a level at all."""
    import struct

    recorder = _recorder()
    path = tmp_path / "levels"
    path.write_bytes(b"")
    recorder._write_level(str(path), _frame_at(32767))

    (value,) = struct.unpack("<i", path.read_bytes())
    assert 0 <= value <= 32767


def test_cold_open_silence_is_skipped():
    """A freshly opened stream delivers zeros, then a ramp: cut to the signal.

    That silence made the floor read as a dead-silent room and the VAD's pad
    kept it in the trimmed file, where it poisoned the first transcription.
    """
    
    recorder = _recorder()
    rate = 16000
    frame = 480
    rng = np.random.default_rng(4)
    n = rate * 2
    audio = np.zeros(n)
    t = np.arange(n) / rate
    span = (t >= 0.5) & (t < 1.6)
    audio[span] = np.sin(2 * np.pi * 1800 * t[span]) * 80 + rng.normal(0, 30, span.sum())
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    frames = [audio[i:i+frame].tobytes() for i in range(0, len(audio)-frame+1, frame)]

    skipped = recorder._skip_cold_open_silence(frames)
    assert len(skipped) < len(frames)
    first = np.frombuffer(skipped[0], dtype=np.int16)
    assert abs(first).mean() > 30.0, "the signal onset must survive"


def test_whisper_with_a_small_mean_is_not_wrongly_skipped():
    """A genuine quiet whisper (frame mean ~50) must not look like warm-up."""
    
    recorder = _recorder()
    rate = 16000
    frame = 480
    rng = np.random.default_rng(5)
    n = rate * 2
    # No cold-open zeros: the stream is warm, the whisper is just quiet.
    audio = rng.normal(0, 8, n)
    t = np.arange(n) / rate
    span = (t >= 0.4) & (t < 1.6)
    audio[span] += np.sin(2 * np.pi * 1800 * t[span]) * 40
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    frames = [audio[i:i+frame].tobytes() for i in range(0, len(audio)-frame+1, frame)]

    skipped = recorder._skip_cold_open_silence(frames)
    # Room tone at mean ~8 is below the 30 threshold: nothing should be cut,
    # because the clip never had a warm-up to drop.
    assert len(skipped) == len(frames)


# ------------------------------------------------- input switching
def _pa_with(inputs, default):
    """A fake PortAudio: inputs is [(name, is_input)], default the name."""
    pa = Mock()
    infos = {}
    for index, (name, is_input) in enumerate(inputs):
        infos[index] = {"index": index, "name": name,
                        "maxInputChannels": 1 if is_input else 0,
                        "defaultSampleRate": 44100.0, "hostApi": 0}
    pa.get_device_count.return_value = len(infos)
    pa.get_device_info_by_index.side_effect = lambda i: infos[i]
    pa.get_default_input_device_info.return_value = {
        "index": next(i for i, (n, _) in enumerate(inputs) if n == default),
        "name": default, "defaultSampleRate": 44100.0}
    return pa


def test_input_signature_names_choice_default_and_offered():
    recorder = _recorder()
    recorder.pa = _pa_with([("Speakers", False), ("Mic A", True),
                            ("Mic B", True)], "Mic A")
    recorder.config.mic_device_index = None
    assert recorder.input_signature() == (
        None, "Mic A", frozenset({"Mic A", "Mic B"}))


def test_input_signature_none_without_audio():
    recorder = _recorder()
    assert recorder.pa is None
    assert recorder.input_signature() is None


def test_drop_warm_stream_releases_the_old_device():
    recorder = _recorder()
    stream = Mock()
    recorder._warm_stream = stream
    recorder.drop_warm_stream()
    assert recorder._warm_stream is None
    stream.close.assert_called_once()


def test_drop_warm_stream_is_a_noop_when_idle():
    recorder = _recorder()
    recorder.drop_warm_stream()  # must not raise with nothing held
    assert recorder._warm_stream is None


# ------------------------------------------------- dead-device fallback
def _live_recorder(pinned=7, rate=48000):
    """A recorder whose only mic is pinned, running hot like the BRIO."""
    from unittest.mock import Mock, patch

    config = Mock()
    config.vad_mode = 2
    config.frame_ms = 30
    config.sample_rate = 16000
    config.mic_device_index = pinned
    config.speedup_audio = 1.0
    with patch("whisper_flow.audio.pyaudio", Mock()):
        recorder = AudioRecorder(config, Mock())
    # The container has no audio stack; keep the (mocked) module present
    # so the per-recording check passes.
    recorder._check_pyaudio = lambda: True
    infos = {7: {"index": 7, "name": "Test Mic",
                 "maxInputChannels": 1, "defaultSampleRate": float(rate),
                 "hostApi": 0}}
    pa = Mock()
    pa.get_device_count.return_value = max(infos) + 1
    pa.get_device_info_by_index.side_effect = lambda i: infos[i]
    pa.get_default_input_device_info.return_value = {
        "index": 30, "name": "default", "defaultSampleRate": 44100.0}
    recorder.pa = pa
    return recorder


def _zero_stream(frames_before_stop, stop_event, samples_per_read=1440):
    """A stream that opens fine and delivers exact digital zeros."""
    from unittest.mock import Mock

    stream = Mock()
    stream.is_active.return_value = True
    calls = {"n": 0}

    def read(n, *a, **k):
        calls["n"] += 1
        if calls["n"] >= frames_before_stop:
            stop_event.set()
        return b"\x00" * (samples_per_read * 2)

    stream.read.side_effect = read
    return stream


def test_all_zero_capture_avoids_the_device():
    import threading

    recorder = _live_recorder()
    stop = threading.Event()
    recorder.pa.open.return_value = _zero_stream(40, stop)
    path = recorder.record_push_to_talk("super+alt", stop_event=stop)
    assert path is not None
    assert recorder._avoid_device == 7
    # Next press resolves to the platform default instead.
    assert recorder._input_device_index() is None


def test_live_capture_keeps_the_pin():
    import threading

    recorder = _live_recorder()
    stop = threading.Event()
    stream = _zero_stream(40, stop)
    loud = (np.zeros(1440, dtype=np.int16) + 2000).tobytes()
    stream.read.side_effect = lambda n, *a, **k: loud
    recorder.pa.open.return_value = stream
    stop_trigger = threading.Timer(0.5, stop.set)
    stop_trigger.start()
    try:
        path = recorder.record_push_to_talk("super+alt", stop_event=stop)
    finally:
        stop_trigger.cancel()
    assert path is not None
    assert recorder._avoid_device is None
    assert recorder._input_device_index() == 7


def test_silence_on_the_default_has_nowhere_to_fall_back():
    recorder = _live_recorder(pinned=None)
    recorder.config.mic_device_index = None
    recorder._last_open_device = (None, "default")
    frames = [b"\x00" * 960 for _ in range(40)]
    recorder._note_capture_result(frames, 2.0)
    assert recorder._avoid_device is None


def test_short_silence_does_not_count():
    recorder = _live_recorder()
    recorder._last_open_device = (7, "Test Mic")
    frames = [b"\x00" * 960 for _ in range(10)]
    recorder._note_capture_result(frames, 0.3)
    assert recorder._avoid_device is None


def test_a_live_mic_forgives_a_muted_one():
    recorder = _live_recorder()
    recorder._avoid_device = 7
    frames = [(np.zeros(480, dtype=np.int16) + 500).tobytes()
              for _ in range(40)]
    recorder._note_capture_result(frames, 2.0)
    assert recorder._avoid_device is None
    assert recorder._input_device_index() == 7


def test_alsa_plugin_pcms_are_not_listed_as_microphones():
    """PortAudio enumerates lavrate/pulse/default as capture devices.

    They are not microphones. Selecting one is why the Linux settings
    Test sat at zero while the desktop default source was a real USB
    mic PipeWire already held.
    """
    for name in ("default", "sysdefault", "pulse", "pipewire", "lavrate",
                 "samplerate", "speexrate", "speex", "upmix", "vdownmix"):
        assert capture_name_listed(name, "ALSA") is False, name
        assert capture_name_listed(name, "") is False, name
    assert capture_name_listed("surround51", "ALSA") is False
    assert capture_name_listed(
        "HDA Intel PCH: ALC1220 Analog (hw:0,0)", "ALSA") is True
    assert capture_name_listed("Logitech BRIO: USB Audio (hw:2,0)", "ALSA")
    # Windows names must not be filtered by the ALSA plugin list.
    assert capture_name_listed("default", "WASAPI") is True
    assert capture_name_listed("Headset", "MME") is True


def test_opens_name_the_device_and_rate():
    recorder = _live_recorder()
    recorder.pa.open.return_value = Mock()
    logged = []
    import whisper_flow.audio as audio_module
    from unittest.mock import patch
    with patch.object(audio_module, "log", logged.append):
        recorder._open_input_stream(480)
    line = " ".join(str(x) for x in logged)
    assert "Test Mic" in line and "48000Hz" in line


# ------------------------------------------------- hardware rescans
def _fake_pa(names, default):
    """A PortAudio snapshot: names is [str], default one of them."""
    from unittest.mock import Mock

    pa = Mock()
    infos = {i: {"index": i, "name": n, "maxInputChannels": 1,
                 "defaultSampleRate": 44100.0, "hostApi": 0}
             for i, n in enumerate(names)}
    pa.get_device_count.return_value = len(infos)
    pa.get_device_info_by_index.side_effect = lambda i: infos[i]
    pa.get_default_input_device_info.return_value = {
        "index": names.index(default), "name": default,
        "defaultSampleRate": 44100.0}
    return pa


def _rescan_recorder(pa, probes):
    """A recorder on pa; patch pyaudio so PyAudio() yields probes in turn."""
    from unittest.mock import Mock, patch

    import whisper_flow.audio as audio_module

    recorder = _recorder()
    recorder.pa = pa
    fake_pyaudio = Mock()
    fake_pyaudio.PyAudio.side_effect = list(probes)
    return recorder, patch.object(audio_module, "pyaudio", fake_pyaudio)


def test_refresh_keeps_instance_when_unchanged():
    recorder, ctx = _rescan_recorder(
        _fake_pa(["Mic A"], "Mic A"), [_fake_pa(["Mic A"], "Mic A")])
    with ctx:
        assert recorder.refresh_devices() is False
    # Same world: old instance kept, probe discarded.
    assert recorder.pa.get_default_input_device_info()["name"] == "Mic A"


def test_refresh_swaps_instance_on_new_hardware():
    from unittest.mock import Mock

    recorder, ctx = _rescan_recorder(
        _fake_pa(["Mic A"], "Mic A"), [_fake_pa(["Mic A", "Headset"], "Headset")])
    old = recorder.pa
    stream = Mock()
    recorder._warm_stream = stream
    with ctx:
        assert recorder.refresh_devices() is True
    assert recorder._warm_stream is None
    stream.close.assert_called_once()
    old.terminate.assert_called_once()
    assert recorder.input_signature()[1] == "Headset"


def test_refresh_survives_failed_probe():
    from unittest.mock import Mock, patch

    import whisper_flow.audio as audio_module

    recorder = _recorder()
    pa = _fake_pa(["Mic A"], "Mic A")
    recorder.pa = pa
    fake_pyaudio = Mock()
    fake_pyaudio.PyAudio.side_effect = OSError("no audio")
    with patch.object(audio_module, "pyaudio", fake_pyaudio):
        assert recorder.refresh_devices() is False
    assert recorder.pa is pa


def test_refresh_recovers_when_audio_was_down():
    from unittest.mock import Mock, patch

    import whisper_flow.audio as audio_module

    recorder = _recorder()
    assert recorder.pa is None
    probe = _fake_pa(["Mic A"], "Mic A")
    fake_pyaudio = Mock()
    fake_pyaudio.PyAudio.return_value = probe
    with patch.object(audio_module, "pyaudio", fake_pyaudio):
        assert recorder.refresh_devices() is True
    assert recorder.pa is probe
