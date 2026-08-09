"""Guards on the capture-stream warm pool."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_flow.audio import AudioRecorder


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
    import numpy as np

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
    import numpy as np

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
    import numpy as np

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
