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
