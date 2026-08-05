"""Guards on dropping the silence either side of what was actually said."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_flow.audio import (
    MIN_RECORDING_SECONDS,
    TRIM_PAD_MS,
    AudioRecorder,
    trim_silence,
)

RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(RATE * FRAME_MS / 1000) * 2       # 16-bit mono


class Vad:
    """A detector that hears speech in exactly the frames it is told to."""

    def __init__(self, voiced):
        self.voiced = set(voiced)
        self.seen = 0

    def is_speech(self, frame, rate):
        index, self.seen = self.seen, self.seen + 1
        return index in self.voiced


def _frames(count):
    return [bytes(FRAME_BYTES) for _ in range(count)]


def _trim(frames, voiced, pad_ms=0):
    return trim_silence(frames, Vad(voiced), RATE, FRAME_MS, pad_ms=pad_ms)


# ------------------------------------------------------------------ trimming
def test_silence_either_side_of_the_speech_is_dropped():
    """Push-to-talk opens before the sentence and closes after it."""
    assert len(_trim(_frames(40), voiced=range(20, 30))) == 10


def test_the_speech_itself_survives_intact():
    frames = _frames(20)
    for frame_index in range(5, 12):
        frames[frame_index] = bytes([frame_index]) * FRAME_BYTES
    kept = _trim(frames, voiced=range(5, 12))
    assert kept == frames[5:12]


def test_padding_keeps_the_opening_consonant():
    """A detector marks the frame a word became audible, not where it began."""
    kept = _trim(_frames(40), voiced=range(20, 30), pad_ms=150)
    assert len(kept) == 10 + 2 * (150 // FRAME_MS)


def test_padding_cannot_run_off_either_end():
    kept = _trim(_frames(6), voiced=range(0, 6), pad_ms=300)
    assert len(kept) == 6


def test_audio_with_no_speech_in_it_is_left_alone():
    """A dead microphone is something to report, not to truncate to nothing."""
    frames = _frames(30)
    assert _trim(frames, voiced=()) is frames


def test_a_detector_that_throws_leaves_the_audio_alone():
    class Broken:
        def is_speech(self, frame, rate):
            raise RuntimeError("no")

    frames = _frames(10)
    assert trim_silence(frames, Broken(), RATE, FRAME_MS) is frames


def test_frames_the_detector_cannot_judge_are_skipped_not_kept():
    """webrtcvad rejects odd-sized frames; a short tail must not fool it."""
    frames = _frames(10) + [b"\0\0"]
    kept = _trim(frames, voiced=range(2, 4))
    assert kept == frames[2:4]


# ------------------------------------------------------------ the recorder
def _recorder(trim=True):
    config = Mock()
    config.vad_mode = 2
    config.sample_rate = RATE
    config.frame_ms = FRAME_MS
    config.trim_silence = trim
    with patch("whisper_flow.audio.pyaudio", None):
        recorder = AudioRecorder(config, Mock())
    return recorder


def test_the_recorder_trims_through_its_own_detector():
    recorder = _recorder()
    recorder._new_vad = lambda: Vad(range(20, 30))
    kept = recorder.trim_frames(_frames(40))
    assert len(kept) == 10 + 2 * (TRIM_PAD_MS // FRAME_MS)
    assert len(kept) < 40


def test_turning_the_setting_off_keeps_every_frame():
    recorder = _recorder(trim=False)
    frames = _frames(40)
    assert recorder.trim_frames(frames) is frames


def test_a_trim_down_to_nothing_is_refused():
    """Below the length we discard as too short, the untrimmed audio is honest."""
    recorder = _recorder()
    recorder._new_vad = lambda: Vad([0])     # one 30ms frame of "speech"
    frames = _frames(40)
    assert recorder.trim_frames(frames) is frames
    assert MIN_RECORDING_SECONDS > FRAME_MS / 1000


def test_the_same_audio_trims_the_same_way_every_time():
    """webrtcvad adapts as it listens; a shared one made pass one different.

    The first dictation of a session used to keep noticeably more silence
    than the rest, because the detector was still settling on its estimate of
    the room. Real webrtcvad on purpose - a fake one cannot drift, so it
    cannot catch this.
    """
    recorder = _recorder()
    speech, room = _tone_frames(40), _room_frames(30)
    frames = room + speech + room

    kept = [len(recorder.trim_frames(list(frames))) for _ in range(4)]
    assert len(set(kept)) == 1, f"trim drifted across passes: {kept}"
    assert kept[0] < len(frames), "the guard is worthless if nothing is trimmed"


def _tone_frames(count):
    """Frames a voice detector hears speech in."""
    import numpy as np

    samples = int(RATE * FRAME_MS / 1000)
    out = []
    for index in range(count):
        t = np.arange(samples) + index * samples
        wave = (6000 * np.sin(2 * np.pi * 220 * t / RATE)
                + 2000 * np.sin(2 * np.pi * 700 * t / RATE))
        out.append(wave.astype(np.int16).tobytes())
    return out


def _room_frames(count):
    """Frames holding nothing but a quiet room."""
    import numpy as np

    samples = int(RATE * FRAME_MS / 1000)
    rng = np.random.default_rng(7)
    return [rng.normal(0, 20, samples).astype(np.int16).tobytes()
            for _ in range(count)]
