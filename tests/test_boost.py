"""Rescuing a recording that was too quiet to transcribe.

From a real session: a whisper reached the microphone at peak 89-126 out of
32767 and whisper.cpp returned nothing, while peak 281 transcribed fine. The
speech was captured; it was just small. These cover lifting it without
inventing signal, and knowing when not to bother.
"""

import os
import tempfile
import wave

import numpy as np
import pytest

from whisper_flow.boost import (
    DEAD_PEAK,
    MAX_GAIN,
    boost_wav,
    needs_boost,
    rescue_wav,
)


def _wav(peak, seconds=1.0, dc=0, rate=16000, width=2):
    path = tempfile.mktemp(suffix=".wav")
    n = int(rate * seconds)
    t = np.linspace(0, seconds, n, False)
    sig = np.sin(2 * np.pi * 180 * t) * 0.6 + np.sin(2 * np.pi * 900 * t) * 0.4
    if peak:
        sig = sig / np.abs(sig).max() * peak
    sig = sig + dc
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(width)
        f.setframerate(rate)
        f.writeframes(sig.astype(np.int16).tobytes())
    return path


def _peak_of(path):
    with wave.open(path, "rb") as f:
        data = f.readframes(f.getnframes())
    return int(np.abs(np.frombuffer(data, dtype=np.int16)).max())


@pytest.fixture
def out_path():
    path = tempfile.mktemp(suffix=".wav")
    yield path
    if os.path.exists(path):
        os.unlink(path)


# ------------------------------------------------------------ when to bother
def test_a_whisper_is_worth_retrying_louder():
    """The levels that actually failed in a real session."""
    assert needs_boost(89)
    assert needs_boost(126)


def test_ordinary_speech_is_not_retried():
    """It already transcribed; being quiet was not the problem."""
    assert not needs_boost(8000)


def test_a_dead_line_is_not_retried():
    """Amplifying a noise floor only produces louder noise."""
    assert not needs_boost(8)
    assert not needs_boost(DEAD_PEAK - 1)


# ---------------------------------------------------------------- the lift
def test_a_quiet_recording_is_brought_up_to_a_usable_level(out_path):
    src = _wav(126)
    try:
        gain = boost_wav(src, out_path)
        assert gain and gain > 100
        assert _peak_of(out_path) > 20000
    finally:
        os.unlink(src)


def test_nothing_is_written_for_a_dead_recording(out_path):
    src = _wav(8)
    try:
        assert boost_wav(src, out_path) is None
        assert not os.path.exists(out_path)
    finally:
        os.unlink(src)


def test_a_loud_recording_is_left_alone(out_path):
    src = _wav(30000)
    try:
        assert boost_wav(src, out_path) is None
    finally:
        os.unlink(src)


def test_cafe_music_peak_still_gets_speech_rescue(out_path):
    """Music can make the peak high so boost_wav refuses; rescue must run."""
    rate = 16000
    n = rate * 2
    t = np.linspace(0, 2, n, False)
    music = np.sin(2 * np.pi * 100 * t) * 8000
    speech = np.zeros(n)
    speech[int(0.5 * rate):int(1.5 * rate)] = (
        np.sin(2 * np.pi * 1500 * t[int(0.5 * rate):int(1.5 * rate)]) * 200)
    samples = np.clip(music + speech, -32768, 32767).astype(np.int16)
    src = tempfile.mktemp(suffix=".wav")
    with wave.open(src, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(samples.tobytes())
    try:
        assert boost_wav(src, out_path) is None  # peak already loud
        assert rescue_wav(src, out_path) is not None
        assert os.path.exists(out_path)
        assert _peak_of(out_path) > 1000
    finally:
        os.unlink(src)


def test_a_constant_offset_is_removed_rather_than_amplified(out_path):
    """Gain would multiply a DC offset along with the voice, and a whisper
    riding on one would clip flat before the speech got loud enough."""
    src = _wav(100, dc=400)
    try:
        assert boost_wav(src, out_path)
        with wave.open(out_path, "rb") as f:
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        assert abs(int(data.mean())) < 2000     # centred, not offset
    finally:
        os.unlink(src)


def test_the_gain_is_capped(out_path):
    """Past a point the noise floor arrives with the speech."""
    src = _wav(DEAD_PEAK + 1)
    try:
        gain = boost_wav(src, out_path)
        assert gain is None or gain <= MAX_GAIN
    finally:
        os.unlink(src)


def test_the_result_never_clips_to_the_rail(out_path):
    src = _wav(126)
    try:
        boost_wav(src, out_path)
        with wave.open(out_path, "rb") as f:
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        assert np.abs(data).max() < 32767
    finally:
        os.unlink(src)


def test_the_audio_keeps_its_shape(out_path):
    """Amplifying must not change how long it is or how it is sampled."""
    src = _wav(126, seconds=1.5, rate=16000)
    try:
        boost_wav(src, out_path)
        with wave.open(src, "rb") as a, wave.open(out_path, "rb") as b:
            assert a.getnframes() == b.getnframes()
            assert a.getframerate() == b.getframerate()
            assert a.getnchannels() == b.getnchannels()
            assert a.getsampwidth() == b.getsampwidth()
    finally:
        os.unlink(src)


def test_an_unreadable_file_is_survived(out_path):
    assert boost_wav("/nonexistent/nope.wav", out_path) is None


def test_an_empty_recording_is_survived(out_path):
    src = _wav(0, seconds=0)
    try:
        assert boost_wav(src, out_path) is None
    finally:
        os.unlink(src)
