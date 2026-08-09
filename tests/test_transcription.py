"""Tests for the transcription request, and the encoder window it asks for."""

import wave
from unittest.mock import Mock

import pytest

from whisper_flow import transcription as transcription_module
from whisper_flow.transcription import (
    AUDIO_CTX_FULL,
    AUDIO_CTX_MIN,
    TranscriptionService,
    audio_context,
    is_hallucination,
    wav_duration,
)


def test_hallucination_tags_are_rejected():
    assert is_hallucination("*sad music*")
    assert is_hallucination("[Music]")
    assert is_hallucination("BLANK_AUDIO")
    assert not is_hallucination("How are random errors detected?")


class FakeConfig:
    local_whisper_url = "http://127.0.0.1:8082"
    fast_encoder = True
    beam_size = 1
    best_of = 2
    no_speech_thold = 0.4
    suppress_nst = False


def _wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\0\0" * int(rate * seconds))
    return str(path)


# ------------------------------------------------------------ encoder window
def test_a_short_dictation_asks_for_the_smallest_useful_window():
    """Most dictation is a few seconds, and pays for 30 unless it says so."""
    assert audio_context(1.0) == AUDIO_CTX_MIN
    assert audio_context(3.0) == AUDIO_CTX_MIN
    assert audio_context(8.0) == AUDIO_CTX_MIN


def test_the_window_never_goes_below_the_floor():
    """Under ~768 the transcript drifts even when the speech still fits."""
    for seconds in (0.4, 1.0, 2.0, 5.0, 13.0):
        assert audio_context(seconds) >= AUDIO_CTX_MIN


def test_a_longer_clip_gets_a_window_that_still_contains_it():
    """The window is what the encoder can hear; speech past it is lost."""
    for seconds in (16.0, 20.0, 24.0):
        context = audio_context(seconds)
        assert context == 0 or context / (AUDIO_CTX_FULL / 30.0) >= seconds


def test_the_window_is_asked_for_in_multiples_of_256():
    """whisper.cpp pads audio_ctx to 256, so anything else is a silent round."""
    for seconds in (1.0, 6.0, 12.0, 15.0, 18.0, 22.0):
        context = audio_context(seconds)
        assert context % 256 == 0


def test_anything_whisper_would_chunk_gets_the_whole_window():
    """The setting applies per 30s chunk, so a short one deafens every chunk."""
    assert audio_context(30.0) == 0
    assert audio_context(45.0) == 0
    assert audio_context(600.0) == 0


def test_an_unmeasurable_clip_asks_for_nothing_unusual():
    assert audio_context(0.0) == 0
    assert audio_context(-1.0) == 0


# ------------------------------------------------------------- wav duration
def test_duration_is_read_from_the_file(tmp_path):
    assert wav_duration(_wav(tmp_path / "a.wav", 2.5)) == pytest.approx(2.5)


def test_an_unreadable_file_reports_no_duration(tmp_path):
    broken = tmp_path / "broken.wav"
    broken.write_text("not a wav")
    assert wav_duration(str(broken)) == 0.0
    assert wav_duration(str(tmp_path / "missing.wav")) == 0.0


# ----------------------------------------------------------------- the post
def _service_posting(monkeypatch, config):
    posted = {}

    def fake_post(url, files=None, data=None, timeout=None):
        posted["url"] = url
        posted["data"] = data
        response = Mock()
        response.json.return_value = {"text": "hello"}
        return response

    monkeypatch.setattr(transcription_module.requests, "post", fake_post)
    return TranscriptionService(config), posted


def test_the_window_is_sent_with_the_audio(tmp_path, monkeypatch):
    """The server takes it per request, so no restart sizes it."""
    service, posted = _service_posting(monkeypatch, FakeConfig())
    assert service.transcribe_audio(_wav(tmp_path / "a.wav", 3.0)) == "hello"
    assert posted["data"]["audio_ctx"] == str(AUDIO_CTX_MIN)


def test_a_long_recording_sends_no_window_at_all(tmp_path, monkeypatch):
    service, posted = _service_posting(monkeypatch, FakeConfig())
    service.transcribe_audio(_wav(tmp_path / "a.wav", 40.0))
    assert "audio_ctx" not in posted["data"]


def test_turning_the_setting_off_sends_no_audio_ctx(
        tmp_path, monkeypatch):
    """fast_encoder off must not shrink the encoder window.

    Inference still carries temperature / no_speech / language knobs that
    help noisy-room decoding; those are not gated on fast_encoder.
    """
    config = FakeConfig()
    config.fast_encoder = False
    service, posted = _service_posting(monkeypatch, config)
    service.transcribe_audio(_wav(tmp_path / "a.wav", 3.0))
    assert "audio_ctx" not in posted["data"]
    assert posted["data"].get("temperature") == "0.0"
    assert posted["data"].get("language") == "en"


def test_the_shortened_window_is_off_unless_it_is_asked_for():
    """It is worth 1.9x and it changes words. Speed does not get the default.

    Measured against a real whisper.cpp server on base.en, "ask not what your
    country can do for you" came back as "asked not" on two clips of five at
    every window short enough to be worth setting. Someone dictating cannot
    see which words were changed, so this cannot be the behaviour they get
    without choosing it.
    """
    from whisper_flow.config import Config

    assert Config().fast_encoder is False


# ------------------------------------------------------------- decode knobs
def test_defaults_are_greedy_and_send_the_usual_thresholds(tmp_path, monkeypatch):
    """A fresh install must not change what is sent to the server."""
    config = FakeConfig()
    service, posted = _service_posting(monkeypatch, config)
    service.transcribe_audio(_wav(tmp_path / "a.wav", 3.0))
    assert "beam_size" not in posted["data"]
    assert posted["data"].get("no_speech_thold") == "0.4"
    assert "suppress_nst" not in posted["data"]


def test_beam_search_is_sent_with_its_best_of(tmp_path, monkeypatch):
    config = FakeConfig()
    config.beam_size = 5
    config.best_of = 3
    service, posted = _service_posting(monkeypatch, config)
    service.transcribe_audio(_wav(tmp_path / "a.wav", 3.0))
    assert posted["data"].get("beam_size") == "5"
    assert posted["data"].get("best_of") == "3"


def test_no_speech_threshold_comes_from_settings(tmp_path, monkeypatch):
    config = FakeConfig()
    config.no_speech_thold = 0.6
    service, posted = _service_posting(monkeypatch, config)
    service.transcribe_audio(_wav(tmp_path / "a.wav", 3.0))
    assert posted["data"].get("no_speech_thold") == "0.6"


def test_suppress_nst_is_sent_only_when_asked_for(tmp_path, monkeypatch):
    config = FakeConfig()
    config.suppress_nst = True
    service, posted = _service_posting(monkeypatch, config)
    service.transcribe_audio(_wav(tmp_path / "a.wav", 3.0))
    assert posted["data"].get("suppress_nst") == "true"
