"""Audio debug captures must explain a blank whisper without filling the disk."""

import json
from pathlib import Path

import numpy as np

from whisper_flow import audio_debug


RATE = 16000


def _whisper_in_noise(seconds=2.0, speech_peak=120, noise_rms=400):
    """A buried whisper: room louder than the voice peak."""
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    noise = np.random.default_rng(0).normal(0, noise_rms, n)
    speech = np.zeros(n)
    span = (t >= 0.4) & (t < 1.4)
    speech[span] = np.sin(2 * np.pi * 220 * t[span]) * speech_peak
    return np.clip(speech + noise, -32768, 32767).astype(np.int16)


def test_analyze_flags_buried_whisper():
    samples = _whisper_in_noise()
    report = audio_debug.analyze(samples, RATE, gate_threshold=2.2)
    assert report["peak"] > 0
    text = " ".join(report["diagnosis"]).lower()
    assert any(word in text for word in ("snr", "buried", "gate", "room", "quiet"))


def test_save_and_finalize_archives_blank(tmp_path):
    samples = _whisper_in_noise()
    raw = samples.tobytes()
    dest = audio_debug.save_capture(
        tmp_path,
        rate=RATE,
        raw_untrimmed=raw,
        raw_trimmed=raw,
        sent=raw,
        floor=400.0,
        gate_threshold=2.2,
        settings={"smart_voice_amplification": True},
        mode="transcribe",
    )
    assert dest is not None
    assert (dest / "sent.wav").exists()
    assert (dest / "report.txt").exists()
    assert "diagnosis" in (dest / "report.txt").read_text(encoding="utf-8")

    fail = audio_debug.finalize_capture(
        tmp_path, transcript=None, rate=RATE)
    assert fail is not None
    assert fail.name.startswith("fail-")
    assert (fail / "report.json").exists()
    meta = json.loads((fail / "report.json").read_text(encoding="utf-8"))
    assert meta.get("transcript") in (None, "")


def test_successful_transcript_does_not_archive_fail(tmp_path):
    samples = (np.sin(2 * np.pi * 220 * np.arange(RATE) / RATE) * 8000).astype(
        np.int16)
    audio_debug.save_capture(
        tmp_path, rate=RATE,
        raw_untrimmed=samples.tobytes(),
        raw_trimmed=samples.tobytes(),
        sent=samples.tobytes(),
    )
    out = audio_debug.finalize_capture(
        tmp_path, transcript="hello world", rate=RATE)
    assert out is not None
    assert out.name == "last"
    assert not list(Path(tmp_path).glob("fail-*"))


def test_fail_ring_is_bounded(tmp_path):
    samples = _whisper_in_noise(seconds=0.5)
    raw = samples.tobytes()
    for i in range(audio_debug.MAX_FAIL_CAPTURES + 3):
        audio_debug.save_capture(
            tmp_path, rate=RATE,
            raw_untrimmed=raw, raw_trimmed=raw, sent=raw,
            mode=str(i),
        )
        audio_debug.finalize_capture(tmp_path, transcript=None, rate=RATE)
    fails = list(Path(tmp_path).glob("fail-*"))
    assert len(fails) <= audio_debug.MAX_FAIL_CAPTURES
