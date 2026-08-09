"""Capture what the mic heard vs what Whisper was given.

Whispering in a noisy room fails for several different reasons that look
identical to the user (nothing typed). This module writes a small folder of
WAVs and a metrics report after every dictation so the cause is measurable:

  * raw_untrimmed.wav  — everything captured, including room tone
  * raw_trimmed.wav    — after silence trim, before denoise
  * sent.wav           — exactly what went to Whisper
  * report.txt         — peaks, floor, gate open %, estimated SNR, settings

Disk is bounded: always overwrite ``audio-debug/last/``, and keep a short
ring of failed captures under ``audio-debug/fail-*/``.
"""

from __future__ import annotations

import json
import shutil
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from . import denoise
from .boost import DEAD_PEAK, needs_boost
from .logging import log

# Keep this many failed-dictation folders; oldest deleted first.
MAX_FAIL_CAPTURES = 8

# How many captures the samples library may hold before the oldest go.
# Deliberately generous: the whole point of the library is a corpus of real
# café/whisper recordings to test denoise changes against, and each capture
# is a few hundred KB at most.
MAX_SAMPLE_CAPTURES = 200


def debug_root(config_dir: Path) -> Path:
    return Path(config_dir) / "audio-debug"


def last_dir(config_dir: Path) -> Path:
    return debug_root(config_dir) / "last"


def samples_dir(config_dir: Path) -> Path:
    """The library of every capture kept while keep_all_captures is on."""
    return debug_root(config_dir) / "samples"


def _write_wav(path: Path, samples: np.ndarray | bytes, rate: int) -> None:
    if isinstance(samples, (bytes, bytearray)):
        data = bytes(samples)
    else:
        data = np.asarray(samples, dtype=np.int16).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data)


def _peak_mean(samples: np.ndarray) -> tuple[int, int, float]:
    if samples.size == 0:
        return 0, 0, 0.0
    abs_s = np.abs(samples.astype(np.float64))
    peak = int(abs_s.max())
    mean = int(abs_s.mean())
    rms = float(np.sqrt((samples.astype(np.float64) ** 2).mean()))
    return peak, mean, rms


def analyze(
    samples: np.ndarray,
    rate: int,
    *,
    floor: float | None = None,
    gate_threshold: float | None = None,
) -> dict:
    """Metrics that explain a blank or weak transcript."""
    peak, mean, rms = _peak_mean(samples)
    duration = samples.size / rate if rate else 0.0

    filtered, _ = denoise.high_pass(samples, rate) if samples.size else (
        samples, None)
    measured_floor = denoise.measure_floor(filtered, rate) if samples.size else 0.0
    use_floor = float(floor) if floor is not None and floor > 0 else measured_floor
    mult = float(gate_threshold) if gate_threshold is not None else denoise.GATE_THRESHOLD
    if mult < 1.0:
        mult = 1.0
    speech_line = use_floor * mult

    frame = max(1, int(rate * denoise.GATE_FRAME_MS / 1000))
    levels, _ = denoise._frame_levels(
        filtered if hasattr(filtered, "size") else samples, rate, frame)
    if levels.size:
        above = float((levels > speech_line).mean()) if speech_line > 0 else 0.0
        level_p50 = float(np.percentile(levels, 50))
        level_p90 = float(np.percentile(levels, 90))
        level_max = float(levels.max())
    else:
        above = level_p50 = level_p90 = level_max = 0.0

    # Rough SNR: peak speech frame vs floor. Negative means the "voice" never
    # clearly cleared the room.
    snr_db = 0.0
    if use_floor > 1e-6 and level_max > 0:
        snr_db = 20.0 * float(np.log10(max(level_max, 1e-6) / use_floor))

    diagnosis = _diagnose(
        peak=peak,
        floor=use_floor,
        speech_line=speech_line,
        above=above,
        snr_db=snr_db,
        duration=duration,
    )

    return {
        "duration_s": round(duration, 3),
        "peak": peak,
        "mean": mean,
        "rms": round(rms, 1),
        "floor_rms": round(use_floor, 2),
        "floor_measured": round(measured_floor, 2),
        "gate_threshold": mult,
        "speech_line": round(speech_line, 2),
        "gate_open_fraction": round(above, 3),
        "level_p50": round(level_p50, 1),
        "level_p90": round(level_p90, 1),
        "level_max": round(level_max, 1),
        "snr_db_est": round(snr_db, 1),
        "needs_boost": bool(needs_boost(peak)),
        "below_dead_peak": peak < DEAD_PEAK,
        "gate_floor_db": round(denoise.gate_floor_db(False), 1),
        "diagnosis": diagnosis,
    }


def _diagnose(*, peak, floor, speech_line, above, snr_db, duration) -> list[str]:
    reasons = []
    if duration < 0.35:
        reasons.append("clip too short (<0.35s) — discarded or hallucinated")
    if peak < DEAD_PEAK:
        reasons.append(
            f"peak {peak} < dead floor {DEAD_PEAK} — muted mic or no signal")
    if floor > 0 and peak < speech_line:
        reasons.append(
            f"whisper buried: peak {peak} < speech line {speech_line:.0f} "
            f"(floor {floor:.0f} × threshold) — gate ducks everything")
    if above < 0.05 and duration >= 0.35:
        reasons.append(
            f"gate open only {above:.0%} of frames — VAD/gate hears no speech")
    if snr_db < 6 and peak >= DEAD_PEAK:
        reasons.append(
            f"estimated SNR {snr_db:.0f} dB is very low — room masks whisper")
    if needs_boost(peak) and snr_db < 12:
        reasons.append(
            "quiet peak + noisy room: boost amplifies noise with the voice")
    if not reasons:
        reasons.append(
            "levels look usable — if Whisper still blank, model/language or "
            "content issue (not pure amplitude)")
    return reasons


def format_report(meta: dict) -> str:
    """Human-readable report for report.txt and the log ring."""
    lines = [
        f"whisper-flow audio debug  {meta.get('timestamp', '')}",
        f"mode={meta.get('mode', '?')}  rate={meta.get('sample_rate', '?')}Hz",
        f"smart_voice={meta.get('smart_voice_amplification', meta.get('noise_filter'))}",
        f"trim_silence={meta.get('trim_silence')}  "
        f"vad_mode={meta.get('vad_mode')}",
        "",
        "--- raw (untrimmed) ---",
    ]
    raw = meta.get("raw_untrimmed") or {}
    for k in ("duration_s", "peak", "mean", "rms", "floor_rms", "speech_line",
              "gate_open_fraction", "snr_db_est", "level_p50", "level_p90",
              "level_max", "gate_threshold"):
        if k in raw:
            lines.append(f"  {k}: {raw[k]}")
    if raw.get("diagnosis"):
        lines.append("  diagnosis:")
        for d in raw["diagnosis"]:
            lines.append(f"    - {d}")

    lines.append("")
    lines.append("--- sent to whisper ---")
    sent = meta.get("sent") or {}
    for k in ("duration_s", "peak", "mean", "rms", "floor_rms",
              "gate_open_fraction", "snr_db_est", "level_max"):
        if k in sent:
            lines.append(f"  {k}: {sent[k]}")
    if sent.get("diagnosis"):
        lines.append("  diagnosis:")
        for d in sent["diagnosis"]:
            lines.append(f"    - {d}")

    if meta.get("transcript") is not None:
        lines.append("")
        t = meta["transcript"]
        if t:
            lines.append(f"transcript ({len(t)} chars): {t[:200]!r}")
        else:
            lines.append("transcript: (blank / None)")
    if meta.get("boost_gain") is not None:
        lines.append(f"boost_gain: {meta['boost_gain']}")
    if meta.get("boost_transcript") is not None:
        bt = meta["boost_transcript"]
        lines.append(
            f"boost_transcript: {bt[:200]!r}" if bt else "boost_transcript: (blank)")

    lines.append("")
    lines.append(f"files: {meta.get('dir', '')}")
    return "\n".join(lines) + "\n"


def _prune_fails(root: Path) -> None:
    fails = sorted(
        (p for p in root.glob("fail-*") if p.is_dir()),
        key=lambda p: p.name,
    )
    while len(fails) > MAX_FAIL_CAPTURES:
        old = fails.pop(0)
        try:
            shutil.rmtree(old)
        except Exception:
            pass


def _prune_samples(root: Path) -> None:
    samples = sorted(
        (p for p in root.glob("sample-*") if p.is_dir()),
        key=lambda p: p.name,
    )
    while len(samples) > MAX_SAMPLE_CAPTURES:
        old = samples.pop(0)
        try:
            shutil.rmtree(old)
        except Exception:
            pass


def save_capture(
    config_dir: Path,
    *,
    rate: int,
    raw_untrimmed: bytes | np.ndarray | None,
    raw_trimmed: bytes | np.ndarray | None,
    sent: bytes | np.ndarray | None,
    floor: float | None = None,
    gate_threshold: float | None = None,
    settings: dict | None = None,
    mode: str = "",
) -> Path | None:
    """Write last/ capture folder. Returns the directory, or None on failure."""
    try:
        dest = last_dir(config_dir)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        def as_i16(data) -> np.ndarray:
            if data is None:
                return np.array([], dtype=np.int16)
            if isinstance(data, np.ndarray):
                return data.astype(np.int16, copy=False)
            return np.frombuffer(data, dtype=np.int16)

        raw_u = as_i16(raw_untrimmed)
        raw_t = as_i16(raw_trimmed)
        sent_s = as_i16(sent)

        if raw_u.size:
            _write_wav(dest / "raw_untrimmed.wav", raw_u, rate)
        if raw_t.size:
            _write_wav(dest / "raw_trimmed.wav", raw_t, rate)
        if sent_s.size:
            _write_wav(dest / "sent.wav", sent_s, rate)

        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "sample_rate": rate,
            "dir": str(dest),
            **(settings or {}),
            "raw_untrimmed": analyze(
                raw_u, rate, floor=floor, gate_threshold=gate_threshold)
            if raw_u.size else {},
            "raw_trimmed": analyze(
                raw_t, rate, floor=floor, gate_threshold=gate_threshold)
            if raw_t.size else {},
            "sent": analyze(
                sent_s, rate, floor=floor, gate_threshold=gate_threshold)
            if sent_s.size else {},
        }
        (dest / "report.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        report = format_report(meta)
        (dest / "report.txt").write_text(report, encoding="utf-8")
        # Stash for finalize_capture to amend with transcript results.
        (dest / ".meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

        # Always surface the diagnosis in the ring buffer (Copy log).
        for line in report.strip().splitlines():
            log(f"[AUDIO-DEBUG] {line}")

        return dest
    except Exception as e:
        log(f"[AUDIO-DEBUG] save failed: {e}")
        return None


def finalize_capture(
    config_dir: Path,
    *,
    transcript: str | None,
    boost_gain: float | None = None,
    boost_transcript: str | None = None,
    boosted_wav: bytes | np.ndarray | None = None,
    rate: int = 16000,
    keep_sample: bool = False,
) -> Path | None:
    """Attach transcription outcome; archive under fail-* if blank.

    ``keep_sample=True`` archives the capture under samples/ regardless of
    outcome - the "keep every capture" toggle, which turns everyday dictation
    into a corpus of real recordings for testing denoise and decode changes.
    """
    try:
        dest = last_dir(config_dir)
        meta_path = dest / ".meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["transcript"] = transcript
        meta["boost_gain"] = boost_gain
        meta["boost_transcript"] = boost_transcript

        if boosted_wav is not None:
            if isinstance(boosted_wav, np.ndarray):
                data = boosted_wav.astype(np.int16)
            else:
                data = np.frombuffer(boosted_wav, dtype=np.int16)
            if data.size:
                _write_wav(dest / "boosted.wav", data, rate)
                meta["boosted"] = analyze(data, rate)

        report = format_report(meta)
        (dest / "report.txt").write_text(report, encoding="utf-8")
        (dest / "report.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        log(f"[AUDIO-DEBUG] transcript="
            f"{'(blank)' if not transcript else repr(transcript[:80])}")
        if boost_gain:
            log(f"[AUDIO-DEBUG] boost={boost_gain:.0f}x -> "
                f"{'(blank)' if not boost_transcript else repr(boost_transcript[:80])}")

        # Samples library first: it keeps successes too, so a "keep every
        # capture" run must archive before the blank-only branch decides.
        if keep_sample:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            sample = samples_dir(config_dir) / f"sample-{stamp}"
            if sample.exists():
                shutil.rmtree(sample)
            shutil.copytree(dest, sample)
            _prune_samples(samples_dir(config_dir))
            log(f"[AUDIO-DEBUG] sample kept at {sample}")
            meta["dir"] = str(sample)
            (sample / "report.txt").write_text(
                format_report(meta), encoding="utf-8")
            (sample / "report.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8")

        blank = not (transcript or boost_transcript)
        if blank:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            fail = debug_root(config_dir) / f"fail-{stamp}"
            if fail.exists():
                shutil.rmtree(fail)
            shutil.copytree(dest, fail)
            _prune_fails(debug_root(config_dir))
            log(f"[AUDIO-DEBUG] blank capture archived at {fail}")
            # Point report at the fail folder too.
            meta["dir"] = str(fail)
            (fail / "report.txt").write_text(format_report(meta), encoding="utf-8")
            return fail
        return dest
    except Exception as e:
        log(f"[AUDIO-DEBUG] finalize failed: {e}")
        return None
