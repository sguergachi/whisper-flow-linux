"""The noise filter must quieten the room without touching the voice.

Measured as RMS during speech against RMS during the pauses. Band energy
summed over a whole recording is the wrong measure and was briefly believed:
it counts noise correctly removed from the pauses as speech lost, and gain
that varies in time spreads energy over more bins, so the sum of magnitudes
rises while the power falls. Both made a working filter look broken.
"""

import numpy as np
import pytest

from whisper_flow import denoise

RATE = 16000


@pytest.fixture()
def recording():
    """Three bursts of voiced-sounding audio, over hum, rumble and hiss.

    Opens with 400ms of room tone so the pre-roll floor is the noise, not
    the first speech burst (which is what whole-clip percentiles used to
    confuse with continuous café noise).
    """
    rng = np.random.default_rng(7)
    moment = np.arange(int(RATE * 4.0)) / RATE
    speech = np.zeros_like(moment)
    for start, length in ((0.5, 0.7), (1.6, 0.9), (3.0, 0.6)):
        span = (moment >= start) & (moment < start + length)
        formants = sum(np.sin(2 * np.pi * hz * moment[span])
                       for hz in (140, 420, 900, 1800)) / 4
        speech[span] = formants * np.hanning(span.sum())
    speech *= 4000

    index = np.arange(speech.size)
    noise = (np.sin(2 * np.pi * 50 * index / RATE) * 900      # mains hum
             + np.sin(2 * np.pi * 22 * index / RATE) * 700    # desk rumble
             + rng.normal(0, 120, speech.size))               # hiss
    voiced = np.abs(speech) > speech.std()
    # The clean voice too: it is what the filtered result should resemble.
    return (speech + noise).astype(np.int16), voiced, speech.astype(np.int16)


def _rms(samples, mask):
    values = samples.astype(float)[mask]
    return float(np.sqrt((values ** 2).mean())) if values.size else 0.0


def test_the_room_gets_quieter_and_the_voice_does_not(recording):
    noisy, voiced, voice_only = recording
    cleaned = denoise.clean(noisy, RATE, normalize=False)

    quiet_before, quiet_after = _rms(noisy, ~voiced), _rms(cleaned, ~voiced)
    assert quiet_after < quiet_before * 0.7, (
        f"the pauses are still loud: {quiet_before:.0f} -> {quiet_after:.0f}")

    # Against the voice itself, not against the noisy recording. The hum sits
    # on top of the speech as well as between it, so removing it lowers the
    # level during speech too - which is the filter working, not the voice
    # being damaged. What matters is landing near the clean voice.
    spoken = _rms(voice_only, voiced)
    heard = _rms(cleaned, voiced)
    assert 0.85 * spoken < heard < 1.25 * spoken, (
        f"the voice came out at {heard:.0f} against {spoken:.0f} clean")


def test_a_higher_noise_floor_ignores_more_of_the_room(recording):
    """Settings 'Noise floor' is the gate multiplier; higher = stricter."""
    noisy, voiced, _ = recording
    mild = denoise.clean(noisy, RATE, gate_threshold=1.5, normalize=False)
    strict = denoise.clean(noisy, RATE, gate_threshold=4.0, normalize=False)
    quiet_mild = _rms(mild, ~voiced)
    quiet_strict = _rms(strict, ~voiced)
    assert quiet_strict <= quiet_mild * 1.05, (
        f"stricter floor left pauses louder: {quiet_mild:.0f} -> "
        f"{quiet_strict:.0f}")


def test_stricter_noise_floor_digs_pauses_deeper():
    """Higher threshold also lowers GATE_FLOOR_DB (not only the open line)."""
    assert denoise.gate_floor_db(1.2) == pytest.approx(denoise.GATE_FLOOR_DB)
    assert denoise.gate_floor_db(5.0) == pytest.approx(
        denoise.GATE_FLOOR_DB_STRICT)
    assert denoise.gate_floor_db(5.0) < denoise.gate_floor_db(2.2)


def test_the_offset_and_the_rumble_go(recording):
    """A converter's DC offset is not sound, and it eats the boost's headroom."""
    noisy, _, _ = recording
    filtered, _ = denoise.high_pass(noisy, RATE)
    assert abs(float(filtered.mean())) < abs(float(noisy.mean())) + 1
    spectrum = np.abs(np.fft.rfft(filtered.astype(float)))
    frequencies = np.fft.rfftfreq(filtered.size, 1 / RATE)
    raw = np.abs(np.fft.rfft(noisy.astype(float)))
    below = (frequencies > 1) & (frequencies < 60)
    assert spectrum[below].sum() < raw[below].sum() * 0.6, (
        "the high-pass left the hum and rumble behind")


def test_the_filter_runs_in_pieces_without_a_seam():
    """Chunked capture must give the same answer as one pass.

    Without carried state the filter restarts every chunk, and a step at
    each boundary is a click every few milliseconds - louder than what was
    being removed.
    """
    rng = np.random.default_rng(3)
    samples = (rng.normal(0, 3000, RATE)).astype(np.int16)

    whole, _ = denoise.high_pass(samples, RATE)

    pieces, state = [], None
    for start in range(0, samples.size, 480):
        piece, state = denoise.high_pass(samples[start:start + 480],
                                         RATE, state)
        pieces.append(piece)
    joined = np.concatenate(pieces)

    assert joined.size == whole.size
    assert np.allclose(joined, whole, atol=2.0), (
        "chunked filtering does not match a single pass; the state is not "
        "being carried across chunk boundaries")


def test_silence_and_emptiness_survive_it():
    """Nothing here may raise on the edge cases a real recording produces."""
    assert denoise.clean(np.array([], dtype=np.int16), RATE).size == 0
    silence = np.zeros(RATE, dtype=np.int16)
    assert denoise.clean(silence, RATE).size == silence.size
    tiny = np.array([5, -5, 5], dtype=np.int16)
    assert denoise.clean(tiny, RATE).size == tiny.size


def test_the_shortened_kernel_still_is_the_filter():
    """The kernel is truncated for speed; it must not change the answer.

    Written as a convolution, the one-pole's kernel is alpha**k, which is
    spent after a couple of thousand taps. Convolving against a kernel as
    long as the recording costs length-squared - a quarter of a second on a
    few seconds of speech, spent multiplying by zeros - and that lands on
    every transcription. The cut is only safe if it is invisible.
    """
    rng = np.random.default_rng(1)
    samples = (rng.normal(0, 3000, RATE * 2)).astype(np.int16)

    alpha = 1.0 / (2 * np.pi * denoise.HIGHPASS_HZ)
    alpha = alpha / (alpha + 1.0 / RATE)
    exact = np.empty(samples.size)
    carried, previous = 0.0, float(samples[0])
    for i, value in enumerate(samples.astype(float)):
        carried = alpha * (carried + value - previous)
        previous = value
        exact[i] = carried

    got, _ = denoise.high_pass(samples, RATE)
    assert np.abs(got - exact).max() < 0.5, (
        "the truncated kernel no longer matches the one-pole it stands for")


def test_floor_comes_from_the_pre_roll_not_the_whole_clip():
    """Continuous mid-clip noise must not redefine the room floor.

    A whole-clip 20th percentile on a busy recording sits near speech energy.
    Pre-roll uses only the opening room tone, so a café bed after speech
    starts cannot raise the gate line.
    """
    rng = np.random.default_rng(11)
    n = RATE * 3
    # 300ms of soft hiss, then loud continuous noise mixed with speech.
    audio = rng.normal(0, 80, n).astype(np.float64)
    audio[int(0.3 * RATE):] += rng.normal(0, 2000, n - int(0.3 * RATE))
    samples = np.clip(audio, -32768, 32767).astype(np.int16)

    floor = denoise.measure_floor(samples, RATE)
    whole = float(np.percentile(
        denoise._frame_levels(samples, RATE,
                              max(1, int(RATE * denoise.GATE_FRAME_MS / 1000)))[0],
        denoise.NOISE_PERCENTILE))
    assert floor < whole * 0.5, (
        f"pre-roll floor {floor:.0f} should be well below whole-clip "
        f"{whole:.0f}")


def test_peak_normalise_lifts_a_quiet_recording():
    """Always-on mild AGC: a quiet mic should not arrive near digital zero."""
    t = np.arange(RATE) / RATE
    quiet = (np.sin(2 * np.pi * 220 * t) * 200).astype(np.int16)
    out = denoise.normalize_peak(quiet)
    # Cap is NORM_MAX_GAIN (40x) → peak ~8000, still far above the input.
    assert int(np.abs(out).max()) > 5000
    assert int(np.abs(out).max()) > int(np.abs(quiet).max()) * 10
    # Dead signal is left alone (boost territory / nothing to lift).
    dead = (np.sin(2 * np.pi * 220 * t) * 5).astype(np.int16)
    assert int(np.abs(denoise.normalize_peak(dead)).max()) <= 30


def test_peak_normalise_does_not_blow_up_loud_speech():
    t = np.arange(RATE) / RATE
    loud = (np.sin(2 * np.pi * 220 * t) * 20000).astype(np.int16)
    out = denoise.normalize_peak(loud)
    assert int(np.abs(out).max()) <= 32767
    # Already loud enough: left alone (no push toward full scale).
    assert abs(int(np.abs(out).max()) - int(np.abs(loud).max())) < 500


def test_spectral_subtract_cuts_steady_midband_noise():
    """Strong mode: stationary tone under speech should lose energy."""
    rng = np.random.default_rng(3)
    t = np.arange(int(RATE * 2.0)) / RATE
    # Room tone first, then speech + 1kHz fan whine throughout.
    speech = np.zeros_like(t)
    span = (t >= 0.4) & (t < 1.6)
    speech[span] = (np.sin(2 * np.pi * 180 * t[span]) * 0.5
                    + np.sin(2 * np.pi * 900 * t[span]) * 0.5) * 5000
    whine = np.sin(2 * np.pi * 1000 * t) * 1500
    hiss = rng.normal(0, 100, t.size)
    noisy = np.clip(speech + whine + hiss, -32768, 32767).astype(np.int16)

    # noise_ref = opening room tone (whine + hiss, no speech)
    ref = noisy[:int(0.35 * RATE)]
    cleaned = denoise.spectral_subtract(noisy.astype(np.float32), RATE,
                                        noise_ref=ref.astype(np.float32))

    freqs = np.fft.rfftfreq(noisy.size, 1 / RATE)
    before = np.abs(np.fft.rfft(noisy.astype(float)))
    after = np.abs(np.fft.rfft(cleaned.astype(float)))
    band = (freqs > 950) & (freqs < 1050)
    assert after[band].sum() < before[band].sum() * 0.7, (
        "1kHz stationary noise should be reduced by spectral subtraction")


def test_clean_with_spectral_still_preserves_speech(recording):
    noisy, voiced, voice_only = recording
    # Untrimmed pre-roll for the noise profile.
    filtered, _ = denoise.high_pass(noisy, RATE)
    ref = filtered[:int(RATE * denoise.PRE_ROLL_MS / 1000)]
    cleaned = denoise.clean(
        noisy, RATE, spectral=True, normalize=False, noise_ref=ref)
    spoken = _rms(voice_only, voiced)
    heard = _rms(cleaned, voiced)
    assert heard > 0.5 * spoken, (
        f"strong mode wiped the voice: {heard:.0f} vs {spoken:.0f}")


def test_live_prefixes_share_a_stable_floor():
    """As the capture grows, the pre-roll floor must not drift.

    Live LocalAgreement only works if prepare is deterministic on a shared
    prefix; a floor recomputed from the whole clip would move every pass.
    """
    rng = np.random.default_rng(5)
    full = rng.normal(0, 100, RATE * 4).astype(np.float64)
    full[RATE:] += rng.normal(0, 3000, RATE * 3)  # loud later
    samples = np.clip(full, -32768, 32767).astype(np.int16)

    floor_1s = denoise.measure_floor(samples[:RATE], RATE)
    floor_4s = denoise.measure_floor(samples, RATE)
    assert abs(floor_1s - floor_4s) < 5.0, (
        f"floor drifted as the clip grew: {floor_1s:.1f} -> {floor_4s:.1f}")


def _cafe_whisper(seconds=2.5, speech_peak=180, music_rms=900):
    """Whisper under continuous mid/low music (café-like)."""
    rng = np.random.default_rng(42)
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    # Continuous music bed (bass + mid).
    music = (np.sin(2 * np.pi * 90 * t) * music_rms * 0.6
             + np.sin(2 * np.pi * 220 * t) * music_rms * 0.3
             + rng.normal(0, music_rms * 0.25, n))
    speech = np.zeros(n)
    span = (t >= 0.45) & (t < 2.0)
    # Whisper-ish: weaker, more mid/high formants than bass.
    speech[span] = (
        np.sin(2 * np.pi * 500 * t[span]) * 0.4
        + np.sin(2 * np.pi * 1400 * t[span]) * 0.4
        + np.sin(2 * np.pi * 2400 * t[span]) * 0.2
    ) * speech_peak * np.hanning(span.sum())
    return np.clip(music + speech, -32768, 32767).astype(np.int16), span


def test_cafe_whisper_triggers_low_snr_rescue():
    noisy, _ = _cafe_whisper()
    filtered, _ = denoise.high_pass(noisy, RATE)
    floor = denoise.measure_floor(filtered, RATE)
    assert denoise.is_low_snr(filtered, RATE, floor=floor, gate_threshold=2.2)


def _band_rms(samples, rate, lo_hz, hi_hz, mask=None):
    """Energy in a frequency band, optionally restricted to a time mask."""
    x = samples.astype(float)
    if mask is not None:
        # Zero outside the mask so the FFT still matches length.
        x = x.copy()
        x[~mask] = 0.0
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, 1 / rate)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    return float(np.sqrt((spec[band] ** 2).mean())) if band.any() else 0.0


def test_cafe_whisper_rescue_lifts_speech_band():
    """Rescue should raise mid-band (whisper) energy vs ordinary gate+peak."""
    noisy, span = _cafe_whisper(speech_peak=200, music_rms=1200)
    filtered, _ = denoise.high_pass(noisy, RATE)
    floor = denoise.measure_floor(filtered, RATE)

    ordinary = denoise.clean(
        noisy, RATE, gate_threshold=2.2, normalize=True,
        floor=floor, whisper_rescue=False)
    rescued = denoise.clean(
        noisy, RATE, gate_threshold=2.2, normalize=True,
        floor=floor, whisper_rescue=True)

    # Whisper formants sit ~500–2500 Hz; bass music is lower.
    mid_ord = _band_rms(ordinary, RATE, 500, 2500, span)
    mid_res = _band_rms(rescued, RATE, 500, 2500, span)
    assert mid_res > mid_ord * 1.2, (
        f"rescue did not lift speech band: ordinary {mid_ord:.0f} "
        f"vs rescue {mid_res:.0f}")


def test_normalize_speech_lifts_under_music_peak():
    """Global peak norm skips when music is loud; speech norm must still lift."""
    noisy, span = _cafe_whisper(speech_peak=120, music_rms=4000)
    # Peak is music-dominated so normalize_peak is a no-op.
    after_peak = denoise.normalize_peak(noisy)
    assert int(np.abs(after_peak).max()) >= int(np.abs(noisy).max()) * 0.9

    floor = denoise.measure_floor(noisy.astype(np.float32), RATE)
    after_speech = denoise.normalize_speech(noisy, RATE, floor=floor)
    speech_before = _rms(noisy, span)
    speech_after = _rms(after_speech, span)
    assert speech_after > speech_before * 1.5, (
        f"speech norm failed under music: {speech_before:.0f} -> "
        f"{speech_after:.0f}")
