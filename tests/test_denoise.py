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
    """Three bursts of voiced-sounding audio, over hum, rumble and hiss."""
    rng = np.random.default_rng(7)
    moment = np.arange(int(RATE * 4.0)) / RATE
    speech = np.zeros_like(moment)
    for start, length in ((0.4, 0.7), (1.6, 0.9), (3.0, 0.6)):
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
    cleaned = denoise.clean(noisy, RATE)

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
    mild = denoise.clean(noisy, RATE, gate_threshold=1.5)
    strict = denoise.clean(noisy, RATE, gate_threshold=4.0)
    quiet_mild = _rms(mild, ~voiced)
    quiet_strict = _rms(strict, ~voiced)
    assert quiet_strict <= quiet_mild * 1.05, (
        f"stricter floor left pauses louder: {quiet_mild:.0f} -> "
        f"{quiet_strict:.0f}")


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
