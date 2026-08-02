"""Rescuing a recording that was too quiet to transcribe.

A whisper reaches the microphone at a peak of around 100 out of 32767 - two
orders of magnitude below ordinary speech - and whisper.cpp returns nothing
at all rather than a poor guess. The audio is there; it is simply small.

So when a transcription comes back empty, the recording is amplified and
tried again rather than discarded. This is deliberately a fallback: it costs
another pass, so it runs only on the closing transcription, only when the
first attempt produced nothing, and only when there is something in the
recording to amplify.

Nothing here invents signal. Gain scales what was captured, and a high-pass
removes the offset and rumble that gain would otherwise scale with it.
"""

import wave

import numpy as np

from .logging import log

# Peak of a recording that has no useful signal at all - a muted input or a
# dead device. Below this, amplifying only produces louder noise.
DEAD_PEAK = 30

# Amplify towards this fraction of full scale. Short of 1.0 so that a
# transient does not clip flat.
TARGET_PEAK = 0.89

# Ceiling on the gain. A recording whose peak is 100 needs about 290x to
# reach the target; beyond this the noise floor arrives with the speech.
MAX_GAIN = 300.0

# Speech that matters starts well above this. Removing what is below it
# takes out mains hum, desk rumble and any DC offset, all of which would
# otherwise be amplified alongside the voice.
HIGHPASS_HZ = 80.0


def needs_boost(peak: int) -> bool:
    """Whether a recording is quiet enough to be worth retrying louder."""
    return DEAD_PEAK <= peak < int(TARGET_PEAK * 32767 * 0.25)


def _highpass(samples: np.ndarray, rate: int) -> np.ndarray:
    """One-pole high-pass, to stop gain amplifying offset and rumble."""
    if samples.size < 2:
        return samples
    # Standard one-pole coefficient for the corner frequency.
    dt = 1.0 / rate
    rc = 1.0 / (2 * np.pi * HIGHPASS_HZ)
    alpha = rc / (rc + dt)

    # y[n] = a * (y[n-1] + x[n] - x[n-1]), as a cumulative form so numpy can
    # do it without a Python loop over every sample.
    x = samples.astype(np.float32)
    dx = np.diff(x, prepend=x[0])
    weights = alpha ** np.arange(x.size, dtype=np.float32)
    # Guard against underflow making the whole tail zero.
    weights[weights < 1e-20] = 0.0
    filtered = np.convolve(dx, weights)[: x.size] * alpha
    return filtered


def boost_wav(path: str, out_path: str) -> float | None:
    """Write an amplified copy of a wav. Returns the gain, or None.

    None means there was nothing worth doing: no signal to amplify, or the
    recording was already loud enough that being quiet is not the reason
    transcription failed.
    """
    try:
        with wave.open(path, "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            frames = source.readframes(source.getnframes())

        if width != 2:
            return None                      # only 16-bit is ever recorded here
        samples = np.frombuffer(frames, dtype=np.int16)
        if samples.size == 0:
            return None

        peak = int(np.abs(samples).max())
        if peak < DEAD_PEAK:
            log(f"[BOOST] peak {peak} is below the noise floor; nothing to lift")
            return None

        filtered = _highpass(samples, rate)
        filtered_peak = float(np.abs(filtered).max())
        if filtered_peak < 1.0:
            return None

        gain = min(MAX_GAIN, TARGET_PEAK * 32767.0 / filtered_peak)
        if gain <= 1.05:
            return None                      # already as loud as it needs to be

        louder = np.clip(filtered * gain, -32768, 32767).astype(np.int16)
        with wave.open(out_path, "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(louder.tobytes())

        log(f"[BOOST] amplified {gain:.0f}x (peak {peak} -> "
            f"{int(np.abs(louder).max())})")
        return gain
    except Exception as e:
        log(f"[BOOST] could not amplify: {e}")
        return None
