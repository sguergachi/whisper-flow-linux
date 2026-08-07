"""Taking the room out of a recording, without taking the voice with it.

Whisper is trained on noisy audio and copes with a great deal of it; what it
copes badly with is processing that removes or smears speech. So nothing here
tries to be clever beyond what pays for itself in a real room:

  * A high-pass at 80Hz. Below that is mains hum, desk rumble, fan bearings
    and DC offset from the converter, none of which is voice.

  * An adaptive soft gate. The noise floor is measured from a short pre-roll
    at the start of the clip (room tone before speech), not from the whole
    recording - continuous café noise no longer inflates the floor and ducks
    the voice. Quiet stretches are turned down rather than silenced.

  * Mild peak normalisation so a quiet mic still reaches Whisper at a usable
    level without waiting for a blank transcript and a 300x boost retry.

  * Optional stationary spectral subtraction ("strong" mode) for fans, AC and
    other steady mid-band noise that sits on top of the speech itself.

All of it is cheap, causal where live passes need determinism, and reversible
from settings if a particular machine is worse with it on.
"""

import numpy as np

# Speech that matters starts well above this.
HIGHPASS_HZ = 80.0

# How the gate behaves. The threshold is a multiple of the measured floor,
# so a quiet room and a loud one get the same treatment relative to their
# own noise rather than an absolute level that suits neither.
NOISE_PERCENTILE = 20      # frames this quiet (in the pre-roll) are noise
GATE_THRESHOLD = 2.2       # above this multiple of the floor is speech
GATE_FLOOR_DB = -14.0      # how far down quiet parts go at the default threshold
GATE_FLOOR_DB_STRICT = -24.0  # how far at the max noise_floor setting
GATE_FRAME_MS = 20         # resolution of the level estimate
GATE_ATTACK_MS = 15        # how quickly it opens; short, so onsets survive
GATE_RELEASE_MS = 120      # how slowly it closes, so tails are not clipped

# Room tone is taken from the start of the clip, not the whole recording.
# Push-to-talk opens before the sentence starts, so the first few hundred
# milliseconds are almost always the room. Using the whole clip's quietest
# fifth made continuous background (AC, café) raise the floor until quiet
# speech looked like noise and was ducked.
PRE_ROLL_MS = 300
# Fallback when the pre-roll is shorter than this (very short clips).
MIN_FLOOR_FRAMES = 3

# Always-on peak normalise after cleaning. Milder than the blank-retry boost:
# enough that Whisper hears a sentence, not so much that residual room noise
# is driven into full scale.
NORM_TARGET_PEAK = 0.80
NORM_MAX_GAIN = 40.0
# Below this after filtering there is nothing to lift (muted / dead device).
NORM_DEAD_PEAK = 30.0

# Spectral subtraction (strong mode). Frame size is a power of two so the
# FFT is cheap; hop is half for 50% overlap-add.
SPECTRAL_FRAME_MS = 32
SPECTRAL_OVERSUB = 1.0     # how aggressively to remove the noise spectrum
SPECTRAL_FLOOR = 0.08      # leave this fraction of each bin (avoid musical noise)
SPECTRAL_MAX_ATTEN_DB = 18.0  # never carve a bin quieter than this below input


def high_pass(samples: np.ndarray, rate: int, state=None):
    """One-pole high-pass that can be run over a stream in pieces.

    Returns the filtered samples and the state to pass to the next call.
    Without carried state every chunk restarts the filter, and a step at
    each boundary is a click every few milliseconds - worse than the rumble
    being removed.
    """
    if samples.size == 0:
        return samples, state

    dt = 1.0 / rate
    rc = 1.0 / (2 * np.pi * HIGHPASS_HZ)
    alpha = np.float32(rc / (rc + dt))

    x = samples.astype(np.float32)
    previous_x, previous_y = state if state else (x[0], np.float32(0.0))

    # The kernel is alpha**k, so it is spent long before the recording ends.
    # Convolving against the full length instead costs the length of the
    # audio times itself - a quarter of a second on a few seconds of speech,
    # all of it multiplying by zeros. Cut it where it stops contributing.
    span = min(x.size, int(np.ceil(-46.0 / np.log(alpha))) + 1)

    dx = np.diff(x, prepend=np.float32(previous_x))
    weights = alpha ** np.arange(span, dtype=np.float32)
    y = np.convolve(dx, weights)[:x.size] * alpha
    # Whatever the filter was already carrying, decaying across this chunk.
    decay = (previous_y * alpha) * weights
    y[:decay.size] += decay[:y.size]

    return y, (x[-1], y[-1] if y.size else previous_y)


def _frame_levels(samples: np.ndarray, rate: int, frame: int):
    """RMS per short frame, which is what the gate decides on."""
    usable = samples.size - (samples.size % frame)
    if usable < frame:
        return np.array([]), 0
    blocks = samples[:usable].reshape(-1, frame).astype(np.float32)
    return np.sqrt((blocks ** 2).mean(axis=1)), usable


def measure_floor(samples: np.ndarray, rate: int) -> float:
    """RMS noise floor from the pre-roll (start of the clip).

    Causal and stable as a recording grows: live passes always see the same
    first PRE_ROLL_MS, so the floor - and therefore the gate - does not drift
    between passes and break LocalAgreement.
    """
    frame = max(1, int(rate * GATE_FRAME_MS / 1000))
    pre_n = min(samples.size, max(frame * MIN_FLOOR_FRAMES,
                                  int(rate * PRE_ROLL_MS / 1000)))
    if pre_n < frame:
        return 0.0
    levels, _ = _frame_levels(samples[:pre_n], rate, frame)
    if levels.size == 0:
        return 0.0
    return float(np.percentile(levels, NOISE_PERCENTILE))


def gate_floor_db(threshold: float | None) -> float:
    """How far quiet parts are turned down, scaled by the noise_floor setting.

    Higher threshold (stricter "Noise floor" knob) also digs the pauses
    deeper: someone who asked to ignore more of the room gets quieter
    pauses, not just a harder open decision.
    """
    mult = float(threshold) if threshold is not None else GATE_THRESHOLD
    # Settings clamp noise_floor to [1.2, 5.0]; map that onto gate depth.
    lo, hi = 1.2, 5.0
    t = (mult - lo) / (hi - lo)
    t = float(np.clip(t, 0.0, 1.0))
    return GATE_FLOOR_DB + t * (GATE_FLOOR_DB_STRICT - GATE_FLOOR_DB)


def gate(samples: np.ndarray, rate: int,
         threshold: float | None = None,
         floor: float | None = None) -> np.ndarray:
    """Turn down the stretches where nobody is speaking.

    The floor is measured from the pre-roll (or supplied by the caller) so
    continuous background does not redefine "quiet" as the whole clip grows.
    A recording that is all speech has a high pre-roll floor only if speech
    started immediately - then almost nothing is gated, which is correct.

    `threshold` is how many times the measured floor a frame must exceed to
    count as speech (settings "Noise floor"). Default is GATE_THRESHOLD.
    """
    frame = max(1, int(rate * GATE_FRAME_MS / 1000))
    levels, usable = _frame_levels(samples, rate, frame)
    if levels.size < 3:
        return samples

    if floor is None:
        floor = measure_floor(samples, rate)
    floor = float(floor)
    if floor <= 0:
        # No usable pre-roll: fall back to the quietest fifth of what we have.
        floor = float(np.percentile(levels, NOISE_PERCENTILE))
    if floor <= 0:
        return samples

    mult = float(threshold) if threshold is not None else GATE_THRESHOLD
    if mult < 1.0:
        mult = 1.0
    speaking = levels > floor * mult
    if not speaking.any():
        speaking = levels > floor * 1.5      # nothing loud; keep the loudest

    # Smooth the decision in time: instant gating sounds like chopping and
    # removes the first consonant of a word as reliably as it removes noise.
    attack = max(1, int(GATE_ATTACK_MS / GATE_FRAME_MS))
    release = max(1, int(GATE_RELEASE_MS / GATE_FRAME_MS))
    envelope = np.zeros(levels.size, dtype=np.float32)
    current = 0.0
    for i, open_now in enumerate(speaking):
        target = 1.0 if open_now else 0.0
        step = 1.0 / (attack if target > current else release)
        current += np.clip(target - current, -step, step)
        envelope[i] = current

    quiet = float(10.0 ** (gate_floor_db(threshold) / 20.0))
    gains = quiet + (1.0 - quiet) * envelope

    # Interpolated across the samples, not held per frame. A gain that steps
    # at each frame boundary is a discontinuity every 20ms, and a train of
    # discontinuities is broadband click energy - it measurably put back
    # more low frequency than the high-pass had just taken out, which is the
    # opposite of the point.
    centres = np.arange(gains.size, dtype=np.float32) * frame + frame / 2.0
    positions = np.arange(samples.size, dtype=np.float32)
    per_sample = np.interp(positions, centres, gains).astype(np.float32)
    return samples.astype(np.float32) * per_sample


def normalize_peak(samples: np.ndarray,
                   target: float = NORM_TARGET_PEAK,
                   max_gain: float = NORM_MAX_GAIN,
                   dead_peak: float = NORM_DEAD_PEAK) -> np.ndarray:
    """Bring a quiet but non-dead recording up toward a usable peak level.

    Always-on and mild. Only lifts when the peak is well below the target
    (under half of it) - ordinary speech is left alone. The blank-retry
    boost in boost.py still exists for extreme whispers; this just stops
    ordinary quiet mics from arriving at Whisper near the noise floor.
    """
    if samples.size == 0:
        return samples
    x = samples.astype(np.float32)
    peak = float(np.abs(x).max())
    if peak < dead_peak:
        return samples if samples.dtype == np.int16 else np.clip(
            x, -32768, 32767).astype(np.int16)
    target_amp = target * 32767.0
    # Already loud enough for Whisper; do not push every clip toward FS.
    if peak >= target_amp * 0.5:
        return samples if samples.dtype == np.int16 else np.clip(
            x, -32768, 32767).astype(np.int16)
    gain = min(max_gain, target_amp / peak)
    if gain <= 1.05:
        return samples if samples.dtype == np.int16 else np.clip(
            x, -32768, 32767).astype(np.int16)
    return np.clip(x * gain, -32768, 32767).astype(np.int16)


def spectral_subtract(samples: np.ndarray, rate: int,
                      noise_ref: np.ndarray | None = None) -> np.ndarray:
    """Stationary spectral subtraction using a pre-roll noise estimate.

    Aimed at fans, AC and other steady mid-band noise that rides on the
    speech itself - the gate only helps between words. Over-subtraction is
    capped and a spectral floor is kept so musical-noise artifacts stay mild.

    `noise_ref` should be the high-passed room tone from *before* silence
    trim. After trim the start of the clip is speech, and estimating noise
    from that would subtract the voice.
    """
    if samples.size == 0:
        return samples

    n_fft = 1
    frame = max(1, int(rate * SPECTRAL_FRAME_MS / 1000))
    while n_fft < frame:
        n_fft *= 2
    hop = n_fft // 2
    window = np.hanning(n_fft).astype(np.float32)

    x = samples.astype(np.float32)
    # Pad so the last partial frame is covered; trim back at the end.
    pad = (hop - (x.size % hop)) % hop
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])

    # Noise magnitude from the room-tone reference, or the clip pre-roll.
    ref = noise_ref.astype(np.float32) if noise_ref is not None and noise_ref.size else x
    pre_n = min(ref.size, max(n_fft, int(rate * PRE_ROLL_MS / 1000)))
    noise_mags = []
    # At least one frame even when the pre-roll is shorter than n_fft.
    last = max(1, pre_n - n_fft + 1)
    for start in range(0, last, hop):
        frame_s = ref[start:start + n_fft]
        if frame_s.size < n_fft:
            frame_s = np.pad(frame_s, (0, n_fft - frame_s.size))
        noise_mags.append(np.abs(np.fft.rfft(frame_s * window)))
    if not noise_mags:
        return samples
    noise_mag = np.median(np.stack(noise_mags, axis=0), axis=0).astype(np.float32)

    # If the pre-roll is almost silent, subtraction does nothing useful.
    if float(noise_mag.mean()) < 1e-3:
        return samples

    min_gain = float(10.0 ** (-SPECTRAL_MAX_ATTEN_DB / 20.0))
    out = np.zeros(x.size, dtype=np.float32)
    weight = np.zeros(x.size, dtype=np.float32)

    for start in range(0, x.size - n_fft + 1, hop):
        frame_s = x[start:start + n_fft] * window
        spec = np.fft.rfft(frame_s)
        mag = np.abs(spec)
        # Power-style subtraction with a floor so bins never go fully empty.
        cleaned = mag - SPECTRAL_OVERSUB * noise_mag
        floor_mag = SPECTRAL_FLOOR * mag
        cleaned = np.maximum(cleaned, floor_mag)
        # Relative attenuation cap.
        gain = np.ones_like(mag)
        nonzero = mag > 1e-8
        gain[nonzero] = cleaned[nonzero] / mag[nonzero]
        gain = np.clip(gain, min_gain, 1.0)
        rebuilt = np.fft.irfft(spec * gain, n=n_fft).astype(np.float32)
        out[start:start + n_fft] += rebuilt * window
        weight[start:start + n_fft] += window * window

    weight = np.maximum(weight, 1e-8)
    out = out / weight
    out = out[:samples.size]
    return out


def clean(samples: np.ndarray, rate: int, gated: bool = True,
          gate_threshold: float | None = None,
          spectral: bool = False,
          normalize: bool = True,
          floor: float | None = None,
          noise_ref: np.ndarray | None = None) -> np.ndarray:
    """High-pass, optional spectral subtract, gate, optional peak normalise.

    Returns int16, ready to write.

    `gate_threshold` is the settings noise-floor multiplier passed through
    to ``gate``; None keeps the built-in default.
    `spectral` enables stationary spectral subtraction (settings "Strong
    noise filter"). Off for live passes: STFT is fine for a closing clip,
    but live agreement needs a cheap causal path.
    `floor` / `noise_ref` are pre-measured from untrimmed room tone. When
    silence has already been trimmed off the start, the pre-roll is speech
    and measuring again would duck or subtract the voice - pass them in.
    """
    if samples.size == 0:
        return samples
    filtered, _ = high_pass(samples, rate)
    # Floor once from the high-passed pre-roll so gate and spectral agree,
    # unless the caller already measured it on the untrimmed capture.
    if floor is None and (gated or spectral):
        floor = measure_floor(filtered, rate)
    if spectral:
        filtered = spectral_subtract(filtered, rate, noise_ref=noise_ref)
    if gated:
        filtered = gate(filtered, rate, threshold=gate_threshold, floor=floor)
    if normalize:
        return normalize_peak(filtered)
    return np.clip(filtered, -32768, 32767).astype(np.int16)
