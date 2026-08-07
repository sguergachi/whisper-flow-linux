"""Taking the room out of a recording, without taking the voice with it.

Two things, deliberately mild. Whisper is trained on noisy audio and copes
with a great deal of it; what it copes badly with is processing that removes
or smears speech. So nothing here tries to be clever - it removes energy
that cannot be speech, and turns down the parts where nobody is speaking.

  * A high-pass at 80Hz. Below that is mains hum, desk rumble, fan bearings
    and DC offset from the converter, none of which is voice. On a quiet
    recording this is most of what the level meter was reading, and it is
    the part that eats the headroom the boost step needs.

  * An adaptive gate. The noise floor is measured from the recording itself
    rather than assumed, and the quiet stretches are turned down rather than
    silenced - a hard gate chops the start of words, and digital silence
    between phrases is its own problem for a model that reads context.

Both are cheap enough to run on every recording, and both are reversible
from settings if they ever make things worse on a particular machine.
"""

import numpy as np

# Speech that matters starts well above this.
HIGHPASS_HZ = 80.0

# How the gate behaves. The threshold is a multiple of the measured floor,
# so a quiet room and a loud one get the same treatment relative to their
# own noise rather than an absolute level that suits neither.
NOISE_PERCENTILE = 20      # frames this quiet are taken to be noise
GATE_THRESHOLD = 2.2       # above this multiple of the floor is speech
GATE_FLOOR_DB = -14.0      # how far down the quiet parts go, not to silence
GATE_FRAME_MS = 20         # resolution of the level estimate
GATE_ATTACK_MS = 15        # how quickly it opens; short, so onsets survive
GATE_RELEASE_MS = 120      # how slowly it closes, so tails are not clipped


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


def gate(samples: np.ndarray, rate: int,
         threshold: float | None = None) -> np.ndarray:
    """Turn down the stretches where nobody is speaking.

    The floor is the level of the quietest fifth of the recording, so it
    adapts to the room. A recording that is all speech has a high floor and
    is left alone; one that is all noise has nothing above the threshold and
    is turned down throughout, which is the honest answer for it.

    `threshold` is how many times the measured floor a frame must exceed to
    count as speech (settings "Noise floor"). Default is GATE_THRESHOLD.
    """
    frame = max(1, int(rate * GATE_FRAME_MS / 1000))
    levels, usable = _frame_levels(samples, rate, frame)
    if levels.size < 3:
        return samples

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

    quiet = float(10.0 ** (GATE_FLOOR_DB / 20.0))
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


def clean(samples: np.ndarray, rate: int, gated: bool = True,
          gate_threshold: float | None = None) -> np.ndarray:
    """High-pass, then optionally gate. Returns int16, ready to write.

    `gate_threshold` is the settings noise-floor multiplier passed through
    to ``gate``; None keeps the built-in default.
    """
    if samples.size == 0:
        return samples
    filtered, _ = high_pass(samples, rate)
    if gated:
        filtered = gate(filtered, rate, threshold=gate_threshold)
    return np.clip(filtered, -32768, 32767).astype(np.int16)
