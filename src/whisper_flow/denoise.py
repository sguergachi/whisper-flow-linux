"""Smart voice amplification: one pipeline that prepares audio for Whisper.

Settings expose a single toggle. When it is on, ``enhance`` chooses how hard
to work from the signal itself (adaptive floor, SNR, speak-at-HUD timing):

  * Ordinary speech — high-pass, soft gate, mild spectral subtract, peak lift.
  * Whisper under noise — softer gate, speech-band emphasis, pre-emphasis,
    speech-frame normalise, milder spectral cut so music peaks do not win.

When the toggle is off the recorder sends the raw capture. Nothing here is
meant to be tuned by hand; the dynamics are the product.
"""

import numpy as np

# Speech that matters starts well above this.
HIGHPASS_HZ = 80.0

# How the gate behaves. The threshold is a multiple of the measured floor,
# so a quiet room and a loud one get the same treatment relative to their
# own noise rather than an absolute level that suits neither.
NOISE_PERCENTILE = 20      # frames this quiet (in the pre-roll) are noise
GATE_THRESHOLD = 2.2       # above this multiple of the floor is speech
GATE_FLOOR_DB = -14.0      # how far down quiet parts go (ordinary path)
GATE_FRAME_MS = 20         # resolution of the level estimate
GATE_ATTACK_MS = 15        # how quickly it opens; short, so onsets survive
GATE_RELEASE_MS = 120      # how slowly it closes, so tails are not clipped

# Floor estimation cannot assume pure room tone at the start: the HUD means
# "speak now", and users talk the moment capture begins. Pre-roll alone would
# treat the first words as noise. Adaptive floor uses the quieter of (a) the
# opening window and (b) the quiet percentile of everything heard so far, so
# between-word gaps correct a speech-filled opening, and a true room-tone
# opening still works.
PRE_ROLL_MS = 300
# How soon live SNR may latch rescue (ms of audio). Shorter = quicker switch
# under café noise; needs a few frames of level history.
SNR_LATCH_MIN_MS = 200
# Fallback when the clip is shorter than this (very short clips).
MIN_FLOOR_FRAMES = 3

# Always-on peak normalise after cleaning. Milder than the blank-retry boost:
# enough that Whisper hears a sentence, not so much that residual room noise
# is driven into full scale.
NORM_TARGET_PEAK = 0.80
NORM_MAX_GAIN = 40.0
# Below this after filtering there is nothing to lift (muted / dead device).
NORM_DEAD_PEAK = 30.0

# Spectral subtraction (strong mode / rescue). Frame size is a power of two
# so the FFT is cheap; hop is half for 50% overlap-add.
SPECTRAL_FRAME_MS = 32
SPECTRAL_OVERSUB = 1.0     # how aggressively to remove the noise spectrum
SPECTRAL_FLOOR = 0.08      # leave this fraction of each bin (avoid musical noise)
SPECTRAL_MAX_ATTEN_DB = 18.0  # never carve a bin quieter than this below input

# --- Whisper-in-café rescue -------------------------------------------------
# Estimated SNR (max frame vs pre-roll floor) below this engages rescue.
# Café music + whisper typically lands 0–10 dB; normal speech is 15–30+.
LOW_SNR_DB = 11.0
# Gate that will not bury a whisper under music (configured 2.2× would).
WHISPER_GATE_MULT = 1.2
WHISPER_GATE_FLOOR_DB = -8.0
# Cut more bass than the ordinary 80 Hz path: music energy and HVAC sit
# below here, while whispered consonants live above.
SPEECH_BAND_HPF_HZ = 180.0
# Classic pre-emphasis for unvoiced speech / sibilants.
PREEMPH_COEFF = 0.95
# Milder spectral cut when music is non-stationary (less "musical noise").
RESCUE_SPECTRAL_OVERSUB = 0.75
RESCUE_SPECTRAL_MAX_ATTEN_DB = 12.0
# Normalise so speech-frame energy reaches this, even if music peaks higher.
SPEECH_NORM_TARGET = 0.78
SPEECH_NORM_MAX_GAIN = 90.0


def high_pass(samples: np.ndarray, rate: int, state=None, cutoff_hz: float = HIGHPASS_HZ):
    """One-pole high-pass that can be run over a stream in pieces.

    Returns the filtered samples and the state to pass to the next call.
    Without carried state every chunk restarts the filter, and a step at
    each boundary is a click every few milliseconds - worse than the rumble
    being removed.
    """
    if samples.size == 0:
        return samples, state

    dt = 1.0 / rate
    rc = 1.0 / (2 * np.pi * cutoff_hz)
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


def pre_emphasize(samples: np.ndarray, coeff: float = PREEMPH_COEFF) -> np.ndarray:
    """Lift consonants / sibilants; whispered speech lives in the tilt."""
    if samples.size < 2:
        return samples.astype(np.float32)
    x = samples.astype(np.float32)
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - np.float32(coeff) * x[:-1]
    return y


def _frame_levels(samples: np.ndarray, rate: int, frame: int):
    """RMS per short frame, which is what the gate decides on."""
    usable = samples.size - (samples.size % frame)
    if usable < frame:
        return np.array([]), 0
    blocks = samples[:usable].reshape(-1, frame).astype(np.float32)
    return np.sqrt((blocks ** 2).mean(axis=1)), usable


def measure_floor(samples: np.ndarray, rate: int,
                  strategy: str = "adaptive") -> float:
    """RMS noise floor for gating and SNR.

    strategy:
      * ``adaptive`` (default) — quieter of opening window vs whole-clip quiet
        percentile. Correct when speech starts at HUD show (no pure pre-roll)
        and when the user does wait a beat before talking.
      * ``opening`` — first PRE_ROLL_MS only (spectral noise_ref source).
      * ``full`` — quiet percentile of the entire clip.
    """
    frame = max(1, int(rate * GATE_FRAME_MS / 1000))
    levels, _ = _frame_levels(samples, rate, frame)
    if levels.size == 0:
        return 0.0

    pre_frames = max(MIN_FLOOR_FRAMES,
                     int(PRE_ROLL_MS / GATE_FRAME_MS))
    pre_levels = levels[:min(levels.size, pre_frames)]
    opening = float(np.percentile(pre_levels, NOISE_PERCENTILE))
    whole = float(np.percentile(levels, NOISE_PERCENTILE))

    if strategy == "opening":
        return opening
    if strategy == "full":
        return whole
    # adaptive
    return float(min(opening, whole))


def noise_reference(samples: np.ndarray, rate: int,
                    floor: float | None = None) -> np.ndarray:
    """Samples that best represent the room for spectral subtraction.

    Prefer frames at or below the floor (pauses / music bed). When speech
    starts at HUD open there is no pure pre-roll, so stitching quiet frames
    beats blindly taking the first 300ms of voice.
    """
    if samples.size == 0:
        return samples
    x = samples.astype(np.float32)
    frame = max(1, int(rate * GATE_FRAME_MS / 1000))
    levels, usable = _frame_levels(x, rate, frame)
    if levels.size == 0:
        n = min(x.size, max(frame * MIN_FLOOR_FRAMES,
                            int(rate * PRE_ROLL_MS / 1000)))
        return x[:n].copy()

    if floor is None or floor <= 0:
        floor = measure_floor(x, rate)
    quiet_idx = np.where(levels <= max(floor * 1.2, floor + 1.0))[0]
    if quiet_idx.size == 0:
        # Loud throughout: take the quietest fifth of frames instead.
        order = np.argsort(levels)
        quiet_idx = order[:max(1, levels.size // 5)]

    want = max(frame * MIN_FLOOR_FRAMES, int(rate * PRE_ROLL_MS / 1000))
    chunks = []
    total = 0
    for i in quiet_idx:
        start = int(i) * frame
        end = min(start + frame, usable if usable else x.size)
        chunks.append(x[start:end])
        total += end - start
        if total >= want:
            break
    if not chunks:
        return x[:want].copy()
    ref = np.concatenate(chunks)
    return ref[:want].copy()


def estimate_snr_db(samples: np.ndarray, rate: int,
                    floor: float | None = None) -> float:
    """Rough SNR: loudest frame vs adaptive floor, in dB.

    Negative or low values mean the room is as loud as (or louder than) the
    voice - the café + whisper case. Returns a high number when there is no
    measurable floor so ordinary clean path is used.
    """
    if samples.size == 0:
        return 99.0
    x = samples.astype(np.float32)
    if floor is None or floor <= 0:
        floor = measure_floor(x, rate)
    if floor <= 1e-6:
        return 99.0
    frame = max(1, int(rate * GATE_FRAME_MS / 1000))
    levels, _ = _frame_levels(x, rate, frame)
    if levels.size == 0:
        return 99.0
    return float(20.0 * np.log10(max(float(levels.max()), 1e-6) / floor))


def is_low_snr(samples: np.ndarray, rate: int,
               floor: float | None = None,
               gate_threshold: float | None = None,
               min_ms: float = 0.0) -> bool:
    """Whether the clip looks like a whisper under continuous background.

    `min_ms` skips the call until enough audio is in (live path waits
    SNR_LATCH_MIN_MS so a single click does not latch rescue).
    """
    if samples.size == 0:
        return False
    if min_ms > 0 and samples.size < int(rate * min_ms / 1000):
        return False
    x = samples.astype(np.float32)
    if floor is None or floor <= 0:
        floor = measure_floor(x, rate)
    snr = estimate_snr_db(x, rate, floor=floor)
    if snr < LOW_SNR_DB:
        return True
    # Configured gate line above every frame → ordinary gate would bury voice.
    mult = float(gate_threshold) if gate_threshold is not None else GATE_THRESHOLD
    if mult < 1.0:
        mult = 1.0
    if floor > 0:
        frame = max(1, int(rate * GATE_FRAME_MS / 1000))
        levels, _ = _frame_levels(x, rate, frame)
        if levels.size and float(levels.max()) < floor * mult:
            return True
    return False


def gate_floor_db(whisper_mode: bool = False) -> float:
    """How far quiet parts are turned down (fixed; no user knob)."""
    return WHISPER_GATE_FLOOR_DB if whisper_mode else GATE_FLOOR_DB


def gate(samples: np.ndarray, rate: int,
         threshold: float | None = None,
         floor: float | None = None,
         whisper_mode: bool = False) -> np.ndarray:
    """Turn down the stretches where nobody is speaking.

    Floor is adaptive (or supplied). In whisper_mode the open threshold is
    capped so café music does not keep the gate closed over a soft voice.
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
    if whisper_mode:
        mult = min(mult, WHISPER_GATE_MULT)
    speaking = levels > floor * mult
    if not speaking.any():
        # Keep the loudest third rather than only 1.5×floor: under music the
        # whisper may never clear 1.5×, and turning the whole clip down is
        # how we used to erase it.
        if whisper_mode:
            cut = float(np.percentile(levels, 65))
            speaking = levels >= cut
        else:
            speaking = levels > floor * 1.5
        if not speaking.any():
            speaking = levels >= float(levels.max()) * 0.85

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

    quiet = float(10.0 ** (gate_floor_db(whisper_mode=whisper_mode) / 20.0))
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


def normalize_speech(samples: np.ndarray, rate: int,
                     floor: float | None = None,
                     target: float = SPEECH_NORM_TARGET,
                     max_gain: float = SPEECH_NORM_MAX_GAIN) -> np.ndarray:
    """Lift speech-frame energy toward target, ignoring music peaks.

    Global peak normalise fails in a café: a bass hit or cymbal sets the
    peak so the whisper never gets gain. Using the 90th percentile of
    frames above the floor tracks the voice instead.
    """
    if samples.size == 0:
        return samples
    x = samples.astype(np.float32)
    frame = max(1, int(rate * GATE_FRAME_MS / 1000))
    levels, _ = _frame_levels(x, rate, frame)
    if levels.size == 0:
        return normalize_peak(samples, target=target, max_gain=max_gain)

    use_floor = float(floor) if floor is not None and floor > 0 else 0.0
    if use_floor > 0:
        speechish = levels[levels > use_floor * 1.05]
    else:
        speechish = levels
    if speechish.size < 2:
        speechish = levels
    ref = float(np.percentile(speechish, 90))
    if ref < NORM_DEAD_PEAK:
        return np.clip(x, -32768, 32767).astype(np.int16)

    target_amp = target * 32767.0
    gain = min(max_gain, target_amp / ref)
    if gain <= 1.05:
        return np.clip(x, -32768, 32767).astype(np.int16)
    return np.clip(x * gain, -32768, 32767).astype(np.int16)


def spectral_subtract(samples: np.ndarray, rate: int,
                      noise_ref: np.ndarray | None = None,
                      oversub: float = SPECTRAL_OVERSUB,
                      max_atten_db: float = SPECTRAL_MAX_ATTEN_DB) -> np.ndarray:
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

    min_gain = float(10.0 ** (-max_atten_db / 20.0))
    out = np.zeros(x.size, dtype=np.float32)
    weight = np.zeros(x.size, dtype=np.float32)

    for start in range(0, x.size - n_fft + 1, hop):
        frame_s = x[start:start + n_fft] * window
        spec = np.fft.rfft(frame_s)
        mag = np.abs(spec)
        # Power-style subtraction with a floor so bins never go fully empty.
        cleaned = mag - float(oversub) * noise_mag
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


def enhance(samples: np.ndarray, rate: int, *,
            floor: float | None = None,
            noise_ref: np.ndarray | None = None,
            live: bool = False,
            force_rescue: bool | None = None) -> np.ndarray:
    """Smart voice amplification: prepare a clip for Whisper.

    One entry point for all preprocessing. Chooses ordinary vs low-SNR
    rescue from the signal (or ``force_rescue``). ``live=True`` skips the
    heavier full-band spectral step on the ordinary path so streaming
    LocalAgreement stays cheap and deterministic; rescue still applies a
    mild spectral cut when a noise reference is available.

    Returns int16 samples ready to write.
    """
    if samples.size == 0:
        return samples

    base, _ = high_pass(samples, rate)
    if floor is None:
        floor = measure_floor(base, rate)

    rescue = force_rescue
    if rescue is None:
        rescue = is_low_snr(base, rate, floor=floor)

    if rescue:
        try:
            from .logging import log
            snr = estimate_snr_db(base, rate, floor=floor)
            log(f"[VOICE] smart amplify rescue "
                f"(snr≈{snr:.0f}dB, floor={floor:.0f}, live={live})")
        except Exception:
            pass
        filtered, _ = high_pass(samples, rate, cutoff_hz=SPEECH_BAND_HPF_HZ)
        ref = noise_ref
        if ref is None or (hasattr(ref, "size") and ref.size == 0):
            ref = noise_reference(base, rate, floor=floor)
        if ref is not None and getattr(ref, "size", 0):
            ref, _ = high_pass(ref, rate, cutoff_hz=SPEECH_BAND_HPF_HZ)
            filtered = spectral_subtract(
                filtered, rate, noise_ref=ref,
                oversub=RESCUE_SPECTRAL_OVERSUB,
                max_atten_db=RESCUE_SPECTRAL_MAX_ATTEN_DB,
            )
        filtered = gate(filtered, rate, floor=floor, whisper_mode=True)
        filtered = pre_emphasize(filtered)
        return normalize_speech(filtered, rate, floor=floor)

    # Ordinary speech: mild cleanup. Final pass also peels stationary noise;
    # live stays lighter for agreement latency.
    filtered = base
    if not live:
        ref = noise_ref
        if ref is None or (hasattr(ref, "size") and ref.size == 0):
            ref = noise_reference(base, rate, floor=floor)
        if ref is not None and getattr(ref, "size", 0):
            filtered = spectral_subtract(
                filtered, rate, noise_ref=ref,
                oversub=0.85,
                max_atten_db=14.0,
            )
    filtered = gate(filtered, rate, floor=floor, whisper_mode=False)
    return normalize_peak(filtered)


def clean(samples: np.ndarray, rate: int, gated: bool = True,
          gate_threshold: float | None = None,
          spectral: bool = False,
          normalize: bool = True,
          floor: float | None = None,
          noise_ref: np.ndarray | None = None,
          whisper_rescue: bool | None = None) -> np.ndarray:
    """Compatibility wrapper around ``enhance`` for existing tests.

    Prefer ``enhance`` for new call sites. Unused knobs (gated/spectral/
    normalize) are mapped as best-effort so unit tests keep meaning.
    """
    if not gated and not normalize and not spectral and whisper_rescue is False:
        filtered, _ = high_pass(samples, rate)
        return np.clip(filtered, -32768, 32767).astype(np.int16)
    live = not spectral and whisper_rescue is False
    force = whisper_rescue if whisper_rescue is not None else None
    out = enhance(samples, rate, floor=floor, noise_ref=noise_ref,
                  live=live, force_rescue=force)
    if not normalize:
        # Tests that disable normalise expect pre-gain levels; re-run a
        # thin path without the final lift.
        base, _ = high_pass(samples, rate)
        use_floor = floor if floor is not None else measure_floor(base, rate)
        rescue = force if force is not None else is_low_snr(
            base, rate, floor=use_floor, gate_threshold=gate_threshold)
        if rescue:
            filtered, _ = high_pass(samples, rate, cutoff_hz=SPEECH_BAND_HPF_HZ)
            ref = noise_ref if noise_ref is not None else noise_reference(
                base, rate, floor=use_floor)
            if ref is not None and getattr(ref, "size", 0):
                ref, _ = high_pass(ref, rate, cutoff_hz=SPEECH_BAND_HPF_HZ)
                filtered = spectral_subtract(
                    filtered, rate, noise_ref=ref,
                    oversub=RESCUE_SPECTRAL_OVERSUB,
                    max_atten_db=RESCUE_SPECTRAL_MAX_ATTEN_DB)
            if gated:
                filtered = gate(filtered, rate, threshold=gate_threshold,
                                floor=use_floor, whisper_mode=True)
            filtered = pre_emphasize(filtered)
            return np.clip(filtered, -32768, 32767).astype(np.int16)
        filtered = base
        if spectral:
            ref = noise_ref if noise_ref is not None else noise_reference(
                base, rate, floor=use_floor)
            filtered = spectral_subtract(filtered, rate, noise_ref=ref)
        if gated:
            filtered = gate(filtered, rate, threshold=gate_threshold,
                            floor=use_floor)
        return np.clip(filtered, -32768, 32767).astype(np.int16)
    return out


def rescue_samples(samples: np.ndarray, rate: int,
                   floor: float | None = None,
                   noise_ref: np.ndarray | None = None,
                   gate_threshold: float | None = None) -> np.ndarray:
    """Force the low-SNR path (blank-retry second pass)."""
    return enhance(samples, rate, floor=floor, noise_ref=noise_ref,
                   live=False, force_rescue=True)
