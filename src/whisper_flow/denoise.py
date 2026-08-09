"""Smart voice amplification: one pipeline that prepares audio for Whisper.

Settings expose a single toggle. When it is on, ``enhance``:

  1. Profiles the clip (floor, SNR, bass load, level).
  2. Builds a filter plan — only the steps that input needs.
  3. Applies them.
  4. On the final (non-live) path, if the result still looks weak, runs a
     second pass from the original with an escalated plan.

When the toggle is off the recorder sends the raw capture.
"""

from dataclasses import dataclass, replace

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
WHISPER_GATE_MULT = 1.15     # open on soft speech (2.2× misses whispers)
WHISPER_GATE_FLOOR_DB = -18.0  # duck room hard between syllables before lift
# Cut more bass than the ordinary 80 Hz path: music energy and HVAC sit
# below here, while whispered consonants live above.
SPEECH_BAND_HPF_HZ = 180.0
# Classic pre-emphasis for unvoiced speech / sibilants.
PREEMPH_COEFF = 0.95
# Milder spectral cut when music is non-stationary (less "musical noise").
RESCUE_SPECTRAL_OVERSUB = 0.75
RESCUE_SPECTRAL_MAX_ATTEN_DB = 12.0
# Normalise targets. Global peak-norm stays modest. Whisper path may lift
# *speech frames* harder after a soft gate (see WHISPER_* gains).
SPEECH_NORM_TARGET = 0.65
SPEECH_NORM_MAX_GAIN = 8.0
# Quiet-whisper: gate ducks room first, then lift voiced frames so Whisper
# hears consonants instead of inventing ``*sad music*`` on the bed.
WHISPER_SPEECH_NORM_TARGET = 0.55
WHISPER_SPEECH_NORM_MAX_GAIN = 40.0
NORM_TARGET_PEAK = 0.70
NORM_MAX_GAIN = 6.0
# Leave headroom so speech-norm never rails int16 (clipping → garbage decode).
NORM_CLIP_PEAK = 0.89
# Plain fallback when a speech-frame lift failed to separate voice from room:
# boost the raw high-passed capture to near full scale, preserving the
# voice/room ratio the mic heard (measured: 30x plain gain recovered "Keeps
# you going." where 40x whisper-lift returned [MUSIC]).
PLAIN_BOOST_TARGET = 0.89
PLAIN_BOOST_MAX_GAIN = 300.0

# Dynamic plan thresholds (absolute int16 RMS / ratios).
BASS_RATIO_SPEECH_BAND = 0.35   # bass share alone is not enough (see below)
BASS_RATIO_STRONG = 0.45
# Speech-band HPF only when bass is high *and* the room is actually loud, or
# SNR is poor. Quiet rooms often show high bass% from rumble with no music.
FLOOR_SPEECH_BAND = 400.0
FLOOR_SPECTRAL = 350.0          # room energy worth peeling
FLOOR_GATE = 80.0               # skip gate in near-silent rooms
SNR_MILD_DB = 14.0              # below this, prefer speech-aware path
SNR_CLEAN_DB = 20.0             # high speech-p90 SNR + usable level → light plan
# Speech level (p90 of frames above a soft line), not abs peak — a single
# spike made Windows logs report snr≈31dB while speech p90 was ~130 (whisper).
SPEECH_P90_USABLE = 1200.0      # normal talking
SPEECH_P90_WHISPER = 500.0      # quiet / whispered speech
PEAK_USABLE = 1500.0
PEAK_QUIET = 800.0
# A whisper is quiet speech in a quiet room. Above this floor, a low p90 is a
# voice *masked* by a bed (café music, nature ambience) - whisper-lift's 40x
# gain would amplify the bed with the voice, and Whisper hears the bed.
WHISPER_MAX_FLOOR = 80.0
# Second pass: first output still not good enough for Whisper.
SECOND_PASS_SNR_DB = 12.0
SECOND_PASS_PEAK = 1200.0
SECOND_PASS_SPEECH_P90 = 800.0


@dataclass(frozen=True)
class VoicePlan:
    """Which filters to run for this clip. Built from the signal, not settings."""

    hpf_hz: float = HIGHPASS_HZ
    spectral: bool = False
    spectral_oversub: float = 0.85
    spectral_max_atten_db: float = 14.0
    gate: bool = True
    whisper_gate: bool = False
    preemph: bool = False
    speech_norm: bool = False       # else peak_norm when normalize
    normalize: bool = True
    speech_norm_target: float = SPEECH_NORM_TARGET
    speech_norm_max_gain: float = SPEECH_NORM_MAX_GAIN
    pass_index: int = 1
    # Why this plan (for logs / debug).
    reason: str = ""


@dataclass(frozen=True)
class SignalProfile:
    floor: float
    snr_db: float           # peak-frame SNR (can be spiky)
    speech_snr_db: float    # p90-of-voiced vs floor — real speech loudness
    peak: float
    speech_p90: float       # RMS of voiced-ish frames
    voiced_frac: float      # fraction of frames above soft speech line
    bass_ratio: float
    low_snr: bool
    is_whisper: bool        # quiet voice that needs gate-then-lift


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
    # A cold-opened capture stream delivers ~0.3-0.5s of exact digital silence
    # while the device wakes up, so an opening of pure zeros must not be read
    # as "a dead-silent room" - the room's floor is in the rest of the clip,
    # and treating it as zero sent the smart-voice plan down the whisper-lift
    # path, which then amplified the music bed into the transcript. A warm
    # stream's opening carries real room tone (~0.04+), so only an opening
    # that is (near-)digital silence is overridden by the whole clip.
    if opening < 0.01 and whole > opening * 10:
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
    """Modest peak lift for quiet clips. Hard-capped gain; never rails int16."""
    if samples.size == 0:
        return samples
    x = samples.astype(np.float32)
    peak = float(np.abs(x).max())
    if peak < dead_peak:
        return samples if samples.dtype == np.int16 else np.clip(
            x, -32768, 32767).astype(np.int16)
    # Already usable — leave alone (avoids *sad music* from over-gain).
    if peak >= PEAK_USABLE:
        return samples if samples.dtype == np.int16 else np.clip(
            x, -32768, 32767).astype(np.int16)
    target_amp = min(target, NORM_CLIP_PEAK) * 32767.0
    gain = min(max_gain, target_amp / peak)
    if gain <= 1.05:
        return samples if samples.dtype == np.int16 else np.clip(
            x, -32768, 32767).astype(np.int16)
    return np.clip(x * gain, -32768, 32767).astype(np.int16)


def normalize_speech(samples: np.ndarray, rate: int,
                     floor: float | None = None,
                     target: float = SPEECH_NORM_TARGET,
                     max_gain: float = SPEECH_NORM_MAX_GAIN) -> np.ndarray:
    """Lift speech-frame energy modestly, ignoring music peaks.

    Gain is hard-capped. Windows café logs showed uncapped speech-norm
    slamming peak to 32768 and wrecking the decode.
    """
    if samples.size == 0:
        return samples
    x = samples.astype(np.float32)
    peak = float(np.abs(x).max())
    if peak >= NORM_CLIP_PEAK * 32767 * 0.95:
        return np.clip(x, -32768, 32767).astype(np.int16)

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

    target_amp = min(target, NORM_CLIP_PEAK) * 32767.0
    gain = min(max_gain, target_amp / max(ref, 1.0))
    if gain <= 1.05:
        return np.clip(x, -32768, 32767).astype(np.int16)
    louder = x * gain
    # Keep music peaks under the clip rail after speech-frame gain.
    peak_after = float(np.abs(louder).max())
    cap = NORM_CLIP_PEAK * 32767.0
    if peak_after > cap:
        louder *= cap / peak_after
    return np.clip(louder, -32768, 32767).astype(np.int16)


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


def bass_ratio(samples: np.ndarray, rate: int) -> float:
    """Fraction of energy below ~150 Hz (music bass / rumble vs speech)."""
    if samples.size < 32:
        return 0.0
    # Cap FFT cost on long clips: a 1s mid window is enough to classify.
    x = samples.astype(np.float64)
    if x.size > rate:
        mid = x.size // 2
        half = rate // 2
        x = x[mid - half:mid + half]
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, 1.0 / rate)
    total = float(np.sum(spec ** 2))
    if total <= 1e-12:
        return 0.0
    bass = float(np.sum(spec[freqs < 150.0] ** 2))
    return bass / total


def profile_signal(samples: np.ndarray, rate: int,
                   floor: float | None = None) -> SignalProfile:
    """Measure what the clip needs before choosing filters.

    Uses speech-frame p90 (not abs peak) so a single spike cannot make a
    whisper look like loud clean speech (Windows log: peak SNR 31 dB while
    speech p90 was ~130 — Whisper returned ``*sad music*``).
    """
    empty = SignalProfile(0.0, 99.0, 99.0, 0.0, 0.0, 0.0, 0.0, False, False)
    if samples.size == 0:
        return empty
    base, _ = high_pass(samples, rate)
    use_floor = float(floor) if floor is not None and floor > 0 else measure_floor(
        base, rate)
    snr = estimate_snr_db(base, rate, floor=use_floor)
    peak = float(np.abs(samples.astype(np.float32)).max())
    bass = bass_ratio(base, rate)

    frame = max(1, int(rate * GATE_FRAME_MS / 1000))
    levels, _ = _frame_levels(base, rate, frame)
    if levels.size == 0:
        return SignalProfile(
            use_floor, snr, snr, peak, 0.0, 0.0, bass, snr < LOW_SNR_DB, False)

    # Soft line for "maybe voice" — below hard GATE_THRESHOLD so whispers count.
    soft_line = max(use_floor * 1.25, use_floor + 8.0) if use_floor > 0 else 30.0
    voiced = levels > soft_line
    voiced_frac = float(voiced.mean())
    if voiced.any():
        speech_p90 = float(np.percentile(levels[voiced], 90))
    else:
        speech_p90 = float(np.percentile(levels, 90))
    if use_floor > 1e-6:
        speech_snr = float(20.0 * np.log10(max(speech_p90, 1e-3) / use_floor))
    else:
        speech_snr = 99.0

    low = (is_low_snr(base, rate, floor=use_floor)
           or snr < LOW_SNR_DB
           or speech_snr < LOW_SNR_DB)

    # Quiet voice with real activity — needs gate-then-lift, not "clean/no-gain".
    #
    # A whisper is quiet speech in a *quiet* room. A low p90 over an elevated
    # floor is a masked voice, not a whisper: the 40x speech-frame gain would
    # amplify the bed with the voice and Whisper hears the bed ([MUSIC] over
    # café music, [SOUND] over a nature bed). Those go to the rescue path
    # (spectral subtraction + speech-band + capped gain) instead.
    is_whisper = (
        peak > NORM_DEAD_PEAK * 2
        and voiced_frac >= 0.06
        and speech_p90 < SPEECH_P90_WHISPER
        and speech_p90 > NORM_DEAD_PEAK
        and use_floor < WHISPER_MAX_FLOOR
    )

    return SignalProfile(
        floor=use_floor,
        snr_db=snr,
        speech_snr_db=speech_snr,
        peak=peak,
        speech_p90=speech_p90,
        voiced_frac=voiced_frac,
        bass_ratio=bass,
        low_snr=low,
        is_whisper=is_whisper,
    )


def plan_filters(profile: SignalProfile, *, live: bool = False,
                 force_rescue: bool | None = None,
                 pass_index: int = 1) -> VoicePlan:
    """Turn a signal profile into which filters to enable.

    Quiet-whisper (low speech_p90, some voiced frames) always gets gate +
    speech-frame lift — that is the path that makes Whisper hear the voice
    instead of tagging the room as music.
    """
    rescue = force_rescue if force_rescue is not None else profile.low_snr
    whisper = profile.is_whisper or (force_rescue is True and profile.speech_p90 < SPEECH_P90_USABLE)
    # Truly clean: normal speech level, good speech-SNR, not a whisper.
    clean = (not rescue and not whisper
             and profile.speech_snr_db >= SNR_CLEAN_DB
             and profile.speech_p90 >= SPEECH_P90_USABLE
             and profile.floor < FLOOR_SPECTRAL)
    reasons = []

    want_speech_band = (
        whisper or rescue
        or (profile.bass_ratio >= BASS_RATIO_SPEECH_BAND
            and profile.floor >= FLOOR_SPEECH_BAND)
        or (profile.bass_ratio >= BASS_RATIO_STRONG
            and profile.speech_snr_db < SNR_MILD_DB)
    )
    hpf = SPEECH_BAND_HPF_HZ if want_speech_band else HIGHPASS_HZ
    reasons.append(
        f"speech-band HPF ({profile.bass_ratio:.0%} bass)"
        if want_speech_band else "rumble HPF")

    spectral = False
    oversub = 0.85
    atten = 14.0
    if rescue or profile.floor >= FLOOR_SPECTRAL:
        spectral = True
        reasons.append("spectral")
        if rescue or profile.bass_ratio >= BASS_RATIO_STRONG:
            oversub = RESCUE_SPECTRAL_OVERSUB
            atten = RESCUE_SPECTRAL_MAX_ATTEN_DB
        if pass_index >= 2:
            oversub = min(1.05, oversub + 0.15)
            atten = min(16.0, atten + 2.0)
            reasons.append("spectral+")
    if live and not rescue and not whisper:
        spectral = False
        reasons = [r for r in reasons if not r.startswith("spectral")]

    # Whisper always gates (duck room between syllables) then lifts speech.
    gate_on = whisper or rescue or (
        not clean and (
            profile.floor >= FLOOR_GATE
            or profile.speech_snr_db < SNR_MILD_DB))
    whisper_gate = bool(whisper or rescue or profile.speech_snr_db < LOW_SNR_DB + 2)
    if gate_on:
        reasons.append("soft-gate" if whisper_gate else "gate")

    preemph = bool(whisper or rescue)
    if preemph:
        reasons.append("preemph")

    speech_norm = bool(whisper or rescue or (
        profile.speech_snr_db < SNR_MILD_DB
        and profile.floor >= FLOOR_SPECTRAL * 0.5))
    need_norm = False
    norm_target = SPEECH_NORM_TARGET
    norm_gain = SPEECH_NORM_MAX_GAIN
    if whisper:
        need_norm = True
        speech_norm = True
        norm_target = WHISPER_SPEECH_NORM_TARGET
        norm_gain = WHISPER_SPEECH_NORM_MAX_GAIN
        reasons.append(f"whisper-lift (p90={profile.speech_p90:.0f})")
    elif speech_norm and profile.speech_p90 < SPEECH_P90_USABLE * 2:
        need_norm = True
        reasons.append("speech-norm")
    elif profile.speech_p90 < PEAK_QUIET and not clean:
        need_norm = True
        reasons.append("peak-norm")
    elif profile.peak < PEAK_USABLE and profile.speech_snr_db < SNR_CLEAN_DB:
        need_norm = True
        reasons.append("peak-norm")
    else:
        reasons.append("no-gain")

    if whisper:
        reasons.insert(0, "quiet-whisper")
    elif rescue:
        reasons.insert(0, "low-SNR")
    if clean:
        reasons.insert(0, "clean")

    return VoicePlan(
        hpf_hz=hpf,
        spectral=spectral,
        spectral_oversub=oversub,
        spectral_max_atten_db=atten,
        gate=gate_on,
        whisper_gate=whisper_gate,
        preemph=preemph,
        speech_norm=speech_norm,
        normalize=need_norm,
        speech_norm_target=norm_target,
        speech_norm_max_gain=norm_gain,
        pass_index=pass_index,
        reason=", ".join(reasons) or "passthrough",
    )


def escalate_plan(plan: VoicePlan, profile: SignalProfile) -> VoicePlan:
    """Stronger plan for an audio-domain second pass (usually quiet-whisper)."""
    return replace(
        plan,
        hpf_hz=max(plan.hpf_hz, SPEECH_BAND_HPF_HZ),
        spectral=plan.spectral or profile.floor >= FLOOR_SPECTRAL * 0.5,
        spectral_oversub=min(1.1, max(plan.spectral_oversub, RESCUE_SPECTRAL_OVERSUB) + 0.1),
        spectral_max_atten_db=min(18.0, max(plan.spectral_max_atten_db,
                                            RESCUE_SPECTRAL_MAX_ATTEN_DB) + 2.0),
        gate=True,
        whisper_gate=True,
        preemph=True,
        speech_norm=True,
        normalize=True,
        speech_norm_target=WHISPER_SPEECH_NORM_TARGET,
        speech_norm_max_gain=WHISPER_SPEECH_NORM_MAX_GAIN,
        pass_index=2,
        reason=(plan.reason + " → 2nd: whisper-lift"),
    )


def needs_second_pass(original: np.ndarray, first: np.ndarray, rate: int,
                      first_plan: VoicePlan,
                      floor: float | None) -> bool:
    """Escalate when pass 1 left speech too quiet for Whisper."""
    if first_plan.pass_index >= 2:
        return False
    if first.size == 0:
        return True
    in_prof = profile_signal(original, rate, floor=floor)
    out_prof = profile_signal(first, rate, floor=floor)

    # Normal talking already loud enough — stop.
    if (not in_prof.is_whisper and not in_prof.low_snr
            and in_prof.speech_p90 >= SPEECH_P90_USABLE
            and in_prof.speech_snr_db >= SNR_CLEAN_DB):
        return False

    # Whisper still too quiet after pass 1 (or pass 1 never lifted).
    if in_prof.is_whisper and out_prof.speech_p90 < SECOND_PASS_SPEECH_P90:
        return True
    if in_prof.is_whisper and not first_plan.speech_norm:
        return True
    if in_prof.low_snr and out_prof.speech_snr_db < SECOND_PASS_SNR_DB:
        return True
    if in_prof.floor >= FLOOR_SPECTRAL and out_prof.speech_p90 < SECOND_PASS_SPEECH_P90:
        return True
    return False


def apply_plan(samples: np.ndarray, rate: int, plan: VoicePlan, *,
               floor: float | None = None,
               noise_ref: np.ndarray | None = None) -> np.ndarray:
    """Run only the filters the plan enabled.

    Order matters for whisper: gate ducks the room *before* speech-frame
    gain so we amplify voice, not the bed Whisper called music.
    """
    if samples.size == 0:
        return samples

    filtered, _ = high_pass(samples, rate, cutoff_hz=plan.hpf_hz)
    use_floor = float(floor) if floor is not None and floor > 0 else measure_floor(
        filtered, rate)

    if plan.spectral:
        ref = noise_ref
        if ref is None or (hasattr(ref, "size") and ref.size == 0):
            base80, _ = high_pass(samples, rate)
            ref = noise_reference(base80, rate, floor=use_floor)
        if ref is not None and getattr(ref, "size", 0):
            ref, _ = high_pass(ref, rate, cutoff_hz=plan.hpf_hz)
            filtered = spectral_subtract(
                filtered, rate, noise_ref=ref,
                oversub=plan.spectral_oversub,
                max_atten_db=plan.spectral_max_atten_db,
            )

    if plan.gate:
        filtered = gate(filtered, rate, floor=use_floor,
                        whisper_mode=plan.whisper_gate)

    if plan.preemph:
        filtered = pre_emphasize(filtered)

    if plan.normalize:
        if plan.speech_norm:
            return normalize_speech(
                filtered, rate, floor=use_floor,
                target=plan.speech_norm_target,
                max_gain=plan.speech_norm_max_gain,
            )
        return normalize_peak(filtered)
    return np.clip(filtered, -32768, 32767).astype(np.int16)


def enhance(samples: np.ndarray, rate: int, *,
            floor: float | None = None,
            noise_ref: np.ndarray | None = None,
            live: bool = False,
            force_rescue: bool | None = None) -> np.ndarray:
    """Smart voice amplification: profile → plan → filters → optional 2nd pass.

    ``live=True``: one pass only (streaming). ``live=False``: may re-run from
    the original samples with an escalated plan if pass 1 still looks weak.
    """
    if samples.size == 0:
        return samples

    profile = profile_signal(samples, rate, floor=floor)
    plan = plan_filters(profile, live=live, force_rescue=force_rescue,
                        pass_index=1)
    try:
        from .logging import log
        log(f"[VOICE] pass1 peak_snr≈{profile.snr_db:.0f}dB "
            f"speech_snr≈{profile.speech_snr_db:.0f}dB "
            f"p90={profile.speech_p90:.0f} voiced={profile.voiced_frac:.0%} "
            f"floor={profile.floor:.0f} → {plan.reason}")
    except Exception:
        pass

    out = apply_plan(samples, rate, plan, floor=profile.floor,
                     noise_ref=noise_ref)

    if live:
        return out

    # Verify the boost actually separated voice from room. Whisper-lift's
    # 40x speech-frame gain is aimed at a quiet-room whisper; over a music
    # or nature bed it amplifies the bed with the voice, and Whisper hears
    # the bed ([MUSIC]/[SOUND] tags - the failure in every noisy-room
    # report). If the output's speech p90 still sits close to its floor, the
    # gain lifted the room, not the voice: fall back to a plain peak boost of
    # the raw capture, which preserves the voice/room ratio as the mic heard
    # it and is what actually recovers the words.
    if plan.speech_norm and plan.normalize:
        out_prof = profile_signal(out, rate, floor=profile.floor)
        if out_prof.floor > 1 and out_prof.speech_p90 <= out_prof.floor * 3:
            try:
                from .logging import log
                log(f"[VOICE] lift did not separate speech from room "
                    f"(out p90={out_prof.speech_p90:.0f} floor="
                    f"{out_prof.floor:.0f}); reverting to plain peak boost")
            except Exception:
                pass
            plain, _ = high_pass(samples, rate, cutoff_hz=HIGHPASS_HZ)
            plain_peak = float(np.abs(plain).max())
            if plain_peak > 1:
                gain = min(PLAIN_BOOST_MAX_GAIN,
                           PLAIN_BOOST_TARGET * 32767.0 / plain_peak)
                out = np.clip(plain * gain, -32768, 32767).astype(np.int16)

    if needs_second_pass(samples, out, rate, plan, profile.floor):
        plan2 = escalate_plan(plan, profile)
        try:
            from .logging import log
            log(f"[VOICE] pass2 (audio) → {plan2.reason}")
        except Exception:
            pass
        out = apply_plan(samples, rate, plan2, floor=profile.floor,
                         noise_ref=noise_ref)
    return out


def whisper_lift_plan() -> VoicePlan:
    """Fixed plan for blank/*music* retry: gate room, lift quiet voice hard."""
    return VoicePlan(
        hpf_hz=SPEECH_BAND_HPF_HZ,
        spectral=False,
        gate=True,
        whisper_gate=True,
        preemph=True,
        speech_norm=True,
        normalize=True,
        speech_norm_target=WHISPER_SPEECH_NORM_TARGET,
        speech_norm_max_gain=WHISPER_SPEECH_NORM_MAX_GAIN,
        pass_index=1,
        reason="whisper-lift-retry",
    )


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
