"""The processing animation's arithmetic, kept out of the overlay.

The overlay is a GTK 4 layer-shell surface: importing hud_app needs gi, GTK 4
and - off Windows - gtk4-layer-shell, so nothing inside it can be exercised
on a machine that has no desktop. That is most machines this is developed and
tested on, and it is why the pill's behaviour has historically been checked by
reading its source rather than by running it.

Numbers need none of that. What the bars do while a transcript is being worked
out is arithmetic over elapsed time, so it lives here, where a test can call it.
The live noise-risk score that tints the waveform is the same kind of pure
number work: the HUD only has RMS levels, not the full audio pipeline.

Loaded by hud_app through _load_sibling, for the same reason blur_win and
wayland_blur are: importing it through the package would pull in the daemon,
and with it pystray's GTK 3, which cannot share a process with GTK 4.
"""

import math

# One sweep of the bump along the pill, per second.
SWEEP_HZ = 0.75
# Width of the bump, as a fraction of the pill. Wide enough to read as one
# moving thing rather than a single bar flicking on and off.
WIDTH = 0.16
PEAK = 0.80
# A slow swell under the bump. Without it the bars away from the bump sit at
# zero, and a row of flat bars reads as a stalled overlay rather than a
# working one.
FLOOR = 0.14
# Turns per second of the spinner, and how much of the ring is lit.
SPIN_HZ = 0.9
ARC = 1.9

# --- Live noise / transcription-risk tint ---------------------------------
# Built from the same RMS stream that drives the bars. Aligns with denoise's
# LOW_SNR_DB (~11) so yellow/red on the HUD matches when the café rescue path
# is likely to engage.
RMS_HISTORY = 80          # ~1.6s at 50 samples/s (level file cadence varies)
FLOOR_PERCENTILE = 20
# Absolute room floors (int16 RMS). Above these the room itself is hostile.
NOISY_ROOM_RMS = 450.0
LOUD_ROOM_RMS = 1200.0
# Estimated SNR bands (dB) from peak-ish frame vs quiet floor.
SNR_GOOD_DB = 16.0
SNR_BAD_DB = 9.0
# Risk score 0..1: white → yellow → red.
RISK_YELLOW_AT = 0.40
RISK_RED_AT = 0.78
GOOD_RGB = (0.95, 0.97, 1.0)      # cool white (recording, clean)
WARN_RGB = (1.0, 0.82, 0.18)      # yellow — elevated noise / soft voice
BAD_RGB = (1.0, 0.30, 0.26)       # red — poor transcription likely


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def noise_risk(rms_history: list[float] | tuple[float, ...]) -> float:
    """0..1 risk of poor transcription from a ring of recent RMS levels.

    Uses quiet-percentile floor vs recent peak (SNR) and absolute room level.
    Needs a few samples before it leaves 0; short histories stay calm so the
    pill does not flash red on open.
    """
    if len(rms_history) < 4:
        return 0.0
    values = [float(v) for v in rms_history if v >= 0]
    if len(values) < 4:
        return 0.0

    floor = _percentile(values, FLOOR_PERCENTILE)
    # Recent peak: last quarter of the window, or whole if short.
    tail_n = max(4, len(values) // 4)
    recent = values[-tail_n:]
    peak = max(recent) if recent else max(values)
    # p90 of all samples: continuous café bed raises floor and peak together.
    p90 = _percentile(values, 90)

    if floor <= 1e-3:
        snr_db = 40.0 if peak > 1 else 0.0
    else:
        snr_db = 20.0 * math.log10(max(peak, 1e-3) / floor)

    # Map SNR: good → 0, bad → 1.
    if snr_db >= SNR_GOOD_DB:
        snr_risk = 0.0
    elif snr_db <= SNR_BAD_DB:
        snr_risk = 1.0
    else:
        snr_risk = (SNR_GOOD_DB - snr_db) / (SNR_GOOD_DB - SNR_BAD_DB)

    # Absolute room energy: high floor even with "ok" ratio is still hard.
    if floor <= NOISY_ROOM_RMS:
        room_risk = 0.0
    elif floor >= LOUD_ROOM_RMS:
        room_risk = 1.0
    else:
        room_risk = (floor - NOISY_ROOM_RMS) / (LOUD_ROOM_RMS - NOISY_ROOM_RMS)

    # Flat loud signal (music bed) — peak barely above floor.
    flatness = 0.0
    if p90 > NOISY_ROOM_RMS and floor > 1:
        ratio = p90 / floor
        if ratio < 1.8:
            flatness = min(1.0, (1.8 - ratio) / 0.8)

    risk = max(snr_risk, room_risk * 0.85, flatness * 0.9)
    return float(min(1.0, max(0.0, risk)))


def risk_rgb(risk: float) -> tuple[float, float, float]:
    """Lerp good → yellow → red for a 0..1 risk score."""
    r = float(min(1.0, max(0.0, risk)))
    if r <= RISK_YELLOW_AT:
        t = r / RISK_YELLOW_AT if RISK_YELLOW_AT > 0 else 0.0
        return _lerp_rgb(GOOD_RGB, WARN_RGB, t)
    t = (r - RISK_YELLOW_AT) / max(1e-6, RISK_RED_AT - RISK_YELLOW_AT)
    t = min(1.0, t)
    return _lerp_rgb(WARN_RGB, BAD_RGB, t)


def _lerp_rgb(a, b, t: float) -> tuple[float, float, float]:
    t = min(1.0, max(0.0, t))
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def sweep(elapsed: float, bars: int) -> list[float]:
    """Bar heights at `elapsed` seconds into processing, each 0..1.

    A gaussian bump crossing the pill and coming round again, over a swell.
    It is driven by the clock and by nothing else, because there is nothing
    else to drive it with: whisper.cpp reports no progress, it simply answers
    when it is done. So this says "still working", which is true, rather than
    miming a percentage that would have to be invented.
    """
    if bars <= 0:
        return []
    if bars == 1:
        return [min(1.0, FLOOR + PEAK)]

    span = 1.0 + 2.0 * WIDTH
    centre = -WIDTH + (elapsed * SWEEP_HZ % 1.0) * span
    levels = []
    for index in range(bars):
        position = index / float(bars - 1)
        # Wrapped, so the bump re-enters at the left as it leaves at the
        # right. Measured straight, the pill empties out for a moment at the
        # end of every pass and the animation reads as having stopped -
        # which is the one thing it exists not to say.
        offset = position - centre
        distance = min(abs(offset), abs(offset - span), abs(offset + span))
        swell = 0.5 + 0.5 * math.sin(elapsed * 2.2 + position * 3.0)
        levels.append(min(1.0, FLOOR * swell
                          + PEAK * math.exp(-(distance / WIDTH) ** 2)))
    return levels


def spinner_angle(elapsed: float) -> float:
    """Where the lit arc starts, in radians, at `elapsed` seconds."""
    return elapsed * SPIN_HZ * 2.0 * math.pi
