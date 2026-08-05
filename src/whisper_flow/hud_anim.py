"""The processing animation's arithmetic, kept out of the overlay.

The overlay is a GTK 4 layer-shell surface: importing hud_app needs gi, GTK 4
and - off Windows - gtk4-layer-shell, so nothing inside it can be exercised
on a machine that has no desktop. That is most machines this is developed and
tested on, and it is why the pill's behaviour has historically been checked by
reading its source rather than by running it.

Numbers need none of that. What the bars do while a transcript is being worked
out is arithmetic over elapsed time, so it lives here, where a test can call it.

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
