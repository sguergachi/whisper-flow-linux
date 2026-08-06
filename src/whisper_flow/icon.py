"""The microphone mark, drawn once and used everywhere it is needed.

The tray draws it at 64px with a dark halo so a light glyph still reads
against a light panel. The application icon is the same drawing without the
halo, at the sizes Windows asks for - the halo exists for a background it is
composited onto, and an .ico is composited onto nothing.

Its own module, not daemon.py, because the build needs it: the spec makes the
.ico at package time, and importing the daemon there would pull in pystray,
PIL's tray backend and the whole application behind it. Nothing here imports
anything but PIL.
"""

from PIL import Image, ImageDraw, ImageFilter

ICON_SIZE = 64
ICON_SUPERSAMPLE = 8  # draw large, downscale: PIL has no antialiased primitives
ICON_IDLE = (245, 245, 247, 255)
ICON_RECORDING = (255, 69, 74, 255)
# Width of the soft outer halo, in final icon pixels. Odd, as MaxFilter requires.
HALO_PIXELS = 7
# Near-black stroke under the glyph so a light mic still reads on light trays.
ICON_OUTLINE = (18, 18, 22, 255)
# Outline thickness on the 64px design grid (drawn under the fill).
OUTLINE_UNITS = 2.4

# What an .ico carries. Windows picks per context - 16 in the title bar, 32 in
# the taskbar, 256 in Explorer's large view - and one it has to scale itself
# is the one that looks soft.
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

# What a window needs, which is far less: the title bar and the taskbar ask
# for the small and large system icon sizes and nothing above 64 is ever
# shown. Drawn at open, so the sizes nobody will see are worth skipping -
# 256px alone is most of the cost, and it is Explorer's, not a window's.
WINDOW_ICO_SIZES = (16, 20, 24, 32, 40, 48, 64)

# The mark on its own, without the tray's halo, in a colour that reads on both
# a light and a dark desktop.
APP_COLOR = (236, 238, 242, 255)


def _clamp_byte(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _mix_rgb(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    t: float,
) -> tuple[int, int, int, int]:
    """Linear blend of two RGBA colours; t=0 → a, t=1 → b."""
    return (
        _clamp_byte(a[0] + (b[0] - a[0]) * t),
        _clamp_byte(a[1] + (b[1] - a[1]) * t),
        _clamp_byte(a[2] + (b[2] - a[2]) * t),
        _clamp_byte(a[3] + (b[3] - a[3]) * t),
    )


def _shade_rgb(
    color: tuple[int, int, int, int],
    factor: float,
    lift: float = 0.0,
) -> tuple[int, int, int, int]:
    """Scale RGB toward black (factor < 1) or white (factor > 1), optional lift."""
    r, g, b, a = color
    return (
        _clamp_byte(r * factor + lift),
        _clamp_byte(g * factor + lift),
        _clamp_byte(b * factor + lift),
        a,
    )


def _draw_mic_silhouette(
    draw: ImageDraw.ImageDraw,
    s: float,
    fill: tuple[int, int, int, int],
    stroke: int,
    inflate: float = 0.0,
) -> None:
    """Microphone capsule + cradle + stem on the 64-unit design grid.

    inflate grows the silhouette outward (used for the dark outline pass).
    """
    u = s / 64.0
    cx = s / 2
    pad = inflate * u

    draw.rounded_rectangle(
        [cx - 8 * u - pad, 9 * u - pad, cx + 8 * u + pad, 35 * u + pad],
        radius=8 * u + pad,
        fill=fill,
    )

    # Cradle: lower half-circle under the capsule.
    # PIL angles run clockwise from 3 o'clock, so 0→180 sweeps the bottom.
    cradle_r = 13 * u + pad
    cradle_cy = 34 * u
    draw.arc(
        [cx - cradle_r, cradle_cy - cradle_r, cx + cradle_r, cradle_cy + cradle_r],
        start=0,
        end=180,
        fill=fill,
        width=max(1, round(stroke + 2 * pad)),
    )

    stem_w = max(1, round(stroke + 2 * pad))
    draw.line(
        [cx, cradle_cy + cradle_r, cx, 54 * u + pad],
        fill=fill,
        width=stem_w,
    )
    draw.line(
        [cx - 9 * u - pad, 54 * u + pad, cx + 9 * u + pad, 54 * u + pad],
        fill=fill,
        width=stem_w,
    )


def _gradient_fill(
    mask: Image.Image,
    top: tuple[int, int, int, int],
    bottom: tuple[int, int, int, int],
) -> Image.Image:
    """Vertical highlight→shadow gradient clipped to mask alpha."""
    w, h = mask.size
    # One column of blended colours, then stretch — O(h) not O(w*h) in Python.
    column = Image.new("RGBA", (1, h))
    column.putdata([_mix_rgb(top, bottom, y / max(1, h - 1)) for y in range(h)])
    gradient = column.resize((w, h), Image.NEAREST)
    gradient.putalpha(mask.getchannel("A"))
    return gradient


def draw_mic(size: int, color: tuple[int, int, int, int]) -> Image.Image:
    """The glyph at `size`, antialiased by supersampling.

    Drawn in a 64px design grid scaled up by ICON_SUPERSAMPLE and reduced,
    because PIL's primitives have hard edges and the icon looks ragged at any
    size worth shipping.

    The fill is a top→bottom gradient (a little dimensionality) and sits on a
    dark outline so a light glyph still reads on a light background.
    """
    s = size * ICON_SUPERSAMPLE
    stroke = max(1, round(4.5 * (s / 64.0)))

    # Dark outline drawn slightly larger than the fill.
    outline = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    _draw_mic_silhouette(
        ImageDraw.Draw(outline), s, ICON_OUTLINE, stroke, inflate=OUTLINE_UNITS,
    )

    # Solid mask of the true glyph, then paint a vertical gradient through it.
    mask = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    _draw_mic_silhouette(ImageDraw.Draw(mask), s, (255, 255, 255, 255), stroke)

    # Highlight at the top, deeper shade at the base — reads as slight bevel.
    top = _shade_rgb(color, 1.06, lift=18)
    bottom = _shade_rgb(color, 0.72, lift=0)
    glyph = _gradient_fill(mask, top, bottom)

    image = Image.alpha_composite(outline, glyph)
    return image.resize((size, size), Image.LANCZOS)


def tray_icon(color: tuple[int, int, int, int]) -> Image.Image:
    """The glyph with the halo the tray needs behind it.

    Downscale first, then grow the halo. Growing it at the supersampled size
    meant a 41-pixel kernel over 512x512 - 609ms of the 615ms this used to
    take, for a halo a few pixels wide once reduced to 64. The same operation
    at the final size is 0.3ms, and the supersampling still does its job on
    the glyph, which is the only part that needed it.
    """
    image = draw_mic(ICON_SIZE, color)

    # Soft dark halo outside the crisp outline — extra contrast on light trays.
    # Opacity is higher than the old ~39% fill: the outline does the hard edge,
    # the halo keeps the mark from vanishing on pale panels.
    halo_alpha = image.getchannel("A").filter(ImageFilter.MaxFilter(HALO_PIXELS))
    halo = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    halo.putalpha(halo_alpha.point(lambda v: v * 160 // 255))
    return Image.alpha_composite(halo, image)


def write_ico(path: str, sizes: tuple = ICO_SIZES) -> str:
    """Write the application icon, the same mark the tray shows.

    Every size is drawn at its own resolution rather than left to the .ico
    writer, which would resample one bitmap down to 16px and lose the stem
    and the base of the microphone entirely.

    `sizes` is for the windows, which ask for a handful of small ones at open
    and never for the largest: see WINDOW_ICO_SIZES. The build passes nothing
    and gets the full set, which is what the executable and the shortcuts
    need.
    """
    frames = [draw_mic(size, APP_COLOR) for size in sizes]
    largest = frames[-1]
    largest.save(path, format="ICO",
                 sizes=[(s, s) for s in sizes],
                 append_images=frames[:-1])
    return path
