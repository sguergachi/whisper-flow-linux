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
# Width of the dark halo, in final icon pixels. Odd, as MaxFilter requires.
HALO_PIXELS = 5

# What an .ico carries. Windows picks per context - 16 in the title bar, 32 in
# the taskbar, 256 in Explorer's large view - and one it has to scale itself
# is the one that looks soft.
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)

# The mark on its own, without the tray's halo, in a colour that reads on both
# a light and a dark desktop.
APP_COLOR = (236, 238, 242, 255)


def draw_mic(size: int, color: tuple[int, int, int, int]) -> Image.Image:
    """The glyph at `size`, antialiased by supersampling.

    Drawn in a 64px design grid scaled up by ICON_SUPERSAMPLE and reduced,
    because PIL's primitives have hard edges and the icon looks ragged at any
    size worth shipping.
    """
    s = size * ICON_SUPERSAMPLE
    image = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    u = s / 64.0  # one unit of the 64px design grid
    cx = s / 2
    stroke = max(1, round(4.5 * u))

    # Capsule body
    draw.rounded_rectangle(
        [cx - 8 * u, 9 * u, cx + 8 * u, 35 * u],
        radius=8 * u,
        fill=color,
    )

    # Cradle: the lower half of a circle, wrapping under the capsule.
    # PIL angles run clockwise from 3 o'clock, so 0->180 sweeps the bottom.
    cradle_r = 13 * u
    cradle_cy = 34 * u
    draw.arc(
        [cx - cradle_r, cradle_cy - cradle_r, cx + cradle_r, cradle_cy + cradle_r],
        start=0,
        end=180,
        fill=color,
        width=stroke,
    )

    # Stem down from the cradle, then the base
    draw.line([cx, cradle_cy + cradle_r, cx, 54 * u], fill=color, width=stroke)
    draw.line([cx - 9 * u, 54 * u, cx + 9 * u, 54 * u], fill=color, width=stroke)

    return image.resize((size, size), Image.LANCZOS)


def tray_icon(color: tuple[int, int, int, int]) -> Image.Image:
    """The glyph with the halo the tray needs behind it.

    Downscale first, then grow the halo. Growing it at the supersampled size
    meant a 41-pixel kernel over 512x512 - 609ms of the 615ms this used to
    take, for a halo five pixels wide once reduced to 64. The same operation
    at the final size is 0.3ms, and the supersampling still does its job on
    the glyph, which is the only part that needed it.
    """
    image = draw_mic(ICON_SIZE, color)

    # A dark halo grown from the glyph's own alpha, so a light icon still reads
    # against a light panel without needing to know the tray's theme.
    halo_alpha = image.getchannel("A").filter(ImageFilter.MaxFilter(HALO_PIXELS))
    halo = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    halo.putalpha(halo_alpha.point(lambda v: v * 100 // 255))
    return Image.alpha_composite(halo, image)


def write_ico(path: str) -> str:
    """Write the application icon, the same mark the tray shows.

    Every size is drawn at its own resolution rather than left to the .ico
    writer, which would resample one bitmap down to 16px and lose the stem
    and the base of the microphone entirely.
    """
    frames = [draw_mic(size, APP_COLOR) for size in ICO_SIZES]
    largest = frames[-1]
    largest.save(path, format="ICO",
                 sizes=[(s, s) for s in ICO_SIZES],
                 append_images=frames[:-1])
    return path
