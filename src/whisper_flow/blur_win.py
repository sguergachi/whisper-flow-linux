"""Window composition on Windows 11: transparency, corners, border.

Not blur. The pill is a capsule, and Windows will not blur inside a shape:

  * DWMWA_SYSTEMBACKDROP_TYPE paints the window's whole rectangle and
    ignores its region. The region was applied and GetWindowRgnBox confirmed
    the capsule; the grey slab stayed.
  * The undocumented accent policy does the same, only black.
  * The Visual Layer will honour a clip, and neither of its brushes can
    sample the desktop from a plain Win32 window: CreateHostBackdropBrush
    renders black outside UWP, and CreateBackdropBrush sees only its own
    composition tree. A solid colour brush in that same tree drew a perfect
    capsule, so the plumbing was right and the sampling is simply absent.

So the pill is opaque, and what this module still does is make the pixels
*outside* it disappear: DwmEnableBlurBehindWindow over an empty region,
which has not blurred anything since Windows 8 removed Aero and now only
asks DWM to honour the alpha channel. A window region is kept as the
fallback for when even that fails.

Earlier builds are refused rather than worked around. Having one supported
target is worth more than covering a build nobody here runs.
"""

import ctypes
import ctypes.wintypes as wintypes

# DwmSetWindowAttribute attributes, all Windows 11.
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_BORDER_COLOR = 34
DWMWA_SYSTEMBACKDROP_TYPE = 38

# DWM_SYSTEMBACKDROP_TYPE
DWMSBT_TRANSIENTWINDOW = 3   # acrylic; the flyout/transient variant

# DWM_WINDOW_CORNER_PREFERENCE
DWMWCP_DONOTROUND = 1        # we cut our own shape; DWM must not add corners
DWMWCP_ROUND = 2             # the full radius, as used by system flyouts
DWMWCP_ROUNDSMALL = 3

# DwmEnableBlurBehindWindow, which is what makes the alpha channel real.
DWM_BB_ENABLE = 0x1
DWM_BB_BLURREGION = 0x2


class _DWM_BLURBEHIND(ctypes.Structure):
    _fields_ = [
        ("dwFlags", ctypes.c_uint),
        ("fEnable", ctypes.c_int),
        ("hRgnBlur", ctypes.c_void_p),
        ("fTransitionOnMaximized", ctypes.c_int),
    ]

# Build 22621 is 22H2, the first with DWMWA_SYSTEMBACKDROP_TYPE. 22000 is
# 21H2, which only had the undocumented DWMWA_MICA_EFFECT.
MIN_BUILD = 22621

DWMWA_COLOR_NONE = 0xFFFFFFFE  # sentinel: suppress the border entirely


class _OSVERSIONINFOEXW(ctypes.Structure):
    _fields_ = [
        ("dwOSVersionInfoSize", ctypes.c_ulong),
        ("dwMajorVersion", ctypes.c_ulong),
        ("dwMinorVersion", ctypes.c_ulong),
        ("dwBuildNumber", ctypes.c_ulong),
        ("dwPlatformId", ctypes.c_ulong),
        ("szCSDVersion", ctypes.c_wchar * 128),
        ("wServicePackMajor", ctypes.c_ushort),
        ("wServicePackMinor", ctypes.c_ushort),
        ("wSuiteMask", ctypes.c_ushort),
        ("wProductType", ctypes.c_byte),
        ("wReserved", ctypes.c_byte),
    ]


def windows_build() -> int:
    """The real OS build number.

    Read through RtlGetVersion because GetVersionEx lies to processes without
    a compatibility manifest, reporting Windows 8 on anything newer - which
    would make every Windows 11 machine look unsupported.
    """
    try:
        info = _OSVERSIONINFOEXW()
        info.dwOSVersionInfoSize = ctypes.sizeof(info)
        ctypes.WinDLL("ntdll").RtlGetVersion(ctypes.byref(info))
        return int(info.dwBuildNumber)
    except Exception:
        return 0


def is_supported() -> bool:
    """Whether this build has the composition attributes the HUD relies on."""
    return windows_build() >= MIN_BUILD


def _set_attribute(hwnd: int, attribute: int, value: int) -> bool:
    try:
        dwm = ctypes.WinDLL("dwmapi")
        val = ctypes.c_int(value)
        return dwm.DwmSetWindowAttribute(
            wintypes.HWND(hwnd), ctypes.c_uint(attribute),
            ctypes.byref(val), ctypes.sizeof(val),
        ) == 0
    except Exception:
        return False


def _enable_per_pixel_alpha(hwnd: int) -> bool:
    """Make the window's alpha channel mean something.

    An ordinary window is composited as opaque: whatever cairo puts in the
    alpha channel is thrown away, and a pixel drawn fully transparent comes
    out as whatever the backdrop layer beneath it happened to be - grey with
    the system backdrop, black with the accent policy. Enabling blur-behind
    over an empty region asks DWM to honour the channel instead. It is the
    long-standing way to do this and, since Windows 8 removed Aero, it no
    longer blurs anything: enabling the alpha is all it now does, which is
    exactly what is wanted here.

    The region is empty, not the pill: an empty one means "no blur anywhere",
    and the whole window keeps its per-pixel alpha.
    """
    try:
        gdi32 = ctypes.WinDLL("gdi32")
        dwm = ctypes.WinDLL("dwmapi")
        gdi32.CreateRectRgn.argtypes = [ctypes.c_int] * 4
        gdi32.CreateRectRgn.restype = ctypes.c_void_p
        dwm.DwmEnableBlurBehindWindow.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p]
        dwm.DwmEnableBlurBehindWindow.restype = ctypes.c_long

        empty = gdi32.CreateRectRgn(0, 0, -1, -1)
        blur = _DWM_BLURBEHIND(
            DWM_BB_ENABLE | DWM_BB_BLURREGION, 1, empty, 0)
        ok = dwm.DwmEnableBlurBehindWindow(
            ctypes.c_void_p(hwnd), ctypes.byref(blur)) == 0
        if empty:
            gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
            gdi32.DeleteObject(ctypes.c_void_p(empty))
        return ok
    except Exception:
        return False


def apply_window_style(hwnd: int) -> str | None:
    """Give the window a blurred backdrop, no corners of its own, no border.

    Every backdrop DWM will draw is a rectangle. DWMWA_SYSTEMBACKDROP_TYPE
    paints across the window's whole rect and ignores the window region -
    the region was applied, GetWindowRgnBox confirmed the capsule, and the
    grey slab was still there. The undocumented accent policy behaves the
    same way; it changed the slab's colour and nothing else. A compositor
    blur here is a rectangle or it is not there at all.

    So the backdrop goes and the alpha channel does the work. The corners
    outside the pill are transparent because nothing is drawn there, with
    cairo's antialiasing intact rather than a region's hard cut. What the
    pill loses is the frost behind it; what it gains is being a pill. Its
    own material, sheen, inner shadow and outline are all still drawn.

    The name of what was applied comes back so the caller knows whether it
    still needs the window region: only when the alpha could not be enabled,
    because then transparent pixels are opaque again and cutting the shape
    out of the window is the one remaining way to be rid of them.

    Args:
        hwnd: Native window handle.

    Returns:
        A short description of what was applied, or None on a build that does
        not support it.

    """
    if not hwnd or not is_supported():
        return None

    # Dark first: anything DWM tints, it tints towards the theme, and asking
    # afterwards leaves the first composited frame light.
    _set_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
    # Suppress DWM's own border; the overlay draws its own outline.
    _set_attribute(hwnd, DWMWA_BORDER_COLOR, DWMWA_COLOR_NONE)
    # No corner radius of DWM's. The pill's shape is what cairo draws, and a
    # rounded rectangle behind it is the exact thing being removed.
    _set_attribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_DONOTROUND)

    if _enable_per_pixel_alpha(hwnd):
        return "per-pixel-alpha"

    # Nothing honours the alpha channel here. A frosted rectangle beats a
    # black one, and the caller cuts the window down to the pill instead.
    if not _set_attribute(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW):
        return None
    _set_attribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)
    return "dwm-acrylic"


def set_shape(hwnd: int, points) -> bool:
    """Clip the window to a polygon; everything outside it becomes nothing.

    A top-level window on Windows is a rectangle, and DWM composites its
    backdrop across all of it. The pill was drawn with transparent corners
    and the transparency went nowhere: the acrylic filled the full rect, so
    the capsule sat on an opaque grey slab with the desktop hidden behind it.

    A window region is the one mechanism that removes those pixels rather
    than colouring them - outside the region the window does not exist, so
    the desktop shows through and clicks pass to whatever is underneath.
    The cut is hard-edged, with none of the compositor's antialiasing, which
    is why the caller inflates it past the drawn outline: the ragged edge
    then falls outside the pill instead of eating into its rim.

    Args:
        hwnd: Native window handle.
        points: The outline, in physical pixels relative to the window's
            top-left corner.

    Returns:
        Whether the region was applied.

    """
    if not hwnd or not points:
        return False
    try:
        gdi32 = ctypes.WinDLL("gdi32")
        user32 = ctypes.WinDLL("user32")
        # A region handle is pointer-sized. Untyped, ctypes would marshal it
        # as a 32-bit int and hand SetWindowRgn a handle that is not the one
        # CreatePolygonRgn returned.
        gdi32.CreatePolygonRgn.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        gdi32.CreatePolygonRgn.restype = ctypes.c_void_p
        user32.SetWindowRgn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
        user32.SetWindowRgn.restype = ctypes.c_int

        count = len(points)
        buffer = (wintypes.POINT * count)()
        for i, (x, y) in enumerate(points):
            buffer[i].x = round(x)
            buffer[i].y = round(y)
        WINDING = 2      # correct for a closed outline that never self-crosses
        region = gdi32.CreatePolygonRgn(ctypes.byref(buffer), count, WINDING)
        if not region:
            return False
        # Redraw: without it the old rectangle stays on screen until
        # something else invalidates the window. Windows owns the region
        # after this call, so it must not be deleted here.
        return bool(user32.SetWindowRgn(
            ctypes.c_void_p(hwnd), ctypes.c_void_p(region), True))
    except Exception:
        return False


def unsupported_reason() -> str:
    """Why this machine cannot run the overlay, for the log."""
    build = windows_build()
    if build == 0:
        return "could not determine the Windows build"
    return (f"Windows build {build} is too old; this needs {MIN_BUILD} "
            f"(Windows 11 22H2) for acrylic and rounded corners")
