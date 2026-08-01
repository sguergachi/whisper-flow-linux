"""Window composition on Windows 11: acrylic backdrop, rounded corners, border.

Windows 11 22H2 exposes all three through DwmSetWindowAttribute, so this is
the documented path with no undocumented ordinals and no per-build branching.

Earlier builds are refused rather than worked around. Windows 10 could get a
similar effect through SetWindowCompositionAttribute, but that is undocumented,
looks visibly different, and having one supported target is worth more than
covering a build nobody here runs.

The window must stay layered with a uniform alpha below 1. DWM blurs the
backdrop and composites the window over it, so an opaque window hides the
effect completely - the alpha is what lets the acrylic through.
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
DWMWCP_ROUND = 2             # the full radius, as used by system flyouts
DWMWCP_ROUNDSMALL = 3

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


def apply_window_style(hwnd: int) -> str | None:
    """Give the window an acrylic backdrop, rounded corners and no border.

    Args:
        hwnd: Native window handle.

    Returns:
        A short description of what was applied, or None on a build that does
        not support it.

    """
    if not hwnd or not is_supported():
        return None

    # Dark first: the acrylic tints towards the theme, and asking afterwards
    # leaves the first composited frame light.
    _set_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1)

    if not _set_attribute(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW):
        return None

    # Rounded by the compositor, so the corners are antialiased. The Windows 10
    # route was a window region, which is a hard-edged mask.
    _set_attribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)
    # Suppress DWM's own border; the overlay draws its own outline.
    _set_attribute(hwnd, DWMWA_BORDER_COLOR, DWMWA_COLOR_NONE)
    return "dwm-acrylic"


def unsupported_reason() -> str:
    """Why this machine cannot run the overlay, for the log."""
    build = windows_build()
    if build == 0:
        return "could not determine the Windows build"
    return (f"Windows build {build} is too old; this needs {MIN_BUILD} "
            f"(Windows 11 22H2) for acrylic and rounded corners")
