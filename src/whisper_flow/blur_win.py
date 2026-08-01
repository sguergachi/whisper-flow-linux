"""Backdrop blur on Windows.

Windows has three ways to blur what is behind a window, and which exist
depends on the build:

  * DwmSetWindowAttribute(DWMWA_SYSTEMBACKDROP_TYPE) - Windows 11 22H2 and
    later. The documented one, and the only one Microsoft supports.
  * SetWindowCompositionAttribute with ACCENT_ENABLE_ACRYLICBLURBEHIND -
    Windows 10 1803 and later. Undocumented, exported by ordinal, and what
    most applications actually used before 22H2.
  * ACCENT_ENABLE_BLURBEHIND - earlier Windows 10. Plain Gaussian blur, no
    acrylic noise or tint.

They are tried in that order and the first that takes is used, so a machine
gets the best it supports rather than the lowest common denominator.

The window this is applied to should be layered with a uniform alpha below 1
(tkinter's "-alpha"). DWM blurs the backdrop, the window's own painted
content is composited over it, and the alpha is what lets the blur show
through the tint. Without the alpha the blur is there but hidden behind
opaque paint.
"""

import ctypes
import ctypes.wintypes as wintypes

# DwmSetWindowAttribute
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMSBT_TRANSIENTWINDOW = 3  # acrylic: the right one for a transient overlay

# SetWindowCompositionAttribute
WCA_ACCENT_POLICY = 19
ACCENT_ENABLE_BLURBEHIND = 3
ACCENT_ENABLE_ACRYLICBLURBEHIND = 4

# Acrylic tint, as AABBGGRR - note the byte order is not RGBA. Dark and mostly
# transparent, so the blur stays visible through it.
DEFAULT_TINT = 0x99201814


class ACCENT_POLICY(ctypes.Structure):
    _fields_ = [
        ("AccentState", ctypes.c_int),
        ("AccentFlags", ctypes.c_int),
        ("GradientColor", ctypes.c_uint),
        ("AnimationId", ctypes.c_int),
    ]


class WINCOMPATTRDATA(ctypes.Structure):
    _fields_ = [
        ("Attribute", ctypes.c_int),
        ("Data", ctypes.POINTER(ACCENT_POLICY)),
        ("SizeOfData", ctypes.c_size_t),
    ]


def _try_system_backdrop(hwnd: int) -> bool:
    """Windows 11 acrylic. Documented, and the only supported route."""
    try:
        dwm = ctypes.WinDLL("dwmapi")
        value = ctypes.c_int(DWMSBT_TRANSIENTWINDOW)
        result = dwm.DwmSetWindowAttribute(
            wintypes.HWND(hwnd), ctypes.c_uint(DWMWA_SYSTEMBACKDROP_TYPE),
            ctypes.byref(value), ctypes.sizeof(value),
        )
        if result != 0:
            return False
        # Ask for the dark variant so the acrylic tints towards black rather
        # than white; harmless if it is not supported.
        dark = ctypes.c_int(1)
        dwm.DwmSetWindowAttribute(
            wintypes.HWND(hwnd), ctypes.c_uint(DWMWA_USE_IMMERSIVE_DARK_MODE),
            ctypes.byref(dark), ctypes.sizeof(dark),
        )
        return True
    except Exception:
        return False


def _try_accent(hwnd: int, state: int, tint: int) -> bool:
    """Windows 10 acrylic or blur, via the undocumented accent policy.

    SetWindowCompositionAttribute is exported by name but not declared in any
    header, and is absent on older builds - hence the getattr rather than a
    direct call.
    """
    try:
        user32 = ctypes.WinDLL("user32")
        fn = getattr(user32, "SetWindowCompositionAttribute", None)
        if fn is None:
            return False
        accent = ACCENT_POLICY(state, 2, tint, 0)
        data = WINCOMPATTRDATA(
            WCA_ACCENT_POLICY, ctypes.pointer(accent), ctypes.sizeof(accent),
        )
        return bool(fn(wintypes.HWND(hwnd), ctypes.byref(data)))
    except Exception:
        return False


def enable_blur(hwnd: int, tint: int = DEFAULT_TINT) -> str | None:
    """Blur whatever is behind this window.

    Args:
        hwnd: Native window handle.
        tint: Acrylic tint as AABBGGRR.

    Returns:
        The name of the method that worked, or None if the machine supports
        none of them - in which case the caller should draw an opaque panel.

    """
    if not hwnd:
        return None
    if _try_system_backdrop(hwnd):
        return "dwm-acrylic"
    if _try_accent(hwnd, ACCENT_ENABLE_ACRYLICBLURBEHIND, tint):
        return "accent-acrylic"
    if _try_accent(hwnd, ACCENT_ENABLE_BLURBEHIND, tint):
        return "accent-blur"
    return None


def disable_blur(hwnd: int) -> None:
    """Drop the effect, for symmetry with the Wayland side."""
    _try_accent(hwnd, 0, 0)  # ACCENT_DISABLED
