"""Request compositor background blur via ext-background-effect-v1.

This is the upstream Wayland protocol for backdrop blur. Nothing else works:
KWin ignores the older X11 `_KDE_NET_WM_BLUR_BEHIND_REGION` property in a
Wayland session, and a client cannot read the pixels behind its own surface to
blur them itself.

There are no Python bindings for the protocol, so the two requests we need are
marshalled by hand through libwayland-client. That means declaring the wire
description (wl_interface / wl_message) for the two interfaces exactly as the
XML defines them - the opcodes below are request order within each interface,
and getting them wrong is a protocol error that kills the connection.

The requests run on a private event queue so that dispatching them cannot
interfere with the queue GTK is driving on the same connection.
"""

import ctypes
import ctypes.util

# --- libwayland-client ------------------------------------------------------

_WL = None


def _wl():
    global _WL
    if _WL is None:
        path = ctypes.util.find_library("wayland-client") or "libwayland-client.so.0"
        _WL = ctypes.CDLL(path)
    return _WL


class WlMessage(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("signature", ctypes.c_char_p),
        ("types", ctypes.POINTER(ctypes.c_void_p)),
    ]


class WlInterface(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("version", ctypes.c_int),
        ("method_count", ctypes.c_int),
        ("methods", ctypes.POINTER(WlMessage)),
        ("event_count", ctypes.c_int),
        ("events", ctypes.POINTER(WlMessage)),
    ]


# Registry listener: global(data, registry, name, interface, version)
_GLOBAL_CB = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_char_p, ctypes.c_uint32,
)
_GLOBAL_REMOVE_CB = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
)


class _RegistryListener(ctypes.Structure):
    _fields_ = [("global_", _GLOBAL_CB), ("global_remove", _GLOBAL_REMOVE_CB)]


MANAGER_NAME = b"ext_background_effect_manager_v1"
COMPOSITOR_NAME = b"wl_compositor"

# Request opcodes, in the order the interfaces declare them.
WL_DISPLAY_GET_REGISTRY = 1
WL_REGISTRY_BIND = 0
WL_COMPOSITOR_CREATE_REGION = 1    # after create_surface(0)
WL_REGION_ADD = 1                  # after destroy(0)
MANAGER_GET_BACKGROUND_EFFECT = 1  # after destroy(0)
SURFACE_SET_BLUR_REGION = 1        # after destroy(0)


class _Protocol:
    """Wire descriptions for the two ext_background_effect interfaces."""

    def __init__(self):
        wl = _wl()

        # ext_background_effect_surface_v1: destroy, set_blur_region(region?)
        self._surface_methods = (WlMessage * 2)(
            WlMessage(b"destroy", b"", None),
            WlMessage(b"set_blur_region", b"?o", self._types_region()),
        )
        self.surface_iface = WlInterface(
            b"ext_background_effect_surface_v1", 1,
            2, self._surface_methods, 0, None,
        )

        # ext_background_effect_manager_v1: destroy, get_background_effect(id, surface)
        self._mgr_types = (ctypes.c_void_p * 2)(
            ctypes.addressof(self.surface_iface),
            ctypes.addressof(WlInterface.in_dll(wl, "wl_surface_interface")),
        )
        self._mgr_methods = (WlMessage * 2)(
            WlMessage(b"destroy", b"", None),
            WlMessage(b"get_background_effect", b"no", self._mgr_types),
        )
        self.manager_iface = WlInterface(
            MANAGER_NAME, 1, 2, self._mgr_methods, 1, self._capabilities_events(),
        )

    def _types_region(self):
        wl = _wl()
        arr = (ctypes.c_void_p * 1)(
            ctypes.addressof(WlInterface.in_dll(wl, "wl_region_interface")),
        )
        self._region_types = arr  # keep alive
        return arr

    def _capabilities_events(self):
        self._mgr_event_types = (ctypes.c_void_p * 1)(None)
        self._mgr_events = (WlMessage * 1)(
            WlMessage(b"capabilities", b"u", self._mgr_event_types),
        )
        return self._mgr_events


_PROTO = None


def _proto():
    global _PROTO
    if _PROTO is None:
        _PROTO = _Protocol()
    return _PROTO


class BlurHandle:
    """Keeps the blur objects alive; dropping it would drop the effect."""

    def __init__(self, manager, effect_surface, queue, display, region,
                 empty_region, version, compositor, region_iface,
                 compositor_version):
        self._manager = manager
        self._effect_surface = effect_surface
        self._queue = queue
        self._display = display
        self._region = region
        self._empty_region = empty_region
        self._version = version
        self._compositor = compositor
        self._region_iface = region_iface
        self._cver = compositor_version
        # Regions handed to the compositor stay referenced on this side, so a
        # destroyed proxy can never race a compositor still reading it.
        self._retired = []
        self.active = False

    def set_active(self, on: bool):
        """Turn the effect on or off.

        The blur cannot be faded - the compositor applies it at full strength
        or not at all - so it has to be switched at a moment when the change
        is not visible.

        Switching off uses an empty region rather than NULL. NULL removes the
        effect entirely, and restoring a region afterwards did not reliably
        come back shaped: blur reappeared over the surface's whole bounding
        box, square corners and all. Handing over a real, empty region keeps
        the request symmetric and the shape intact.
        """
        if on == self.active:
            return
        wl = _wl()
        wl.wl_proxy_marshal_flags(
            ctypes.c_void_p(self._effect_surface), SURFACE_SET_BLUR_REGION,
            None, ctypes.c_uint32(self._version), ctypes.c_uint32(0),
            ctypes.c_void_p(self._region if on else self._empty_region),
        )
        wl.wl_display_flush(ctypes.c_void_p(self._display))
        self.active = on

    def set_region(self, rows: list[tuple[int, int, int, int]]):
        """Replace the blur region with a new shape.

        The stop button grows the pill, and the pill shrinks back when the
        recording ends - a region built for one of those shapes blurs
        through the other. The effect surface only accepts one effect, so a
        second enable_blur is a protocol error; this asks the same effect
        for a new region instead.
        """
        wl = _wl()
        region = wl.wl_proxy_marshal_flags(
            ctypes.c_void_p(self._compositor), WL_COMPOSITOR_CREATE_REGION,
            ctypes.c_void_p(self._region_iface), ctypes.c_uint32(self._cver),
            ctypes.c_uint32(0), None,
        )
        if not region:
            return
        wl.wl_proxy_set_queue(ctypes.c_void_p(region), ctypes.c_void_p(self._queue))
        for x, y, w, h in rows:
            wl.wl_proxy_marshal_flags(
                ctypes.c_void_p(region), WL_REGION_ADD,
                None, ctypes.c_uint32(self._cver), ctypes.c_uint32(0),
                ctypes.c_int32(x), ctypes.c_int32(y),
                ctypes.c_int32(w), ctypes.c_int32(h),
            )
        wl.wl_proxy_marshal_flags(
            ctypes.c_void_p(self._effect_surface), SURFACE_SET_BLUR_REGION,
            None, ctypes.c_uint32(self._version), ctypes.c_uint32(0),
            ctypes.c_void_p(region),
        )
        wl.wl_display_flush(ctypes.c_void_p(self._display))
        self._retired.append(self._region)
        self._region = region


def rounded_rect_rows(width: int, height: int, radius: float,
                      inset: int = 0) -> list[tuple[int, int, int, int]]:
    """Rows tracing a rounded rectangle, for an ordinary window's blur region.

    Same idea as pill_rects but with a fixed corner radius rather than fully
    rounded ends: outside the corner zone each row spans the full width,
    inside it the edge follows a quarter circle.
    """
    import math

    width -= 2 * inset
    height -= 2 * inset
    if width <= 0 or height <= 0:
        return []
    radius = min(radius, width / 2.0, height / 2.0)
    rows = []
    for y in range(height):
        dy = min(y + 0.5, height - y - 0.5)   # distance to the nearest edge
        if dy >= radius:
            rows.append((inset, y + inset, width, 1))
            continue
        d = radius - dy
        dx = radius - math.sqrt(max(0.0, radius * radius - d * d))
        x0 = int(math.ceil(dx))
        w = width - 2 * x0
        if w > 0:
            rows.append((x0 + inset, y + inset, w, 1))
    return rows


def pill_rects(width: int, height: int, exponent: float,
               inset: int = 0, cap: float = 1.12) -> list[tuple[int, int, int, int]]:
    """Rows tracing the capsule the HUD draws, as wl_region only takes rects.

    Mirrors _squircle in hud_app: straight sides with superelliptical end caps
    reaching `cap` half-heights along the edge. If this drifts from what is
    drawn, the blur shows outside the pill.

    `inset` shrinks the shape on every side. A region is a hard-edged mask with
    no antialiasing, so its curved ends are a staircase; pulling it in behind
    the drawn outline keeps that staircase from showing.
    """
    import math

    width -= 2 * inset
    height -= 2 * inset
    if width <= 0 or height <= 0:
        return []
    b = height / 2.0
    rx = min(b * cap, width / 2.0)
    rects = []
    for y in range(height):
        dy = abs(y + 0.5 - b) / b
        if dy >= 1.0:
            continue
        f = (1.0 - dy ** exponent) ** (1.0 / exponent)
        x0 = int(math.ceil(rx * (1.0 - f)))
        w = width - 2 * x0
        if w > 0:
            rects.append((x0 + inset, y + inset, w, 1))
    return rects


def rows_translated(rows: list[tuple[int, int, int, int]],
                    dx: int, dy: int) -> list[tuple[int, int, int, int]]:
    """A shape's rows moved to another place on the surface."""
    return [(x + dx, y + dy, w, h) for x, y, w, h in rows]


def enable_blur(wl_display: int, wl_surface: int,
                width: int, height: int, exponent: float,
                inset: int = 0, active: bool = True) -> BlurHandle | None:
    """Ask the compositor to blur behind a pill-shaped surface."""
    return _enable_blur_impl(
        wl_display, wl_surface, pill_rects(width, height, exponent, inset),
        active)


def enable_blur_rows(wl_display: int, wl_surface: int,
                     rows: list[tuple[int, int, int, int]],
                     active: bool = True) -> BlurHandle | None:
    """Ask the compositor to blur behind rows we built.

    The pill's blur region is not always one capsule: the stop button hangs
    below it, and the two shapes blur as one region.
    """
    return _enable_blur_impl(wl_display, wl_surface, rows, active)


def enable_window_blur(wl_display: int, wl_surface: int,
                       width: int, height: int, radius: float,
                       inset: int = 0, active: bool = True) -> BlurHandle | None:
    """Ask the compositor to blur behind a rounded-rectangle window.

    Same contract as enable_blur, with a region matching the window's
    rounded corners instead of the HUD's pill. The window must not be
    resized afterwards - the region is in fixed surface-local coordinates.
    """
    return _enable_blur_impl(
        wl_display, wl_surface,
        rounded_rect_rows(width, height, radius, inset), active)


def _enable_blur_impl(wl_display: int, wl_surface: int,
                      rows: list[tuple[int, int, int, int]],
                      active: bool) -> BlurHandle | None:
    """Ask the compositor to blur behind wl_surface.

    A region must be supplied and it must be non-empty: the protocol starts
    with an empty blur region, and passing a NULL wl_region *removes* the
    effect rather than meaning "all of it" (that is the older org_kde_kwin_blur
    convention). Coordinates are surface-local.

    Args:
        wl_display: Pointer to the struct wl_display GTK is already using
        wl_surface: Pointer to this window's struct wl_surface
        rows: Rectangles (x, y, w, h) tracing the shape to blur

    Returns:
        A handle to retain, or None if the compositor lacks the protocol.

    """
    if not wl_display or not wl_surface or not rows:
        return None

    wl = _wl()
    proto = _proto()

    wl.wl_proxy_marshal_flags.restype = ctypes.c_void_p
    wl.wl_proxy_marshal_flags.argtypes = None  # variadic
    wl.wl_display_create_queue.restype = ctypes.c_void_p
    wl.wl_display_create_queue.argtypes = [ctypes.c_void_p]
    wl.wl_proxy_set_queue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    wl.wl_proxy_add_listener.restype = ctypes.c_int
    wl.wl_proxy_add_listener.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    wl.wl_display_roundtrip_queue.restype = ctypes.c_int
    wl.wl_display_roundtrip_queue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    wl.wl_proxy_get_version.restype = ctypes.c_uint32
    wl.wl_proxy_get_version.argtypes = [ctypes.c_void_p]

    queue = wl.wl_display_create_queue(ctypes.c_void_p(wl_display))
    if not queue:
        return None

    registry_iface = ctypes.addressof(WlInterface.in_dll(wl, "wl_registry_interface"))
    registry = wl.wl_proxy_marshal_flags(
        ctypes.c_void_p(wl_display), WL_DISPLAY_GET_REGISTRY,
        ctypes.c_void_p(registry_iface), ctypes.c_uint32(1), ctypes.c_uint32(0),
        None,
    )
    if not registry:
        return None
    wl.wl_proxy_set_queue(ctypes.c_void_p(registry), ctypes.c_void_p(queue))

    found = {}

    def on_global(data, reg, name, iface, version):
        if iface == MANAGER_NAME:
            found["name"] = name
            found["version"] = version
        elif iface == COMPOSITOR_NAME:
            found["compositor_name"] = name
            found["compositor_version"] = version

    def on_global_remove(data, reg, name):
        pass

    listener = _RegistryListener(_GLOBAL_CB(on_global), _GLOBAL_REMOVE_CB(on_global_remove))
    wl.wl_proxy_add_listener(
        ctypes.c_void_p(registry), ctypes.byref(listener), None,
    )
    wl.wl_display_roundtrip_queue(ctypes.c_void_p(wl_display), ctypes.c_void_p(queue))

    if "name" not in found or "compositor_name" not in found:
        return None

    version = min(1, found["version"])
    manager = wl.wl_proxy_marshal_flags(
        ctypes.c_void_p(registry), WL_REGISTRY_BIND,
        ctypes.c_void_p(ctypes.addressof(proto.manager_iface)),
        ctypes.c_uint32(version), ctypes.c_uint32(0),
        ctypes.c_uint32(found["name"]), MANAGER_NAME, ctypes.c_uint32(version),
        None,
    )
    if not manager:
        return None
    wl.wl_proxy_set_queue(ctypes.c_void_p(manager), ctypes.c_void_p(queue))

    effect = wl.wl_proxy_marshal_flags(
        ctypes.c_void_p(manager), MANAGER_GET_BACKGROUND_EFFECT,
        ctypes.c_void_p(ctypes.addressof(proto.surface_iface)),
        ctypes.c_uint32(version), ctypes.c_uint32(0),
        None, ctypes.c_void_p(wl_surface),
    )
    if not effect:
        return None
    wl.wl_proxy_set_queue(ctypes.c_void_p(effect), ctypes.c_void_p(queue))

    # Build the region to blur. This is mandatory: the protocol's initial blur
    # region is empty, so without one nothing is ever blurred.
    compositor_iface = ctypes.addressof(WlInterface.in_dll(wl, "wl_compositor_interface"))
    cver = min(4, found["compositor_version"])
    compositor = wl.wl_proxy_marshal_flags(
        ctypes.c_void_p(registry), WL_REGISTRY_BIND,
        ctypes.c_void_p(compositor_iface), ctypes.c_uint32(cver), ctypes.c_uint32(0),
        ctypes.c_uint32(found["compositor_name"]), COMPOSITOR_NAME,
        ctypes.c_uint32(cver), None,
    )
    if not compositor:
        return None
    wl.wl_proxy_set_queue(ctypes.c_void_p(compositor), ctypes.c_void_p(queue))

    region_iface = ctypes.addressof(WlInterface.in_dll(wl, "wl_region_interface"))
    region = wl.wl_proxy_marshal_flags(
        ctypes.c_void_p(compositor), WL_COMPOSITOR_CREATE_REGION,
        ctypes.c_void_p(region_iface), ctypes.c_uint32(cver), ctypes.c_uint32(0),
        None,
    )
    if not region:
        return None
    wl.wl_proxy_set_queue(ctypes.c_void_p(region), ctypes.c_void_p(queue))

    empty_region = wl.wl_proxy_marshal_flags(
        ctypes.c_void_p(compositor), WL_COMPOSITOR_CREATE_REGION,
        ctypes.c_void_p(region_iface), ctypes.c_uint32(cver), ctypes.c_uint32(0),
        None,
    )
    if empty_region:
        wl.wl_proxy_set_queue(ctypes.c_void_p(empty_region), ctypes.c_void_p(queue))

    for x, y, w, h in rows:
        wl.wl_proxy_marshal_flags(
            ctypes.c_void_p(region), WL_REGION_ADD,
            None, ctypes.c_uint32(cver), ctypes.c_uint32(0),
            ctypes.c_int32(x), ctypes.c_int32(y),
            ctypes.c_int32(w), ctypes.c_int32(h),
        )

    wl.wl_proxy_marshal_flags(
        ctypes.c_void_p(effect), SURFACE_SET_BLUR_REGION,
        None, ctypes.c_uint32(version), ctypes.c_uint32(0),
        ctypes.c_void_p(region),
    )
    wl.wl_display_flush(ctypes.c_void_p(wl_display))

    handle = BlurHandle(manager, effect, queue, wl_display, region,
                        empty_region, version, compositor, region_iface, cver)
    handle.active = True
    handle._listener = listener  # the callbacks must outlive the registry
    if not active:
        handle.set_active(False)
    return handle
