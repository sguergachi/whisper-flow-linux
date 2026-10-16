"""Surface-independent pointer deltas for dragging a Wayland layer surface.

GTK exposes surface-local motion, which changes when the surface itself moves.
The relative-pointer protocol supplies accelerated deltas in the same logical
units as wl_pointer.motion, without that feedback. Our own proxies use GTK's
default event queue; we neither dispatch GTK's connection nor replace any of
its listeners. No pointer lock, global cursor access, or round-trip is needed.
"""

import ctypes
import ctypes.util


class _Message(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("signature", ctypes.c_char_p),
                ("types", ctypes.POINTER(ctypes.c_void_p))]


class _Interface(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("version", ctypes.c_int),
                ("method_count", ctypes.c_int), ("methods", ctypes.POINTER(_Message)),
                ("event_count", ctypes.c_int), ("events", ctypes.POINTER(_Message))]


_Global = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32)
_Remove = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
_Motion = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_uint32, ctypes.c_uint32,
                          ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)


class _RegistryListener(ctypes.Structure):
    _fields_ = [("global_", _Global), ("remove", _Remove)]


class _MotionListener(ctypes.Structure):
    _fields_ = [("motion", _Motion)]


class RelativePointer:
    """Own the protocol descriptions, proxies and callbacks until close()."""

    def __init__(self, display, pointer, on_motion):
        if not display or not pointer:
            raise ValueError("Wayland display and pointer are required")
        self._wl = ctypes.CDLL(ctypes.util.find_library("wayland-client")
                               or "libwayland-client.so.0")
        wl = self._wl
        wl.wl_proxy_marshal_flags.restype = ctypes.c_void_p
        wl.wl_proxy_marshal_flags.argtypes = None  # variadic
        wl.wl_proxy_add_listener.argtypes = [ctypes.c_void_p] * 3
        wl.wl_proxy_add_listener.restype = ctypes.c_int
        wl.wl_proxy_destroy.argtypes = [ctypes.c_void_p]
        self._manager = self._relative = self._registry = None
        self._on_motion = on_motion
        self._pointer = pointer
        self._destroy = (_Message * 1)(_Message(b"destroy", b"", None))
        self._events = (_Message * 1)(_Message(b"relative_motion", b"uuffff", None))
        self._relative_iface = _Interface(b"zwp_relative_pointer_v1", 1,
                                         1, self._destroy, 1, self._events)
        self._types = (ctypes.c_void_p * 2)(
            ctypes.addressof(self._relative_iface),
            ctypes.addressof(_Interface.in_dll(wl, "wl_pointer_interface")))
        self._methods = (_Message * 2)(
            _Message(b"destroy", b"", None),
            _Message(b"get_relative_pointer", b"no", self._types))
        self._manager_iface = _Interface(b"zwp_relative_pointer_manager_v1", 1,
                                        2, self._methods, 0, None)
        self._registry_listener = _RegistryListener(
            _Global(self._global), _Remove(lambda *args: None))
        self._motion_listener = _MotionListener(_Motion(self._motion))
        iface = _Interface.in_dll(wl, "wl_registry_interface")
        self._registry = self._request(display, 1, iface, None)
        wl.wl_proxy_add_listener(ctypes.c_void_p(self._registry),
                                 ctypes.byref(self._registry_listener), None)

    @property
    def available(self):
        return self._relative is not None

    def _request(self, proxy, opcode, iface, *args):
        return self._wl.wl_proxy_marshal_flags(
            ctypes.c_void_p(proxy), ctypes.c_uint32(opcode),
            ctypes.byref(iface), ctypes.c_uint32(1), ctypes.c_uint32(0), *args)

    def _global(self, data, registry, name, interface, version):
        if interface != self._manager_iface.name or self._manager is not None:
            return
        self._manager = self._request(
            registry, 0, self._manager_iface, ctypes.c_uint32(name),
            interface, ctypes.c_uint32(1), None)
        self._relative = self._request(
            self._manager, 1, self._relative_iface, None, ctypes.c_void_p(self._pointer))
        self._wl.wl_proxy_add_listener(ctypes.c_void_p(self._relative),
                                      ctypes.byref(self._motion_listener), None)

    def _motion(self, data, proxy, hi, lo, dx, dy, raw_dx, raw_dy):
        # wl_fixed_t is signed 24.8. Use accelerated motion, as the cursor does.
        self._on_motion(dx / 256.0, dy / 256.0)

    def close(self):
        for attr in ("_relative", "_manager"):
            proxy = getattr(self, attr)
            if proxy:
                self._wl.wl_proxy_marshal_flags(
                    ctypes.c_void_p(proxy), ctypes.c_uint32(0), None,
                    ctypes.c_uint32(1), ctypes.c_uint32(1))  # destructor
                setattr(self, attr, None)
        if self._registry:
            self._wl.wl_proxy_destroy(ctypes.c_void_p(self._registry))
            self._registry = None
