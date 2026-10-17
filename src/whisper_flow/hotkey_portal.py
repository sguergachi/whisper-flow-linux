"""Wayland global shortcuts through the XDG GlobalShortcuts portal.

The fail-open evdev observer deliberately lets the physical hotkey reach the
compositor and the focused app, so injecting text while a modifier-only chord
such as the default ``super+alt`` is held would turn its letters into desktop
shortcuts. The portal is the Wayland-native answer to that: shortcuts bound
through it are intercepted by the compositor (KWin registers them through
KGlobalAccel) and never delivered to applications. Holding the chord is then
invisible to the focused field, live words stream cleanly, the release edge
arrives as a ``Deactivated`` signal, and the transcript pastes after release
with focus exactly where it started.

No input-device access is needed at all, so this backend also works without
the ``input`` group or udev rules the evdev backends require.

When the portal is unavailable, denied, or unusable, the manager falls back
to the fail-open observer (buffered final paste, no live preview). The portal
backend therefore never leaves the session without hotkeys by itself.
"""

from __future__ import annotations

import itertools
import logging
import os
import queue
import threading
import time

_log = logging.getLogger(__name__)

PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
PORTAL_IFACE = "org.freedesktop.portal.GlobalShortcuts"
REQUEST_IFACE = "org.freedesktop.portal.Request"
SESSION_IFACE = "org.freedesktop.portal.Session"
DBUS_BUS = "org.freedesktop.DBus"
DBUS_PATH = "/org/freedesktop/DBus"
DBUS_IFACE = "org.freedesktop.DBus"

RESPONSE_SUCCESS = 0
RESPONSE_CANCELLED = 1

CREATE_TIMEOUT_S = 10.0
BIND_TIMEOUT_S = 300.0
PENDING_NOTICE_S = 4.0


class PortalUnavailable(Exception):
    """The portal cannot be used here (fast, synchronous signal)."""


class PortalBindError(Exception):
    """Session creation or shortcut binding failed (async signal)."""


# ---------------------------------------------------------------------------
# trigger mapping (pure, no D-Bus)


_MOD_TO_XKB = {
    "ctrl": "CTRL", "control": "CTRL", "ctl": "CTRL",
    "alt": "ALT", "option": "ALT",
    "shift": "SHIFT",
    "super": "LOGO", "win": "LOGO", "meta": "LOGO",
    "cmd": "LOGO", "command": "LOGO", "windows": "LOGO",
}

_MOD_KEYSYM = {
    "CTRL": ("Control_L", "Control_R"),
    "ALT": ("Alt_L", "Alt_R"),
    "SHIFT": ("Shift_L", "Shift_R"),
    "LOGO": ("Super_L", "Super_R"),
}

_NAMED_KEYS = {
    "space": "space",
    "return": "Return", "enter": "Return",
    "tab": "Tab",
    "escape": "Escape", "esc": "Escape",
    "backspace": "BackSpace",
    "delete": "Delete",
    "insert": "Insert",
    "home": "Home", "end": "End",
    "pageup": "Page_Up", "pagedown": "Page_Down",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
}


def _split_side(part: str) -> tuple[str, str | None]:
    for prefix in ("left_", "right_"):
        if part.startswith(prefix):
            return part[len(prefix):], prefix
    return part, None


def config_to_trigger(hotkey: str) -> str | None:
    """Map a config chord (``super+alt``) to a shortcuts-spec trigger.

    Modifiers become CTRL/ALT/SHIFT/LOGO and the final key an XKB keysym, so
    ``super+alt`` becomes ``LOGO+Alt_L`` and ``ctrl+alt+space`` becomes
    ``CTRL+ALT+space``. Returns None when the chord cannot be expressed, in
    which case the caller binds the shortcut without a preferred trigger and
    the portal dialog lets the user pick one.
    """
    parts = [p.strip().lower() for p in (hotkey or "").split("+") if p.strip()]
    if not parts:
        return None
    *mod_parts, last = parts
    xkb_mods = []
    for mod in mod_parts:
        base, _side = _split_side(mod)
        xkb = _MOD_TO_XKB.get(base)
        if xkb is None:
            return None
        if xkb not in xkb_mods:
            xkb_mods.append(xkb)
    base, side = _split_side(last)
    if base in _MOD_TO_XKB:
        pair = _MOD_KEYSYM[_MOD_TO_XKB[base]]
        key = pair[1] if side == "right_" else pair[0]
    elif base in _NAMED_KEYS:
        key = _NAMED_KEYS[base]
    elif len(base) == 1 and (base.isalnum() or base == "_"):
        key = base
    elif base.startswith("f") and base[1:].isdigit() and 1 <= int(base[1:]) <= 24:
        key = base.upper()
    else:
        return None
    if not xkb_mods:
        return key
    if key in ("Control_L", "Control_R", "Alt_L", "Alt_R",
               "Shift_L", "Shift_R", "Super_L", "Super_R"):
        # A bare modifier as the trigger key is exactly the default-chord
        # case; keep it expressible and let the bind result prove whether
        # the compositor grabbed it.
        pass
    return "+".join(xkb_mods + [key])


# ---------------------------------------------------------------------------
# bind plan (pure, no D-Bus)


class PortalBindPlan:
    """The request/response choreography, without any transport.

    Kept free of Gio so the whole flow is unit-testable with literal dicts.
    """

    def __init__(self, bindings: list[tuple[str, str | None, str]]):
        """Bindings as (shortcut id, preferred trigger or None, description)."""
        self.bindings = list(bindings)

    def shortcut_arguments(self) -> list[tuple[str, dict]]:
        args = []
        for shortcut_id, trigger, description in self.bindings:
            entry: dict = {"description": description}
            if trigger:
                entry["preferred_trigger"] = trigger
            args.append((shortcut_id, entry))
        return args

    @staticmethod
    def create_response_ok(results: dict) -> str:
        session_handle = (results or {}).get("session_handle")
        if not session_handle:
            raise PortalBindError("portal CreateSession returned no session")
        return session_handle

    def bind_response_ok(self, response: int, results: dict) -> dict[str, str]:
        """Validate a BindShortcuts response.

        Returns shortcut id -> trigger description. All-or-nothing: any
        requested shortcut missing or left without a trigger fails the whole
        bind, so the manager falls back to one uniform backend instead of a
        mix where some hotkeys stream and others buffer.
        """
        if response != RESPONSE_SUCCESS:
            raise PortalBindError(
                f"portal BindShortcuts refused (response {response})")
        bound = {}
        for shortcut_id, info in ((results or {}).get("shortcuts") or []):
            trigger = (info or {}).get("trigger_description")
            if trigger:
                bound[shortcut_id] = trigger
        missing = [sid for sid, _t, _d in self.bindings if sid not in bound]
        if missing:
            raise PortalBindError(
                "portal left shortcuts unbound: " + ", ".join(missing))
        return bound


# ---------------------------------------------------------------------------
# transport abstraction (Gio in production, fakes in tests)


_token_counter = itertools.count(1)


def make_token() -> str:
    return f"whisper_flow_{os.getpid()}_{next(_token_counter)}"


class RealPortalIO:
    """Gio-based transport. All methods run on the listener worker thread."""

    def __init__(self):
        from gi.repository import Gio, GLib  # noqa: PLC0415
        self._Gio = Gio
        self._GLib = GLib
        self._loop = None
        self._tick_id = None

    # -- fast synchronous pre-check (caller thread) -------------------------
    def quick_check(self) -> None:
        """Raise PortalUnavailable fast when there is no portal to talk to."""
        try:
            from gi.repository import Gio  # noqa: PLC0415
            conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        except Exception as exc:
            raise PortalUnavailable(f"no D-Bus session bus: {exc}")
        try:
            (owned,) = conn.call_sync(
                DBUS_BUS, DBUS_PATH, DBUS_IFACE, "NameHasOwner",
                self._params("s", PORTAL_BUS), self._reply("b"),
                Gio.DBusCallFlags.NONE, 3000, None).unpack()
        except Exception as exc:
            raise PortalUnavailable(f"portal bus check failed: {exc}")
        if not owned:
            raise PortalUnavailable("GlobalShortcuts portal is not running")

    @staticmethod
    def _params(fmt, *args):
        from gi.repository import GLib  # noqa: PLC0415
        return GLib.Variant(f"({fmt})", args)

    # -- worker-thread transport ---------------------------------------------
    def connect(self):
        return self._Gio.bus_get_sync(self._Gio.BusType.SESSION, None)

    def unique_suffix(self, conn) -> str:
        return conn.get_unique_name().lstrip(":").replace(".", "_")

    def request_path(self, suffix: str, token: str) -> str:
        return f"/org/freedesktop/portal/desktop/request/{suffix}/{token}"

    def subscribe_response(self, conn, path, callback):
        return conn.signal_subscribe(
            None, REQUEST_IFACE, "Response", path, None,
            self._Gio.DBusSignalFlags.NONE, callback, None)

    def subscribe_signal(self, conn, member, arg0, callback):
        return conn.signal_subscribe(
            None, PORTAL_IFACE, member, PORTAL_PATH, arg0,
            self._Gio.DBusSignalFlags.NONE, callback, None)

    def unsubscribe(self, conn, sub_id):
        try:
            conn.signal_unsubscribe(sub_id)
        except Exception:
            pass

    def call_create_session(self, conn, request_path, handle_token,
                            session_token):
        options = {"handle_token": self._var("s", handle_token),
                   "session_handle_token": self._var("s", session_token)}
        return conn.call_sync(
            PORTAL_BUS, PORTAL_PATH, PORTAL_IFACE, "CreateSession",
            self._GLib.Variant("(a{sv})", (options,)),
            self._GLib.VariantType.new("(o)"),
            self._Gio.DBusCallFlags.NONE, 10000, None)

    def call_bind_shortcuts(self, conn, session_handle,
                            shortcuts, request_token):
        args = []
        for shortcut_id, entry in shortcuts:
            vardict = {k: self._var("s", v) for k, v in entry.items()}
            args.append((shortcut_id, vardict))
        options = {"handle_token": self._var("s", request_token)}
        return conn.call_sync(
            PORTAL_BUS, PORTAL_PATH, PORTAL_IFACE, "BindShortcuts",
            self._GLib.Variant("(oa(sa{sv})sa{sv})", (
                session_handle, args, "", options)),
            self._GLib.VariantType.new("(o)"),
            self._Gio.DBusCallFlags.NONE, 10000, None)

    def call_session_close(self, conn, session_handle):
        try:
            conn.call_sync(
                PORTAL_BUS, session_handle, SESSION_IFACE, "Close", None, None,
                self._Gio.DBusCallFlags.NONE, 2000, None)
        except Exception:
            pass

    def _var(self, fmt, value):
        return self._GLib.Variant(fmt, value)

    def run_loop(self):
        self._loop = self._GLib.MainLoop()
        self._loop.run()

    def quit_loop(self):
        loop, self._loop = self._loop, None
        if loop is not None:
            try:
                loop.quit()
            except Exception:
                pass

    def tick_each_second(self, callback):
        self._tick_id = self._GLib.timeout_add_seconds(1, callback)

    def untick(self):
        tick_id, self._tick_id = self._tick_id, None
        if tick_id is not None:
            try:
                self._GLib.source_remove(tick_id)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# listener (observer-compatible interface)


class PortalHotkeyListener:
    """Global-shortcut hotkeys with compositor-consumed triggers.

    Interface-compatible with the evdev backends (register/start/stop,
    is_alive, status_snapshot, escape_callback) so HotkeyManager can select
    it without branching.
    """

    backend_name = "portal"

    def __init__(self, on_bind_complete=None, io_factory=None,
                 bind_timeout: float = BIND_TIMEOUT_S):
        self._bindings: dict = {}
        self._callbacks: "queue.Queue" = queue.Queue()
        self._running = False
        self._bound = False
        self._bind_error: str | None = None
        self._bound_triggers: dict[str, str] = {}
        self._pending_notice_sent = False
        self._started_at = 0.0
        self._worker = None
        self._dispatch_thread = None
        self._on_bind_complete = on_bind_complete
        self._io_factory = io_factory or RealPortalIO
        self._bind_timeout = bind_timeout
        self._lock = threading.Lock()
        self.escape_callback = None
        self.on_emergency = None
        # Compositor-consumed shortcuts never reach applications: injection
        # needs no modifier cleanup and no Alt-menu repair afterwards.
        self.shortcuts_consumed = True

    # -- registration --------------------------------------------------------
    def register_hotkey(self, name, key_string, callback_press,
                        callback_release=None, release_modifiers=False):
        del release_modifiers
        trigger = config_to_trigger(key_string)
        self._bindings[name] = (trigger, callback_press, callback_release)

    # -- lifecycle ------------------------------------------------------------
    def start(self):
        if self._running:
            return
        if not self._bindings:
            raise PortalUnavailable("no shortcuts registered")
        io = self._io_factory()
        quick_check = getattr(io, "quick_check", None)
        if callable(quick_check):
            try:
                quick_check()
            except PortalUnavailable:
                raise
            except Exception as exc:
                raise PortalUnavailable(f"portal pre-check failed: {exc}")
        self._running = True
        self._started_at = time.monotonic()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True,
            name="whisper-flow-portal-dispatch")
        self._dispatch_thread.start()
        self._worker = threading.Thread(
            target=self._bind_loop, args=(io,), daemon=True,
            name="whisper-flow-portal-bind")
        self._worker.start()

    def stop(self):
        self._running = False
        io = getattr(self, "_io", None)
        session = getattr(self, "_session_handle", None)
        conn = getattr(self, "_conn", None)
        if conn is not None and session:
            try:
                io.call_session_close(conn, session)
            except Exception:
                pass
        try:
            if io is not None:
                io.quit_loop()
        except Exception:
            pass
        worker, self._worker = self._worker, None
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3)
        self._close_dispatch()
        self._conn = None

    # -- queries ---------------------------------------------------------------
    def is_alive(self):
        # Pending binds are alive-but-not-ready: neither heal loop may
        # restart (and re-dialog) them. Only a failed bind reads as dead,
        # and the manager answers that with the observer fallback, once.
        if self._bound:
            worker = self._worker
            return bool(self._running and worker is not None
                        and worker.is_alive())
        if self._bind_error is not None or not self._running:
            return False
        worker = self._worker
        return bool(worker is not None and worker.is_alive())

    @property
    def bind_failed(self) -> bool:
        return self._bind_error is not None

    @property
    def is_bound(self) -> bool:
        with self._lock:
            return self._bound

    @property
    def bind_pending(self) -> bool:
        return self._running and not self._bound and self._bind_error is None

    def pump_age_seconds(self):
        return None

    def sweep_unheld_modifiers(self) -> int:
        return 0

    def maybe_sweep_unheld_modifiers(self) -> int:
        return 0

    def status_snapshot(self) -> dict:
        return {
            "backend": "portal",
            "alive": self.is_alive(),
            "bound": len(self._bound_triggers),
            "pending": self.bind_pending,
            "failed": self._bind_error,
        }

    # -- worker -----------------------------------------------------------------
    def _bind_loop(self, io):
        plan = PortalBindPlan([
            (name, trigger, self._describe(name))
            for name, (trigger, _p, _r) in self._bindings.items()
        ])
        try:
            conn = io.connect()
        except Exception as exc:
            self._fail(f"portal connect failed: {exc}")
            return
        self._io = io
        self._conn = conn
        suffix = io.unique_suffix(conn)
        create_token = make_token()
        bind_token = make_token()
        session_token = make_token()
        create_path = io.request_path(suffix, create_token)
        bind_path = io.request_path(suffix, bind_token)
        state = {"create_path": create_path, "bind_path": bind_path,
                 "session": None, "done": False}

        def on_response(conn, _sender, _path, _iface, _signal, params, _data):
            try:
                response, results = params.unpack()
            except Exception as exc:
                self._fail(f"portal response unreadable: {exc}")
                return
            self._on_response(io, plan, state, _path, response, results or {})

        def on_shortcut(conn, _sender, _path, _iface, signal, params, _data):
            try:
                session, shortcut_id, _ts, _opts = params.unpack()
            except Exception:
                return
            if session != state["session"]:
                return
            self._on_shortcut(signal, shortcut_id)

        subs = []
        try:
            subs.append(io.subscribe_response(conn, create_path, on_response))
            subs.append(io.subscribe_response(conn, bind_path, on_response))
            subs.append(io.subscribe_signal(conn, "Activated", None, on_shortcut))
            subs.append(io.subscribe_signal(conn, "Deactivated", None,
                                            on_shortcut))
            io.tick_each_second(lambda: self._tick(io, state, plan))
            io.call_create_session(conn, create_path, create_token,
                                   session_token)
        except Exception as exc:
            for sub in subs:
                try:
                    io.unsubscribe(conn, sub)
                except Exception:
                    pass
            self._fail(f"portal session creation failed: {exc}")
            return
        try:
            io.run_loop()
        finally:
            try:
                io.untick()
            except Exception:
                pass
            for sub in subs:
                try:
                    io.unsubscribe(conn, sub)
                except Exception:
                    pass

    def _describe(self, name: str) -> str:
        return {
            "transcribe": "Push-to-talk transcription (hold to dictate)",
            "auto_transcribe": "Auto-stop transcription (speak, pause to finish)",
            "command": "Command dictation (speak, pause to finish)",
        }.get(name, name)

    def _on_response(self, io, plan, state, path, response, results):
        if state["done"]:
            return
        if path == state["create_path"]:
            if state["session"] is not None:
                return
            try:
                session = plan.create_response_ok(results)
            except PortalBindError as exc:
                self._fail(str(exc))
                return
            state["session"] = session
            self._session_handle = session
            try:
                io.call_bind_shortcuts(
                    self._conn, session, plan.shortcut_arguments(), make_token())
            except Exception as exc:
                self._fail(f"portal bind call failed: {exc}")
        elif path == state["bind_path"]:
            try:
                bound = plan.bind_response_ok(response, results)
            except PortalBindError as exc:
                self._fail(str(exc))
                return
            state["done"] = True
            with self._lock:
                self._bound = True
                self._bound_triggers = dict(bound)
            _log.info(f"[HOTKEY] portal shortcuts bound: {sorted(bound)}")
            self._report(True, f"bound {len(bound)} shortcut(s)")

    def _on_shortcut(self, signal: str, shortcut_id: str):
        binding = self._bindings.get(shortcut_id)
        if not binding:
            return
        _trigger, press_cb, release_cb = binding
        callback = press_cb if signal == "Activated" else release_cb
        kind = "press" if signal == "Activated" else "release"
        if callback is None:
            return
        if not self._bound:
            return
        self._callbacks.put((shortcut_id, kind, callback))

    def _tick(self, io, state, plan) -> bool:
        # True keeps the per-second tick alive; False stops it.
        if not self._running or state["done"] or self._bind_error:
            return False
        elapsed = time.monotonic() - self._started_at
        if not self._pending_notice_sent and elapsed >= PENDING_NOTICE_S:
            self._pending_notice_sent = True
            self._report_pending()
        if elapsed >= self._bind_timeout:
            self._fail(f"portal bind timed out after {self._bind_timeout:.0f}s")
            return False
        return True

    def _fail(self, reason: str):
        with self._lock:
            if self._bound or self._bind_error is not None:
                return
            self._bind_error = reason
        _log.info(f"[HOTKEY] portal backend failed: {reason}")
        try:
            io = getattr(self, "_io", None)
            if io is not None:
                io.quit_loop()
        except Exception:
            pass
        self._report(False, reason)

    def _report(self, ok: bool, detail: str):
        callback = self._on_bind_complete
        if callback is None:
            return
        try:
            callback(ok, detail)
        except Exception:
            _log.exception("portal bind completion callback failed")

    def _report_pending(self):
        # Reported through the same channel with ok=None: still trying, and
        # the user may need to approve a dialog.
        callback = self._on_bind_complete
        if callback is None:
            return
        try:
            callback(None, "waiting for approval")
        except Exception:
            _log.exception("portal pending callback failed")

    def _dispatch_loop(self):
        while True:
            item = self._callbacks.get()
            if item is None:
                return
            name, kind, callback = item
            try:
                callback()
            except Exception:
                _log.exception("hotkey %s %s callback failed", name, kind)

    def _close_dispatch(self):
        if self._dispatch_thread is not None:
            try:
                self._callbacks.put(None)
                self._dispatch_thread.join(timeout=2)
            except Exception:
                pass
            self._dispatch_thread = None
