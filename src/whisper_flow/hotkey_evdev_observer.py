"""Fail-open Wayland hotkey observation.

This module has deliberately one-way access to input: it reads hardware event
devices alongside the compositor and has no uinput device, no ``EVIOCGRAB``,
and no event-forwarding path. If this process stalls or crashes, KDE keeps
receiving the physical keyboard directly.
"""

from __future__ import annotations

import logging
import os
import queue
import select
import threading
import time

import evdev
from evdev import ecodes

from .logging import log as app_log

NAME_TO_CODE = {
    "super": ecodes.KEY_LEFTMETA, "cmd": ecodes.KEY_LEFTMETA,
    "win": ecodes.KEY_LEFTMETA, "meta": ecodes.KEY_LEFTMETA,
    "ctrl": ecodes.KEY_LEFTCTRL, "control": ecodes.KEY_LEFTCTRL,
    "alt": ecodes.KEY_LEFTALT,
    "shift": ecodes.KEY_LEFTSHIFT,
    "space": ecodes.KEY_SPACE,
}

CODE_ALIASES = {
    ecodes.KEY_RIGHTMETA: ecodes.KEY_LEFTMETA,
    ecodes.KEY_RIGHTCTRL: ecodes.KEY_LEFTCTRL,
    ecodes.KEY_RIGHTALT: ecodes.KEY_LEFTALT,
    ecodes.KEY_RIGHTSHIFT: ecodes.KEY_LEFTSHIFT,
}

BUS_VIRTUAL = 0x06
RESCAN_SECONDS = 3.0
RECONCILE_SECONDS = 1.0
ESC_RESET_COUNT = 3
ESC_RESET_WINDOW = 1.2
DEBUG_KEYS = os.environ.get("WHISPER_FLOW_HOTKEY_DEBUG") == "1"

log = logging.getLogger(__name__)


class EvdevHotkeyObserver:
    """Observe global hotkeys without owning or modifying keyboard delivery."""

    def __init__(self):
        self._kbd_devices = []
        self._bindings = {}
        self._key_state: set[int] = set()
        self._press_triggered: set[str] = set()
        self._active_hotkey = None
        self._running = False
        self._thread = None
        self._dispatch_thread = None
        self._callbacks = queue.Queue()
        self._last_pump_at = 0.0
        self._esc_press_times: list[float] = []
        self._open_failures = []
        self.escape_callback = None
        self.on_emergency = None

    def register_hotkey(self, name, key_string, callback_press,
                        callback_release=None, release_modifiers=False):
        # Retained for interface compatibility only. Observer mode never
        # synthesises input in either direction.
        del release_modifiers
        self._bindings[name] = (
            self._parse_key_string(key_string),
            callback_press,
            callback_release,
        )

    @property
    def _binding_codes(self) -> set[int]:
        codes = set()
        for keys, *_ in self._bindings.values():
            codes |= set(keys)
        return codes

    @staticmethod
    def _parse_key_string(key_string):
        codes = set()
        for part in [item.strip().lower() for item in key_string.split("+")]:
            code = NAME_TO_CODE.get(
                part, getattr(ecodes, f"KEY_{part.upper()}", None))
            if code:
                codes.add(code)
        return frozenset(codes)

    def _find_keyboard_devices(self):
        devices = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                # Never observe virtual injectors such as ydotool: dictated
                # text must not feed back into the hotkey matcher.
                if dev.info.bustype == BUS_VIRTUAL:
                    dev.close()
                    continue
                caps = dev.capabilities()
                if ecodes.EV_KEY in caps and len(caps[ecodes.EV_KEY]) >= 80:
                    devices.append(path)
                dev.close()
            except OSError:
                continue
        return devices

    def _open_devices(self) -> bool:
        """Open parallel readers. This function must never call ``grab``."""
        self._close_devices()
        opened = []
        failures = []
        for path in self._find_keyboard_devices():
            try:
                opened.append(evdev.InputDevice(path))
            except OSError as exc:
                failures.append((path, exc))
        self._kbd_devices = opened
        self._open_failures = failures
        if opened:
            app_log(f"[HOTKEY] observing {len(opened)} keyboard(s), "
                    "exclusive grabs disabled")
        elif failures:
            for path, exc in failures:
                app_log(f"[HOTKEY] could not observe {path}: {exc}")
        return bool(opened)

    def start(self):
        if self._running:
            return
        if not self._open_devices():
            raise RuntimeError("Cannot open any keyboard devices for observation")
        self._running = True
        self._last_pump_at = time.monotonic()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True,
            name="whisper-flow-hotkey-dispatch",
        )
        self._dispatch_thread.start()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True,
            name="whisper-flow-hotkey-observer",
        )
        self._thread.start()

    def _dispatch_loop(self):
        while True:
            item = self._callbacks.get()
            if item is None:
                return
            name, kind, callback = item
            try:
                callback()
            except Exception:
                log.exception("hotkey %s %s callback failed", name, kind)

    def _read_loop(self):
        try:
            while self._running:
                if not self._kbd_devices and not self._open_devices():
                    time.sleep(RESCAN_SECONDS)
                    continue
                self._pump_current_devices()
                self._close_devices()
                self._reset_callback_state()
        except Exception:
            # A dead observer can disable hotkeys, but it cannot disable the
            # keyboard. The manager heartbeat will replace this thread.
            log.exception("keyboard observer stopped")
        finally:
            self._close_devices()

    def _pump_current_devices(self):
        fds = {dev.fd: dev for dev in self._kbd_devices}
        next_scan = time.monotonic() + RESCAN_SECONDS
        next_reconcile = time.monotonic() + RECONCILE_SECONDS
        while self._running:
            self._last_pump_at = time.monotonic()
            try:
                readable, _, _ = select.select(list(fds), [], [], 0.1)
            except (OSError, ValueError):
                return
            for fd in readable:
                try:
                    for event in fds[fd].read():
                        if event.type == ecodes.EV_KEY:
                            self._handle_key(event)
                except BlockingIOError:
                    pass
                except OSError:
                    return
                except Exception:
                    log.exception("error reading keyboard observer")

            now = time.monotonic()
            if now >= next_reconcile:
                next_reconcile = now + RECONCILE_SECONDS
                self._reconcile_with_kernel()
            if now >= next_scan:
                next_scan = now + RESCAN_SECONDS
                if {dev.path for dev in self._kbd_devices} != set(
                        self._find_keyboard_devices()):
                    return

    def _kernel_held_keys(self) -> set[int] | None:
        if not self._kbd_devices:
            return None
        merged: set[int] = set()
        any_ok = False
        for dev in self._kbd_devices:
            try:
                for code in dev.active_keys():
                    merged.add(CODE_ALIASES.get(code, code))
                any_ok = True
            except OSError:
                continue
        return merged if any_ok else None

    def _sync_key_state_from_devices(self) -> bool:
        held = self._kernel_held_keys()
        if held is None:
            return False
        self._key_state = held
        return True

    def _reconcile_with_kernel(self) -> int:
        held = self._kernel_held_keys()
        if held is None or held == self._key_state:
            return 0
        old = set(self._key_state)
        self._key_state = set(held)
        self._check_bindings(rising=False)
        repaired = len(old - held)
        if repaired:
            app_log(f"[HOTKEY] reconciled {repaired} missed key release(s) "
                    "inside observer state")
        return repaired

    def _handle_key(self, event):
        code = CODE_ALIASES.get(event.code, event.code)
        value = event.value
        if value == 2:
            return
        if value == 1:
            if not self._sync_key_state_from_devices():
                self._key_state.add(code)
            if code == ecodes.KEY_ESC:
                self._track_escape()
                if self.escape_callback:
                    self._callbacks.put(("escape", "press", self.escape_callback))
            else:
                self._esc_press_times.clear()
            if DEBUG_KEYS and code in self._binding_codes:
                app_log(f"[HOTKEY-DEBUG] hotkey key down code={code} "
                        f"state={sorted(self._key_state)}")
            self._check_bindings(rising=True)
            return

        if not self._sync_key_state_from_devices():
            self._key_state.discard(code)
        self._check_bindings(rising=False)

    def _track_escape(self):
        now = time.monotonic()
        self._esc_press_times = [
            stamp for stamp in self._esc_press_times
            if now - stamp <= ESC_RESET_WINDOW
        ]
        self._esc_press_times.append(now)
        if len(self._esc_press_times) >= ESC_RESET_COUNT:
            self._esc_press_times.clear()
            self._reset_callback_state()
            app_log("[HOTKEY] triple-Esc reset observer state; physical "
                    "keyboard was never intercepted")

    def _check_bindings(self, rising: bool):
        extras = self._key_state - self._binding_codes
        winner = None
        if not extras:
            satisfied = [
                (name, keys)
                for name, (keys, *_rest) in self._bindings.items()
                if keys and keys.issubset(self._key_state)
            ]
            if satisfied:
                winner = max(satisfied, key=lambda item: len(item[1]))[0]

        for name, (_keys, press_cb, release_cb) in self._bindings.items():
            if name == winner:
                if rising and name not in self._press_triggered:
                    self._press_triggered.add(name)
                    self._active_hotkey = name
                    if press_cb:
                        self._callbacks.put((name, "press", press_cb))
            elif name in self._press_triggered:
                self._press_triggered.discard(name)
                if self._active_hotkey == name:
                    self._active_hotkey = None
                if release_cb:
                    self._callbacks.put((name, "release", release_cb))

    def _reset_callback_state(self):
        for name in list(self._press_triggered):
            binding = self._bindings.get(name)
            self._press_triggered.discard(name)
            if binding and binding[2]:
                self._callbacks.put((name, "release", binding[2]))
        self._active_hotkey = None
        self._key_state.clear()

    def _close_devices(self):
        for dev in self._kbd_devices:
            try:
                dev.close()
            except OSError:
                pass
        self._kbd_devices.clear()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        self._close_devices()
        self._reset_callback_state()
        if self._dispatch_thread:
            self._callbacks.put(None)
            self._dispatch_thread.join(timeout=2)
            self._dispatch_thread = None

    def is_alive(self):
        return bool(self._running and self._thread and self._thread.is_alive())

    def pump_age_seconds(self) -> float | None:
        if not self._last_pump_at:
            return None
        return time.monotonic() - self._last_pump_at

    def sweep_unheld_modifiers(self) -> int:
        return 0

    def maybe_sweep_unheld_modifiers(self) -> int:
        return 0

    def status_snapshot(self) -> dict:
        age = self.pump_age_seconds()
        return {
            "backend": "evdev-observer",
            "alive": self.is_alive(),
            "observed": len(self._kbd_devices),
            "grabbed": 0,
            "pump_age_s": round(age, 1) if age is not None else None,
            "muted": 0,
            "held": len(self._key_state),
            "forwarded": 0,
            "disabled": False,
            "recoveries_60s": 0,
            "healed_60s": 0,
        }
