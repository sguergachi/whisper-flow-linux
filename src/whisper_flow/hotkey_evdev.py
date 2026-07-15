"""Keyboard hotkey listener using evdev with uinput proxy for Wayland."""

import logging
import os
import select
import struct
import threading
import time
from collections.abc import Callable

import evdev
from evdev import ecodes

log = logging.getLogger(__name__)

NAME_TO_CODE = {
    "super": ecodes.KEY_LEFTMETA, "cmd": ecodes.KEY_LEFTMETA,
    "win": ecodes.KEY_LEFTMETA, "meta": ecodes.KEY_LEFTMETA,
    "ctrl": ecodes.KEY_LEFTCTRL, "control": ecodes.KEY_LEFTCTRL,
    "alt": ecodes.KEY_LEFTALT,
    "shift": ecodes.KEY_LEFTSHIFT,
    "space": ecodes.KEY_SPACE,
}


class EvdevHotkeyListener:
    """Reads keyboard events via evdev, forwarding all events through uinput
    so the compositor continues to receive keyboard input."""

    def __init__(self):
        self._kbd_devices = []
        self._uinput = None
        self._key_state = set()
        self._bindings = {}
        self._running = False
        self._thread = None
        self._active_hotkey = None
        self._press_triggered = set()  # hotkeys whose press callback was already fired

    def register_hotkey(self, name, key_string, callback_press, callback_release=None):
        keys = self._parse_key_string(key_string)
        self._bindings[name] = (keys, callback_press, callback_release)

    def _parse_key_string(self, key_string):
        codes = set()
        for p in [x.strip().lower() for x in key_string.split("+")]:
            code = NAME_TO_CODE.get(p, getattr(ecodes, f"KEY_{p.upper()}", None))
            if code:
                codes.add(code)
        return frozenset(codes)

    def _find_keyboard_devices(self):
        devices = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                caps = dev.capabilities()
                if ecodes.EV_KEY in caps and len(caps[ecodes.EV_KEY]) >= 80:
                    ecodes_by_type = {ecodes.EV_KEY: caps[ecodes.EV_KEY]}
                    devices.append((path, ecodes_by_type))
            except Exception:
                continue
        return devices

    def start(self):
        kbd_info = self._find_keyboard_devices()
        if not kbd_info:
            raise RuntimeError("No keyboard devices found. Are you in the 'input' group?")

        # Create uinput proxy from the first keyboard's capabilities
        try:
            self._uinput = evdev.UInput.from_device(
                kbd_info[0][0], name="whisper-flow-keyboard-proxy"
            )
        except Exception as e:
            # If UInput fails, try just key capabilities
            try:
                self._uinput = evdev.UInput(
                    events={ecodes.EV_KEY: kbd_info[0][1][ecodes.EV_KEY]},
                    name="whisper-flow-keyboard-proxy",
                )
            except Exception:
                raise RuntimeError(f"Cannot create uinput proxy: {e}") from e

        # Grab and open real keyboard devices
        self._kbd_devices = []
        for path, _ in kbd_info:
            try:
                dev = evdev.InputDevice(path)
                dev.grab()
                self._kbd_devices.append(dev)
            except Exception:
                continue

        if not self._kbd_devices:
            if self._uinput:
                self._uinput.close()
            raise RuntimeError("Cannot grab any keyboard devices")

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        fds = {dev.fd: dev for dev in self._kbd_devices}

        while self._running:
            r, _, _ = select.select(list(fds), [], [], 0.1)
            for fd in r:
                try:
                    for event in fds[fd].read():
                        if event.type == ecodes.EV_KEY:
                            self._handle_key(event)
                        # Forward event to uinput proxy (including EV_SYN for sync)
                        try:
                            self._uinput.write_event(event)
                            # Only emit SYN for the real SYN events, not after every event
                            if event.type == ecodes.EV_SYN:
                                self._uinput.syn()
                        except Exception:
                            pass
                except BlockingIOError:
                    pass
                except Exception:
                    pass

    def _handle_key(self, event):
        code, value = event.code, event.value
        if value == 1:
            self._key_state.add(code)
            self._check_bindings()
        elif value == 0:
            self._check_bindings()
            self._key_state.discard(code)

    def _check_bindings(self):
        for name, (keys, cb_press, cb_release) in self._bindings.items():
            if keys and keys.issubset(self._key_state):
                if name not in self._press_triggered:
                    self._press_triggered.add(name)
                    self._active_hotkey = name
                    if cb_press:
                        try:
                            cb_press()
                        except Exception:
                            pass
            else:
                if name in self._press_triggered:
                    self._press_triggered.discard(name)
                    if self._active_hotkey == name:
                        self._active_hotkey = None
                    if cb_release:
                        try:
                            cb_release()
                        except Exception:
                            pass

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        for dev in self._kbd_devices:
            try:
                dev.ungrab()
                dev.close()
            except Exception:
                pass
        self._kbd_devices.clear()
        if self._uinput:
            try:
                self._uinput.close()
            except Exception:
                pass
            self._uinput = None
