"""Keyboard hotkey listener using evdev with uinput proxy for Wayland."""

import logging
import queue
import select
import threading

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

# Both physical sides of a modifier map onto the same logical key.
CODE_ALIASES = {
    ecodes.KEY_RIGHTMETA: ecodes.KEY_LEFTMETA,
    ecodes.KEY_RIGHTCTRL: ecodes.KEY_LEFTCTRL,
    ecodes.KEY_RIGHTALT: ecodes.KEY_LEFTALT,
    ecodes.KEY_RIGHTSHIFT: ecodes.KEY_LEFTSHIFT,
}

PROXY_NAME = "whisper-flow-keyboard-proxy"
BUS_VIRTUAL = 0x06  # uinput devices; see linux/input.h BUS_VIRTUAL


class EvdevHotkeyListener:
    """Reads keyboard events via evdev, forwarding all events through uinput
    so the compositor continues to receive keyboard input.

    Callbacks never run on the reader thread: they are handed to a dispatch
    thread, because anything slow on the reader thread stalls event forwarding
    and freezes the user's keyboard.
    """

    def __init__(self):
        self._kbd_devices = []
        self._uinput = None
        self._key_state = set()
        self._bindings = {}
        self._running = False
        self._thread = None
        self._dispatch_thread = None
        self._callbacks = queue.Queue()
        self._active_hotkey = None
        self._press_triggered = set()  # hotkeys whose press callback was already fired
        self._swallowed = set()   # codes hidden from the compositor while held
        self._pending = []        # codes held back pending a combination
        self._participants = set()  # codes that could start a swallowing combo

    def register_hotkey(self, name, key_string, callback_press, callback_release=None,
                        swallow=False):
        """Register a binding.

        `swallow` hides the combination's own keys from the compositor while it
        is held. Push-to-talk needs this: the app types the transcription with
        ydotool while the user is still holding the hotkey, and if the
        compositor still sees those modifiers down, every injected character
        arrives as Super+Alt+<key> and nothing gets typed.
        """
        keys = self._parse_key_string(key_string)
        self._bindings[name] = (keys, callback_press, callback_release, swallow)
        if swallow:
            self._participants |= set(keys)

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
                # Only ever grab real hardware. Virtual devices are keystroke
                # injectors - our own proxy, and crucially ydotoold, which is
                # what this app types transcriptions with. Grabbing that pulls
                # every injected character back through here and re-emits it
                # under whatever modifiers the user is holding, so dictated
                # text arrives as hotkey combinations instead of text.
                if dev.info.bustype == BUS_VIRTUAL or dev.name == PROXY_NAME:
                    dev.close()
                    continue
                caps = dev.capabilities()
                if ecodes.EV_KEY in caps and len(caps[ecodes.EV_KEY]) >= 80:
                    ecodes_by_type = {ecodes.EV_KEY: caps[ecodes.EV_KEY]}
                    devices.append((path, ecodes_by_type))
                dev.close()
            except Exception:
                continue
        return devices

    def start(self):
        kbd_info = self._find_keyboard_devices()
        if not kbd_info:
            raise RuntimeError("No keyboard devices found. Are you in the 'input' group?")

        # Create uinput proxy from the first keyboard's capabilities
        try:
            self._uinput = evdev.UInput.from_device(kbd_info[0][0], name=PROXY_NAME)
        except Exception as e:
            # If UInput fails, try just key capabilities
            try:
                self._uinput = evdev.UInput(
                    events={ecodes.EV_KEY: kbd_info[0][1][ecodes.EV_KEY]},
                    name=PROXY_NAME,
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
                self._uinput = None
            raise RuntimeError("Cannot grab any keyboard devices")

        self._running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="whisper-flow-hotkey-dispatch",
        )
        self._dispatch_thread.start()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="whisper-flow-hotkey-reader",
        )
        self._thread.start()

    def is_alive(self):
        """True while the reader thread is actually pumping events."""
        return bool(self._running and self._thread and self._thread.is_alive())

    def _dispatch_loop(self):
        """Run hotkey callbacks off the reader thread."""
        while True:
            item = self._callbacks.get()
            if item is None:
                return
            name, kind, cb = item
            try:
                cb()
            except Exception:
                log.exception("hotkey %s %s callback failed", name, kind)

    def _read_loop(self):
        fds = {dev.fd: dev for dev in self._kbd_devices}
        try:
            while self._running:
                try:
                    r, _, _ = select.select(list(fds), [], [], 0.1)
                except (OSError, ValueError):
                    # A device disappeared (unplugged); stop cleanly so the
                    # supervisor can rebuild against the current device set.
                    return
                for fd in r:
                    try:
                        for event in fds[fd].read():
                            if event.type == ecodes.EV_KEY:
                                # _handle_key owns forwarding for key events:
                                # whether one reaches the compositor depends on
                                # state it is in the middle of updating.
                                self._handle_key(event)
                            else:
                                self._forward(event)
                    except BlockingIOError:
                        pass
                    except OSError:
                        return
                    except Exception:
                        log.exception("error reading keyboard events")
        finally:
            # Devices must never stay grabbed after this thread exits, or the
            # user loses their keyboard entirely.
            self._release_devices()

    def _forward(self, event):
        """Pass an event through to the compositor via the uinput proxy."""
        try:
            self._uinput.write_event(event)
            if event.type == ecodes.EV_SYN:
                self._uinput.syn()
        except Exception:
            pass

    def _emit(self, code, value):
        """Synthesise a key event on the proxy (used to hide/restore keys)."""
        try:
            self._uinput.write(ecodes.EV_KEY, code, value)
            self._uinput.syn()
        except Exception:
            pass

    def _flush_pending(self):
        """Deliver speculatively-held keys, in the order they were pressed."""
        for code in self._pending:
            self._emit(code, 1)
        self._pending.clear()

    def _begin_swallow(self, keys):
        """Hide a combination's keys for as long as they are held."""
        for code in keys:
            if code in self._pending:
                self._pending.remove(code)
            self._swallowed.add(code)

    def _handle_key(self, event):
        code = CODE_ALIASES.get(event.code, event.code)
        value = event.value

        if value == 2:
            # Auto-repeat. Never a state change: holding a push-to-talk combo
            # generates these continuously, and treating one as a release ends
            # the recording about half a second after it starts.
            if code not in self._swallowed and code not in self._pending:
                self._forward(event)
            return

        if value == 1:
            if code in self._swallowed:
                return
            self._key_state.add(code)

            if code in self._participants:
                # Hold it back rather than forwarding: if this turns out to be
                # the start of a push-to-talk combination, the compositor must
                # never have seen it down, or releasing it later reads as a
                # bare modifier tap and opens the launcher.
                self._pending.append(code)
                self._check_bindings(rising=True)
                return

            # An unrelated key: whatever was held back was a real modifier
            # after all, so deliver it first to keep combinations working.
            self._flush_pending()
            self._check_bindings(rising=True)
            self._forward(event)
            return

        # Release. Drop the key before re-evaluating, otherwise the
        # combination still looks held and the release callback never fires.
        self._key_state.discard(code)
        self._check_bindings(rising=False)

        if code in self._swallowed:
            self._swallowed.discard(code)
            return  # the compositor never saw it go down
        if code in self._pending:
            # Held back but never part of a combination: replay it now.
            self._pending.remove(code)
            self._emit(code, 1)
            self._emit(code, 0)
            return
        self._forward(event)

    def _check_bindings(self, rising: bool):
        # Only the most specific satisfied binding wins, so cmd+shift+alt does
        # not also fire the cmd+alt binding nested inside it.
        satisfied = [
            (name, keys)
            for name, (keys, _, _, _) in self._bindings.items()
            if keys and keys.issubset(self._key_state)
        ]
        winner = None
        if satisfied:
            winner = max(satisfied, key=lambda item: len(item[1]))[0]

        for name, (keys, cb_press, cb_release, swallow) in self._bindings.items():
            if name == winner:
                # Only arm on a key-down. Otherwise releasing shift out of
                # cmd+shift+alt would "fall through" and fire cmd+alt, starting
                # a dictation the user never asked for.
                if name not in self._press_triggered and rising:
                    self._press_triggered.add(name)
                    self._active_hotkey = name
                    if swallow:
                        self._begin_swallow(keys)
                    if cb_press:
                        self._callbacks.put((name, "press", cb_press))
            elif name in self._press_triggered:
                self._press_triggered.discard(name)
                if self._active_hotkey == name:
                    self._active_hotkey = None
                if cb_release:
                    self._callbacks.put((name, "release", cb_release))

    def _release_devices(self):
        for dev in self._kbd_devices:
            try:
                dev.ungrab()
            except Exception:
                pass
            try:
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

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        # _read_loop releases devices on its way out; do it here too in case it
        # never started.
        self._release_devices()
        if self._dispatch_thread:
            self._callbacks.put(None)
            self._dispatch_thread.join(timeout=2)
            self._dispatch_thread = None
        self._key_state.clear()
        self._press_triggered.clear()
        self._swallowed.clear()
        self._active_hotkey = None
