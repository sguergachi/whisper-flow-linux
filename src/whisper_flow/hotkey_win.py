"""Global hotkey detection for Windows, by polling key state.

This deliberately does not install a WH_KEYBOARD_LL hook, which is the usual
way to do this and is the wrong tool in Python.

A low-level hook's callback runs on the thread that installed it, and a
Python callback needs the GIL. Whenever another thread holds it - during
transcription, an HTTP request, audio work - the callback cannot run, and
Windows blocks the *entire system's* input waiting for it, up to the
LowLevelHooksTimeout. The visible result is the mouse and keyboard freezing
until the app is killed. A hook is also free to swallow keystrokes, which is
how the Linux side once left a modifier stuck down system-wide.

GetAsyncKeyState has neither problem. It reads state rather than
intercepting events, so it cannot swallow input and cannot stall anything:
if this thread is starved of the GIL the only consequence is that a hotkey
is noticed late. The cost is that a press shorter than the poll interval can
be missed, which does not matter for push-to-talk.
"""

import ctypes
import logging
import queue
import threading
import time

log = logging.getLogger(__name__)

# Virtual key codes, under the names the config uses.
NAME_TO_VK = {
    "ctrl": 0x11, "control": 0x11,
    "alt": 0x12,
    "shift": 0x10,
    "cmd": 0x5B, "super": 0x5B, "win": 0x5B, "meta": 0x5B,
    "space": 0x20,
    "esc": 0x1B, "escape": 0x1B,
    "tab": 0x09,
    "enter": 0x0D, "return": 0x0D,
    "capslock": 0x14,
}
for _c in "abcdefghijklmnopqrstuvwxyz":
    NAME_TO_VK[_c] = ord(_c.upper())
for _d in "0123456789":
    NAME_TO_VK[_d] = ord(_d)
for _n in range(1, 25):
    NAME_TO_VK[f"f{_n}"] = 0x6F + _n

POLL_INTERVAL = 0.016   # ~60Hz; a tap shorter than this can be missed
HELD_MASK = 0x8000      # high bit of GetAsyncKeyState means currently down

# While this app is injecting keystrokes of its own, the key state is not the
# user's and must not be read as if it were.
#
# Typing releases every modifier first, so a held Alt cannot turn dictated
# text into shortcuts, and puts back whatever was down afterwards. Both go
# through the same state table this module polls, so between the two there is
# a window - a few milliseconds - in which the keys the user is holding read
# as released. At a 16ms poll that window is caught often enough to end a
# recording at the first committed word, which is exactly what it did.
#
# Restoring the keys is not enough on its own: the poll can land inside the
# gap however narrow it is. This closes it instead of racing it.
_suppressed_until = 0.0
_suppress_lock = threading.Lock()

# How long the state table is given to settle after the last injected event.
# The events are queued, so the keys we put back are not necessarily readable
# the instant SendInput returns.
SETTLE_SECONDS = 0.05

_typing_depth = 0
_settled_at = 0.0


def begin_typing() -> None:
    """Mark the start of a batch of injected keystrokes."""
    global _typing_depth
    with _suppress_lock:
        _typing_depth += 1


def end_typing() -> None:
    """Mark the end of one, and start the settling window."""
    global _typing_depth, _settled_at
    with _suppress_lock:
        _typing_depth = max(0, _typing_depth - 1)
        if _typing_depth == 0:
            _settled_at = time.monotonic() + SETTLE_SECONDS


def suppress(seconds: float = 0.25) -> None:
    """Ignore the keyboard for a fixed moment.

    Kept for callers that cannot bracket their own injection. Bracketing is
    better: a guess long enough to cover a long commit is also a guess that
    goes on ignoring the keyboard well after the typing has finished, and
    that reads as a hotkey that does not respond. Half a second of that,
    right after a dictation, is exactly when the next one is pressed.
    """
    global _suppressed_until
    with _suppress_lock:
        _suppressed_until = max(_suppressed_until, time.monotonic() + seconds)


def _suppressed() -> bool:
    with _suppress_lock:
        if _typing_depth > 0:
            return True
        now = time.monotonic()
        return now < _settled_at or now < _suppressed_until

# Both physical sides of a modifier mean the same thing, exactly as the Linux
# listener treats them. Ctrl, Alt and Shift need no entry: 0x11, 0x12 and 0x10
# are the combined codes and already report either side. Windows has no
# combined code for the Windows key, so without this the right-hand one
# silently did nothing.
VK_RWIN = 0x5C
VK_ALIASES = {VK_RWIN: NAME_TO_VK["super"]}


class WinHotkeyListener:
    """Watches for hotkey combinations by sampling keyboard state."""

    def __init__(self, poll_interval: float = POLL_INTERVAL):
        self._bindings = {}
        self._press_triggered = set()
        self._callbacks = queue.Queue()
        self._thread = None
        self._dispatch_thread = None
        self._running = False
        self._poll_interval = poll_interval
        # Cancel is not a binding: bindings resolve most-specific-wins, so a
        # lone Escape could never fire while a push-to-talk combination is
        # held - which is the whole point of a cancel key. Set by the manager.
        self.escape_callback = None
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.GetAsyncKeyState.restype = ctypes.c_short
        self._user32.GetAsyncKeyState.argtypes = [ctypes.c_int]

    @staticmethod
    def parse_keys(key_string: str) -> frozenset:
        codes = set()
        for part in (p.strip().lower() for p in key_string.split("+")):
            vk = NAME_TO_VK.get(part)
            if vk:
                codes.add(vk)
        return frozenset(codes)

    def register_hotkey(self, name, key_string, callback_press, callback_release=None):
        self._bindings[name] = (
            self.parse_keys(key_string), callback_press, callback_release,
        )

    def is_alive(self):
        return bool(self._running and self._thread and self._thread.is_alive())

    def start(self):
        self._running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True,
            name="whisper-flow-hotkey-dispatch",
        )
        self._dispatch_thread.start()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="whisper-flow-hotkey-poll",
        )
        self._thread.start()

    def _dispatch_loop(self):
        """Run callbacks off the polling thread, so a slow one cannot stall it."""
        while True:
            item = self._callbacks.get()
            if item is None:
                return
            name, kind, cb = item
            try:
                cb()
            except Exception:
                log.exception("hotkey %s %s callback failed", name, kind)

    def _held(self) -> set:
        """Which of the keys we care about are down right now."""
        down = set()
        for vk in self._watched:
            if self._user32.GetAsyncKeyState(vk) & HELD_MASK:
                down.add(VK_ALIASES.get(vk, vk))
        return down

    @property
    def _watched(self):
        watched = set()
        for keys, _, _ in self._bindings.values():
            watched |= set(keys)
        # Poll the alternate physical key for anything aliased onto it.
        for physical, logical in VK_ALIASES.items():
            if logical in watched:
                watched.add(physical)
        return watched

    def _poll_loop(self):
        watched = self._watched          # fixed once; bindings do not change
        get = self._user32.GetAsyncKeyState
        esc = NAME_TO_VK["esc"]
        previous = set()
        esc_was_down = False
        while self._running:
            try:
                # Our own injected input is in the state table too. Reading it
                # while typing is how a held hotkey looked released.
                if _suppressed():
                    time.sleep(self._poll_interval)
                    continue
                down = {VK_ALIASES.get(vk, vk)
                        for vk in watched if get(vk) & HELD_MASK}
                if down != previous:
                    self._check_bindings(down, rising=len(down) > len(previous))
                    previous = down
                # Escape is tracked beside the bindings, not among them: it
                # must fire even while a push-to-talk combination is held.
                esc_down = bool(get(esc) & HELD_MASK)
                if esc_down and not esc_was_down and self.escape_callback:
                    self._callbacks.put(("escape", "press", self.escape_callback))
                esc_was_down = esc_down
            except Exception:
                log.exception("error polling keyboard state")
            time.sleep(self._poll_interval)

    @staticmethod
    def _spoil_start_menu_if_needed(keys) -> None:
        """Stop letting go of a Super hotkey from opening the Start menu.

        Windows opens Start when its key is released and nothing was pressed
        while it was held. A push-to-talk combination ends with exactly that
        release, so the menu appeared and the dictated text went into its
        search box.

        Done here, at the moment the combination fires, rather than only
        before typing: a press with nothing to type still ends in a release.
        """
        if NAME_TO_VK["super"] not in keys:
            return
        try:
            from . import system_win

            system_win.spoil_start_menu()
        except Exception as e:
            log.debug("could not suppress the Start menu: %s", e)

    def _check_bindings(self, down: set, rising: bool):
        satisfied = [
            (name, keys) for name, (keys, _, _) in self._bindings.items()
            if keys and keys.issubset(down)
        ]
        # Most specific wins, so ctrl+shift+alt does not also fire ctrl+alt.
        winner = max(satisfied, key=lambda i: len(i[1]))[0] if satisfied else None

        for name, (_keys, cb_press, cb_release) in self._bindings.items():
            if name == winner:
                # Only arm while keys are going down, or releasing one key of a
                # larger combination would fall through and fire a smaller one.
                if name not in self._press_triggered and rising:
                    self._press_triggered.add(name)
                    self._spoil_start_menu_if_needed(_keys)
                    if cb_press:
                        self._callbacks.put((name, "press", cb_press))
            elif name in self._press_triggered:
                self._press_triggered.discard(name)
                if cb_release:
                    self._callbacks.put((name, "release", cb_release))

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._dispatch_thread:
            self._callbacks.put(None)
            self._dispatch_thread.join(timeout=2)
            self._dispatch_thread = None
        self._press_triggered.clear()
