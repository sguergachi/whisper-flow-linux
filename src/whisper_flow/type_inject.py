"""Typing that does not depend on the host's input injector.

The transcripts have to arrive as keystrokes, and the usual route - ydotool
through its daemon - can silently stop working: KWin ignores the daemon's
device outright on some setups (seen here: keystrokes reach the kernel and
are dropped by the compositor), and ydotool still exits 0, so nothing looks
wrong from the outside and the words simply never appear.

This module owns its injection device instead. It is the same mechanism
ydotool uses - a uinput keyboard - with one difference that matters: the
device is created by this process, so it is new to the compositor and
subject to none of the per-device state (disabled devices, stale settings)
that can accumulate for the daemon's device. The app already creates a
uinput proxy for its hotkey grabs, so this is not a new kind of dependency,
and it works on any compositor that accepts injected input at all.

Delivery is still verified the same way as everything else the app types:
if writing fails, the callers fall back to ydotool, wtype, or the clipboard.
"""

from __future__ import annotations

import time

try:
    import evdev  # noqa: F401  (imported for availability check)
    from evdev import UInput, ecodes

    HAVE_EVDEV = True
except Exception:  # pragma: no cover - evdev is a hard dependency elsewhere
    HAVE_EVDEV = False

# Characters that need no modifier on a US layout.
_PLAIN = {
    " ": ecodes.KEY_SPACE if HAVE_EVDEV else 0,
    "-": ecodes.KEY_MINUS if HAVE_EVDEV else 0,
    "=": ecodes.KEY_EQUAL if HAVE_EVDEV else 0,
    "[": ecodes.KEY_LEFTBRACE if HAVE_EVDEV else 0,
    "]": ecodes.KEY_RIGHTBRACE if HAVE_EVDEV else 0,
    "\\": ecodes.KEY_BACKSLASH if HAVE_EVDEV else 0,
    ";": ecodes.KEY_SEMICOLON if HAVE_EVDEV else 0,
    "'": ecodes.KEY_APOSTROPHE if HAVE_EVDEV else 0,
    ",": ecodes.KEY_COMMA if HAVE_EVDEV else 0,
    ".": ecodes.KEY_DOT if HAVE_EVDEV else 0,
    "/": ecodes.KEY_SLASH if HAVE_EVDEV else 0,
    "`": ecodes.KEY_GRAVE if HAVE_EVDEV else 0,
    "\n": ecodes.KEY_ENTER if HAVE_EVDEV else 0,
    "\t": ecodes.KEY_TAB if HAVE_EVDEV else 0,
}

# Characters that are the shifted form of another key.
_SHIFTED = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5",
    "^": "6", "&": "7", "*": "8", "(": "9", ")": "0",
    "_": "-", "+": "=", "{": "[", "}": "]", "|": "\\",
    ":": ";", '"': "'", "<": ",", ">": ".", "?": "/", "~": "`",
}

_DEVICE: "UInput | None" = None
_DEVICE_FAILED = False


def _codes_for(char: str):
    """(keycode, needs_shift) for a character, or None if untypable."""
    if char in _PLAIN:
        return _PLAIN[char], False
    if "a" <= char <= "z":
        return getattr(ecodes, f"KEY_{char.upper()}"), False
    if "A" <= char <= "Z":
        return getattr(ecodes, f"KEY_{char.upper()}"), True
    if "0" <= char <= "9":
        return getattr(ecodes, f"KEY_{char}"), False
    if char in _SHIFTED:
        base, _ = _codes_for(_SHIFTED[char])
        return base, True
    return None


def _device():
    """The long-lived injection keyboard, created on first use."""
    global _DEVICE, _DEVICE_FAILED
    if _DEVICE is not None or _DEVICE_FAILED or not HAVE_EVDEV:
        return _DEVICE
    try:
        keys = set(_PLAIN.values()) | {ecodes.KEY_LEFTSHIFT}
        for c in "abcdefghijklmnopqrstuvwxyz0123456789":
            keys.add(getattr(ecodes, f"KEY_{c.upper()}"))
        for base in _SHIFTED.values():
            code, _ = _codes_for(base)
            if code:
                keys.add(code)
        _DEVICE = UInput(
            {ecodes.EV_KEY: sorted(keys)},
            name="whisper-flow-typing",
            # BUS_VIRTUAL keeps the hotkey grabber away from it: that one
            # grabs real keyboards only, and a device it grabbed would be
            # swallowed and re-emitted under the user's held modifiers.
            bustype=0x06,
        )
        # Small settle before the first keystroke, so the compositor has the
        # device in hand when the first character arrives.
        time.sleep(0.15)
    except Exception:
        _DEVICE = None
        _DEVICE_FAILED = True
    return _DEVICE


def type_text(text: str) -> bool:
    """Type `text` as keystrokes. False if nothing could be delivered.

    Characters with no place on a US layout (smart quotes, em dashes, and
    the like - which transcripts do contain) are dropped rather than allowed
    to corrupt the sentence; the caller still has the clipboard fallback for
    a faithful copy.
    """
    ui = _device()
    if ui is None or not text:
        return False
    shift = ecodes.KEY_LEFTSHIFT
    typed = 0
    for char in text:
        found = _codes_for(char)
        if found is None:
            continue
        code, needs_shift = found
        try:
            if needs_shift:
                ui.write(ecodes.EV_KEY, shift, 1)
            ui.write(ecodes.EV_KEY, code, 1)
            ui.write(ecodes.EV_KEY, code, 0)
            if needs_shift:
                ui.write(ecodes.EV_KEY, shift, 0)
            ui.syn()
        except Exception:
            return typed > 0
        typed += 1
        # Long strings are queued faster than a terminal can consume them;
        # a hair of pacing keeps the order intact without feeling slow.
        if typed % 32 == 0:
            time.sleep(0.01)
    return typed > 0
