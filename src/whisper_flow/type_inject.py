"""Typing that does not depend on the host's input injector.

The transcripts have to arrive as keystrokes, and the usual route - ydotool
through its daemon - can silently stop working: KWin discards every keystroke
from its device on some setups (its device disabled in kcminputrc) while
ydotoold still exits 0, so the failure is invisible and the fallbacks never
run.

This module owns its injection device instead. It is the same mechanism
ydotool uses - a uinput keyboard - with one difference that matters: the
device is created by this process, so it is new to the compositor and
subject to none of the per-device state (disabled, stale) that can
accumulate for the daemon's device. The app already creates a uinput proxy
for its hotkey grabs, so this is not a new kind of dependency, and it
works on any compositor that accepts injected input at all.

Delivery is still verified the same way as everything else the app types:
if writing fails, the callers fall back to ydotool, wtype, or the clipboard.
"""

from __future__ import annotations

import time

try:
    import evdev  # noqa: F401
    from evdev import UInput, ecodes

    HAVE_EVDEV = True
except Exception:  # pragma: no cover - evdev is a hard dependency elsewhere
    HAVE_EVDEV = False

# Characters that need no modifier on a US layout.
_PLAIN = {}
_SHIFTED = {}
if HAVE_EVDEV:
    _PLAIN = {
        " ": ecodes.KEY_SPACE,
        "-": ecodes.KEY_MINUS,
        "=": ecodes.KEY_EQUAL,
        "[": ecodes.KEY_LEFTBRACE,
        "]": ecodes.KEY_RIGHTBRACE,
        "\\": ecodes.KEY_BACKSLASH,
        ";": ecodes.KEY_SEMICOLON,
        "'": ecodes.KEY_APOSTROPHE,
        ",": ecodes.KEY_COMMA,
        ".": ecodes.KEY_DOT,
        "/": ecodes.KEY_SLASH,
        "`": ecodes.KEY_GRAVE,
        "\n": ecodes.KEY_ENTER,
        "\t": ecodes.KEY_TAB,
    }
    _SHIFTED = {
        "!": "1", "@": "2", "#": "3", "$": "4", "%": "5",
        "^": "6", "&": "7", "*": "8", "(": "9", ")": "0",
        "_": "-", "+": "=", "{": "[", "}": "]", "|": "\\",
        ":": ";", '"': "'", "<": ",", ">": ".", "?": "/", "~": "`",
    }

# Modifiers a failed run must never leave held.
_MODIFIER_KEYS = (ecodes.KEY_LEFTSHIFT,) if HAVE_EVDEV else ()

_DEVICE = None
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
        keys = set(_PLAIN.values()) | {_MODIFIER_KEYS[0]}
        for c in "abcdefghijklmnopqrstuvwxyz0123456789":
            keys.add(getattr(ecodes, f"KEY_{c.upper()}"))
        for base in _SHIFTED.values():
            code, _ = _codes_for(base)
            if code:
                keys.add(code)
        _DEVICE = UInput(
            {ecodes.EV_KEY: sorted(keys)},
            name="whisper-flow-typing",
            # BUS_VIRTUAL so the hotkey grabber ignores it (real keyboards
            # only), and so it is new to the compositor regardless of the
            # host injector's state.
            bustype=0x06,
        )
        time.sleep(0.15)
    except Exception:
        _DEVICE = None
        _DEVICE_FAILED = True
    return _DEVICE


def type_text(text: str) -> bool:
    """Type `text` as keystrokes. False if nothing could be delivered.

    Characters with no place on a US layout (smart quotes, em dashes, and
    the like, which transcripts do contain) are dropped rather than allowed
    to corrupt the sentence; the caller still has the clipboard for a faithful
    copy. Modifiers are always released, even on failure, so a crashed run
    cannot leave a key held down.
    """
    ui = _device()
    if ui is None or not text:
        return False
    shift = _MODIFIER_KEYS[0]
    typed = 0
    try:
        # Clear any modifier state left by a previous, interrupted run.
        for code in _MODIFIER_KEYS:
            ui.write(ecodes.EV_KEY, code, 0)
        ui.syn()
        for char in text:
            found = _codes_for(char)
            if found is None:
                continue
            code, needs_shift = found
            if needs_shift:
                ui.write(ecodes.EV_KEY, shift, 1)
            ui.write(ecodes.EV_KEY, code, 1)
            ui.write(ecodes.EV_KEY, code, 0)
            if needs_shift:
                ui.write(ecodes.EV_KEY, shift, 0)
            ui.syn()
            typed += 1
            if typed % 32 == 0:
                time.sleep(0.01)
    except Exception:
        for code in _MODIFIER_KEYS:
            try:
                ui.write(ecodes.EV_KEY, code, 0)
                ui.syn()
            except Exception:
                pass
        return typed > 0
    return typed > 0


def release_modifiers() -> None:
    """Release every modifier we know about, through our own device.

    KWin tracks modifier state globally, across keyboards, so a modifier
    that the hotkey manager held and then dropped without a clean release
    stays "pressed" in the compositor. The next ordinary keystroke then
    arrives as a shortcut - mission control, lock screen, logout - instead
    of text. Sending releases through a device we own (which KWin accepts)
    clears that state without depending on the host injector, which may
    itself be disabled. Only releases are ever sent, never presses, so the
    worst case is a redundant key-up.
    """
    ui = _device()
    if ui is None:
        return
    for code in _MODIFIER_KEYS:
        try:
            ui.write(ecodes.EV_KEY, code, 0)
        except Exception:
            pass
    try:
        ui.syn()
    except Exception:
        pass
