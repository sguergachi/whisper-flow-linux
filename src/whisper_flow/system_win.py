"""Typing, clipboard and notifications on Windows.

Text goes in through SendInput with KEYEVENTF_UNICODE, which delivers a
character rather than a keystroke, so it needs no keyboard layout mapping and
is unaffected by which modifiers happen to be held. Every event carries a tag
in dwExtraInfo that the hotkey hook filters out, or dictated text would be
read straight back in as hotkey input.
"""

import ctypes
import ctypes.wintypes as wintypes
import subprocess

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

# Must match hotkey_win.INJECTED_TAG.
INJECTED_TAG = 0x5748464C

VK_CONTROL, VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN = 0x11, 0x12, 0x10, 0x5B, 0x5C
VK_V = 0x56
MODIFIERS = (VK_CONTROL, VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN)

_user32 = ctypes.WinDLL("user32", use_last_error=True)


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("padding", ctypes.c_ubyte * 24)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTunion)]


_TAG = ctypes.pointer(wintypes.ULONG(INJECTED_TAG))


def _key_event(vk: int, scan: int, flags: int) -> INPUT:
    return INPUT(
        type=INPUT_KEYBOARD,
        union=_INPUTunion(ki=KEYBDINPUT(
            wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=_TAG,
        )),
    )


def _send(events: list) -> bool:
    if not events:
        return True
    arr = (INPUT * len(events))(*events)
    sent = _user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))
    return sent == len(events)


def release_modifiers() -> None:
    """Tell Windows no modifiers are held before injecting text.

    Live transcription types while the push-to-talk keys are physically down,
    and a held Alt or Ctrl would turn every character into a shortcut. Only
    releases are sent, never presses, so nothing can be left stuck.
    """
    _send([_key_event(vk, 0, KEYEVENTF_KEYUP) for vk in MODIFIERS])


def type_text(text: str) -> bool:
    """Type text as characters, not keystrokes."""
    if not text:
        return True
    release_modifiers()
    events = []
    for ch in text:
        for code in _utf16_units(ch):
            events.append(_key_event(0, code, KEYEVENTF_UNICODE))
            events.append(_key_event(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
    # SendInput takes the whole batch atomically, so nothing can interleave.
    return _send(events)


def _utf16_units(ch: str):
    """UTF-16 code units for a character; astral chars need a surrogate pair."""
    encoded = ch.encode("utf-16-le")
    return [int.from_bytes(encoded[i:i + 2], "little")
            for i in range(0, len(encoded), 2)]


def send_paste() -> bool:
    """Ctrl+V, for the clipboard fallback path."""
    release_modifiers()
    return _send([
        _key_event(VK_CONTROL, 0, 0),
        _key_event(VK_V, 0, 0),
        _key_event(VK_V, 0, KEYEVENTF_KEYUP),
        _key_event(VK_CONTROL, 0, KEYEVENTF_KEYUP),
    ])


def copy_to_clipboard(text: str) -> bool:
    """Put text on the clipboard using the built-in clip.exe."""
    try:
        proc = subprocess.run(
            ["clip.exe"], input=text.encode("utf-16-le"),
            capture_output=True, timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        return False


def notify(title: str, message: str) -> None:
    """Best-effort desktop notification via PowerShell toast."""
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications,"
        " ContentType=WindowsRuntime] > $null;"
        "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(0);"
        f"$t.GetElementsByTagName('text').Item(0).AppendChild($t.CreateTextNode('{title}')) > $null;"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        "'whisper-flow').Show([Windows.UI.Notifications.ToastNotification]::new($t))"
    )
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
    except Exception:
        print(f"[whisper-flow] {title}: {message}")
