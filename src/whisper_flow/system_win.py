"""Typing, clipboard and notifications on Windows.

Text goes in through SendInput with KEYEVENTF_UNICODE, which delivers a
character rather than a keystroke, so it needs no keyboard layout mapping and
is unaffected by which modifiers happen to be held.
"""

import ctypes
import ctypes.wintypes as wintypes
import subprocess

from .logging import log

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

# Stamped on every event this module injects. Nothing reads it back
# today - the hotkey listener polls key state rather than hooking the
# stream, so it cannot see its own injections anyway - but it costs
# nothing and identifies our events in a trace.
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
        # ULONG_PTR is an integer wide enough to hold a pointer, not a
        # pointer. Declaring it as POINTER(ULONG) put the address of a tag
        # in the field instead of the tag, which is meaningless to anything
        # reading it back.
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("padding", ctypes.c_ubyte * 24)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTunion)]


_TAG = INJECTED_TAG


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
    """Type text as characters, not keystrokes.

    Falls back to the clipboard when SendInput is refused. It is refused
    whenever the focused window belongs to a more privileged process - an
    elevated editor or terminal - because UIPI blocks synthetic input from
    a lower integrity level. Returning False there meant a dictation that
    recorded, animated, and typed nothing.
    """
    if not text:
        return True
    release_modifiers()
    events = []
    for ch in text:
        for code in _utf16_units(ch):
            events.append(_key_event(0, code, KEYEVENTF_UNICODE))
            events.append(_key_event(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
    # SendInput takes the whole batch atomically, so nothing can interleave.
    if _send(events):
        return True

    error = ctypes.get_last_error()
    log(f"[WIN] SendInput refused {len(text)} chars (error {error}); "
        f"falling back to the clipboard")
    return copy_to_clipboard(text) and send_paste()


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


CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


def copy_to_clipboard(text: str) -> bool:
    """Put text on the clipboard through the Win32 API.

    Not clip.exe. That meant spawning a process for every copy, and it was
    being fed UTF-16 with no byte-order mark, which clip.exe reads as the
    console code page - so anything outside ASCII arrived as mojibake. The
    failure reports this carries are exactly the text that must survive
    intact.
    """
    try:
        kernel32, user32 = ctypes.windll.kernel32, ctypes.windll.user32
        buffer = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(buffer)

        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            return False
        locked = kernel32.GlobalLock(handle)
        if not locked:
            kernel32.GlobalFree(handle)
            return False
        try:
            ctypes.memmove(locked, buffer, size)
        finally:
            kernel32.GlobalUnlock(handle)

        if not user32.OpenClipboard(None):
            kernel32.GlobalFree(handle)
            return False
        try:
            user32.EmptyClipboard()
            # Ownership passes to the clipboard on success; freeing it after
            # that would be a double free.
            if not user32.SetClipboardData(CF_UNICODETEXT, handle):
                kernel32.GlobalFree(handle)
                return False
        finally:
            user32.CloseClipboard()
        return True
    except Exception as e:
        log(f"[WIN] clipboard failed: {e}")
        return _copy_via_clip_exe(text)


def _copy_via_clip_exe(text: str) -> bool:
    """Fallback for a locked clipboard. utf-16 here carries its own BOM."""
    try:
        proc = subprocess.run(
            ["clip.exe"], input=text.encode("utf-16"),
            capture_output=True, timeout=5, check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _ps_literal(text: str) -> str:
    """Quote text for a PowerShell single-quoted string.

    Doubling the quote is the only escape such a string has. Without this an
    apostrophe ended the literal and the remainder of the message was parsed
    as code - so an error like "can't open device" silently produced no
    notification at all, and any text reaching here decided what ran.
    """
    flattened = " ".join(str(text).split())     # newlines would break the line
    return flattened.replace("'", "''")


def notify(title: str, message: str) -> None:
    """Best-effort desktop notification via PowerShell toast."""
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications,"
        " ContentType=WindowsRuntime] > $null;"
        "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(0);"
        f"$t.GetElementsByTagName('text').Item(0).AppendChild("
        f"$t.CreateTextNode('{_ps_literal(title)}')) > $null;"
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
