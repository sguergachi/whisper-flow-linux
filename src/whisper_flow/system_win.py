"""Typing, clipboard and notifications on Windows.

Text goes in through SendInput with KEYEVENTF_UNICODE, which delivers a
character rather than a keystroke, so it needs no keyboard layout mapping and
is unaffected by which modifiers happen to be held.
"""

import contextlib
import ctypes
import ctypes.wintypes as wintypes
import subprocess
import time

from .logging import log

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
HELD_MASK = 0x8000      # high bit of GetAsyncKeyState means currently down

# Stamped on every event this module injects. Nothing reads it back
# today - the hotkey listener polls key state rather than hooking the
# stream, so it cannot see its own injections anyway - but it costs
# nothing and identifies our events in a trace.
INJECTED_TAG = 0x5748464C

VK_CONTROL, VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN = 0x11, 0x12, 0x10, 0x5B, 0x5C
VK_V = 0x56
# Reserved by Windows as a key that does nothing. Used to mark the
# Windows key as having been combined with something, so releasing it
# does not open the Start menu.
VK_NONAME = 0xFC
MODIFIERS = (VK_CONTROL, VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


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


class MOUSEINPUT(ctypes.Structure):
    """Never sent, but it is the largest member and therefore sets the size."""

    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTunion(ctypes.Union):
    # Spelled out rather than padded to a guessed width. It used to be
    # `c_ubyte * 24`, which is the size of this union on 32-bit Windows; on
    # x64 MOUSEINPUT is 32 bytes, so INPUT is 40 and not the 32 that padding
    # produced. SendInput is handed sizeof(INPUT) as cbSize and rejects a
    # value that is not exactly right, so every call failed with
    # ERROR_INVALID_PARAMETER and no dictated text was ever typed.
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTunion)]


_user32.SendInput.restype = wintypes.UINT
_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]

# Declared for the same reason SendInput is: ctypes assumes every function
# returns a C int, and an HWND on 64-bit Windows does not fit in one. A
# truncated window handle compares equal to nothing and is refused by every
# call it is passed to, which would make restoring focus a silent no-op.
try:
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetForegroundWindow.argtypes = []
    _user32.IsWindow.argtypes = [wintypes.HWND]
    _user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.c_void_p]
    _user32.AttachThreadInput.restype = wintypes.BOOL
    _user32.AttachThreadInput.argtypes = [
        wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    _kernel32.GetCurrentThreadId.restype = wintypes.DWORD
    _kernel32.GetCurrentThreadId.argtypes = []
except Exception:                       # a stubbed loader off Windows
    pass


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


def foreground_window() -> int:
    """The window that had focus, to type back into later. 0 if there is none."""
    try:
        return int(_user32.GetForegroundWindow() or 0)
    except Exception:
        return 0


def _window_thread(hwnd: int) -> int:
    return int(_user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), None))


def focus_window(hwnd: int) -> bool:
    """Put focus back on the window the dictation was started in.

    Text is typed into whatever has focus at the moment it is typed, which is
    not necessarily where the user was when they spoke: the closing
    transcription takes as long as it takes, and a click during it sends the
    whole utterance into whichever field is focused when it lands. It arrived
    in the wrong window entirely, which is worse than arriving late - a
    transcript can end up in a chat box or a terminal.

    SetForegroundWindow refuses a caller that does not own the foreground
    window, which is exactly this case. Attaching to the foreground thread's
    input queue first makes Windows treat the call as coming from the
    foreground itself, which is the documented way round it.
    """
    if not hwnd:
        return False
    try:
        if not _user32.IsWindow(wintypes.HWND(hwnd)):
            return False
        current = foreground_window()
        if current == hwnd:
            return True

        ours = int(_kernel32.GetCurrentThreadId())
        theirs = _window_thread(current) if current else 0
        attached = bool(
            theirs and theirs != ours
            and _user32.AttachThreadInput(ours, theirs, True))
        try:
            _user32.SetForegroundWindow(wintypes.HWND(hwnd))
        finally:
            if attached:
                _user32.AttachThreadInput(ours, theirs, False)

        # Focus changes are asynchronous; typing into a window that has not
        # taken it yet loses the first characters.
        for _ in range(10):
            if foreground_window() == hwnd:
                return True
            time.sleep(0.01)
        return False
    except Exception as e:
        log(f"[WIN] could not restore focus to {hwnd}: {e}")
        return False


def spoil_start_menu() -> None:
    """Stop the Start menu opening when the user lets go of the Windows key.

    Windows opens Start on a Windows-key release if no other key was pressed
    while it was held. Push-to-talk on Super+Alt ends with exactly that
    release, so letting go opened Start and the dictated text went into its
    search box.

    Pressing any key while Windows is still down marks the press as used, and
    the release is then ignored. VK_NONAME is reserved as a no-op precisely
    for this: it reaches no application and does nothing on its way past.
    """
    _send([
        _key_event(VK_NONAME, 0, 0),
        _key_event(VK_NONAME, 0, KEYEVENTF_KEYUP),
    ])


@contextlib.contextmanager
def _hotkeys_blinded():
    """Have the hotkey listener ignore the keyboard while we inject into it.

    Imported here rather than at module scope: this is the typing path and
    the listener is the keyboard path, and neither should need the other to
    exist. A build without it still types, it just races again.

    Bracketed around the injection rather than timed from a guess at how long
    it will take. The guess was 0.2s plus a millisecond per character, which
    is wrong in both directions: too short for a long commit, so the race it
    exists to close could reopen, and far too long for a short one - the
    listener went on ignoring the keyboard for up to half a second after the
    last word of a dictation was typed, which is precisely when the next
    hotkey is pressed. Pressing and holding then did nothing visible until
    the window lapsed.
    """
    try:
        from . import hotkey_win
    except Exception:
        yield
        return
    try:
        hotkey_win.begin_typing()
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            hotkey_win.end_typing()
        except Exception:
            pass


def release_modifiers() -> tuple:
    """Tell Windows no modifiers are held before injecting text.

    Live transcription types while the push-to-talk keys are physically down,
    and a held Alt or Ctrl would turn every character into a shortcut.

    Returns the modifiers that were actually down, for restore_modifiers to
    put back. They must go back: the hotkey listener reads the same state
    table this writes to, so releasing the keys the user is holding tells it
    the push-to-talk was let go. It ended the recording at the first
    committed word, every time, while the key was still down - the overlay
    vanishing mid-sentence was this, not a crash.

    The no-op key goes first, while Windows still believes its own key is
    down. Sent afterwards it would be too late: the release below is what
    arms the Start menu.
    """
    held = tuple(vk for vk in MODIFIERS
                 if _user32.GetAsyncKeyState(vk) & HELD_MASK)
    spoil_start_menu()
    _send([_key_event(vk, 0, KEYEVENTF_KEYUP) for vk in MODIFIERS])
    return held


def restore_modifiers(held) -> None:
    """Put back the modifiers the user never let go of.

    Only the ones that were down before we interfered, so a recording that
    ends between the snapshot and here cannot leave a key stuck: nothing is
    pressed that was not already pressed.
    """
    if held:
        _send([_key_event(vk, 0, 0) for vk in held])


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
    # Blind the hotkey listener for the whole of this. Everything below writes
    # to the key state table it polls, and the gap between releasing the
    # modifiers and putting them back reads as the user letting go of the
    # hotkey. Putting them back is not enough by itself - a 16ms poll can land
    # inside any gap - so the listener is told not to look at all.
    with _hotkeys_blinded():
        held = release_modifiers()
        try:
            events = []
            for ch in text:
                for code in _utf16_units(ch):
                    events.append(_key_event(0, code, KEYEVENTF_UNICODE))
                    events.append(
                        _key_event(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
            # SendInput takes the whole batch atomically, so nothing can
            # interleave.
            if _send(events):
                return True

            error = ctypes.get_last_error()
            log(f"[WIN] SendInput refused {len(text)} chars (error {error}); "
                f"falling back to the clipboard")
            return copy_to_clipboard(text) and send_paste(release_first=False)
        finally:
            # Whatever happened above, the user is still holding the hotkey and
            # the state table has to say so again.
            restore_modifiers(held)


def _utf16_units(ch: str):
    """UTF-16 code units for a character; astral chars need a surrogate pair."""
    encoded = ch.encode("utf-16-le")
    return [int.from_bytes(encoded[i:i + 2], "little")
            for i in range(0, len(encoded), 2)]


def send_paste(release_first: bool = True) -> bool:
    """Ctrl+V, for the clipboard fallback path.

    release_first is False when type_text calls this: it has already
    released the modifiers and holds the list to restore afterwards, and a
    second release would lose the record of what the user was holding.
    """
    if release_first:
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
        # Declare the signatures. Without these ctypes assumes every function
        # returns a C int, so on 64-bit Windows the HGLOBAL from GlobalAlloc
        # and the pointer from GlobalLock are truncated to 32 bits and every
        # subsequent call is handed a bad handle - which is exactly why this
        # returned False on the first real Windows run.
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalLock.restype = wintypes.LPVOID
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalFree.restype = wintypes.HGLOBAL
        kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.SetClipboardData.restype = wintypes.HANDLE
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]

        buffer = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(buffer)

        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            log(f"[WIN] GlobalAlloc failed: {ctypes.get_last_error()}")
            return False
        locked = kernel32.GlobalLock(handle)
        if not locked:
            log(f"[WIN] GlobalLock failed: {ctypes.get_last_error()}")
            kernel32.GlobalFree(handle)
            return False
        try:
            ctypes.memmove(locked, buffer, size)
        finally:
            kernel32.GlobalUnlock(handle)

        # Another process can hold the clipboard open; it is worth a retry
        # rather than losing the report over a moment's contention.
        opened = False
        for _ in range(10):
            if user32.OpenClipboard(None):
                opened = True
                break
            time.sleep(0.02)
        if not opened:
            log(f"[WIN] OpenClipboard failed: {ctypes.get_last_error()}")
            kernel32.GlobalFree(handle)
            return False
        try:
            user32.EmptyClipboard()
            # Ownership passes to the clipboard on success; freeing it after
            # that would be a double free.
            if not user32.SetClipboardData(CF_UNICODETEXT, handle):
                log(f"[WIN] SetClipboardData failed: {ctypes.get_last_error()}")
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
