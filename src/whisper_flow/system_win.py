"""Typing, clipboard and notifications on Windows.

Text goes in through SendInput with KEYEVENTF_UNICODE, which delivers a
character rather than a keystroke, so it needs no keyboard layout mapping and
is unaffected by which modifiers happen to be held.
"""

import contextlib
import ctypes
import ctypes.wintypes as wintypes
import subprocess
import threading
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

# Modifiers this process is holding down on the user's behalf, and which it
# therefore owes a key-up. See restore_modifiers.
_injected_lock = threading.Lock()
_injected_down: set[int] = set()


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
    _user32.GetWindowRect.argtypes = [wintypes.HWND,
                                      ctypes.POINTER(wintypes.RECT)]
    _user32.GetWindowRect.restype = wintypes.BOOL
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


def describe_foreground() -> str:
    """What has the foreground, as class and title, for a log line.

    The class is the useful half: the Start menu and the search box are
    Windows.UI.Core.CoreWindow, the taskbar is Shell_TrayWnd, and our own
    overlay is a GTK window - three different problems that all read as
    "focus went somewhere" without it.
    """
    hwnd = foreground_window()
    if not hwnd:
        return "nothing"
    try:
        name = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(wintypes.HWND(hwnd), name, 256)
        title = ctypes.create_unicode_buffer(256)
        _user32.GetWindowTextW(wintypes.HWND(hwnd), title, 256)
        return f"{name.value or '?'} {title.value!r} (hwnd {hwnd})"
    except Exception:
        return f"hwnd {hwnd}"


def window_center(hwnd: int) -> tuple[int, int] | None:
    """Centre of a window, in physical desktop pixels. None if there is none.

    This is what tells the overlay which screen to appear on. Without it the
    overlay had no answer on Windows at all - the only implementation was
    kdotool's, which does not exist here - so it fell back to the first
    monitor GDK listed and stayed there. On one screen that is invisibly
    correct; on two it put the pill on the other one, every time.

    Physical, because GetWindowRect is: the caller converts.
    """
    if not hwnd:
        return None
    try:
        if not _user32.IsWindow(wintypes.HWND(hwnd)):
            return None
        rect = wintypes.RECT()
        if not _user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None
        if rect.right <= rect.left or rect.bottom <= rect.top:
            return None
        return ((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)
    except Exception:
        return None


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


def _spoil_events() -> list:
    """The keystroke that marks a held modifier as already used.

    Returned rather than sent so callers can put it in the same batch as the
    presses or releases it guards. SendInput takes a batch atomically, and
    two calls are not: between them the user's own key-up can land, which is
    precisely the event being guarded against.

    Sent whatever is being held, not only Super. A lone Alt tap has the same
    shape and opens the focused window's menu bar, and our own press-and-
    release of the modifiers is exactly that shape.
    """
    return [
        _key_event(VK_NONAME, 0, 0),
        _key_event(VK_NONAME, 0, KEYEVENTF_KEYUP),
    ]


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
    _send(_spoil_events())


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
    _send(_spoil_events()
          + [_key_event(vk, 0, KEYEVENTF_KEYUP) for vk in MODIFIERS])
    return held


def _restorable(held) -> tuple:
    """Of the modifiers that were down, the ones we may press back.

    Only the keys of the push-to-talk combination the user is actually
    holding. The snapshot covers every modifier that happened to be down,
    and pressing all of them back is how an incidental Shift became one we
    were responsible for releasing.
    """
    try:
        from . import hotkey_win

        combination = hotkey_win.held_combination()
    except Exception:
        return tuple(held)
    if combination is None:
        return tuple(held)          # no listener to protect; as before
    aliases = getattr(hotkey_win, "VK_ALIASES", {})
    return tuple(vk for vk in held if aliases.get(vk, vk) in combination)


def restore_modifiers(held) -> None:
    """Put back the modifiers the push-to-talk is holding down.

    Recorded as ours while they are down. This is the one thing in here that
    presses a key with no user behind it, and if the user lets go during the
    injection - after the snapshot, before this - their real key-up lands on
    a key already released, and the press below then has no release coming
    at all. Windows goes on believing the key is down: with Super that turns
    the next Shift into Win+Shift, and the desktop starts minimising itself.

    So the daemon clears these at the end of every dictation through
    release_injected_modifiers(), and the window in which a stuck key can
    exist is bounded by the dictation rather than by the session.

    Spoiled straight after the press, in the same batch. This press is a
    fresh, unused Windows-key press as far as Windows is concerned, and the
    next key-up completes it into a lone Super tap - whichever key-up that
    is. release_injected_modifiers() already guards the one we send, but the
    user's own release is the one that usually arrives first: they let go
    mid-dictation, while we are still holding the key back down on their
    behalf, and Start opened under the words still being typed. One batch
    rather than two because a second SendInput is a gap their release can
    land in, which is the very event this exists to disarm.
    """
    wanted = _restorable(held)
    if not wanted:
        return
    with _injected_lock:
        _injected_down.update(wanted)
    _send([_key_event(vk, 0, 0) for vk in wanted] + _spoil_events())


def release_injected_modifiers() -> tuple:
    """Let go of every modifier we pressed back down, and forget them.

    Called when a dictation ends. Sending a key-up for something the user is
    still physically holding is harmless - their own release just arrives at
    a key that is already up - while not sending it leaves the key down for
    every application on the desktop.

    Spoiled first, for the same reason release_modifiers() is and this was
    not. restore_modifiers() presses Super back down, and from Windows' point
    of view that is a fresh press with nothing after it; the release below
    then completes a lone Super tap and opens the Start menu. It only showed
    when the pass that restored the keys committed no words, because any
    character typed in between counts as the intervening press - which is
    what made it look random, and what put the rest of the dictation into the
    Start menu's search box once focus had gone there.
    """
    with _injected_lock:
        ours = tuple(sorted(_injected_down))
        _injected_down.clear()
    if ours:
        log(f"[WIN] releasing {len(ours)} modifier(s) we were holding down")
        _send(_spoil_events()
              + [_key_event(vk, 0, KEYEVENTF_KEYUP) for vk in ours])
    return ours


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
