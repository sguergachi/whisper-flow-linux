"""Global hotkey listener for Windows, via a low-level keyboard hook.

RegisterHotKey is the usual way to claim a global shortcut, but it only ever
reports a press. Push-to-talk needs the release too, so this installs a
WH_KEYBOARD_LL hook instead and tracks key state itself.

The hook callback runs on the thread that installed it, inside Windows'
message loop, and Windows silently removes hooks whose callback is slow. So
this mirrors the evdev listener: observe, hand the work to another thread,
return immediately. It also never swallows a key - the same rule the Linux
side learned the hard way, since a hook that eats input breaks the keyboard
system-wide.
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import queue
import threading

log = logging.getLogger(__name__)

WH_KEYBOARD_LL = 13
WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101
WM_SYSKEYDOWN, WM_SYSKEYUP = 0x0104, 0x0105
HC_ACTION = 0

# Virtual key codes, mapped onto the same names the config uses.
NAME_TO_VK = {
    "ctrl": 0x11, "control": 0x11,
    "alt": 0x12,
    "shift": 0x10,
    "cmd": 0x5B, "super": 0x5B, "win": 0x5B, "meta": 0x5B,
    "space": 0x20,
    "esc": 0x1B, "escape": 0x1B,
    "tab": 0x09,
    "enter": 0x0D, "return": 0x0D,
}
for _c in "abcdefghijklmnopqrstuvwxyz":
    NAME_TO_VK[_c] = ord(_c.upper())
for _d in "0123456789":
    NAME_TO_VK[_d] = ord(_d)
for _n in range(1, 25):
    NAME_TO_VK[f"f{_n}"] = 0x6F + _n

# Both sides of a modifier collapse onto one logical key, as on Linux.
VK_ALIASES = {
    0xA0: 0x10, 0xA1: 0x10,  # L/R shift
    0xA2: 0x11, 0xA3: 0x11,  # L/R control
    0xA4: 0x12, 0xA5: 0x12,  # L/R alt
    0x5C: 0x5B,              # right win
}


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM,
)

# Marks events this process injected, so the hook ignores its own typing.
INJECTED_TAG = 0x5748464C  # 'WHFL'


class WinHotkeyListener:
    """Watches the keyboard globally and fires push-to-talk callbacks."""

    def __init__(self):
        self._bindings = {}
        self._key_state = set()
        self._press_triggered = set()
        self._callbacks = queue.Queue()
        self._thread = None
        self._dispatch_thread = None
        self._hook = None
        self._thread_id = None
        self._running = False
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

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
            target=self._dispatch_loop, daemon=True, name="whisper-flow-hotkey-dispatch",
        )
        self._dispatch_thread.start()
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._hook_loop, args=(ready,), daemon=True,
            name="whisper-flow-hotkey-hook",
        )
        self._thread.start()
        if not ready.wait(timeout=5):
            raise RuntimeError("keyboard hook did not install")

    def _dispatch_loop(self):
        """Run callbacks off the hook thread; a slow hook gets uninstalled."""
        while True:
            item = self._callbacks.get()
            if item is None:
                return
            name, kind, cb = item
            try:
                cb()
            except Exception:
                log.exception("hotkey %s %s callback failed", name, kind)

    def _hook_loop(self, ready):
        # Held as an attribute so the trampoline is not garbage collected while
        # Windows still holds a pointer to it.
        self._proc = HOOKPROC(self._on_event)
        self._thread_id = self._kernel32.GetCurrentThreadId()
        self._hook = self._user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._proc, None, 0,
        )
        ready.set()
        if not self._hook:
            log.error("SetWindowsHookExW failed: %s", ctypes.get_last_error())
            return

        msg = wintypes.MSG()
        while self._running and self._user32.GetMessageW(
            ctypes.byref(msg), None, 0, 0,
        ) > 0:
            self._user32.TranslateMessage(ctypes.byref(msg))
            self._user32.DispatchMessageW(ctypes.byref(msg))

        self._user32.UnhookWindowsHookEx(self._hook)
        self._hook = None

    def _on_event(self, code, wparam, lparam):
        try:
            if code == HC_ACTION:
                kb = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                # Skip anything this process injected, or dictated text would
                # be read back as hotkey input.
                if kb.dwExtraInfo and kb.dwExtraInfo.contents.value == INJECTED_TAG:
                    pass
                else:
                    vk = VK_ALIASES.get(kb.vkCode, kb.vkCode)
                    if wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                        if vk not in self._key_state:
                            self._key_state.add(vk)
                            self._check_bindings(rising=True)
                    elif wparam in (WM_KEYUP, WM_SYSKEYUP):
                        self._key_state.discard(vk)
                        self._check_bindings(rising=False)
        except Exception:
            log.exception("keyboard hook callback failed")
        # Always pass the event on. Returning non-zero would swallow the key.
        return self._user32.CallNextHookEx(None, code, wparam, lparam)

    def _check_bindings(self, rising: bool):
        satisfied = [
            (name, keys) for name, (keys, _, _) in self._bindings.items()
            if keys and keys.issubset(self._key_state)
        ]
        winner = max(satisfied, key=lambda i: len(i[1]))[0] if satisfied else None

        for name, (_keys, cb_press, cb_release) in self._bindings.items():
            if name == winner:
                if name not in self._press_triggered and rising:
                    self._press_triggered.add(name)
                    if cb_press:
                        self._callbacks.put((name, "press", cb_press))
            elif name in self._press_triggered:
                self._press_triggered.discard(name)
                if cb_release:
                    self._callbacks.put((name, "release", cb_release))

    def stop(self):
        self._running = False
        if self._thread_id:
            # Nudge GetMessageW so the loop notices _running went false.
            self._user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)  # WM_QUIT
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._dispatch_thread:
            self._callbacks.put(None)
            self._dispatch_thread.join(timeout=2)
            self._dispatch_thread = None
        self._key_state.clear()
        self._press_triggered.clear()
