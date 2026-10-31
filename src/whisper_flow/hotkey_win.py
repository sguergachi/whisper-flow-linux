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

# The right-hand Windows key. Windows has no combined code for the Windows
# key the way it has 0x11/0x12/0x10 for the others, so it is aliased onto the
# left one in VK_ALIASES below.
VK_RWIN = 0x5C

POLL_INTERVAL = 0.016   # ~60Hz; a tap shorter than this can be missed
HELD_MASK = 0x8000      # high bit of GetAsyncKeyState means currently down

# How often the Start menu is disarmed again while a Super hotkey is held.
# Windows arms it on a Windows-key press and disarms it on any other key, so
# one keystroke at the moment the combination fires covers the hold only for
# as long as nothing presses Super again - and something does. See
# _keep_start_menu_spoiled.
SPOIL_INTERVAL = 0.1

# How often the foreground is checked, while a Super hotkey is held, for a
# Start menu that opened anyway. See _close_start_menu_if_it_opened.
START_MENU_CHECK = 0.2
START_RELEASE_GRACE = 0.75

# The Windows key cannot stay down while the app types. Every keystroke
# injected while it is down is read as a shell combination rather than text:
# Win+Down minimises the focused window, Win+D shows the desktop, Win+E
# opens Explorer. Push-to-talk on Super+Alt therefore takes the Windows key
# out of the OS's key state once, at the press, and leaves it out for the
# whole hold. See _take_super_out_of_the_hold.
#
# The release is then inferred from the rest of the chord, because once the
# Windows key has been released GetAsyncKeyState cannot tell the user's
# key-up from the one we sent. A Super-only chord has no rest of the chord
# to watch, so it is left alone: the alternative is a recording that nothing
# can end.
SUPER_VK = NAME_TO_VK["super"]
WIN_KEYS = frozenset((SUPER_VK, VK_RWIN))

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
    """Mark the end of one, and start the settling window.

    This replaced a fixed window - 0.2s plus a millisecond per character -
    guessed at before the typing began. A guess long enough to cover a long
    commit is also a guess that goes on ignoring the keyboard well after the
    typing has finished, and that reads as a hotkey that does not respond.
    Half a second of it, right after a dictation, is exactly when the next
    one is pressed.
    """
    global _typing_depth, _settled_at
    with _suppress_lock:
        _typing_depth = max(0, _typing_depth - 1)
        if _typing_depth == 0:
            _settled_at = time.monotonic() + SETTLE_SECONDS


def _suppressed() -> bool:
    with _suppress_lock:
        return _typing_depth > 0 or time.monotonic() < _settled_at

# Both physical sides of a modifier mean the same thing, exactly as the Linux
# listener treats them. Ctrl, Alt and Shift need no entry: 0x11, 0x12 and 0x10
# are the combined codes and already report either side. Windows has no
# combined code for the Windows key, so without this the right-hand one
# silently did nothing.
VK_ALIASES = {VK_RWIN: NAME_TO_VK["super"]}

# The listener currently running, so the typing path can ask what the user is
# actually holding. Typing has to put those keys back down after releasing
# them, and only those: re-pressing a key the user is not holding leaves it
# down for every application on the desktop, with no release ever to come.
_active_listener = None


def held_combination():
    """VKs of the push-to-talk being held right now.

    None - not an empty set - when there is no listener to ask. The two mean
    different things to the caller: "nothing is held, restore nothing" versus
    "nobody knows, so do what you would have done anyway".
    """
    listener = _active_listener
    if listener is None:
        return None
    try:
        return listener.triggered_keys()
    except Exception:
        return None


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
        self._spoil_due = 0.0
        self._start_menu_check_due = 0.0
        self._start_release_until = 0.0
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.GetAsyncKeyState.restype = ctypes.c_short
        self._user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        # Keys of a held chord that the app has released in the OS's key
        # state on the user's behalf. They keep the chord satisfied until
        # the rest of it comes up. See _take_super_out_of_the_hold.
        self._synthetic_down = set()
        # Whether a released key has been seen up since, which is the only
        # thing that makes a second release worth sending.
        self._super_came_back = False

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

    def status_snapshot(self) -> dict:
        return {"backend": "windows-poll", "alive": self.is_alive(),
                "held": len(self._press_triggered),
                "callbacks_pending": self._callbacks.qsize()}

    def triggered_keys(self) -> frozenset:
        """Every key of every binding currently firing."""
        keys = set()
        for name in self._press_triggered:
            binding = self._bindings.get(name)
            if binding:
                keys |= set(binding[0])
        return frozenset(keys)

    def start(self):
        global _active_listener
        _active_listener = self
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
        return down | self._synthetic_down

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
                physical = {VK_ALIASES.get(vk, vk)
                            for vk in watched if get(vk) & HELD_MASK}
                self._keep_super_out(physical)
                down = physical | self._synthetic_down
                super_vk = NAME_TO_VK["super"]
                if super_vk in down and super_vk not in previous:
                    # A new Windows-key press belongs to the user. Do not
                    # close a deliberately opened menu after dictation.
                    self._start_release_until = 0.0
                if down != previous:
                    self._check_bindings(down, rising=len(down) > len(previous))
                    previous = down
                self._keep_start_menu_spoiled(down)
                if self._close_start_menu_if_it_opened(down):
                    # The Escape that closed it is still in the key state,
                    # and Escape is the cancel key. Read it now and this
                    # would cancel the dictation it just rescued.
                    time.sleep(self._poll_interval)
                    continue
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

    def _keep_start_menu_spoiled(self, down: set) -> None:
        """Disarm the Start menu again, and keep doing it while the key is held.

        Windows arms Start on a Windows-key press and disarms it on any other
        key, so the single keystroke sent when the combination fires holds
        only until something presses Super again. Over a long dictation
        something does: the key chatters, or a hand shifts its grip, and
        neither is a release this listener can see - both are shorter than
        the poll interval, so the recording carries on while Windows has
        quietly armed the menu behind it. The next such blip opens Start,
        which takes the foreground, and the rest of the dictation has nowhere
        to go. Forty seconds of held key is forty seconds of that exposure,
        which is why it looked like something that happened after a while.

        Sending it again on a cadence bounds that to the cadence. It is a
        keystroke Windows reserves as doing nothing, and it goes out only
        while one of our own Super combinations is actually held - never
        while the user is merely holding the Windows key themselves, whose
        Start menu is theirs to open.

        Not at all once the Windows key has been taken out of the state
        table for the hold: then there is no Super press left to arm
        anything, the release that took it out is already wrapped in this
        same no-op key, and a key that comes back is released again the same
        way. See _take_super_out_of_the_hold.

        The poll loop skips this entirely while text is being injected, which
        is where the typing path does its own disarming.
        """
        keys = self.triggered_keys()
        super_vk = NAME_TO_VK["super"]
        if super_vk not in keys or super_vk not in down:
            return
        if self._synthetic_down:
            # The Windows key is out of the state table (see
            # _take_super_out_of_the_hold), so there is nothing left for
            # Windows to arm Start with. The release that took it out was
            # wrapped in the no-op key, and a key that comes back is
            # released again - also wrapped. Sending the keystroke on a
            # cadence here would only put ten unnecessary key events a
            # second into whatever the user is dictating text into.
            return
        now = time.monotonic()
        if now < self._spoil_due:
            return
        self._spoil_due = now + SPOIL_INTERVAL
        self._spoil_start_menu_if_needed(keys)

    def _close_start_menu_if_it_opened(self, down: set) -> bool:
        """Shut the Start menu if it appeared while the hotkey is held.

        The typing path already does this when it has words to deliver, and
        that is too late and too conditional: the menu opens during holds
        that commit nothing at all - a long pause, a quiet room - and then
        there is no pass to notice it. It stays up, takes the foreground, and
        the words that do arrive have nowhere to go.

        Here it is noticed within a fifth of a second of appearing, whatever
        the dictation is doing, and closing it puts the foreground back where
        it was. That is the symptom gone regardless of which keystroke opened
        it, which matters because three fixes aimed at those keystrokes have
        not stopped it; the log line it writes names what this process last
        injected and how long before, which is the evidence that has been
        missing.

        Returns whether a menu was actually closed, so the caller can leave
        the injected Escape alone.
        """
        keys = self.triggered_keys()
        super_vk = NAME_TO_VK["super"]
        now = time.monotonic()
        held = super_vk in keys and super_vk in down
        if not held and now >= self._start_release_until:
            return False
        if now < self._start_menu_check_due:
            return False
        self._start_menu_check_due = now + START_MENU_CHECK
        try:
            from . import system_win

            return system_win.dismiss_start_menu()
        except Exception as e:
            log.debug("could not close the Start menu: %s", e)
            return False

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
                    # Before the recording starts, not before the first word
                    # is typed: from here on the user is going to be speaking
                    # into an application with the chord still held.
                    self._take_super_out_of_the_hold(_keys)
                    if cb_press:
                        self._callbacks.put((name, "press", cb_press))
            elif name in self._press_triggered:
                self._press_triggered.discard(name)
                if not self._press_triggered:
                    # Nothing of ours is firing any more, so there is no
                    # chord left to keep the released Windows key held for.
                    self._synthetic_down.clear()
                    self._super_came_back = False
                if NAME_TO_VK["super"] in _keys:
                    # Start opens on release, precisely when triggered_keys
                    # stops containing Super. Keep the check alive briefly.
                    self._start_release_until = time.monotonic() + START_RELEASE_GRACE
                if cb_release:
                    self._callbacks.put((name, "release", cb_release))

    def _take_super_out_of_the_hold(self, keys) -> None:
        """Release the Windows key for the rest of a Super chord's hold.

        A held Windows key turns every keystroke into a shell combination:
        the text the app types goes to the shell's shortcut handling instead
        of the focused window, which minimises windows, opens Start, and
        switches desktops during a dictation. Releasing it once, here, is
        the only place it can be done without a re-press to follow - and a
        synthetic re-press is itself a fresh press whose release opens Start.

        Only for a chord with another key in it. The user's own release is
        then read from that other key, because GetAsyncKeyState cannot tell
        their key-up from ours once the Windows key is out of the table. A
        Super-only chord is left alone: it would be a recording that nothing
        can end.

        Best effort. If the release fails the dictation still runs, exactly
        as it did before this existed.
        """
        if not (keys - WIN_KEYS):
            return
        self._release_super()

    def _keep_super_out(self, physical: set) -> None:
        """Take Super back out if it came back while the chord is held.

        Once released, the Windows key is out of the state table, so it can
        only reappear if the user pressed it again or the key chattered.
        Either way the chord is still held and text is still being typed,
        and from that moment every character is a shell shortcut again - so
        it is released again.

        Only once it has actually been seen up in between. Releasing on every
        poll that still reports it down would be sixty keystroke pairs a
        second into the window being dictated into, and it would buy nothing:
        a release that the state table ignored is not going to start working
        on the next poll.
        """
        if not self._synthetic_down:
            return
        if SUPER_VK not in physical:
            self._super_came_back = True
            return
        if not self._super_came_back:
            return
        self._release_super()

    def _release_super(self) -> None:
        """Take every held Windows key out of the OS's key state, once."""
        try:
            from . import system_win

            released = system_win.strip_super()
        except Exception as e:
            log.debug("could not release the Windows key for the hold: %s", e)
            return
        if released:
            # Keep the chord looked held until the rest of it comes up.
            self._synthetic_down.add(SUPER_VK)
            # Nothing to take out again until it has been seen up once.
            self._super_came_back = False
            log.info("[HOTKEY] released the Windows key for the hold; "
                     "the chord stays held until the other keys are let go")

    def stop(self):
        global _active_listener
        if _active_listener is self:
            _active_listener = None
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._dispatch_thread:
            self._callbacks.put(None)
            self._dispatch_thread.join(timeout=2)
            self._dispatch_thread = None
        self._press_triggered.clear()
        self._synthetic_down.clear()
        self._super_came_back = False
