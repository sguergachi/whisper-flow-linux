"""Keyboard hotkey listener using evdev with uinput proxy for Wayland."""

import errno
import logging
import os
import queue
import select
import threading
import time

import evdev
from evdev import ecodes

from .logging import log as app_log

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

# Binding keys use the left code; hardware may have sent either side.
# Synthetic releases must clear BOTH, or the compositor keeps the right-side
# Super held while we type - Meta+B becomes power profile, Meta+W overview,
# Meta+D desktop, etc. That is how "super+alt dictation locked the desktop".
MODIFIER_SIDES = {
    ecodes.KEY_LEFTMETA: (ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA),
    ecodes.KEY_LEFTCTRL: (ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL),
    ecodes.KEY_LEFTALT: (ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT),
    ecodes.KEY_LEFTSHIFT: (ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT),
}

PROXY_NAME = "whisper-flow-keyboard-proxy"
BUS_VIRTUAL = 0x06  # uinput devices; see linux/input.h BUS_VIRTUAL

# How often to look for keyboards that were not there at startup. Cheap: it
# stats /dev/input and compares paths, and only rebuilds when the set differs.
RESCAN_SECONDS = 3.0

# The pump loop iterates ~10x a second even with no keys pressed (the select
# has a 0.1s timeout). If grabbed keyboards go longer than this without a
# pump iteration, the reader is wedged while holding exclusive access - the
# exact shape of "the desktop stopped responding to the keyboard".
PUMP_STALL_SECONDS = 1.5

# A soft stall only matters when there is input at risk. Freeing the keyboard
# means closing the grabbed fds, and closing them discards whatever the user
# typed during the stall - so doing it for a merely slow loop while nothing is
# held loses keystrokes and re-grabs in a cycle, which is exactly "keys
# randomly stop working". Past this much silence the loop is genuinely wedged
# and the keyboard is freed whether or not a key is down.
PUMP_HARD_STALL_SECONDS = 8.0

# Consecutive forwarded-event write failures that prove the uinput proxy is
# dead rather than glitching. One failure is noise; dozens in a row with
# events flowing means every keystroke is being swallowed.
FORWARD_FAIL_THRESHOLD = 30

# Emergency input reset: this many Escape presses inside the window, counted
# on raw key-downs (auto-repeat excluded), independent of any binding.
ESC_RESET_COUNT = 3
ESC_RESET_WINDOW = 1.2

# Recoveries inside a minute beyond this stand the listener down entirely:
# hotkeys stay dead but the keyboard is left free. A usable desktop always
# beats working hotkeys.
MAX_RECOVERIES_PER_MINUTE = 3

# Kernel reports a key held but no key event of any kind has arrived in this
# long: the reader is missing input (a device starved behind a stale fd set,
# a lost key-up) while still holding the grabs. A physically held key always
# produces auto-repeat events, so sustained silence with something held is
# never "the user sitting still" - it is the shape of a locked keyboard.
# Only fires while no hotkey is armed, so an exotic repeat-disabled setup can
# never cancel a live dictation; and like every other emergency it counts
# against the stand-down throttle, so a repeat-off machine flaps at most
# briefly before the keyboard is left free.
KEY_EVENT_STARVATION_SECONDS = 5.0

# Set WHISPER_FLOW_HOTKEY_DEBUG=1 to log the keys a hotkey is built from, and
# the state they are matched against, when one is pressed.
#
# Only keys that appear in a binding - the modifiers and space - are ever
# named. Logging every key would make this a keylogger writing into the
# journal, which is not a thing to leave running on someone's machine while
# they are away from it.
DEBUG_KEYS = os.environ.get("WHISPER_FLOW_HOTKEY_DEBUG") == "1"


def _grab_error(failures: list[tuple[str, BaseException]]) -> str:
    """Why start() could see keyboards and still grab none of them.

    The usual case is EBUSY: another whisper-flow already holds EVIOCGRAB,
    or a remapper (keyd, kanata). The old message named neither, so the
    tray just said the keyboards could not be grabbed.
    """
    if not failures:
        return "Cannot grab any keyboard devices"
    busy = any(getattr(e, "errno", None) == errno.EBUSY for _, e in failures)
    if busy:
        paths = ", ".join(path for path, _ in failures[:4])
        return (
            "Cannot grab the keyboard: another program already has it "
            "(another whisper-flow, keyd, or kanata). "
            f"Busy: {paths}. The holder shows in: "
            f"sudo lsof {failures[0][0]}"
        )
    detail = "; ".join(f"{path}: {err}" for path, err in failures[:3])
    return f"Cannot grab any keyboard devices ({detail})"


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
        self._muted = set()  # codes whose auto-repeat is suppressed while held
        # Muted modifiers pressed back for as long as a key outside every
        # binding is held: the compositor believes them down again so the
        # desktop shortcut that key forms with them can fire. They are
        # released and muted again the moment the extra key goes up.
        self._restored: set = set()
        # (binding_name, frozenset of held codes) already reported as almost
        # matching - cleared when no binding key is held.
        self._near_miss_logged: set = set()
        self._grab_failures: list[tuple[str, BaseException]] = []
        # Cancel is not a binding: bindings resolve most-specific-wins, so a
        # lone Escape could never fire while a push-to-talk combination is
        # held - which is the whole point of a cancel key. Set by the manager.
        self.escape_callback = None
        # Called as on_emergency(reason) when input had to be forcibly reset
        # (stalled reader, dead proxy). Set by the manager so the user gets
        # a tray notification instead of silence.
        self.on_emergency = None
        # Pump heartbeat: stamped every pass through the read loop, even with
        # no keys pressed. An idle keyboard and a wedged reader both forward
        # nothing, so only the stamp tells them apart.
        self._last_pump_at = 0.0
        self._forward_failures = 0
        self._esc_press_times: list = []
        self._supervisor_thread = None
        self._started_at = 0.0
        self._last_sweep_at = 0.0
        # Never, not zero: monotonic() is uptime, so 0.0 reads as "just
        # now" on a machine booted minutes ago (fresh CI runners included)
        # and the idle sweeper would fire having seen no hotkey at all.
        self._last_input_risk_at = float("-inf")
        self._recovery_times: list = []
        # Stood down after repeated failed recoveries: grabs released, reader
        # stopped, keyboard left entirely alone until the app restarts.
        self._listener_disabled = False
        # Compositor-view tracking for the reconciler below: logical codes we
        # forwarded as down and have not yet told the compositor are up
        # (real key-ups, or synthetic releases for muted binding keys, clear
        # them). Compared against kernel truth on every supervisor tick; a
        # key-up the compositor never got leaves it believing the key is held
        # - auto-repeating a single letter, or holding Super so every key is
        # a shortcut - which is exactly the reported "locked keyboard".
        self._forwarded_down: set = set()
        # Last arrival of any real key event (down, up, or auto-repeat) on
        # the reader thread. Unlike the pump stamp, which turns on an idle
        # keyboard too, this only moves when the hardware speaks - so kernel
        # truth saying "held" while this stands still proves input is being
        # missed, not that nobody is typing.
        self._last_key_event_at = float("-inf")
        # Last kernel-truth snapshot taken by the reconciler, for the
        # starvation check in the same tick without a second round of ioctls.
        self._last_kernel_held = None
        # Silent heals (reconciled stuck keys) inside the last minute, for
        # logs and diagnostics. Loud emergencies keep their own counter and
        # their tray notification; a heal that worked needs no toast.
        self._heal_times: list = []
        # A soft stall with nothing held was already reported (once per
        # stall): a slow reader on a loaded machine must not fill the log.
        self._stall_deferred = False

    def register_hotkey(self, name, key_string, callback_press, callback_release=None,
                        release_modifiers=False):
        """Register a binding.

        `release_modifiers` tells the compositor the combination's keys are no
        longer held once it fires. Push-to-talk needs it: the app types the
        transcription while the user is still holding the hotkey, and if the
        compositor thinks Super and Alt are down, every injected character
        arrives as a global shortcut - opening the launcher, starting a screen
        recording - instead of as text.
        """
        keys = self._parse_key_string(key_string)
        self._bindings[name] = (keys, callback_press, callback_release, release_modifiers)

    @property
    def _binding_codes(self) -> set:
        """Every key code that appears in some binding. Nothing else is named."""
        codes = set()
        for keys, *_ in self._bindings.values():
            codes |= set(keys)
        return codes

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

    def _create_proxy(self):
        """(Re)create the uinput proxy from the first keyboard's capabilities.

        Factored out of start() because recovery needs it too: every
        emergency frees the keyboard by releasing the grabs and dropping the
        proxy, and without re-creation here the read loop would re-grab the
        keyboards with nowhere to forward to - swallowing every keystroke
        while the threshold re-fired the emergency in a loop. Raises
        RuntimeError when no proxy can be made; the caller leaves the
        keyboard free and retries later.
        """
        kbd_info = self._find_keyboard_devices()
        if not kbd_info:
            raise RuntimeError("No keyboard devices found. Are you in the 'input' group?")
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
                self._uinput = None
                raise RuntimeError(f"Cannot create uinput proxy: {e}") from e

    def start(self):
        self._create_proxy()

        # Grab and open real keyboard devices
        if not self._open_devices():
            if self._uinput:
                self._uinput.close()
                self._uinput = None
            raise RuntimeError(_grab_error(self._grab_failures))

        self._running = True
        self._listener_disabled = False
        self._started_at = time.monotonic()
        self._last_pump_at = time.monotonic()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="whisper-flow-hotkey-dispatch",
        )
        self._dispatch_thread.start()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="whisper-flow-hotkey-reader",
        )
        self._thread.start()
        self._supervisor_thread = threading.Thread(
            target=self._supervisor_loop, daemon=True, name="whisper-flow-hotkey-supervisor",
        )
        self._supervisor_thread.start()

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
        """Pump events, rebuilding the device set whenever it changes.

        Keyboards come and go: a Bluetooth one reconnects, a dock or KVM is
        switched, a receiver is replugged. This used to return on the first
        such event and leave a comment about a supervisor rebuilding the
        device set - but nothing supervised it, so hotkeys stayed dead until
        the daemon was restarted, while typing kept working because the
        devices had been released. That is precisely the failure that looks
        like "the hotkey stopped working and I cannot see why".

        A keyboard plugged in after startup was never grabbed at all.
        """
        try:
            while self._running:
                if not self._run_once():
                    time.sleep(RESCAN_SECONDS)      # nothing to read yet
        finally:
            self._release_devices()

    def _run_once(self) -> bool:
        """One reader pass. False means "back off and try again".

        A pass can fail transiently - evdev.list_devices during a device
        re-enumeration, an InputDevice that vanished between the scan and the
        open. That must never kill the reader while it holds the grabs: a
        dead reader with a live grab is the locked keyboard this class exists
        to prevent, and the auto-heal that revives it tears down and rebuilds
        the grab and proxy, which is the churn that drops the user's
        keystrokes. Free the keyboard and let the loop try again instead.
        Never raises.
        """
        try:
            if not self._ensure_forwarding_path():
                return False
            self._pump_until_devices_change()
            # Always ungrab before rebuilding. Devices must never stay
            # grabbed by a loop that is no longer reading them, or the
            # user loses their keyboard entirely.
            self._abandon_devices()
            return True
        except Exception:
            log.exception("keyboard reader pass failed; freeing the keyboard")
            try:
                self._abandon_devices()
            except Exception:
                log.exception("failed to release devices after a reader error")
            return False

    def _ensure_forwarding_path(self) -> bool:
        """Proxy alive plus keyboards grabbed. False when there is nothing
        to read and the caller should sleep and retry.

        Two rules keep recovery from becoming a second outage. While stood
        down after repeated failures the keyboard is left free, period: no
        re-grab, so the stand-down notification ("keyboard left working")
        stays true instead of re-grabbing with no proxy and swallowing every
        keystroke until the next emergency. And the proxy is (re)created
        before grabbing, never after: grabs without a proxy to forward
        through are precisely the shape of "the keyboard went dead".
        """
        if self._listener_disabled:
            return False
        if self._uinput is None:
            try:
                self._create_proxy()
            except Exception as e:
                app_log(f"[HOTKEY] uinput proxy unavailable ({e}) - "
                        "leaving keyboard free, will retry")
                return False
        if not self._kbd_devices and not self._open_devices():
            return False
        return True

    def _pump_until_devices_change(self):
        """Read events until a device fails or the keyboard set changes."""
        fds = {dev.fd: dev for dev in self._kbd_devices}
        next_scan = time.monotonic() + RESCAN_SECONDS
        while self._running:
            # Heartbeat first: even an idle pass proves the loop is turning.
            # The supervisor treats a missing stamp with grabs held as a
            # wedged reader and frees the keyboard.
            self._last_pump_at = time.monotonic()
            try:
                r, _, _ = select.select(list(fds), [], [], 0.1)
            except (OSError, ValueError):
                log.info("keyboard device went away; rebuilding")
                return
            for fd in r:
                try:
                    for event in fds[fd].read():
                        if event.type == ecodes.EV_KEY:
                            self._handle_key(event)
                        else:
                            self._forward(event)
                except BlockingIOError:
                    pass
                except OSError:
                    log.info("keyboard device read failed; rebuilding")
                    return
                except Exception:
                    log.exception("error reading keyboard events")

            if time.monotonic() >= next_scan:
                next_scan = time.monotonic() + RESCAN_SECONDS
                if self._grabbed_paths() != self._keyboard_paths():
                    log.info("keyboard set changed; rebuilding")
                    return

    def _keyboard_paths(self) -> set:
        return {path for path, _ in self._find_keyboard_devices()}

    def _grabbed_paths(self) -> set:
        return {dev.path for dev in self._kbd_devices}

    def _abandon_devices(self):
        """Release the current devices and forget any state tied to them.

        A rebuild happens with keys possibly held. Leaving them in _key_state
        would mean the combination still looks pressed against a device that
        no longer exists, so it could never fire again - and any active
        push-to-talk has to be ended, or the recording it started never stops.
        """
        self._ungrab_devices()          # never the proxy: see _ungrab_devices
        # A rebuild with keys held is a desync-risk moment by definition.
        self._mark_input_risk()
        # Restored modifiers are compositor-held with no device left to prove
        # it: release them before the tracking that would have matched them
        # to a real key-up is gone, or they stay down forever.
        if self._restored:
            self._emit_key_ups(tuple(self._restored))
            self._restored.clear()
        for name in list(self._press_triggered):
            binding = self._bindings.get(name)
            self._press_triggered.discard(name)
            if binding and binding[2]:
                self._callbacks.put((name, "release", binding[2]))
        self._active_hotkey = None
        self._key_state.clear()
        self._muted.clear()
        self._forwarded_down.clear()
        self._last_kernel_held = None
        self._near_miss_logged.clear()

    def _open_devices(self) -> bool:
        """Grab every keyboard currently present. True if any were grabbed."""
        opened = []
        failures: list[tuple[str, BaseException]] = []
        for path, _ in self._find_keyboard_devices():
            try:
                dev = evdev.InputDevice(path)
                dev.grab()
                opened.append(dev)
            except Exception as e:
                failures.append((path, e))
                continue
        self._kbd_devices = opened
        self._grab_failures = failures
        if opened:
            log.info("grabbed %d keyboard(s)", len(opened))
        elif failures:
            for path, e in failures:
                app_log(f"[HOTKEY] could not grab {path}: {e}")
        return bool(opened)

    def _forward(self, event):
        """Pass an event through to the compositor via the uinput proxy."""
        try:
            self._uinput.write_event(event)
            if event.type == ecodes.EV_SYN:
                self._uinput.syn()
        except Exception:
            # Count, then act: one lost write is noise, but dozens in a row
            # with events flowing means the proxy is dead and every keystroke
            # is being swallowed while the grabs are still held.
            self._forward_failures += 1
            if self._forward_failures >= FORWARD_FAIL_THRESHOLD:
                failures = self._forward_failures
                self._forward_failures = 0
                if self._listener_disabled:
                    # Stood down already: the read loop no longer re-grabs,
                    # so these are strays, not a new outage. Recounting here
                    # turned one stand-down into a tray notification every
                    # second for the rest of the session.
                    return
                self._emergency_recover(
                    f"uinput proxy failing ({failures} consecutive writes lost)")
            return
        self._forward_failures = 0

    def pump_age_seconds(self) -> float | None:
        """Seconds since the reader last turned, or None before first pump."""
        if not self._last_pump_at:
            return None
        return time.monotonic() - self._last_pump_at

    def status_snapshot(self) -> dict:
        """Listener health for logs and diagnostics. No key names, only counts."""
        try:
            age = self.pump_age_seconds()
        except Exception:
            age = None
        return {
            "backend": "evdev",
            "alive": self.is_alive(),
            "grabbed": len(self._kbd_devices),
            "pump_age_s": round(age, 1) if age is not None else None,
            "muted": len(self._muted),
            "held": len(self._key_state),
            "forwarded": len(self._forwarded_down),
            "disabled": self._listener_disabled,
            "recoveries_60s": len([
                t for t in self._recovery_times
                if time.monotonic() - t < 60.0]),
            "healed_60s": len([
                t for t in self._heal_times
                if time.monotonic() - t < 60.0]),
        }

    def _supervisor_loop(self):
        """Free the keyboard if the reader ever wedges while holding grabs.

        Runs on its own thread because the thing being watched is the reader
        thread itself: a stall there must not also stall the rescue. Fires
        only while devices are actually grabbed - with nothing held there is
        nothing to save, however quiet the loop is.
        """
        while self._running:
            time.sleep(1.0)
            if not self._running:
                return
            try:
                self._supervise_once()
            except Exception:
                log.exception("hotkey supervisor failed")

    def _supervise_once(self) -> None:
        if not self._running or self._listener_disabled:
            return
        now = time.monotonic()
        thread = self._thread
        if thread is None or not thread.is_alive():
            # Grace for normal startup: the thread needs a moment to exist.
            if now - getattr(self, "_started_at", 0.0) < 3.0:
                return
            self._emergency_recover("reader thread died")
            return
        if not self._kbd_devices:
            return
        age = self.pump_age_seconds()
        if age is None or age <= PUMP_STALL_SECONDS:
            # The loop is turning. Reconcile what the compositor believes
            # against what the kernel reports, heal any stuck key silently,
            # and run the bounded unheld-modifier sweep - here, on the
            # listener's own thread, so it also runs mid-recording, when the
            # daemon's idle watchdog deliberately stays out of the way and a
            # lockup would otherwise sit until the dictation ends.
            try:
                self._reconcile_forwarded_with_kernel()
            except Exception:
                log.exception("input reconciliation failed")
            try:
                self.maybe_sweep_unheld_modifiers()
            except Exception:
                log.exception("input sweep failed")
            self._detect_event_starvation(now)
            self._stall_deferred = False
            return
        # A soft stall with nothing held is not an emergency. The reader is
        # slow - a loaded machine, the GIL, a long ioctl - but no input is
        # being lost by waiting, and freeing the keyboard would close the
        # grabbed fds (discarding anything typed in the meantime) only to
        # re-grab moments later. That cycle is the churn that reads as "keys
        # randomly stop working"; it is why this branch is silent. Only past
        # the hard threshold, or with a key actually held, is the loop
        # provably wedged and the keyboard freed.
        if age < PUMP_HARD_STALL_SECONDS:
            held = self._kernel_held_keys()
            if held is not None and not held:
                if not self._stall_deferred:
                    self._stall_deferred = True
                    app_log(f"[HOTKEY] reader slow ({age:.1f}s) with no key "
                            "held - keeping the keyboard, will retry")
                return
        self._emergency_recover(f"reader stalled ({age:.1f}s without pumping)")

    def _detect_event_starvation(self, now: float) -> None:
        """Recover when the kernel holds keys but no events arrive.

        A physically held key always produces auto-repeat events, so kernel
        truth saying "something is down" while no key event of any kind has
        arrived for seconds proves the reader is missing input behind grabs
        it still holds - the keyboard looks locked and no hotkey can fire
        because the release that would end it never arrives either.
        Skipped while a hotkey is armed (never cancel a live dictation on a
        repeat-disabled setup's account) and shortly after startup.
        """
        if self._press_triggered or self._active_hotkey is not None:
            return
        if now - getattr(self, "_started_at", 0.0) < KEY_EVENT_STARVATION_SECONDS:
            return
        held = getattr(self, "_last_kernel_held", None)
        if not held:
            return
        silent = now - getattr(self, "_last_key_event_at", float("-inf"))
        if silent <= KEY_EVENT_STARVATION_SECONDS:
            return
        self._emergency_recover(
            f"keyboard events starved ({len(held)} key(s) held, "
            f"no events for {silent:.0f}s)")

    def _notify_emergency(self, reason: str) -> None:
        cb = self.on_emergency
        if cb is None:
            return
        try:
            cb(reason)
        except Exception:
            log.exception("emergency callback failed")

    def _emergency_recover(self, reason: str, *, count_recovery: bool = True,
                             keep_proxy: bool = False) -> None:
        """Last resort for wedged input: free the keyboard first, ask later.

        Order is deliberate. Ungrabbing restores the user's keyboard
        immediately no matter what else is broken; everything after that is
        best-effort recovery. Pending push-to-talk releases are still fired
        so a recording started before the wedge stops instead of running on.

        `keep_proxy` is for user-invoked resets (triple-Esc): the proxy is
        presumably healthy, so it is kept and only the grabs are released -
        closing it would leave the read loop re-grabbing with nowhere to
        forward to, swallowing every keystroke until the failure threshold
        re-fired this very function in a loop. Genuine proxy deaths pass
        False: the read loop recreates the proxy before re-grabbing.
        """
        if self._listener_disabled:
            # Already stood down: make sure the keyboard is free and return
            # silently. No recount, no re-notify, no restart - without this,
            # the forward-failure path re-entered here on every 30th
            # swallowed keystroke and one stand-down became a tray
            # notification every second for the rest of the session.
            try:
                self._ungrab_devices()
            except Exception:
                log.exception("emergency device release failed")
            return
        app_log(f"[HOTKEY] EMERGENCY input reset: {reason} — releasing all grabs")
        now = time.monotonic()
        if count_recovery:
            self._recovery_times = [
                t for t in self._recovery_times if now - t < 60.0]
            self._recovery_times.append(now)
        # 1. Free the keyboard immediately, whatever else is broken.
        try:
            if keep_proxy:
                # Keeps the uinput proxy: _abandon_devices fires the pending
                # push-to-talk releases itself, so step 2 below is a no-op.
                self._abandon_devices()
            else:
                self._release_devices()
        except Exception:
            log.exception("emergency device release failed")
        # 2. Drop all tracking state, firing pending releases first.
        for name in list(self._press_triggered):
            binding = self._bindings.get(name)
            self._press_triggered.discard(name)
            if binding and binding[2]:
                self._callbacks.put((name, "release", binding[2]))
        self._active_hotkey = None
        self._key_state.clear()
        self._muted.clear()
        # _abandon_devices / _release_devices already owed the compositor
        # those releases; this is only so no stale code survives a rebuild.
        self._restored.clear()
        self._forwarded_down.clear()
        self._near_miss_logged.clear()
        self._esc_press_times.clear()
        self._forward_failures = 0
        # 3. Too many recoveries: stand down entirely. Hotkeys stay dead but
        # the keyboard is left free - a usable desktop always wins.
        if count_recovery and len(self._recovery_times) > MAX_RECOVERIES_PER_MINUTE:
            self._listener_disabled = True
            app_log("[HOTKEY] input recovery keeps failing — hotkeys disabled, "
                    "keyboard left free. Restart the app to re-enable.")
            self._notify_emergency(
                "hotkeys disabled after repeated input failures — "
                "keyboard left working, restart the app")
            return
        # 4. Come back: restart the reader if it died; the read loop
        # recreates the proxy and re-grabs devices by itself on its rescan
        # path (and stays stood down without re-grabbing when disabled).
        try:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._thread = threading.Thread(
                    target=self._read_loop, daemon=True,
                    name="whisper-flow-hotkey-reader",
                )
                self._thread.start()
                app_log("[HOTKEY] reader thread restarted after emergency reset")
        except Exception:
            log.exception("emergency reader restart failed")
        self._notify_emergency(f"keyboard input reset ({reason})")

    def _kernel_held_keys(self) -> set[int] | None:
        """Kernel truth across every grabbed device, aliased. None if unknown.

        EVIOCGKEY per device, merged - the same source _sync uses. A device
        without active_keys (test fakes) is skipped rather than trusted; None
        means "could not ask", never "nothing is held".
        """
        if not self._kbd_devices:
            return None
        merged: set[int] = set()
        any_ok = False
        for dev in self._kbd_devices:
            active = getattr(dev, "active_keys", None)
            if active is None:
                continue
            try:
                for code in active():
                    merged.add(CODE_ALIASES.get(code, code))
                any_ok = True
            except Exception:
                continue
        return merged if any_ok else None

    def _sync_key_state_from_devices(self) -> bool:
        """Rebuild held keys from the kernel across every grabbed device.

        Dual-HID boards (SONiX KN85, many Apple-vendor keyboards) put
        modifiers on one interface and other keys on another. Tracking only
        the events we see can desync when interfaces disagree or a key-up is
        lost; EVIOCGKEY is the kernel's combined truth per device, and
        merging them is what makes ctrl+alt+space match when Space arrives
        on a different node than Ctrl and Alt.

        Returns True when at least one device answered (state was replaced).
        Tests and the brief window before any grab keep event-tracked state.
        """
        held = self._kernel_held_keys()
        if held is None:
            return False
        self._key_state = held
        return True

    def _emit_key_ups(self, codes) -> None:
        """Best-effort key-ups via the proxy. A duplicate up is ignored by
        the compositor; a missing one strands the key - so errors are
        swallowed and only ups are ever synthesised here, never presses."""
        if self._uinput is None:
            return
        for code in codes:
            for side in MODIFIER_SIDES.get(code, (code,)):
                try:
                    self._uinput.write(ecodes.EV_KEY, side, 0)
                except Exception:
                    pass
        try:
            self._uinput.syn()
        except Exception:
            pass

    def _restore_muted_modifiers(self, code) -> int:
        """Give a muted push-to-talk chord back to the compositor.

        While a binding with release_modifiers is held, its keys are muted at
        the compositor so dictated text is text, not shortcuts. That also
        hides the chord from every desktop shortcut built on the same
        modifiers: Super+Alt held for dictation means Meta+Alt+Arrow
        ("Switch Window") never fires, and the desktop looks deaf to Super.
        A key outside every binding says the user is driving the desktop, not
        dictating, so the still-held muted modifiers are pressed back - before
        that key is forwarded, or the compositor matches a bare arrow and
        nothing happens. Only modifiers are restored, and only ones the kernel
        reports physically down: the muted Space of a single-press binding is
        not a key anyone holds through a desktop detour, and pressing it back
        would type a space.
        """
        if not self._muted or code in self._binding_codes:
            return 0
        restored = 0
        for logical in list(self._muted):
            if logical not in MODIFIER_SIDES or logical not in self._key_state:
                continue
            self._muted.discard(logical)
            self._restored.add(logical)
            # Tracked as compositor-held so a lost release is reconciled
            # against kernel truth like any forwarded key.
            self._forwarded_down.add(logical)
            for side in MODIFIER_SIDES[logical]:
                try:
                    self._uinput.write(ecodes.EV_KEY, side, 1)
                except Exception:
                    pass
            restored += 1
        if restored:
            try:
                self._uinput.syn()
            except Exception:
                pass
            self._mark_input_risk()
            app_log(f"[HOTKEY] restored {restored} muted modifier(s) for a "
                    "desktop shortcut while dictation was held")
        return restored

    def _settle_restored(self, code) -> None:
        """Undo the temporary restore around a real key-up.

        Two obligations. A restored modifier the user just released must be
        told up on both sides - the real key-up only covers the physical side,
        and the other one would otherwise stay held in the compositor, which
        is exactly the stranded modifier this class exists to avoid. And once
        the last extra key is up, the restored modifiers go back to being
        muted: the dictation this detour cancelled may still type its text a
        moment later, and with the chord back down every character would
        arrive as a global shortcut instead.
        """
        if not self._restored:
            return
        logical = CODE_ALIASES.get(code, code)
        released = []
        if logical in self._restored:
            self._restored.discard(logical)
            self._forwarded_down.discard(logical)
            released.append(logical)
        if self._key_state - self._binding_codes:
            # Another extra key is still held: the chord stays live so its
            # shortcut works, and this key's release is all that is owed.
            if released:
                self._emit_key_ups(released)
            return
        for restored_logical in list(self._restored):
            self._restored.discard(restored_logical)
            self._forwarded_down.discard(restored_logical)
            if restored_logical in self._key_state:
                # Physically held: tell the compositor it is up again, and
                # mute it so the auto-repeat and the coming transcription
                # cannot re-assert it. The real release still clears it.
                self._muted.add(restored_logical)
            released.append(restored_logical)
        if released:
            self._emit_key_ups(released)
            self._mark_input_risk()
            log.info("muted %d modifier(s) again after the desktop shortcut",
                     len(released))

    def _reconcile_forwarded_with_kernel(self) -> int:
        """Heal keys the compositor believes held but the kernel reports up.

        Two lockup shapes land here. A forwarded key-down whose key-up never
        reached the proxy (lost between a device rebuild, a wedged moment, a
        missed read) leaves the compositor auto-repeating a single letter -
        the "second key held forever" half of the report. A muted modifier
        whose synthetic release was lost on a glitching proxy - or whose
        stale mute outlived its real release - leaves Super held so every
        keystroke fires a global shortcut - the "desktop looks locked" half.
        Both heal the same way: the kernel says UP, so send a key-up. The
        compositor ignores a duplicate; a key still physically held (kernel
        DOWN) is never touched, so the intentional mute of a held push-to-talk
        chord survives this untouched. Silent by design: a heal that worked
        is a line in the log, not a toast.
        """
        if self._uinput is None or not self._running:
            return 0
        held = self._kernel_held_keys()
        self._last_kernel_held = held
        if held is None:
            return 0
        # Kernel truth is newer than our edge tracking (events originate in
        # the kernel, so it always knows first): adopt it, as a key event
        # would. Otherwise a ghost modifier lingering here keeps the sweep
        # skipping it and the matcher believing it, until some unrelated key
        # happens to resync.
        self._key_state = set(held)
        healed = 0
        for code in list(self._forwarded_down):
            if code in held or code in self._muted:
                # Still held, or synthetically released already: the muted
                # loop below owns that case, so it is healed exactly once.
                continue
            self._emit_key_ups((code,))
            self._forwarded_down.discard(code)
            # A restore whose real key-up was missed: the compositor is owed
            # the release, and the logical is not live any more either.
            self._restored.discard(code)
            healed += 1
        for code in list(self._muted):
            if code in held:
                continue
            self._muted.discard(code)
            self._forwarded_down.discard(code)
            self._emit_key_ups((code,))
            healed += 1
        if healed:
            now = time.monotonic()
            self._heal_times = [t for t in self._heal_times if now - t < 60.0]
            self._heal_times.append(now)
            self._mark_input_risk()
            app_log(f"[HOTKEY] healed {healed} stuck key(s) against kernel truth")
        return healed

    def _handle_key(self, event):
        """Track state and forward.

        Every real key-down and key-up is forwarded untouched. Synthetic
        presses exist in exactly one, bounded place: a muted push-to-talk
        modifier is pressed back for as long as an extra key is held, because
        the shortcut that key forms with the chord must fire (see
        _restore_muted_modifiers). Those presses are only ever made for keys
        the kernel reports physically down, and are released on the extra
        key's key-up, the physical release, a device rebuild, or an emergency
        reset - an earlier version replayed held-back keys and one missed
        release stranded a modifier system-wide, which is the failure class
        every path here is written against.
        """
        code = CODE_ALIASES.get(event.code, event.code)
        value = event.value
        self._last_key_event_at = time.monotonic()

        if value == 2:
            # Auto-repeat. Never a state change: holding a push-to-talk
            # combination generates these continuously, and treating one as a
            # release ends the recording about half a second after it starts.
            # Suppressed for muted keys so they are not re-asserted as held
            # after we told the compositor they were released.
            if code not in self._muted:
                self._forward(event)
            return

        if value == 1:
            # Kernel state first (multi-device); fall back to edge tracking
            # when nothing is grabbed yet (unit tests). The extra-key check
            # below reads it, and must run before anything is forwarded: the
            # compositor has to see the chord's modifiers down before the key
            # that forms the shortcut with them.
            if not self._sync_key_state_from_devices():
                self._key_state.add(code)
            self._restore_muted_modifiers(code)
            self._forward(event)
            # The compositor believes this key down until a matching up.
            self._forwarded_down.add(code)
            if code in self._binding_codes:
                # Hotkey-adjacent activity re-arms the idle hygiene sweep.
                self._mark_input_risk()
            if code == ecodes.KEY_ESC:
                self._track_escape_for_reset()
                if self.escape_callback:
                    self._callbacks.put(("escape", "press", self.escape_callback))
            else:
                self._reset_esc_counter_on_other_key(code)
            if DEBUG_KEYS and code in self._binding_codes:
                from .logging import log as _log
                _log(f"[HOTKEY-DEBUG] hotkey key down code={code} "
                     f"state={sorted(self._key_state)} "
                     f"want={{{', '.join(f'{n}:{sorted(k)}' for n, (k, *_) in self._bindings.items())}}}")
            self._check_bindings(rising=True)
            return

        # Drop the key before re-evaluating, otherwise the combination still
        # looks held and the release callback never fires.
        if not self._sync_key_state_from_devices():
            self._key_state.discard(code)
        self._muted.discard(code)
        self._forwarded_down.discard(code)
        self._check_bindings(rising=False)
        # Forwarded even if we already sent a synthetic release: a duplicate
        # key-up is harmless, a missing one is not.
        self._forward(event)
        # After the real key-up, so the compositor sees the key go up with
        # the chord still down - the mirror of the restore before its press.
        self._settle_restored(code)

    def _track_escape_for_reset(self) -> None:
        """Count quick Escape taps for the emergency input reset.

        Raw key-downs only (auto-repeat is filtered before this runs), and
        any other key-down in between restarts the count - it is in
        _handle_key's value==1 path next to this call. On the third tap
        inside the window, reset all input state unconditionally. Each tap
        still fires the normal cancel callback as well; during a real
        lockup that callback may go nowhere, but the reset below runs
        synchronously on the reader thread, which is what frees the user.
        """
        now = time.monotonic()
        self._esc_press_times = [
            t for t in self._esc_press_times if now - t <= ESC_RESET_WINDOW]
        self._esc_press_times.append(now)
        if len(self._esc_press_times) >= ESC_RESET_COUNT:
            self._esc_press_times.clear()
            self._emergency_input_reset()

    def _reset_esc_counter_on_other_key(self, code) -> None:
        if code != ecodes.KEY_ESC and self._esc_press_times:
            self._esc_press_times.clear()

    def _emergency_input_reset(self) -> None:
        """User-invoked full input reset (triple-Esc).

        Releases every modifier on both physical sides without trusting any
        tracked state - the whole point is that the state may be the broken
        part - then runs the same release-and-recover path as a detected
        stall, keeping a presumably healthy proxy: closing it here is what
        used to turn a manual rescue into a proxy-death spiral, the read
        loop re-grabbing with nowhere to forward to. Does not count against
        the failure throttle: a person asking for a reset is not a fault.
        """
        app_log("[HOTKEY] triple-Esc emergency input reset requested")
        if self._uinput is not None:
            for _logical, sides in MODIFIER_SIDES.items():
                for side in sides:
                    try:
                        self._uinput.write(ecodes.EV_KEY, side, 0)
                    except Exception:
                        pass
            try:
                self._uinput.syn()
            except Exception:
                pass
        self._emergency_recover("triple-Esc requested", count_recovery=False,
                                keep_proxy=True)

    def sweep_unheld_modifiers(self) -> int:
        """Release modifiers we do NOT think are held.

        Heals the compositor-side desync where it holds Super (or friends)
        while our tracked state is clean - the shape of "every key opens a
        shortcut and the desktop looks locked". A key-up for an already-up
        key is ignored by the compositor, so a healthy machine cannot tell
        this ran; physically held keys are in _key_state and are skipped, so
        a shortcut the user is holding is never broken. Safe any time,
        including mid-recording: held keys are skipped, and only key-ups -
        never presses - are synthesised.
        """
        if self._uinput is None or not self._running:
            return 0
        released = 0
        for logical, sides in MODIFIER_SIDES.items():
            if logical in self._key_state:
                continue
            for side in sides:
                try:
                    self._uinput.write(ecodes.EV_KEY, side, 0)
                    released += 1
                except Exception:
                    pass
            # Whatever we just told the compositor is up is no longer
            # compositor-held by our account, nor worth muting: a mute for an
            # unheld key would skip the synthetic release the next hold
            # needs, stranding that next hold instead.
            self._forwarded_down.discard(logical)
            self._muted.discard(logical)
            self._restored.discard(logical)
        if released:
            try:
                self._uinput.syn()
            except Exception:
                pass
        return released

    # Sweep policy: at most one hygiene pass a minute, and only within ten
    # minutes of touching hotkey state ourselves. Unbounded sweeping would
    # log forever on an idle desktop; no window at all would leave a stuck
    # modifier unswept exactly when nobody is pressing anything. Mashing
    # Super to unstick things re-arms the window via the key-down stamp.
    SWEEP_MIN_INTERVAL = 60.0
    SWEEP_RISK_WINDOW = 600.0

    def maybe_sweep_unheld_modifiers(self) -> int:
        """Bounded version of the sweep for the watchdog paths.

        At most one hygiene pass a minute, and only shortly after hotkey
        activity re-armed it - unbounded sweeping would log forever on an
        idle desktop. Runs on the listener's supervisor thread as well as
        the daemon's idle watchdog, so a stuck modifier heals mid-recording
        too instead of sitting until the dictation ends.
        """
        now = time.monotonic()
        if now - getattr(self, "_last_sweep_at", 0.0) < self.SWEEP_MIN_INTERVAL:
            return 0
        if now - getattr(self, "_last_input_risk_at", 0.0) > self.SWEEP_RISK_WINDOW:
            return 0
        self._last_sweep_at = now
        swept = self.sweep_unheld_modifiers()
        if swept:
            app_log(f"[HOTKEY] swept {swept} unheld modifier releases (idle hygiene)")
        return swept

    def _mark_input_risk(self) -> None:
        self._last_input_risk_at = time.monotonic()

    def _release_to_compositor(self, keys):
        """Tell the compositor a held combination has been released.

        Only releases are synthesised, never presses. By the time this runs
        the combination is complete, so the compositor has seen at least two
        keys go down - it reads as Super+Alt being released, not as a bare
        Super tap, which is what opens the launcher.

        Each logical modifier is released on both physical sides. Bindings
        store LEFTMETA/LEFTALT; if the keyboard actually sent RIGHTMETA (or
        the dual-HID path did), releasing only the left code leaves Super
        stuck held. The next typed character then fires Meta+letter global
        shortcuts (power profile, Overview, Peek at Desktop, …) instead of
        landing as text - and the desktop looks "locked" until Super is
        released for real.
        """
        self._mark_input_risk()
        for code in keys:
            logical = CODE_ALIASES.get(code, code)
            if logical not in self._key_state or logical in self._muted:
                continue
            self._muted.add(logical)
            # The compositor believes this key up now, not down - and a
            # temporary restore of the same logical is over either way.
            self._forwarded_down.discard(logical)
            self._restored.discard(logical)
            for side in MODIFIER_SIDES.get(logical, (logical,)):
                try:
                    self._uinput.write(ecodes.EV_KEY, side, 0)
                    self._uinput.syn()
                except Exception:
                    pass

    def _check_bindings(self, rising: bool):
        # Only the most specific satisfied binding wins, so cmd+shift+alt does
        # not also fire the cmd+alt binding nested inside it.
        #
        # Any held key outside every binding cancels: the user is pressing
        # something else - typically a virtual-desktop shortcut sharing our
        # modifiers (Super+Alt held for dictation, then an arrow to switch
        # desktops). While such an extra key is held no binding wins, so an
        # active push-to-talk is released instead of continuing with its
        # modifiers muted - which forwarded the extra key bare, breaking the
        # desktop shortcut and repeating the bare key like a held-key lock
        # ("the shortcut I pressed along with a second key to switch virtual
        # desktops locked my keyboard and held a single key"). ESC counts as
        # an extra here too: it still fires its own cancel callback from
        # _handle_key, and ending the held push-to-talk alongside it leaves
        # nothing stranded. Keys that belong to some binding - Shift stepping
        # transcribe up to command - are not extras and keep nested behavior.
        extras = self._key_state - self._binding_codes
        if extras:
            winner = None
            # Cancelling with keys held is a desync-risk moment by definition:
            # the extra key was just forwarded bare while the binding's
            # modifiers stay muted, so arm the idle hygiene sweep.
            if self._press_triggered:
                self._mark_input_risk()
        else:
            satisfied = [
                (name, keys)
                for name, (keys, _, _, _) in self._bindings.items()
                if keys and keys.issubset(self._key_state)
            ]
            winner = None
            if satisfied:
                winner = max(satisfied, key=lambda item: len(item[1]))[0]

        if rising and winner is None and not extras:
            self._log_near_miss()

        for name, (keys, cb_press, cb_release, release_mods) in self._bindings.items():
            if name == winner:
                # Only arm on a key-down. Otherwise releasing shift out of
                # cmd+shift+alt would "fall through" and fire cmd+alt, starting
                # a dictation the user never asked for.
                if name not in self._press_triggered and rising:
                    self._press_triggered.add(name)
                    self._active_hotkey = name
                    if release_mods:
                        self._release_to_compositor(keys)
                    if cb_press:
                        self._callbacks.put((name, "press", cb_press))
            elif name in self._press_triggered:
                self._press_triggered.discard(name)
                if self._active_hotkey == name:
                    self._active_hotkey = None
                if cb_release:
                    self._callbacks.put((name, "release", cb_release))

    def _log_near_miss(self) -> None:
        """When a chord is one key short, say so - once per incomplete hold.

        Auto-transcribe looked dead when Ctrl+Alt+Space never all landed in
        state at once (dual-HID boards, or the user releasing a modifier
        early). Without this, the journal only showed isolated key downs and
        no "why didn't it fire".
        """
        if not self._key_state or not self._bindings:
            return
        # Clear memory when nothing from any binding is held.
        binding_keys = set()
        for keys, *_ in self._bindings.values():
            binding_keys |= set(keys)
        if not (self._key_state & binding_keys):
            self._near_miss_logged.clear()
            return
        from .logging import log as _log
        for name, (keys, *_rest) in self._bindings.items():
            if not keys or keys.issubset(self._key_state):
                continue
            held = frozenset(keys & self._key_state)
            if not held or len(held) < len(keys) - 1:
                continue
            missing = sorted(keys - self._key_state)
            tag = (name, held)
            if tag in self._near_miss_logged:
                continue
            self._near_miss_logged.add(tag)
            _log(f"[HOTKEY] almost {name}: held {sorted(held)} "
                 f"missing {missing} (need {sorted(keys)})")

    def _ungrab_devices(self):
        """Let go of the keyboards, keeping the uinput proxy alive.

        The proxy must outlive a rebuild. Everything read from a grabbed
        keyboard is forwarded through it, so closing it while any device is
        still grabbed would swallow the user's typing entirely - the failure
        this code has already caused once and must never cause again.
        """
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

    def _release_devices(self):
        """Full teardown: let go of the keyboards and drop the proxy."""
        self._ungrab_devices()
        if self._uinput:
            try:
                self._uinput.close()
            except Exception:
                pass
            self._uinput = None
        # Closing the proxy removes it from the compositor, so any restored
        # modifier is forgotten there along with every other key it held.
        self._restored.clear()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._supervisor_thread:
            self._supervisor_thread.join(timeout=2)
            self._supervisor_thread = None
        # _read_loop releases devices on its way out; do it here too in case it
        # never started.
        self._release_devices()
        if self._dispatch_thread:
            self._callbacks.put(None)
            self._dispatch_thread.join(timeout=2)
            self._dispatch_thread = None
        self._key_state.clear()
        self._press_triggered.clear()
        self._muted.clear()
        self._restored.clear()
        self._forwarded_down.clear()
        self._last_kernel_held = None
        self._active_hotkey = None
