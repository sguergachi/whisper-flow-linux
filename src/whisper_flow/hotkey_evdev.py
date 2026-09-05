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
        return (
            "Cannot grab the keyboard: another program already has it "
            "(another whisper-flow, keyd, or kanata)"
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

    def start(self):
        kbd_info = self._find_keyboard_devices()
        if not kbd_info:
            raise RuntimeError("No keyboard devices found. Are you in the 'input' group?")

        # Create uinput proxy from the first keyboard's capabilities
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
                raise RuntimeError(f"Cannot create uinput proxy: {e}") from e

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
                if not self._kbd_devices and not self._open_devices():
                    time.sleep(RESCAN_SECONDS)      # nothing to read yet
                    continue
                self._pump_until_devices_change()
                # Always ungrab before rebuilding. Devices must never stay
                # grabbed by a loop that is no longer reading them, or the
                # user loses their keyboard entirely.
                self._abandon_devices()
        finally:
            self._release_devices()

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
        for name in list(self._press_triggered):
            binding = self._bindings.get(name)
            self._press_triggered.discard(name)
            if binding and binding[2]:
                self._callbacks.put((name, "release", binding[2]))
        self._active_hotkey = None
        self._key_state.clear()
        self._muted.clear()
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
            "disabled": self._listener_disabled,
            "recoveries_60s": len([
                t for t in self._recovery_times
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
            return
        self._emergency_recover(f"reader stalled ({age:.1f}s without pumping)")

    def _notify_emergency(self, reason: str) -> None:
        cb = self.on_emergency
        if cb is None:
            return
        try:
            cb(reason)
        except Exception:
            log.exception("emergency callback failed")

    def _emergency_recover(self, reason: str, *, count_recovery: bool = True) -> None:
        """Last resort for wedged input: free the keyboard first, ask later.

        Order is deliberate. Ungrabbing restores the user's keyboard
        immediately no matter what else is broken; everything after that is
        best-effort recovery. Pending push-to-talk releases are still fired
        so a recording started before the wedge stops instead of running on.
        """
        app_log(f"[HOTKEY] EMERGENCY input reset: {reason} — releasing all grabs")
        now = time.monotonic()
        if count_recovery:
            self._recovery_times = [
                t for t in self._recovery_times if now - t < 60.0]
            self._recovery_times.append(now)
        # 1. Free the keyboard immediately, whatever else is broken.
        try:
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
        # re-grabs devices by itself on its rescan path.
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
        if not self._kbd_devices:
            return False
        merged: set[int] = set()
        any_ok = False
        for dev in self._kbd_devices:
            try:
                for code in dev.active_keys():
                    merged.add(CODE_ALIASES.get(code, code))
                any_ok = True
            except Exception:
                continue
        if any_ok:
            self._key_state = merged
        return any_ok

    def _handle_key(self, event):
        """Track state and forward.

        Two rules keep this safe. Every key-down and key-up is forwarded
        untouched, and no key-down is ever synthesised. An earlier version
        held combination keys back and replayed them later; one missed release
        left a synthetic press with no matching release, stranding a modifier
        down system-wide and making the keyboard unusable. A dropped
        auto-repeat cannot do that - the key is already down as far as the
        compositor is concerned, and its real release is still coming.
        """
        code = CODE_ALIASES.get(event.code, event.code)
        value = event.value

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
            self._forward(event)
            # Kernel state first (multi-device); fall back to edge tracking
            # when nothing is grabbed yet (unit tests).
            if not self._sync_key_state_from_devices():
                self._key_state.add(code)
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
        self._check_bindings(rising=False)
        # Forwarded even if we already sent a synthetic release: a duplicate
        # key-up is harmless, a missing one is not.
        self._forward(event)

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
        stall. Does not count against the failure throttle: a person asking
        for a reset is not a fault.
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
        self._emergency_recover("triple-Esc requested", count_recovery=False)

    def sweep_unheld_modifiers(self) -> int:
        """Release modifiers we do NOT think are held.

        Heals the compositor-side desync where it holds Super (or friends)
        while our tracked state is clean - the shape of "every key opens a
        shortcut and the desktop looks locked". A key-up for an already-up
        key is ignored by the compositor, so a healthy machine cannot tell
        this ran; physically held keys are in _key_state and are skipped, so
        a shortcut the user is holding is never broken. Only called while
        the daemon is idle, never mid-recording.
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
        """Bounded version of the sweep for the idle watchdog path."""
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
            for side in MODIFIER_SIDES.get(logical, (logical,)):
                try:
                    self._uinput.write(ecodes.EV_KEY, side, 0)
                    self._uinput.syn()
                except Exception:
                    pass

    def _check_bindings(self, rising: bool):
        # Only the most specific satisfied binding wins, so cmd+shift+alt does
        # not also fire the cmd+alt binding nested inside it.
        satisfied = [
            (name, keys)
            for name, (keys, _, _, _) in self._bindings.items()
            if keys and keys.issubset(self._key_state)
        ]
        winner = None
        if satisfied:
            winner = max(satisfied, key=lambda item: len(item[1]))[0]

        if rising and winner is None:
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
        self._active_hotkey = None
