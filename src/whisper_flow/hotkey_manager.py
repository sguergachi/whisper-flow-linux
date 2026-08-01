"""Advanced hotkey management for whisper-flow with optimized key handling."""

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from pynput import keyboard

from .logging import log


def _is_wayland():
    """Check if running under Wayland."""
    return os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland"


def _parse_hotkey(keys_str, key_mapping):
    """Parse a hotkey string like 'cmd+alt' into a set of normalized key names."""
    parts = keys_str.lower().split("+")
    key_set = set()
    for part in parts:
        part = part.strip()
        mapped = key_mapping.get(part, part)
        key_set.add(mapped)
    return key_set


class HotkeyMode(Enum):
    """Hotkey activation modes."""

    SINGLE_PRESS = "single_press"  # Triggered once on key combination
    PUSH_TO_TALK = "push_to_talk"  # Triggered on press, released on key release


@dataclass
class HotkeyBinding:
    """Configuration for a hotkey binding."""

    keys: set[str]
    mode: HotkeyMode
    callback_press: Callable[[], None] | None = None
    callback_release: Callable[[], None] | None = None
    priority: int = 0  # Higher priority = checked first
    description: str = ""


class HotkeyManager:
    """Simple, robust hotkey manager with minimal complexity."""

    def __init__(self, debounce_delay: float = 0.05):
        """Initialize the hotkey manager.

        Args:
            debounce_delay: Minimum time between key events (seconds)

        """
        self.debounce_delay = debounce_delay

        # State tracking
        self.pressed_keys: set[str] = set()
        self.active_bindings: dict[str, HotkeyBinding] = {}
        self.is_running = False
        self.keyboard_listener: keyboard.Listener | None = None
        self._evdev_listener = None

        # Simple state
        self.last_key_times: dict[str, float] = {}  # Track per-key debouncing
        self.key_press_times: dict[str, float] = {}  # Track when each key was pressed (for stuck detection)
        self.active_combination: str | None = None
        self.current_push_to_talk: str | None = None
        self.processing_callback: Callable[[], bool] | None = None

        # Heartbeat monitoring
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 10.0  # 10 seconds between heartbeats
        self.heartbeat_thread = None
        self.stuck_key_timeout = 15.0  # Max time a key can be held before considered stuck

        # Key mapping for consistent naming (runtime keys have _l/_r suffixes)
        self.key_mapping = {
            "ctrl_l": "ctrl",
            "ctrl_r": "ctrl",
            "cmd_l": "cmd",
            "cmd_r": "cmd",
            "super_l": "cmd",
            "super_r": "cmd",
            "alt_l": "alt",
            "alt_r": "alt",
            "shift_l": "shift",
            "shift_r": "shift",
            "space": "space",
            # Shorthand names used in config
            "super": "cmd",
            "win": "cmd",
            "command": "cmd",
        }

        log("[HOTKEY] HotkeyManager initialized")

    def register_processing_callback(self, callback: Callable[[], bool]) -> None:
        """Register callback to check if system is processing.

        Args:
            callback: Function that returns True if system is processing

        """
        self.processing_callback = callback
        log("[HOTKEY] Processing callback registered")

    def register_hotkey(
        self,
        name: str,
        keys: str,
        mode: HotkeyMode,
        callback_press: Callable[[], None] | None = None,
        callback_release: Callable[[], None] | None = None,
        priority: int = 0,
        description: str = "",
    ) -> None:
        """Register a new hotkey binding.

        Args:
            name: Unique identifier for this hotkey
            keys: Key combination string (e.g., "ctrl+cmd+alt")
            mode: Hotkey activation mode
            callback_press: Function to call when hotkey is pressed
            callback_release: Function to call when hotkey is released (push-to-talk only)
            priority: Priority for conflict resolution (higher = checked first)
            description: Human-readable description

        """
        key_set = self._parse_hotkey_combination(keys)
        binding = HotkeyBinding(
            keys=key_set,
            mode=mode,
            callback_press=callback_press,
            callback_release=callback_release,
            priority=priority,
            description=description,
        )
        self.active_bindings[name] = binding
        log(
            f"[HOTKEY] Registered hotkey '{name}': {keys} ({mode.value}) - keys: {key_set}",
        )

    def start(self) -> None:
        """Start the hotkey listener."""
        if self.is_running:
            log("[HOTKEY] HotkeyManager is already running")
            return

        try:
            log("[HOTKEY] Starting HotkeyManager...")

            # Log registered hotkeys for debugging
            log(f"[HOTKEY] Registered hotkeys: {list(self.active_bindings.keys())}")
            for name, binding in self.active_bindings.items():
                log(f"[HOTKEY]   {name}: {binding.keys} ({binding.mode.value})")

            # Auto-detect Wayland and use evdev backend if so
            if _is_wayland():
                log("[HOTKEY] Wayland detected, using evdev backend")
                self._start_evdev()
            else:
                self.keyboard_listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                )
                self.keyboard_listener.start()

            self.is_running = True
            self._start_heartbeat()
            log("[HOTKEY] HotkeyManager started successfully")
        except Exception as e:
            log(f"[HOTKEY] Failed to start HotkeyManager: {e}")
            raise

    def _listener_alive(self) -> bool:
        """Whether the active backend's listener thread is still pumping events."""
        if self._evdev_listener:
            return self._evdev_listener.is_alive()
        if self.keyboard_listener:
            return self.keyboard_listener.is_alive()
        return False

    def _start_evdev(self):
        """Start the evdev-based keyboard listener for Wayland."""
        from .hotkey_evdev import EvdevHotkeyListener

        self._evdev_listener = EvdevHotkeyListener()

        for name, binding in self.active_bindings.items():
            # Build key string from the parsed set (which is normalized)
            key_str = "+".join(sorted(binding.keys))
            press_cb = binding.callback_press
            release_cb = binding.callback_release if binding.mode == HotkeyMode.PUSH_TO_TALK else None

            self._evdev_listener.register_hotkey(
                name, key_str, press_cb, release_cb,
                swallow=binding.mode == HotkeyMode.PUSH_TO_TALK,
            )

        self._evdev_listener.start()
        log("[HOTKEY] Evdev hotkey listener started")

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if not self.is_running:
            return

        try:
            log("[HOTKEY] Stopping HotkeyManager...")
            # Stop evdev listener if active
            if hasattr(self, "_evdev_listener") and self._evdev_listener:
                self._evdev_listener.stop()
                self._evdev_listener = None
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
            self.is_running = False

            # Reset state
            self.pressed_keys.clear()
            self.last_key_times.clear()
            self.key_press_times.clear()
            self.active_combination = None
            self.current_push_to_talk = None

            log("[HOTKEY] HotkeyManager stopped")
        except Exception as e:
            log(f"[HOTKEY] Error stopping HotkeyManager: {e}")

    def _parse_hotkey_combination(self, hotkey_str: str) -> set[str]:
        """Parse hotkey string into a set of key names.

        Args:
            hotkey_str: Hotkey combination string (e.g., "ctrl+cmd+alt")

        Returns:
            Set of standardized key names

        """
        parts = hotkey_str.lower().split("+")
        key_set = set()

        for part in parts:
            part = part.strip()
            # Apply key mapping so config names match runtime pressed keys
            mapped = self.key_mapping.get(part, part)
            key_set.add(mapped)

        return key_set

    def _get_key_name(self, key) -> str:
        """Get standardized key name from pynput key object.

        Args:
            key: Pynput key object

        Returns:
            Standardized key name string

        """
        try:
            # Validate that this is a proper key object
            if key is None:
                log("[HOTKEY] Received None key object")
                return None

            # Log the raw key object for debugging
            log(f"[HOTKEY] Raw key object: {key} (type: {type(key)})")

            if hasattr(key, "name"):
                key_name = key.name.lower()
                log(f"[HOTKEY] Key has name attribute: {key_name}")
            elif hasattr(key, "char") and key.char:
                key_name = key.char.lower()
                log(f"[HOTKEY] Key has char attribute: {key_name}")
            elif hasattr(key, "vk"):
                # Virtual key code
                vk = key.vk
                log(f"[HOTKEY] Key has vk attribute: {vk}")
                # Map common virtual key codes
                vk_mapping = {
                    16: "shift",  # VK_SHIFT
                    17: "ctrl",  # VK_CONTROL
                    18: "alt",  # VK_MENU
                    91: "cmd",  # VK_LWIN
                    92: "cmd",  # VK_RWIN
                }
                key_name = vk_mapping.get(vk, f"vk_{vk}")
                log(f"[HOTKEY] Mapped vk {vk} to {key_name}")
            else:
                # Not a valid key object
                log(
                    f"[HOTKEY] Invalid key object: {key} - no name, char, or vk attributes",
                )
                return None

            # Validate key name - reject empty or invalid names
            if not key_name or key_name.strip() == "":
                log(f"[HOTKEY] Empty key name from object: {key}")
                return None

            # Apply key mapping for consistency
            mapped_name = self.key_mapping.get(key_name, key_name)
            log(f"[HOTKEY] Final mapped key name: {key_name} -> {mapped_name}")
            return mapped_name
        except Exception as e:
            log(f"[HOTKEY] Error in _get_key_name: {e}")
            return None

    def _on_key_press(self, key) -> None:
        """Handle key press events with robust error handling.

        Args:
            key: Pynput key object

        """
        try:
            current_time = time.time()
            key_name = self._get_key_name(key)

            # Validate key name - ignore invalid keys
            if key_name is None:
                log(f"[HOTKEY] Ignoring invalid key: {key}")
                return

            log(f"[HOTKEY] Key PRESS: {key_name}")

            # Check if system is processing - ignore keys if so
            if self.processing_callback:
                is_processing = self.processing_callback()
                if is_processing:
                    log(f"[HOTKEY] Ignoring key {key_name} - system is processing")
                    return

            # Handle special keys (escape, etc.)
            if key_name in ["esc", "escape"]:
                log("[HOTKEY] Escape key detected")
                self._handle_escape_key()
                return

            # Check if this key is part of any registered hotkey combination
            is_hotkey_key = False
            matching_hotkeys = []
            for binding in self.active_bindings.values():
                if key_name in binding.keys:
                    is_hotkey_key = True
                    matching_hotkeys.append(list(binding.keys))
                    break

            if not is_hotkey_key:
                log(
                    f"[HOTKEY] Ignoring key {key_name} - not part of any hotkey combination",
                )
                return

            log(
                f"[HOTKEY] Key {key_name} is part of hotkey combinations: {matching_hotkeys}",
            )

            # Per-key debouncing (only debounce repeated presses of the SAME key)
            if key_name in self.last_key_times:
                time_since_last = current_time - self.last_key_times[key_name]
                if time_since_last < self.debounce_delay:
                    log(
                        f"[HOTKEY] Debouncing key {key_name} (last press was {time_since_last:.3f}s ago)",
                    )
                    return

            # Don't trigger new combinations while one is active
            if self.active_combination is not None:
                log(
                    f"[HOTKEY] Ignoring {key_name} - combination {self.active_combination} is already active",
                )
                return

            # Log before adding to pressed_keys
            log(
                f"[HOTKEY] Adding {key_name} to pressed_keys. Before: {self.pressed_keys}",
            )
            self.pressed_keys.add(key_name)
            self.key_press_times[key_name] = current_time
            log(f"[HOTKEY] After adding {key_name}: {self.pressed_keys}")

            # Check for hotkey matches
            self._check_hotkey_combinations()

            # Update per-key timing
            self.last_key_times[key_name] = current_time

        except Exception as e:
            log(f"[HOTKEY] Error in key press handler: {e}")
            # Reset to known good state
            self.pressed_keys.clear()
            self.key_press_times.clear()
            self.active_combination = None

    def _on_key_release(self, key) -> None:
        """Handle key release events with robust error handling.

        Args:
            key: Pynput key object

        """
        try:
            key_name = self._get_key_name(key)

            # Validate key name - ignore invalid keys
            if key_name is None:
                return

            log(f"[HOTKEY] Key RELEASE: {key_name}")

            # Only process release if the key is actually in pressed_keys
            if key_name not in self.pressed_keys:
                log(f"[HOTKEY] Ignoring release of {key_name} - not in pressed_keys")
                return

            # Log before removing from pressed_keys
            log(
                f"[HOTKEY] Removing {key_name} from pressed_keys. Before: {self.pressed_keys}",
            )
            self.pressed_keys.discard(key_name)
            self.key_press_times.pop(key_name, None)
            log(f"[HOTKEY] After removing {key_name}: {self.pressed_keys}")

            # Handle push-to-talk release - only when ALL keys in the combination are released
            if self.current_push_to_talk:
                binding = self.active_bindings[self.current_push_to_talk]
                # Check if ANY key from the combination is still pressed
                still_pressed = any(key in self.pressed_keys for key in binding.keys)
                if not still_pressed:
                    log(
                        f"[HOTKEY] Releasing push-to-talk hotkey: {self.current_push_to_talk}",
                    )
                    self._trigger_hotkey_release(self.current_push_to_talk)
                    self.current_push_to_talk = None
                else:
                    log(
                        f"[HOTKEY] Push-to-talk {self.current_push_to_talk} still active - keys still pressed: {[k for k in binding.keys if k in self.pressed_keys]}",
                    )

            # Reset combination state when all keys released
            if len(self.pressed_keys) == 0:
                if self.active_combination:
                    log(
                        f"[HOTKEY] All keys released, clearing active combination: {self.active_combination}",
                    )
                self.active_combination = None

        except Exception as e:
            log(f"[HOTKEY] Error in key release handler: {e}")
            # Reset to known good state
            self.pressed_keys.clear()
            self.key_press_times.clear()
            self.active_combination = None
            self.current_push_to_talk = None

    def _check_hotkey_combinations(self) -> None:
        """Check for complete hotkey combinations and trigger callbacks."""
        try:
            # Find all matching combinations
            matching_bindings = []
            for name, binding in self.active_bindings.items():
                if binding.keys.issubset(self.pressed_keys):
                    matching_bindings.append((name, binding))
                    log(
                        f"[HOTKEY] Found matching hotkey: {name} with keys {binding.keys}",
                    )

            if not matching_bindings:
                log(
                    f"[HOTKEY] No matching hotkeys for pressed keys: {self.pressed_keys}",
                )
                # Clear active combination if no matches found
                if self.active_combination:
                    log(
                        f"[HOTKEY] Clearing active combination {self.active_combination} - no matches found",
                    )
                    self.active_combination = None
                    self.current_push_to_talk = None
                return

            # Sort by most specific first (most keys), then by priority
            sorted_bindings = sorted(
                matching_bindings,
                key=lambda x: (len(x[1].keys), x[1].priority),
                reverse=True,
            )

            # Trigger the most specific matching combination
            name, binding = sorted_bindings[0]

            # Only set active combination if we're not already in that state
            if self.active_combination != name:
                log(f"[HOTKEY] Activating hotkey: {name} (mode: {binding.mode.value})")
                self.active_combination = name

                if binding.mode == HotkeyMode.PUSH_TO_TALK:
                    self.current_push_to_talk = name
                    log(f"[HOTKEY] Set current push-to-talk: {name}")

                self._trigger_hotkey_press(name)
            else:
                log(f"[HOTKEY] Hotkey {name} already active, not re-triggering")

        except Exception as e:
            log(f"[HOTKEY] Error in hotkey combination check: {e}")
            # Reset to known good state
            self.active_combination = None
            self.current_push_to_talk = None

    def _trigger_hotkey_press(self, name: str) -> None:
        """Trigger hotkey press callback.

        Args:
            name: Name of hotkey to trigger

        """
        try:
            binding = self.active_bindings[name]
            log(f"[HOTKEY] Triggering press callback for: {name}")
            if binding.callback_press:
                binding.callback_press()
                log(f"[HOTKEY] Press callback completed for: {name}")
            else:
                log(f"[HOTKEY] No press callback registered for: {name}")
        except Exception as e:
            log(f"[HOTKEY] Error triggering hotkey press '{name}': {e}")

    def _trigger_hotkey_release(self, name: str) -> None:
        """Trigger hotkey release callback.

        Args:
            name: Name of hotkey to trigger

        """
        try:
            binding = self.active_bindings[name]
            log(f"[HOTKEY] Triggering release callback for: {name}")
            if binding.callback_release:
                binding.callback_release()
                log(f"[HOTKEY] Release callback completed for: {name}")
            else:
                log(f"[HOTKEY] No release callback registered for: {name}")
        except Exception as e:
            log(f"[HOTKEY] Error triggering hotkey release '{name}': {e}")

    def force_reset(self) -> None:
        """Force reset all hotkey state. Call this when recording stops to prevent stuck keys."""
        log(
            f"[HOTKEY] Force reset - clearing state. pressed_keys was: {self.pressed_keys}, active: {self.active_combination}, ptt: {self.current_push_to_talk}",
        )
        self.pressed_keys.clear()
        self.key_press_times.clear()
        self.last_key_times.clear()
        self.active_combination = None
        self.current_push_to_talk = None

    def _handle_escape_key(self) -> None:
        """Handle escape key press for canceling operations."""
        # This can be overridden or configured with a callback
        log("[HOTKEY] Escape key pressed - calling escape handler")

    def _start_heartbeat(self):
        """Start heartbeat monitoring thread."""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return

        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="HotkeyManager-Heartbeat",
        )
        self.heartbeat_thread.start()
        log("[HOTKEY] HotkeyManager heartbeat started")

    def _heartbeat_loop(self):
        """Heartbeat loop to monitor hotkey manager health."""
        while self.is_running:
            try:
                current_time = time.time()

                # Update heartbeat timestamp
                self.last_heartbeat = current_time

                # Log heartbeat status
                log(
                    f"[HOTKEY] Heartbeat - running: {self.is_running}, listener_alive: {self._listener_alive()}, pressed_keys: {len(self.pressed_keys)}, active_combination: {self.active_combination}, push_to_talk: {self.current_push_to_talk}",
                )

                # Safety check: if we have an active combination but no pressed keys, clear it
                if self.active_combination and not self.pressed_keys:
                    log(
                        f"[HOTKEY] Safety check: clearing active combination {self.active_combination} - no keys pressed",
                    )
                    self.active_combination = None
                    self.current_push_to_talk = None

                # Stuck key detection: remove keys held longer than the timeout
                stuck_keys = []
                for key_name, press_time in list(self.key_press_times.items()):
                    held_duration = current_time - press_time
                    if held_duration > self.stuck_key_timeout:
                        stuck_keys.append((key_name, held_duration))

                if stuck_keys:
                    log(
                        f"[HOTKEY] Stuck key detection: keys held too long: {[(k, f'{d:.1f}s') for k, d in stuck_keys]}",
                    )
                    for key_name, _ in stuck_keys:
                        self.pressed_keys.discard(key_name)
                        self.key_press_times.pop(key_name, None)
                        self.last_key_times.pop(key_name, None)
                    log(
                        f"[HOTKEY] Stuck keys removed. pressed_keys now: {self.pressed_keys}",
                    )
                    # If pressed_keys is now empty, clear the active state
                    if not self.pressed_keys:
                        log(
                            "[HOTKEY] Stuck key recovery: all keys cleared, resetting active state",
                        )
                        self.active_combination = None
                        self.current_push_to_talk = None

                # Check for inconsistent state: active_combination set but not all combo keys are pressed
                if self.active_combination and self.pressed_keys:
                    binding = self.active_bindings.get(self.active_combination)
                    if binding and not binding.keys.issubset(self.pressed_keys):
                        log(
                            f"[HOTKEY] Inconsistent state: active combination {self.active_combination} keys {binding.keys} not all in pressed_keys {self.pressed_keys}",
                        )
                        # Only reset if the missing keys aren't coming back (they were released)
                        # If released keys are missing, this means we missed release events
                        log(
                            "[HOTKEY] Resetting active state due to inconsistent key state",
                        )
                        self.active_combination = None
                        self.current_push_to_talk = None

                # Check if the keyboard listener is still alive
                if not self._listener_alive():
                    log("[HOTKEY] Warning: Keyboard listener died, attempting restart")
                    self._restart_listener()

                time.sleep(self.heartbeat_interval)

            except Exception as e:
                log(f"[HOTKEY] Heartbeat error: {e}")
                time.sleep(self.heartbeat_interval)

    def _restart_listener(self):
        """Restart the keyboard listener if it died, using the same backend."""
        try:
            log("[HOTKEY] Restarting keyboard listener...")
            # Stop existing listener
            if self._evdev_listener:
                try:
                    self._evdev_listener.stop()
                except Exception as e:
                    log(f"[HOTKEY] Error stopping dead evdev listener: {e}")
                self._evdev_listener = None
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None

            # Reset state
            self.pressed_keys.clear()
            self.last_key_times.clear()
            self.key_press_times.clear()
            self.active_combination = None
            self.current_push_to_talk = None

            # Start a new listener on whichever backend this session uses
            if _is_wayland():
                self._start_evdev()
            else:
                self.keyboard_listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                )
                self.keyboard_listener.start()
            log("[HOTKEY] Keyboard listener restarted successfully")

        except Exception as e:
            log(f"[HOTKEY] Failed to restart keyboard listener: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
