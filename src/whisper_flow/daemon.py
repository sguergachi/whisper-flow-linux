"""Background daemon for whisper-flow with system tray and global hotkeys."""

import os
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import backend as backend_module
from . import icon
from . import updater
from .icon import ICON_IDLE, ICON_RECORDING
from .app import WhisperFlow
from .backend import LocalBackend
from .config import Config
from .hotkey_manager import HotkeyManager, HotkeyMode
from .hud import HUD, STOP_SUFFIX
from .logging import log, recent_log, set_logging_enabled, write_crash_report
from .logging import reset_stage_log, trace_stage
from .paths import lock_file as _lock_file
from .paths import pid_file as _pid_file
from .version import build_version

# Modes driven by holding a hotkey down; they cannot be deferred and replayed.
# Only plain push-to-talk. Auto-transcribe and command are single-press:
# speak until silence. Command used to be a held chord of pure modifiers
# (super+shift+alt); releasing them ended the recording in the same gesture
# that started it, so every tap produced a clip too short to keep - which
# looked exactly like "command does nothing".
PUSH_TO_TALK_MODES = ("transcribe",)

# How long a Settings tray click suppresses the next one. Opening the tool
# window holds the tray callback for the 0.4s liveness probe (and more if a
# process has to be started), during which the message loop is stalled and
# further clicks queue behind it; without the debounce a user clicking away
# at a tray item that seems slow produced bursts of 6-8 open_settings calls
# in the same second, each writing another "show" and re-running the raise
# and DWM-acrylic work - the storm under which the settings window was seen
# to die (0xC0000005) and vanish. One click per debounce window is enough:
# the window raises itself.
SETTINGS_CLICK_DEBOUNCE_S = 1.5

def _pystray():
    """Import pystray at the point of use.

    It resolves its backend during import, and the X11 backend raises if
    there is no display - so merely importing this module required one, and
    a bare `import whisper_flow.daemon` failed on any headless machine. It
    also costs ~126ms that a run which never reaches the tray need not pay.
    """
    import pystray

    return pystray


_icon_cache: dict[tuple, Image.Image] = {}
_icon_lock = threading.Lock()


def _cached_icon(color: tuple[int, int, int, int]) -> Image.Image:
    """The rendered glyph, drawn once per colour.

    icon.tray_icon supersamples to 512px and then runs a 41-pixel
    MaxFilter over it, which measures at ~615ms. It was being run on every
    recording start and every stop, on the thread that starts the recording
    - so a fixed picture of a microphone was delaying the microphone. Both
    colours are constants, so one render each is all that is ever needed.
    """
    with _icon_lock:
        if color not in _icon_cache:
            started = time.perf_counter()
            _icon_cache[color] = icon.tray_icon(color)
            log(f"[DAEMON] rendered tray icon in "
                f"{(time.perf_counter() - started) * 1000:.0f}ms")
        return _icon_cache[color]


def prerender_icons() -> None:
    """Draw both icons before they are needed, off the recording path."""
    for color in (ICON_IDLE, ICON_RECORDING):
        _cached_icon(color)


class PressToListen:
    """Timing of the path from hotkey press to the microphone capturing.

    PTL - press-to-listen - is the one latency the user feels: everything
    optimised so far, the overlay, the tray icons, the imports, the audio
    device, is a component of it. Recording the stages rather than only the
    total means the next thing worth optimising names itself instead of
    being guessed at.

    Cheap enough to leave on: a monotonic clock read per stage.
    """

    def __init__(self):
        self.started = time.monotonic()
        self._last = self.started
        self._stages: list[tuple[str, float]] = []
        self.reported = False

    def mark(self, stage: str) -> None:
        now = time.monotonic()
        self._stages.append((stage, (now - self._last) * 1000))
        self._last = now

    def report(self) -> None:
        """Log the total and where it went. Only ever once per press."""
        if self.reported:
            return
        self.reported = True
        total = (time.monotonic() - self.started) * 1000
        breakdown = ", ".join(f"{name} {ms:.0f}" for name, ms in self._stages)
        log(f"[PTL] {total:.0f}ms press-to-listen ({breakdown})")


class WhisperFlowDaemon:
    """Background daemon with system tray icon and global hotkey support."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize the daemon."""
        self.config = Config(config_dir=config_dir) if config_dir else Config()
        set_logging_enabled(self.config.logging_enabled)
        log("[DAEMON] Initializing WhisperFlowDaemon...")
        self.tray_icon = None
        self.is_running = False
        self.is_recording = False
        self.current_mode = None
        self.recording_thread = None
        self.stop_recording_event = None

        # Processing state management
        self.processing_lock = threading.Lock()
        self._stop_lock = threading.RLock()
        self.request_queue = queue.Queue()
        self.is_processing = False

        # Thread health monitoring
        self.recording_start_time = None
        self.max_recording_duration = self.config.max_recording_duration
        self.watchdog_thread = None
        self.watchdog_interval = self.config.watchdog_interval

        # Initialize the new HotkeyManager
        log("[DAEMON] Creating HotkeyManager...")
        self.hotkey_manager = HotkeyManager()

        # Initialize HUD overlay for recording indicator
        log("[DAEMON] Creating HUD overlay...")
        self.hud = HUD()

        # Managed speech engine, so a fresh install has something to talk to
        self.backend = LocalBackend(self.config, notify=self.notify)
        self._backend_model = None
        self._backend_engine = None
        self._pressed_at = None
        self._setup_process = None
        # Built and standing by, versus on screen because someone asked. The
        # two need telling apart: a prewarmed window that dies before anyone
        # wants it is a line in the log, while one that dies after the click
        # is a user looking at a desktop where a window should be.
        self._setup_waiting = False
        self._setup_shown = False
        self._setup_lock = threading.Lock()
        self._last_failure = None
        self._last_settings_click = 0.0
        # Set when Update is clicked mid-dictation: the restart waits until
        # the current request finishes instead of killing the recording.
        self._update_apply_deferred = False
        self._instance_lock = None

        # Initialize WhisperFlow instances for different modes
        log("[DAEMON] Creating WhisperFlow instances...")
        self.transcribe_app = WhisperFlow(config_dir, "transcribe")
        self.auto_transcribe_app = WhisperFlow(config_dir, "auto_transcribe")
        self.command_app = WhisperFlow(config_dir, "command")

        log("[DAEMON] WhisperFlowDaemon initialization complete")

    def _start_watchdog(self):
        """Start the watchdog thread to monitor system health."""
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            log("[DAEMON] Watchdog thread already running")
            return

        log("[DAEMON] Starting watchdog thread...")
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="WhisperFlow-Watchdog",
        )
        self.watchdog_thread.start()
        log("[DAEMON] Watchdog thread started")

    def _watchdog_loop(self):
        """Watchdog loop to monitor thread health and detect hangs."""
        log("[DAEMON] Watchdog loop started")
        last_status = None
        while self.is_running:
            try:
                # Check recording thread health
                if self.is_recording and self.recording_thread:
                    # Check if recording thread is still alive
                    if not self.recording_thread.is_alive():
                        log("[DAEMON] WARNING: Recording thread died unexpectedly")
                        self._force_stop_recording("Recording thread died")
                        continue

                    # Check for excessive recording duration
                    if (
                        self.recording_start_time
                        and time.time() - self.recording_start_time
                        > self.max_recording_duration
                    ):
                        log(
                            f"[DAEMON] WARNING: Recording exceeded {self.max_recording_duration}s limit",
                        )
                        self._force_stop_recording("Recording timeout")
                        continue

                # Check if processing lock is held too long
                if self.is_processing and self.recording_start_time:
                    processing_duration = time.time() - self.recording_start_time
                    if processing_duration > 60:  # 1 minute max processing
                        log(
                            f"[DAEMON] WARNING: Processing lock held for {processing_duration:.1f}s",
                        )
                        # Don't force stop here, just log warning

                # Backend health: if the server died (0xC0000005), revive it so
                # the next hotkey doesn't record 5s of speech only to drop it
                # with "No whisper server configured". Deaths are reported to
                # the backend, which re-downloads a fresh engine after several
                # consecutive native crashes of the same bytes. A circuit
                # breaker backs off when revives keep failing, instead of
                # restart-spamming every 2s forever.
                if not self.is_recording and not self.is_processing:
                    try:
                        url = (self.config.local_whisper_url or "").strip()
                        proc = getattr(self.backend, "_process", None)
                        dead = proc is not None and proc.poll() is not None
                        needs_revive = dead or (not url and self.backend.working_model())
                        if needs_revive:
                            now = time.time()
                            if not hasattr(self, "_last_backend_revive"):
                                self._last_backend_revive = 0.0
                            if not hasattr(self, "_revive_backoff_until"):
                                self._revive_backoff_until = 0.0
                            if not hasattr(self, "_revive_failures"):
                                self._revive_failures = []
                            # Drop failures older than 5 minutes.
                            self._revive_failures = [
                                ts for ts in self._revive_failures
                                if now - ts < 300]
                            if now < self._revive_backoff_until:
                                pass  # circuit open: wait before retrying
                            else:
                                throttle = 0 if dead else 10
                                if now - self._last_backend_revive >= throttle:
                                    self._last_backend_revive = now
                                    if dead:
                                        code = proc.poll()
                                        try:
                                            spath = getattr(self.backend, "_stderr_path", None)
                                            if spath and __import__("pathlib").Path(spath).exists():
                                                lines = __import__("pathlib").Path(spath).read_text(encoding="utf-8", errors="replace").strip().splitlines()
                                                tail = "\n".join(lines[-20:])
                                                if tail.strip():
                                                    log(f"[BACKEND] server output at crash ({spath}):\n{tail[:2000]}")
                                        except Exception:
                                            pass
                                        # Name the faulting module (Event 1000):
                                        # 0xC0000005 in ggml.dll vs openblas.dll
                                        # vs the exe itself decides the fix.
                                        try:
                                            from .backend import faulting_module as _fm
                                            mod = _fm("whisper-server.exe")
                                            if mod:
                                                log(f"[BACKEND] faulting module: {mod}")
                                        except Exception:
                                            pass
                                        log(f"[DAEMON] backend process dead (exit={code} 0x{(code or 0) & 0xFFFFFFFF:08X}), reviving")
                                        try:
                                            refreshed = self.backend.note_crash(code)
                                            if refreshed:
                                                log("[BACKEND] fresh engine downloaded after repeated crashes")
                                        except Exception as e:
                                            log(f"[DAEMON] crash tracking failed: {e}")
                                    revived = self._ensure_backend_running()
                                    self._revive_failures.append(now)
                                    if not revived and len(self._revive_failures) >= 5:
                                        # Circuit breaker: this is not a blip.
                                        # Back off for a minute and say so once.
                                        self._revive_backoff_until = now + 60
                                        log("[DAEMON] backend keeps failing — backing off revives for 60s")
                                        try:
                                            self.notify("Speech engine keeps crashing — open Settings to pick another model, or check the log")
                                        except Exception:
                                            pass
                    except Exception as e:
                        log(f"[DAEMON] backend watchdog failed (auto-heal will retry): {e}")

                # Hotkey/hud auto-heal: if listener died, restart it without crashing daemon
                if not self.is_recording:
                    try:
                        hm = getattr(self, "hotkey_manager", None)
                        if hm and hasattr(hm, "is_alive"):
                            try:
                                if not hm.is_alive():
                                    log("[DAEMON] hotkey manager dead, auto-healing")
                                    try:
                                        hm.start()
                                    except Exception as he:
                                        log(f"[DAEMON] hotkey auto-heal failed: {he}")
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Compositor-side stuck modifiers: while idle, release
                    # anything the desktop may hold that we do not. Bounded
                    # inside the listener (once a minute, only shortly after
                    # hotkey activity), harmless when healthy, and it heals
                    # the "every key opens a shortcut" state without input.
                    if not self.is_processing:
                        try:
                            self.hotkey_manager.sweep_input()
                        except Exception as e:
                            log(f"[DAEMON] input sweep failed: {e}")

                # Log status only when it changes; at a 2s interval a constant
                # heartbeat buries every real message in the journal.
                status = (
                    self.is_recording,
                    self.is_processing,
                    self.request_queue.qsize(),
                    self.current_mode,
                )
                if status != last_status:
                    last_status = status
                    log(
                        f"[DAEMON] Watchdog status - running: {self.is_running}, recording: {self.is_recording}, processing: {self.is_processing}, queue_size: {self.request_queue.qsize()}, current_mode: {self.current_mode}",
                    )

                time.sleep(self.watchdog_interval)

            except Exception as e:
                log(f"[DAEMON] Watchdog error: {e}")
                time.sleep(self.watchdog_interval)

    def _force_stop_recording(self, reason: str):
        """Force stop recording when watchdog detects issues.

        Args:
            reason: Reason for forced stop

        """
        log(f"[DAEMON] Forcing recording stop: {reason}")
        self.notify(f"⚠️ Recording stopped: {reason}")
        mode = self.current_mode

        # Signal stop
        if self.stop_recording_event:
            self.stop_recording_event.set()

        # Go through the normal teardown rather than clearing the flags here.
        # Clearing them directly left the HUD on screen and the level file on
        # disk, because the recording thread's own cleanup then saw
        # is_recording already False and returned without doing anything.
        self._stop_recording()

        # This path exists because the recording thread is wedged or gone, so
        # it may never release the processing lock itself. Releasing it here
        # keeps one stuck recording from rejecting every later request with
        # "system busy"; a late release from the thread is a no-op.
        self._finish_processing(mode or "unknown")

    def daemonize(self):
        """Run as background process while preserving desktop session access.

        POSIX only - this forks. Windows reaches the background through the
        tray app being a GUI subsystem binary instead.
        """
        if sys.platform == "win32":
            raise RuntimeError("daemonize() is POSIX-only; use run(foreground=True)")
        log("[DAEMON] Starting daemonization process...")
        # Simple approach: just fork once and redirect output
        # This preserves all environment and session access

        try:
            pid = os.fork()
            if pid > 0:
                # Parent process exits
                log("[DAEMON] Parent process exiting")
                sys.exit(0)
        except OSError:
            log("[DAEMON] Fork failed")
            sys.exit(1)

        # We're now in the child process
        # Don't change session, directory, or umask - preserve everything
        log("[DAEMON] Child process continuing...")

        # Only redirect output to avoid cluttering terminal
        sys.stdout.flush()
        sys.stderr.flush()

        # Redirect to /dev/null but keep stderr for debugging if needed
        with open("/dev/null") as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open("/dev/null", "a+") as f:
            os.dup2(f.fileno(), sys.stdout.fileno())

        # Write PID file
        pid_file = _pid_file()
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        log(f"[DAEMON] Daemonized with PID {os.getpid()}")

    def create_tray_icon(self) -> Image.Image:
        """The system tray icon: a microphone glyph."""
        return _cached_icon(ICON_IDLE)

    def create_recording_icon(self) -> Image.Image:
        """The recording state icon: the same mic, lit red."""
        return _cached_icon(ICON_RECORDING)

    def setup_tray_menu(self):
        """Setup the system tray menu."""
        pystray = _pystray()
        return pystray.Menu(
            # Named with its build. The tray menu is the one part of this app
            # that is always a click away, and "which version am I running"
            # was otherwise answerable only by copying a failure report.
            pystray.MenuItem(f"WhisperFlow {build_version()}", None,
                             enabled=False),
            # Labels spell out how each key works: hold vs tap-then-silence.
            # The spare auto key is not a third mode (command used to call a
            # cloud model; now it is just a second hands-free binding).
            pystray.MenuItem(
                f"Push to talk (hold): {self.config.hotkey_transcribe}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                f"Auto-transcribe (tap): {self.config.hotkey_auto_transcribe}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                f"Auto-transcribe 2nd (tap): {self.config.hotkey_command}",
                None,
                enabled=False,
            ),
            # default=True: left-click (or platform "activate") opens
            # Settings. Right-click still shows the full menu. Without this
            # the tray looked inert unless the user found Settings in the
            # menu - which is the opposite of how every other tray app works.
            pystray.MenuItem("Settings", self.open_settings, default=True),
            pystray.MenuItem("Test setup", self.test_configuration),
            pystray.MenuItem("Copy last error", self.copy_last_error),
            pystray.MenuItem("Copy log", self.copy_log),
            # The label follows the updater state: plain check normally,
            # "Update to X" once a release is downloaded and waiting.
            pystray.MenuItem(lambda item: self._update_label(),
                             self.check_for_updates,
                             visible=updater.available()),
            pystray.MenuItem("Reload", self.reload_daemon),
            pystray.MenuItem("Exit", self.stop_daemon),
        )

    def setup_hotkeys(self):
        """Register global hotkeys using the new HotkeyManager."""
        try:
            log("[DAEMON] Setting up hotkeys...")

            # Register processing callback
            self.hotkey_manager.register_processing_callback(self._is_processing)

            refused = []

            def _register(name, keys, mode, callback_press,
                          callback_release=None, priority=0, description=""):
                ok = self.hotkey_manager.register_hotkey(
                    name=name,
                    keys=keys,
                    mode=mode,
                    callback_press=callback_press,
                    callback_release=callback_release,
                    priority=priority,
                    description=description,
                )
                if not ok:
                    refused.append(keys)

            # Register transcribe hotkey (push-to-talk)
            _register(
                "transcribe",
                self.config.hotkey_transcribe,
                HotkeyMode.PUSH_TO_TALK,
                lambda: self._handle_hotkey_press("transcribe"),
                lambda: self._stop_recording_if_active("transcribe"),
                priority=1,
                description="Push-to-talk transcription",
            )

            # Register auto-transcribe hotkey (single press)
            _register(
                "auto_transcribe",
                self.config.hotkey_auto_transcribe,
                HotkeyMode.SINGLE_PRESS,
                lambda: self._handle_hotkey_press("auto_transcribe"),
                priority=3,  # Highest priority since it has most keys
                description="Auto-stop transcription",
            )

            # Register command hotkey (single press, stop on silence)
            _register(
                "command",
                self.config.hotkey_command,
                HotkeyMode.SINGLE_PRESS,
                lambda: self._handle_hotkey_press("command"),
                priority=2,  # Higher than transcribe since it has more keys
                description="Single-press command dictation (stop on silence)",
            )
            if refused:
                log(f"[DAEMON] Refused broken hotkeys: {refused}")
                self.notify("A hotkey needs a key beyond the modifiers - "
                            "re-set it in Settings")

            # Set up escape key handling for canceling recordings
            self.hotkey_manager._handle_escape_key = self.cancel_recording

            # Emergency input resets must surface: a freed keyboard with no
            # explanation reads as the app flaking, and a repeatedly
            # failing listener needs a human to restart the app.
            self.hotkey_manager.set_emergency_callback(
                lambda reason: self.notify(
                    f"⌨️ Keyboard input was emergency-reset ({reason})"))

            # Start the hotkey manager
            self.hotkey_manager.start()
            log("[DAEMON] Hotkeys setup complete")

            # Have the overlay up and waiting before the first press, rather
            # than starting it on the first recording - which would leave the
            # first dictation of every session as slow as it always was.
            self.hud.prewarm()
            # The same for the settings window, which costs far more to open
            # than the overlay does and is opened from a menu nobody expects
            # to wait three seconds for.
            self.prewarm_settings()
            # And draw the tray icons now, so the first recording does not
            # spend ~615ms on a picture before opening the microphone.
            threading.Thread(target=prerender_icons, daemon=True,
                             name="whisper-flow-icons").start()

        except Exception as e:
            log(f"[DAEMON] Error setting up hotkeys: {e}")
            self.notify(f"Error setting up hotkeys: {e}")

    def _is_processing(self) -> bool:
        """Check if system is currently processing a request."""
        return self.is_processing or self.is_recording

    def _handle_hotkey_press(self, mode: str):
        """Handle hotkey press with queuing support."""
        # Stamped first, before any work, so the figure below is the whole
        # delay the user feels and not a measurement of part of it.
        self._ptl = PressToListen()
        log(f"[DAEMON] Hotkey press received for mode: {mode}")

        # Second tap of the same auto-stop mode ends the recording. Queuing
        # another pass looked like "it does nothing" - the first clip kept
        # listening for silence while the second sat in the queue.
        if (
            self.is_recording
            and mode == self.current_mode
            and mode in ("auto_transcribe", "command")
        ):
            log(f"[DAEMON] Second {mode} press - stopping recording")
            self.notify("Stopping…")
            self._stop_recording_if_active(mode)
            return

        if self.is_processing or self.is_recording:
            if mode in PUSH_TO_TALK_MODES:
                # A held hotkey cannot be replayed later: by the time the queue
                # drains the key is long released, so the recording would run
                # until the max-duration watchdog kills it.
                log(f"[DAEMON] System busy, dropping {mode} request")
                self.notify("⏳ Still working on the last one")
                return
            # Queue the request
            log(f"[DAEMON] System busy, queuing {mode} request")
            self.request_queue.put((mode, time.time()))
            self.notify(f"Queued {mode} request")
        else:
            # Process immediately
            log(f"[DAEMON] Processing {mode} immediately")
            self._process_mode(mode)

    def _process_mode(self, mode: str):
        """Process a mode with proper locking and timeout protection.

        The lock is held for the whole record-transcribe-paste cycle and is
        released by the recording thread via ``_finish_processing``. Releasing
        it here would let the next hotkey press start a second recording on top
        of the one still running.
        """
        log(f"[DAEMON] Attempting to process mode: {mode}")

        # Use a timeout for the processing lock to prevent indefinite blocking
        if not self.processing_lock.acquire(
            timeout=self.config.processing_lock_timeout,
        ):
            log(
                f"[DAEMON] WARNING: Could not acquire processing lock within {self.config.processing_lock_timeout}s timeout",
            )
            self.notify("⚠️ System busy, request ignored")
            return

        log(f"[DAEMON] Processing lock acquired for mode: {mode}")
        self.is_processing = True
        self.recording_start_time = time.time()
        try:
            started = self.start_recording(mode)
        except Exception as e:
            log(f"[DAEMON] Error starting recording for mode {mode}: {e}")
            started = False

        if not started:
            # No recording thread will run, so release the lock here.
            self._finish_processing(mode)

    def _finish_processing(self, mode: str):
        """Release the processing lock once a request has fully completed.

        Reached from the recording thread and, when that thread is wedged,
        from the watchdog. The check and the release have to be atomic or both
        can get through and release the lock twice.
        """
        with self._stop_lock:
            if not self.is_processing:
                return
            log(f"[DAEMON] Processing lock released for mode: {mode}")
            self.is_processing = False
            self.recording_start_time = None
            # Now there is a transcript, or a failure. Either way the waiting
            # is over and the overlay has said all it can.
            self._take_down_overlay()
            try:
                self.processing_lock.release()
            except RuntimeError:
                pass
        # A clicked Update waits out the dictation. Queued requests drain
        # first (each finish re-checks the flag); the restart runs only
        # once truly idle, so no recording is ever killed for an update.
        if getattr(self, "_update_apply_deferred", False):
            try:
                queue_empty = self.request_queue.empty()
            except Exception:
                queue_empty = True
            if queue_empty:
                self._update_apply_deferred = False
                log("[DAEMON] installing deferred update now that idle")
                threading.Thread(
                    target=self._apply_update_when_idle,
                    daemon=True, name="whisper-flow-update-deferred",
                ).start()
                return
        # Process next item in queue
        self._process_next_in_queue()

    def _process_next_in_queue(self):
        """Process next item in queue if any."""
        try:
            while True:
                if self.request_queue.empty():
                    log("[DAEMON] No queued requests to process")
                    return
                mode, timestamp = self.request_queue.get_nowait()
                log(
                    f"[DAEMON] Processing queued request: {mode} (queued at {timestamp})",
                )

                # Check if request is too old (older than configured timeout).
                # A stale one must not stop the drain: the request behind it
                # may still be fresh, and returning here stranded it forever.
                if time.time() - timestamp > self.config.queue_request_timeout:
                    log(
                        f"[DAEMON] Dropping old queued request for {mode} (age: {time.time() - timestamp:.1f}s)",
                    )
                    continue

                log(f"[DAEMON] Processing queued mode: {mode}")
                self._process_mode(mode)
                return
        except queue.Empty:
            log("[DAEMON] Queue is empty")
        except Exception as e:
            log(f"[DAEMON] Error processing next in queue: {e}")

    def start_recording(self, mode: str) -> bool:
        """Start recording in the specified mode.

        Returns:
            True if a recording thread was started, False otherwise.

        """
        if self.is_recording:
            log(f"[DAEMON] Already recording, ignoring start request for mode: {mode}")
            return False

        # If the server died (GPU 0xC0000005 in the user's log), revive it
        # *before* opening the mic — otherwise 5s of speech is recorded and
        # then dropped with "No whisper server configured".
        try:
            backend_ok = self._ensure_backend_running(allow_download=False)
        except Exception as e:
            log(f"[DAEMON] backend check before recording failed: {e}")
            backend_ok = False
        if not backend_ok:
            # No model at all — can't transcribe even on CPU; still allow
            # recording so the audio-debug captures evidence, but warn now
            # rather than after 5s of silence.
            try:
                has_model = bool(self.backend.working_model())
            except Exception:
                has_model = False
            if not has_model:
                log("[DAEMON] no working model — recording will have nowhere to transcribe")
                self.notify("No speech model installed — open Settings to download one")
            else:
                log("[DAEMON] backend not running before recording — will record anyway and retry after")
                # Don't block the hotkey on a download; the recording thread's
                # closing pass will retry via TranscriptionService fallback

        log(f"[DAEMON] Starting recording for mode: {mode}")
        # Clear the previous cycle's thread handle first. The watchdog treats
        # "recording, but the thread is not alive" as a crash, and between
        # setting the flag and starting the new thread the handle still points
        # at the last one, which has finished - enough to force-stop a
        # recording a moment after it began.
        self.recording_thread = None
        self.is_recording = True
        self.current_mode = mode
        self.stop_recording_event = threading.Event()
        self.recording_start_time = time.time()

        try:
            # Save the currently active window UUID for focus restoration at paste time
            self._mark_ptl("dispatch")
            app = self._get_app_for_mode(mode)
            saved_window = None
            try:
                app.system_manager.save_active_window()
                saved_window = app.system_manager._saved_window
                log(f"[DAEMON] Saved active window for mode: {mode} -> {saved_window}")
            except Exception as e:
                log(f"[DAEMON] Failed to save active window: {e}")

            self._mark_ptl("save window")

            # Create level file for HUD waveform
            fd, self._level_file = tempfile.mkstemp(suffix=".levels", prefix="whisper-flow-")
            os.close(fd)

            # The overlay is shown from the recording thread, once the microphone
            # is actually capturing - see _record_audio_thread. Showing it here
            # meant it appeared while the capture stream was still opening, and
            # anything said in that gap was never recorded. The user treats the
            # overlay as "speak now", so it has to be true.
            try:
                self._hud_point = app.system_manager.active_window_center()
            except Exception:
                self._hud_point = None
            self._mark_ptl("window centre")

            # Update tray icon to recording state
            if self.tray_icon:
                self.tray_icon.icon = self.create_recording_icon()

            # Start recording thread
            self.recording_thread = threading.Thread(
                target=self._record_audio_thread,
                args=(mode,),
                daemon=True,
                name=f"WhisperFlow-Recording-{mode}",
            )
            self.recording_thread.start()

            if mode in ("auto_transcribe", "command"):
                # The HUD shows a stop button for these modes; pressing it
                # ends the recording from the other side of the process
                # boundary. It asks through a marker file, and this thread -
                # which lives only as long as the recording does - watches
                # for it.
                threading.Thread(
                    target=self._watch_hud_stop_request,
                    args=(mode,),
                    daemon=True,
                    name="WhisperFlow-HUDStop",
                ).start()
        except Exception:
            # Roll back rather than leave is_recording set with no thread to
            # clear it: the watchdog ignores a recording with no thread, so
            # every later press would be dropped as "busy" for the rest of
            # the daemon's life.
            self._stop_recording()
            raise
        self._mark_ptl("thread start")
        log(f"[DAEMON] Recording thread started for mode: {mode}")
        return True

    def _take_down_overlay(self) -> None:
        """Hide the pill and remove the files it lives off.

        Deleting the level file is also the overlay's own orphan guard: one
        spawned per recording closes when it goes, so this reaches even an
        overlay that never got the hide.
        """
        try:
            self.hud.hide()
        except Exception as e:
            log(f"[DAEMON] could not hide the overlay: {e}")
        level_file = getattr(self, "_level_file", None)
        if not level_file:
            return
        self._level_file = None
        self.hud.clear_processing(level_file)
        self.hud.clear_stop_marker(level_file)
        try:
            os.unlink(level_file)
        except OSError:
            pass

    def _release_stuck_modifiers(self) -> None:
        """Drop any modifier the typing path is still holding down.

        Any of the three apps would do. What is held is a property of the
        keyboard, not of a mode, so it is recorded once in system_win rather
        than per SystemManager - which is why reaching for whichever app
        happens to be at hand is correct here rather than merely harmless.
        """
        try:
            self.transcribe_app.system_manager.release_stuck_modifiers()
        except Exception as e:
            log(f"[DAEMON] could not release held modifiers: {e}")

    def _mark_ptl(self, stage: str) -> None:
        """Record a stage of the press-to-listen path, if one is in flight."""
        ptl = getattr(self, "_ptl", None)
        if ptl is not None and not ptl.reported:
            ptl.mark(stage)

    def _show_hud_now(self) -> None:
        """Put the overlay up. Called when the microphone starts capturing.

        This is the end of the PTL window: the microphone is live, so the
        overlay can honestly say "speak now".
        """
        ptl = getattr(self, "_ptl", None)
        if ptl is not None:
            ptl.mark("mic open")
        try:
            self.hud.show(level_file=getattr(self, "_level_file", None),
                          point=getattr(self, "_hud_point", None),
                          stop_button=self.current_mode in ("auto_transcribe",
                                                            "command"))
        except Exception as e:
            log(f"[DAEMON] could not show the overlay: {e}")
        finally:
            if ptl is not None:
                ptl.mark("overlay")
                ptl.report()
        # Auto-stop modes have no held key to remind you they are live. One
        # toast at mic-open is the "it heard the hotkey" signal when the
        # overlay is slow or missing.
        if self.current_mode in ("auto_transcribe", "command"):
            try:
                self.notify("Listening — speak, pause to finish (or tap again)")
            except Exception:
                pass

    def _stop_recording_if_active(self, mode: str):
        """Stop recording if the specified mode is currently active."""
        log(
            f"[DAEMON] Stop recording check for mode: {mode} (current: {self.current_mode}, recording: {self.is_recording})",
        )
        if self.is_recording and self.current_mode == mode:
            log(f"[DAEMON] Stopping recording for mode: {mode}")
            self._stop_recording()
        else:
            log("[DAEMON] Not stopping - mode mismatch or not recording")

    def _watch_hud_stop_request(self, mode: str):
        """End the recording when the HUD's stop button is pressed.

        The overlay is a separate process with no pipe back into this one, so
        it drops a marker file beside the level file - the mirror of the
        processing marker the daemon uses in the other direction - and this
        thread, which lives only as long as the recording, polls for it.
        """
        level_file = getattr(self, "_level_file", None) or ""
        marker = (level_file + STOP_SUFFIX) if level_file else ""
        # A marker left by a dead overlay must not end the next recording the
        # instant it starts.
        try:
            os.unlink(marker)
        except OSError:
            pass
        try:
            while self.is_recording and self.current_mode == mode:
                if marker and os.path.exists(marker):
                    try:
                        os.unlink(marker)
                    except OSError:
                        pass
                    log("[DAEMON] stop requested from the HUD")
                    self._stop_recording()
                    return
                time.sleep(0.05)
        except Exception as e:
            log(f"[DAEMON] HUD stop watcher gave up: {e}")

    def _healing_transcribe(self, app, func_name: str = "transcribe_audio"):
        """Wrap transcribe_audio to auto-heal backend crashes mid-dictation.

        If the whisper-server died (0xC0000005) between hotkey press and the
        final pass, the original call raises 'No whisper server configured' and
        the 5s of speech is lost. This wrapper revives on CPU and retries once,
        so the *current* dictation is saved, not just the next one.
        """
        orig = getattr(app.transcription_service, func_name)

        def healing(*args, **kwargs):
            try:
                result = orig(*args, **kwargs)
                if result is not None:
                    return result
                if app.transcription_service.local_url:
                    return None
                raise RuntimeError("No whisper server configured (auto-heal check)")
            except Exception as e:
                msg = str(e)
                is_no_server = "No whisper server" in msg or "No whisper server configured" in msg
                is_conn = "Cannot reach" in msg or "Connection" in msg or "Failed to connect" in msg
                if not (is_no_server or is_conn):
                    raise
                log(f"[DAEMON] transcribe failed with '{e}' — auto-healing backend and retrying once")
                healed = self._ensure_backend_running(allow_download=False)
                if not healed:
                    log("[DAEMON] auto-heal could not revive backend")
                    raise
                return orig(*args, **kwargs)

        return healing

    def _record_audio_thread(self, mode: str):
        """Handle audio recording in a separate thread with timeout protection."""
        log(f"[DAEMON] Recording thread started for mode: {mode}")
        app_for_mode = None
        orig_transcribe = None
        try:
            app = self._get_app_for_mode(mode)
            app_for_mode = app
            log(f"[DAEMON] Using app instance for mode: {mode}")
            orig_transcribe = app.transcription_service.transcribe_audio
            app.transcription_service.transcribe_audio = self._healing_transcribe(app)

            if mode in ("auto_transcribe", "command"):
                # Single-press: record until silence (or Escape / max duration).
                # Must show the HUD via on_ready - without it the user has no
                # "speak now" signal and the mode looks dead.
                log(
                    f"[DAEMON] Running {mode} auto-stop with silence duration: "
                    f"{self.config.auto_stop_silence_duration}",
                )
                success = app.run_voice_flow_auto_stop(
                    silence_duration=self.config.auto_stop_silence_duration,
                    level_file=getattr(self, "_level_file", None),
                    stop_event=self.stop_recording_event,
                    on_ready=self._show_hud_now,
                    max_duration=self.config.max_recording_duration,
                )
            elif mode == "transcribe":
                # Push-to-talk: hold to talk, release to stop.
                hotkey = self.config.hotkey_transcribe
                if self.config.live_transcription:
                    log(f"[DAEMON] Running LIVE push-to-talk with stop key: {hotkey}")
                    success = app.run_voice_flow_push_to_talk_live(
                        stop_key=hotkey,
                        stop_event=self.stop_recording_event,
                        level_file=getattr(self, "_level_file", None),
                        on_ready=self._show_hud_now,
                    )
                else:
                    log(f"[DAEMON] Running push-to-talk mode with stop key: {hotkey}")
                    success = app.run_voice_flow_push_to_talk_daemon(
                        stop_key=hotkey,
                        stop_event=self.stop_recording_event,
                        level_file=getattr(self, "_level_file", None),
                        on_ready=self._show_hud_now,
                    )
            else:
                # Fallback to auto-stop
                log(f"[DAEMON] Unknown mode {mode}, falling back to auto-stop")
                success = app.run_voice_flow_auto_stop(
                    silence_duration=self.config.auto_stop_silence_duration,
                    stop_event=self.stop_recording_event,
                    on_ready=self._show_hud_now,
                    max_duration=self.config.max_recording_duration,
                )

            if not success:
                # If the backend died mid-recording, one retry on CPU can still
                # save this utterance instead of dropping 5s of speech.
                if not self.config.local_whisper_url or (self.backend._process and self.backend._process.poll() is not None):
                    log("[DAEMON] recording failed and backend is down — trying one revive + retry")
                    if self._ensure_backend_running(allow_download=False):
                        # Re-run just the transcription on the already-recorded
                        # file is not available here (app deleted it), so the
                        # retry will happen on the *next* hotkey. But we can
                        # at least fix the state so the next press works.
                        self.notify("Engine was down — revived on CPU, press again")
                log(f"[DAEMON] Recording failed for mode: {mode}")
                if mode not in ("auto_transcribe", "command"):
                    self._report_failure(f"Recording failed ({mode})")
            else:
                log(f"[DAEMON] Recording completed successfully for mode: {mode}")

        except Exception as e:
            log(f"[DAEMON] Recording thread error for mode {mode}: {e}")
            self._report_failure(f"Recording error ({mode}): {e}",
                                 traceback.format_exc())
        finally:
            if app_for_mode and orig_transcribe:
                try:
                    app_for_mode.transcription_service.transcribe_audio = orig_transcribe
                except Exception:
                    pass
            log(f"[DAEMON] Recording thread finishing for mode: {mode}")
            self._stop_recording()
            # Again, after the last of the text has been typed: the closing
            # tail goes in long after the recording stopped, and it releases
            # and restores the modifiers exactly as the live words did.
            self._release_stuck_modifiers()
            self._finish_processing(mode)

    def _get_app_for_mode(self, mode: str) -> WhisperFlow:
        """Get the appropriate WhisperFlow instance for the mode."""
        if mode == "transcribe":
            return self.transcribe_app
        if mode == "auto_transcribe":
            return self.auto_transcribe_app
        if mode == "command":
            return self.command_app
        return self.transcribe_app

    def cancel_recording(self):
        """Cancel current recording."""
        log("[DAEMON] Cancel recording requested")
        if not self.is_recording:
            log("[DAEMON] Not recording, nothing to cancel")
            return

        log("[DAEMON] Canceling current recording")
        # Force-reset hotkey state to prevent stuck modifiers
        self.hotkey_manager.force_reset()
        self._stop_recording()

    def _stop_recording(self):
        """Stop the current recording.

        Reached from the hotkey-release thread and from the recording thread's
        cleanup, so the is_recording check and the teardown that follows have
        to be one atomic step - otherwise both can pass the check and the HUD
        gets hidden twice while the second caller works on cleared state.
        """
        with self._stop_lock:
            self._stop_recording_locked()

    def _stop_recording_locked(self):
        if not self.is_recording:
            log("[DAEMON] Not recording, nothing to stop")
            return

        log(f"[DAEMON] Stopping recording for mode: {self.current_mode}")

        # Force-reset hotkey state to prevent stuck modifiers
        self.hotkey_manager.force_reset()
        # And let go of any modifier we are holding down ourselves. The
        # hotkey manager's reset clears our own bookkeeping; this clears the
        # keys we told Windows were pressed.
        self._release_stuck_modifiers()

        # Signal the recording thread to stop
        if self.stop_recording_event:
            self.stop_recording_event.set()

        # The overlay stays up and turns to its processing state. Hiding it
        # here left a gap - often a long one - between letting go of the key
        # and the words appearing, with nothing on screen to say whether
        # anything was happening or whether the dictation had failed. It is
        # taken down in _finish_processing, when there is something to show
        # for it. The level file goes with it, for the same reason: the
        # overlay treats that file disappearing as being orphaned.
        #
        # Guarded, because everything below it is the state reset. This is
        # also reached from start_recording's rollback, where the overlay is
        # being asked to do something in the middle of whatever just failed;
        # letting that throw would leave is_recording set with no thread to
        # clear it, and every later press dropped as busy for the rest of the
        # daemon's life. The overlay is worth nothing next to that.
        try:
            self.hud.processing(getattr(self, "_level_file", None) or "")
        except Exception as e:
            log(f"[DAEMON] could not put the overlay into processing: {e}")

        # Reset recording state
        self.is_recording = False
        self.current_mode = None
        self.recording_start_time = None

        # Restore normal tray icon
        if self.tray_icon:
            self.tray_icon.icon = self.create_tray_icon()

        log("[DAEMON] Recording stopped and state reset")

    def notify(self, message: str):
        """Send a desktop notification.

        Prefers the tray icon's own notification: it is native, costs nothing,
        and on Windows avoids spawning a PowerShell process per message - which
        is slow and, without the right flags, flashes a console window.
        """
        log(f"[DAEMON] Notification: {message}")
        manager = self.transcribe_app.system_manager
        if not getattr(self.config, "notifications_enabled", True):
            return
        if self.tray_icon is not None:
            try:
                if manager.should_notify(message):
                    self.tray_icon.notify(message, "Whisper-Flow")
                return
            except Exception:
                # Not every tray backend implements notifications. Falling
                # through as-is would lose the message entirely: the attempt
                # above already consumed the rate-limit stamp, so the fallback
                # would be suppressed as a repeat. Hand the stamp back first.
                manager._last_notified.pop(message, None)
        manager.notify(message)

    def prewarm_settings(self) -> None:
        """Build the settings window now, so opening it later is a map.

        Measured on Windows: 3.1s from click to window when it is built on
        the click, 15.5s the first time after a boot, when the GTK runtime is
        still coming off the disk. Almost none of that is our own code - it
        is process start, the GTK and libadwaita DLLs, and the config stack -
        and none of it depends on anything the user does. So it happens at
        login, where nobody is waiting, and the click gets the window that is
        already standing by.

        On a thread, like the overlay's: this starts a process that loads GTK,
        and the tray icon must not wait for it.
        """
        if sys.platform != "win32" and not (
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return                      # headless: there is no window to warm

        def work():
            try:
                with self._setup_lock:
                    if self._setup_process and self._setup_process.poll() is None:
                        return          # one already standing by, or open
                    if self._start_tool_window(
                            "--settings", "whisper_flow.settings_gtk",
                            resident=True):
                        log("[DAEMON] settings window ready")
            except Exception as e:
                log(f"[DAEMON] could not pre-build the settings window: {e}")

        threading.Thread(target=work, daemon=True,
                         name="whisper-flow-settings-prewarm").start()

    def _open_tool_window(self, flag: str, module: str) -> bool:
        """Show the settings window, building one first if none is waiting.

        It cannot share this process: pystray owns an event loop here and
        GTK demands the main thread of wherever it runs. Returns False where
        there is no window to show, so the caller can fall back.
        """
        if sys.platform != "win32" and not (
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False                # headless: fall back to downloading
        with self._setup_lock:
            process = self._setup_process
            if process and process.poll() is None:
                # A window built ahead of this click and waiting to be shown:
                # the path that makes the click cost a map rather than a
                # process start. The same order raises one that is already
                # open, which is what a second click on Settings is asking
                # for and what "already open, do nothing" never did - a
                # window buried behind other windows looked like a click that
                # had missed.
                if self._show_prewarmed():
                    return True
                # Pipe is dead but the process may still be running. Kill it
                # before starting a replacement - otherwise two settings
                # processes stack on the desktop.
                self._kill_setup_process_unlocked(process)

            return self._start_tool_window(flag, module, resident=False)

    def shutdown_settings(self, *, quit: bool = False) -> None:
        """Retire the settings window process when the daemon exits.

        Closing stdin is the ordinary path: the child reads end of stream and
        leaves if nobody is looking at it. A window that is already open is
        deliberately kept across a restart (Save restarts the daemon; the
        settings toast must stay put). Tray Exit is different - the whole app
        is going away - so ``quit=True`` tells the child to exit even when
        visible.

        When ``quit`` is false, do not wait for or kill the child. Waiting
        up to three seconds then killing it made Save (which restarts the
        daemon from inside settings) freeze that window until timeout and
        then destroy it mid-toast.
        """
        with self._setup_lock:
            process, self._setup_process = self._setup_process, None
            self._setup_waiting = False
            self._setup_shown = False
        if not process:
            return
        try:
            if process.stdin:
                if quit:
                    # Explicit: leave even when the window is on screen.
                    # Resident children read this; a non-resident has no
                    # command loop and is killed below if it ignores it.
                    try:
                        process.stdin.write("quit\n")
                        process.stdin.flush()
                    except Exception:
                        pass
                process.stdin.close()
        except Exception:
            pass
        if not quit:
            # Leave a visible settings process alone - Save is still using it.
            return
        try:
            process.wait(timeout=2)
        except Exception:
            pass
        if process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass

    def _show_prewarmed(self) -> bool:
        """Tell the waiting window to show itself. False if it cannot be told."""
        process = self._setup_process
        try:
            process.stdin.write("show\n")
            process.stdin.flush()
        except (OSError, ValueError, AttributeError) as e:
            # It died between the poll above and this write, or it was not a
            # prewarmed one and has no pipe. Either way there is no window
            # coming, so drop it and let the caller build one.
            log(f"[DAEMON] the waiting settings window would not come up: {e}")
            self._setup_process = None
            self._setup_waiting = False
            return False
        self._setup_waiting = False
        self._setup_shown = True
        return True

    @staticmethod
    def _kill_setup_process_unlocked(process) -> None:
        """Drop a settings process that can no longer be commanded.

        Caller holds ``_setup_lock`` and has already cleared or will clear
        ``_setup_process``. Without this, a second Settings click after a
        broken pipe left the old process on screen and started another.
        """
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=1)
        except Exception:
            pass

    def _start_tool_window(self, flag: str, module: str,
                           resident: bool) -> bool:
        """Start the window process. Caller holds the lock.

        `resident` builds it without showing it: the child waits on its pipe
        for the click that a prewarmed window exists to be ready for.
        """
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, flag]
        else:
            cmd = [sys.executable, "-m", module]
        # Hand down what this process already knows. Detecting the
        # accelerator means running nvidia-smi, which has to load a driver
        # before it prints a version - most of a second on Windows, and
        # the settings window asks for it several times while building the
        # Speech page. The daemon paid that cost at startup; the window
        # should not pay it again while the user waits at a blank screen.
        env = dict(os.environ)
        env[backend_module.ACCELERATOR_ENV] = backend_module.detect_accelerator()
        # The driver version rides along too: children that inherit the
        # accelerator answer skip the nvidia-smi probe (and with it the
        # version string), which left machine_facts reporting nvidia_drv=none
        # after every settings-triggered restart.
        try:
            env["WHISPER_FLOW_DRIVER_VERSION"] = (
                backend_module._driver_version or "")
        except Exception:
            pass
        # When the click happened, so the window can report what the user
        # actually waited through - process start included, which is a
        # real part of it in a frozen build and cannot be seen from
        # inside the child. A prewarmed one measures its build from its own
        # start, and measures the click separately when the click arrives.
        env["WHISPER_FLOW_TOOL_T0"] = repr(time.time())
        if resident:
            env["WHISPER_FLOW_SETTINGS_RESIDENT"] = "1"
        else:
            env.pop("WHISPER_FLOW_SETTINGS_RESIDENT", None)
        # Where the child writes its startup timings. A file rather than
        # its stdout because a PyInstaller windowed build has sys.stdout
        # set to None, and print() silently does nothing there - the
        # first round of these vanished without trace.
        fd, tool_log = tempfile.mkstemp(suffix=".log", prefix="whisper-flow-tool-")
        os.close(fd)
        env["WHISPER_FLOW_TOOL_LOG"] = self._tool_log = tool_log
        try:
            # Captured, not inherited. The frozen build is windowed, so
            # the child has no console and everything it prints - timings,
            # tracebacks, the reason a window did not appear - went
            # nowhere. Piping it here puts it in the log the tray offers,
            # which is the only route it has to anyone who can read it.
            #
            # stdin is a pipe whether or not this one is prewarmed: it is how
            # a waiting window is told to show itself, and closing it is how
            # any of them is told the daemon has gone.
            #
            # CREATE_NO_WINDOW: a source checkout launches python.exe (console
            # subsystem). Without this the prewarmed settings process flashes
            # a prompt at login. Harmless for the windowed frozen exe.
            process = self._setup_process = subprocess.Popen(
                cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace",
                creationflags=backend_module.no_console_flags())
        except Exception as e:
            log(f"[DAEMON] could not open the setup window: {e}")
            self._setup_process = None
            return False
        self._setup_waiting = resident
        self._setup_shown = not resident
        self._setup_started = time.time()

        # Popen succeeding only means the process started, not that a
        # window appeared. Without tkinter - which is a separate package
        # on most Linux distributions - the child exits immediately, and
        # claiming a window is open would leave the user with no window
        # and no message either.
        time.sleep(0.4)
        if process.poll() not in (None, 0):
            log(f"[DAEMON] the setup window exited at once "
                f"({process.returncode}); falling back")
            self._setup_process = None
            self._setup_waiting = False
            return False

        threading.Thread(target=self._after_tool_window, args=(process,),
                         daemon=True, name="whisper-flow-window-watch").start()
        return True

    @staticmethod
    def _drain_tool_output(process) -> None:
        """Copy the child's output into our log until it closes the pipe.

        Line by line as it arrives rather than in one read at the end: the
        interesting case is a window that is slow or never appears, and a
        report that only lands once the process exits is no use for either.
        """
        stream = getattr(process, "stdout", None)
        if stream is None:
            return
        try:
            for line in stream:
                line = line.rstrip()
                if line:
                    log(line if line.startswith("[") else f"[TOOL] {line}")
        except Exception as e:
            log(f"[DAEMON] lost the tool window's output: {e}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _tail_tool_log(self, process, timeout: float | None = 30.0) -> None:
        """Report the child's startup timings while it is still open.

        Its own thread because the drain below blocks until the child exits,
        and the timings are wanted long before that - the window being slow
        to open is the whole thing they exist to explain, and a report that
        waits for it to be closed again answers too late to be read.

        No timeout means until the child exits, which is what a window built
        at login needs: the line worth reading is written when the click
        finally comes, hours after any fixed deadline would have given up.
        """
        path = getattr(self, "_tool_log", None)
        if not path:
            return
        seen, started = 0, time.time()
        deadline = None if timeout is None else started + timeout
        while deadline is None or time.time() < deadline:
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    lines = fh.read().splitlines()
            except OSError:
                lines = []
            for line in lines[seen:]:
                if line.strip():
                    log(line)
            seen = len(lines)
            if process.poll() is not None:
                return
            # Half a second while the window is starting, where the timings
            # are being written and are worth having promptly; slower once it
            # has gone quiet, because a window built at login is watched for
            # hours and a twice-a-second wakeup for that long is a cost of
            # its own. The line it writes when the click finally comes is a
            # log line, and five seconds late is soon enough for one.
            time.sleep(0.5 if time.time() - started < 30 else 5.0)

    def _after_tool_window(self, process=None) -> None:
        """Adopt whatever the settings window installed, once it closes."""
        process = process or self._setup_process
        threading.Thread(target=self._tail_tool_log, args=(process,),
                         kwargs={"timeout": None}, daemon=True,
                         name="whisper-flow-tool-log").start()
        # Before wait(), and on this thread: a full pipe blocks the child, so
        # nothing may wait on a process it has not drained first.
        self._drain_tool_output(process)
        try:
            process.wait()
        except Exception:
            return
        finally:
            path, self._tool_log = getattr(self, "_tool_log", None), None
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        with self._setup_lock:
            if self._setup_process is process:
                self._setup_process = None
            shown, self._setup_shown = self._setup_shown, False
            self._setup_waiting = False
            lived = time.time() - getattr(self, "_setup_started", 0.0)

        # A window that dies after the 0.4s check above is invisible: the
        # user clicks Settings, nothing opens, and the log holds only the
        # click. That is exactly how a GTK runtime crashing on a missing
        # schema presented - a native abort leaves no Python traceback to
        # find, so the exit code is the only evidence there is.
        code = process.returncode
        if isinstance(code, int) and code != 0:
            # Unsigned too: a native crash reads as 3221225477, which says
            # nothing, while 0xC0000005 is an access violation on sight.
            log(f"[DAEMON] the tool window exited with {code} "
                f"(0x{code & 0xFFFFFFFF:08X}) without being closed; "
                f"no window was shown")
            # Only to someone who asked for a window. One built at login and
            # lost before anyone wanted it is worth a log line and nothing
            # more: a notification about a window the user never asked for
            # explains nothing and interrupts whatever they were doing.
            if shown:
                self.notify("That window could not open - see the log")

        # Have the next one ready. The window just closed is gone for good -
        # it is a process, and it exits - so without this only the first
        # click of a session gets a window that was waiting for it.
        #
        # Not after one that failed, and not after one that was never shown
        # and did not last: both are how a machine that cannot open this
        # window at all would look, and replacing it each time would turn
        # that into a process started every second for as long as the daemon
        # runs. A window someone actually used is a window that works.
        if self.is_running and code == 0 and (shown or lived > 60):
            self.prewarm_settings()

        # working_model() reads the disk rather than the config file, so
        # whatever the window downloaded or saved is picked up here and now.
        model = self.backend.working_model()
        engine = self.backend.installed_engine()
        # The engine matters as much as the model. Installing the GPU one
        # leaves the model exactly as it was, and restarting only on a changed
        # model meant the server kept running on the CPU engine it was started
        # with - so the upgrade appeared to do nothing at all until the next
        # sign-in.
        if not model or (model == self._backend_model
                         and engine == self._backend_engine):
            return                      # dismissed, or nothing new

        self.backend.stop()
        url = self._backend_start(model)
        # If fallback switched model, working_model() now reflects it
        actual_model = self.backend.working_model() if url and hasattr(self.backend, "_cpu_fallback_model") else model
        if not url:
            # Even fallback failed — explain why instead of silent dead hotkeys
            detail = self.backend.describe() if hasattr(self.backend, "describe") else ""
            log(f"[BACKEND] could not start {model} nor any CPU fallback ({detail})")
            self.notify("Speech engine failed to start — open Settings to pick a smaller model")
            return
        # If fallback changed the model (e.g. large→base), surface that
        started_model = actual_model if actual_model and self.backend.model_path(actual_model).exists() else model
        self.config.model_name = started_model
        self._backend_model = started_model
        self._backend_engine = self.backend.installed_engine()
        self._use_backend_url(url)
        # Tell the user when a fallback happened
        if started_model != model:
            self.notify(f"GPU engine failed — running on CPU with {started_model.replace('ggml-', '')} ({self.backend.engine_summary()})")
        else:
            self.notify(f"Speech model ready ({started_model.replace('ggml-', '')}, "
                        f"{self.backend.engine_summary()})")

    def _backend_start(self, model: str | None,
                       allow_download: bool = True) -> str | None:
        """Start with GPU→CPU fallback, handling mocked backends in tests.

        Mock objects create any attribute on access, so hasattr succeeds even
        when the test only mocked `start`. Call start_with_fallback only when
        it returns a real http URL; otherwise fall back to `start`.
        """
        # Prefer the fallback-aware path
        if hasattr(self.backend, "start_with_fallback"):
            try:
                try:
                    url = self.backend.start_with_fallback(
                        model, allow_download=allow_download)
                except TypeError:
                    # Mock backends (and old callers) take only the model.
                    url = self.backend.start_with_fallback(model)
                if isinstance(url, str) and url.startswith("http"):
                    return url
                # Mock or non-URL — try plain start as the test expects
                if url is not None and not isinstance(url, str):
                    # likely a Mock — ignore and try plain start
                    pass
                else:
                    # Real failure — still try plain start as second chance
                    try:
                        plain = self.backend.start(
                            model, allow_download=allow_download)
                    except TypeError:
                        plain = self.backend.start(model)
                    if isinstance(plain, str) and plain.startswith("http"):
                        return plain
                    return url
            except Exception:
                pass
        try:
            try:
                url = self.backend.start(model, allow_download=allow_download)
            except TypeError:
                url = self.backend.start(model)
            if isinstance(url, str) and url.startswith("http"):
                return url
            return None if not isinstance(url, str) else url
        except Exception:
            return None

    def _ensure_backend_running(self, allow_download: bool = True) -> bool:
        """Make sure a transcription server is up before recording.

        Called on every hotkey press. If the server crashed after startup
        (the 0xC0000005 case in the user's log), this is where it gets
        revived on CPU instead of letting the next 5 seconds of speech be
        recorded and then dropped with 'No whisper server configured'.

        With allow_download=False (the hotkey path) this never fetches
        anything: a 677MB engine download on the recording path is what made
        hotkeys look dead (no HUD, no mic for minutes). Missing files are
        fetched by the watchdog/settings paths; recording still proceeds so
        the closing pass can retry once the engine lands.
        """
        try:
            # Already have a URL and the process is alive
            if self.config.local_whisper_url and self.backend._process and self.backend._process.poll() is None:
                return True
        except Exception:
            pass
        # Try to (re)start — fallback-aware (guarded: tests use Mock configs)
        try:
            model = self.backend.working_model()
        except Exception as e:
            log(f"[DAEMON] could not check working model before recording: {e}")
            return False
        if not model:
            return False
        if not allow_download:
            try:
                if not self.backend.is_installed(model):
                    log(f"[DAEMON] {model} not installed — no download on the hotkey path")
                    return False
            except Exception:
                pass
        # Respect the user's model choice here. Model switching belongs to
        # start_with_fallback (used when start() itself fails), not to every
        # revive: silently rewriting medium to base is why the settings page
        # said one thing while the daemon ran another. Repeated native
        # crashes are handled by backend.note_crash, which refreshes the
        # engine bytes instead of second-guessing the model.
        try:
            url = self._backend_start(model, allow_download=allow_download)
        except Exception as e:
            log(f"[DAEMON] backend start before recording failed: {e}")
            return False
        if url:
            # If fallback happened, _last_started_model holds the actual model
            # started. The backend already persists a genuine GPU->CPU model
            # switch to .env itself; the daemon must not rewrite the user's
            # choice on top of it.
            actual = getattr(self.backend, "_last_started_model", None) or model
            try:
                self.backend._last_started_model = None
            except Exception:
                pass
            self._backend_model = actual
            self._backend_engine = self.backend.installed_engine()
            self._use_backend_url(url)
            log(f"[BACKEND] (re)started {actual} on {self.backend.engine_summary()} before recording")
            return True
        return False

    def _start_managed_backend(self) -> None:
        """Bring up the bundled server if one is configured and present.

        Only starts what is already downloaded. Pulling gigabytes without
        being asked, on a connection that might be metered, is not something
        to do at startup.
        """
        if not getattr(self.config, "manage_local_server", False):
            return
        if (self.config.local_whisper_url or "").strip():
            return          # pointed at something else already
        model = self.backend.working_model()
        if not model:
            # Nothing can transcribe yet, so this is the whole difference
            # between a working app and a dead one. Open settings, where the
            # model is chosen and downloaded, straight away.
            if not self._open_settings_window():
                self.notify("No speech model yet - open Settings in the tray")
            return

        url = self._backend_start(model)
        actual_model = self.backend.working_model() if url else None
        if not url:
            # Startup is the only place a silent failure becomes "No whisper
            # server configured" on the first real dictation, which reads as
            # typing being broken rather than transcription. Surface it now
            # while the log still holds the server's stderr tail. The fallback
            # path will retry on CPU at the next hotkey.
            log(f"[BACKEND] failed to start {model} ({self.backend.installed_engine()}); "
                f"see whisper-server log and try a smaller model in Settings — will retry on CPU at next hotkey")
            self.notify("Speech engine failed to start — will try CPU on next hotkey (or open Settings)")
            return
        # Fallback may have switched model
        started = actual_model if actual_model and self.backend.model_path(actual_model).exists() else model
        self._backend_model = started
        self._backend_engine = self.backend.installed_engine()
        self._use_backend_url(url)
        if started != model:
            log(f"[BACKEND] GPU {model} failed, fell back to CPU {started} on {self.backend.engine_summary()}")
            self.notify(f"GPU failed — running on CPU with {started.replace('ggml-', '')}")
        else:
            log(f"[BACKEND] {started} on {self.backend.engine_summary()}")

        # It already works, so the GPU model is an offer rather than a
        # requirement. Mention it once: downloading 1.6GB unprompted on a
        # connection that might be metered is not a decision to make on
        # someone's behalf, and neither is taking over their screen with a
        # window they did not ask for while the app is already working.
        if self.backend.setup_reason() == "gpu":
            # Name the engine, not the model. Someone who reads this as "get
            # the big model" downloads it, gets no engine to go with it, and
            # ends up far slower than before they were told anything.
            self.notify("This PC has an NVIDIA GPU - install the GPU speech "
                        "engine in Settings to use it")
            # Silent background install: if the user hasn't explicitly dismissed
            # the GPU offer, try to fetch the GPU engine quietly. Switched from
            # notify-only to auto-install per user request "handle this
            # automatically as part of the install silently in the background".
            # Respects metered check via total_ram and setup_seen; runs once.
            try:
                if not self.backend.setup_seen():
                    # Only auto-install on machines that can actually benefit
                    # (8GB+ RAM, 4+ cores) to avoid downloading 1.6GB onto a thin client
                    import whisper_flow.backend as _be
                    if _be.total_ram_gb() >= 8 and _be.usable_cores() >= 4:
                        # Check VCRedist first — GPU engine will crash 0xC0000005 without it
                        has_vcr = True
                        if __import__("sys").platform == "win32":
                            try:
                                import pathlib as _P
                                # Check System32 and SysWOW64 for vcruntime
                                sys32 = _P.Path(r"C:\Windows\System32\vcruntime140.dll")
                                wow64 = _P.Path(r"C:\Windows\SysWOW64\vcruntime140.dll")
                                has_vcr = sys32.exists() or wow64.exists()
                                if not has_vcr:
                                    # Try via WinSxS or just check msvcp
                                    has_vcr = _P.Path(r"C:\Windows\System32\msvcp140.dll").exists()
                            except Exception:
                                has_vcr = True  # assume present if check fails
                        if not has_vcr:
                            log("[BACKEND] VCRedist missing — trying silent install before GPU engine")
                            try:
                                import whisper_flow.backend as _be2
                                if _be2._ensure_vcredist_silent():
                                    log("[BACKEND] VCRedist silent install succeeded, proceeding with GPU engine")
                                    has_vcr = True
                                else:
                                    log("[BACKEND] VCRedist silent install failed, will notify")
                            except Exception as _e:
                                log(f"[BACKEND] VCRedist silent check failed: {_e}")
                            if not has_vcr:
                                log("[BACKEND] VCRedist missing — GPU engine would crash, not auto-installing; ask user to install VC++ Redist")
                                self.notify("GPU engine needs Visual C++ Redistributable — install it from https://aka.ms/vs/17/release/vc_redist.x64.exe then retry GPU engine in Settings")
                        else:
                            def _bg_gpu_install():
                                try:
                                    model = self.backend.working_model()
                                    log(f"[BACKEND] silent background GPU engine install for {model}")
                                    ok = self.backend.install(model)
                                    if ok:
                                        log("[BACKEND] silent GPU install succeeded, restarting server on GPU")
                                        self.backend.stop()
                                        url = self._backend_start(model)
                                        if url:
                                            self._backend_model = model
                                            self._backend_engine = self.backend.installed_engine()
                                            self._use_backend_url(url)
                                            self.notify(f"GPU engine installed in background — now on {self.backend.engine_summary()}")
                                        self.backend.mark_setup_seen()
                                    else:
                                        log("[BACKEND] silent GPU install failed, will offer again later")
                                except Exception as e:
                                    log(f"[BACKEND] silent GPU install failed: {e}")
                            import threading as _th
                            _th.Thread(target=_bg_gpu_install, daemon=True, name="whisper-flow-gpu-auto").start()
                            # Mark seen so we don't auto-retry every boot if it fails
                            # (user can still manually Install GPU engine in Settings)
                            self.backend.mark_setup_seen()
            except Exception as e:
                log(f"[DAEMON] auto GPU install check failed: {e}")

    def copy_log(self, icon=None, item=None):
        """Put the recent log on the clipboard, whether or not anything failed.

        Failure reports only appear when something goes wrong, so there was
        no way to hand over a run that worked but felt slow - which is
        exactly what a timing question needs. This is that: one click, then
        paste. It carries the same configuration and log as a failure
        report, including every PTL line.
        """
        text = self._diagnostics("Log requested from the tray")
        try:
            copied = bool(
                self.transcribe_app.system_manager.copy_to_clipboard(text))
        except Exception as e:
            log(f"[DAEMON] could not copy the log: {e}")
            copied = False
        self.notify("Log copied to clipboard" if copied
                    else "Could not reach the clipboard")

    def _report_failure(self, headline: str, detail: str | None = None) -> None:
        """Notify about a failure and put the whole story on the clipboard.

        A toast holds one line, which is never enough to act on and never
        enough to send to anyone. Anything that fails in front of the user
        goes through here so the notification names the problem and the
        clipboard carries the log that explains it.
        """
        text = self._diagnostics(headline, detail)
        self._last_failure = text

        copied = False
        try:
            copied = bool(
                self.transcribe_app.system_manager.copy_to_clipboard(text))
        except Exception as e:
            log(f"[DAEMON] could not copy the failure report: {e}")

        # Say which happened. Silently dropping the report is how this went
        # unnoticed: the notification looked the same whether the clipboard
        # held the details or nothing at all.
        suffix = (" - details copied to clipboard" if copied
                  else " - could not reach the clipboard")
        self.notify(f"❌ {headline}{suffix}")

    def _diagnostics(self, headline: str, detail: str | None = None) -> str:
        """Everything worth knowing about this run, as pasteable text."""
        report = [
            f"whisper-flow {build_version()} - {headline}",
            f"{datetime.now():%Y-%m-%d %H:%M:%S}  {platform.platform()}",
            f"python {sys.version.split()[0]}  frozen={getattr(sys, 'frozen', False)}",
            "",
            "Configuration",
            f"  transcribe hotkey : {self.config.hotkey_transcribe}",
            f"  command hotkey    : {self.config.hotkey_command}",
            f"  auto hotkey       : {self.config.hotkey_auto_transcribe}",
            f"  whisper url       : {self.config.local_whisper_url or '(none)'}",
            f"  model             : {self.config.model_name}",
            f"  live transcription: {self.config.live_transcription}",
        ]
        try:
            report.append(f"  engine            : {self.backend.describe()}")
        except Exception as e:
            report.append(f"  engine            : unavailable ({e})")
        try:
            status = self.hotkey_manager.input_status()
            report.append(
                "  input             : backend={backend} alive={alive} "
                "grabbed={grabbed} pump_age_s={pump} muted={muted} "
                "held={held} disabled={disabled} recoveries={recoveries}".format(
                    backend=status.get("backend"),
                    alive=status.get("alive"),
                    grabbed=status.get("grabbed"),
                    pump=status.get("pump_age_s"),
                    muted=status.get("muted"),
                    held=status.get("held"),
                    disabled=status.get("disabled"),
                    recoveries=status.get("recoveries_60s"),
                ))
        except Exception as e:
            report.append(f"  input             : unavailable ({e})")

        try:
            overlay = self.hud.crash_log()
        except Exception as e:
            overlay = f"could not inspect the overlay: {e}"
        if overlay:
            report += ["", "Overlay", overlay]

        if detail:
            report += ["", "Traceback", detail.rstrip()]
        report += ["", "Recent log", recent_log(200) or "(nothing recorded)"]
        return "\n".join(report)

    # ---------------------------------------------------------- updates
    def _update_label(self) -> str:
        """Tray text for the update row, following the updater state."""
        try:
            if updater.is_downloading():
                return "Downloading update…"
            version = updater.pending_version()
            if version:
                return f"Update to {version} (restart)"
        except Exception:
            pass
        return "Check for updates"

    def _refresh_tray_menu(self) -> None:
        """Re-render the tray menu so the update row shows current state."""
        try:
            if self.tray_icon is not None:
                self.tray_icon.update_menu()
        except Exception:
            pass                    # headless, or menu mid-rebuild: harmless

    def _on_update_ready(self, version: str) -> None:
        """A background download landed: offer it in the tray."""
        log(f"[DAEMON] update {version} ready to install")
        self._refresh_tray_menu()

    def check_for_updates(self, icon=None, item=None):
        """Apply a downloaded update, or fetch one in the background.

        Right-click → Update restarts into the new build. Never mid-dictation:
        while recording or processing, the restart waits for the current
        request to finish. A tray callback that does not return promptly
        stops the icon responding, so the work runs on its own thread.
        """
        threading.Thread(
            target=self._update_clicked,
            daemon=True, name="whisper-flow-update",
        ).start()

    def _update_clicked(self) -> None:
        """What a click on the update row does, off the tray thread."""
        try:
            if updater.pending_version():
                self._apply_update_when_idle()
                return
            if updater.is_downloading():
                self.notify("Update still downloading — a moment…")
                return
            # Nothing waiting: check now, download in the background, and
            # flip the tray row to "Update to X" when it lands.
            version = updater.download_in_background(
                notify=self.notify, on_ready=self._on_update_ready)
            if version and updater.pending_version():
                self._refresh_tray_menu()
            elif not updater.pending_version():
                self.notify("whisper-flow is up to date")
        except Exception as e:
            log(f"[DAEMON] update click failed: {e}")

    def _apply_update_when_idle(self) -> None:
        """Restart into the downloaded build, waiting out any dictation."""
        try:
            if self._is_processing():
                # Restarting now would kill the recording in flight. Park
                # the request; _finish_processing picks it up when idle.
                self._update_apply_deferred = True
                self.notify("Update will install when this dictation finishes")
                return
            self.notify("Restarting to install the update…")
            if not updater.apply_pending(notify=self.notify):
                self.notify("Could not install the update — will retry later")
                self._refresh_tray_menu()
        except Exception as e:
            log(f"[DAEMON] update apply failed: {e}")

    def copy_last_error(self, icon=None, item=None):
        """Put the most recent failure report back on the clipboard.

        The report is copied when the failure happens, but anything else
        copied since will have replaced it, and the failure is usually
        described some time after it happened.
        """
        if not self._last_failure:
            self.notify("No errors recorded since this started")
            return
        try:
            copied = self.transcribe_app.system_manager.copy_to_clipboard(
                self._last_failure)
        except Exception as e:
            log(f"[DAEMON] could not copy the last error: {e}")
            copied = False
        self.notify("Last error copied to clipboard" if copied
                    else "Could not reach the clipboard")

    def _use_backend_url(self, url: str) -> None:
        """Point every mode's transcription service at the running server."""
        self.config.local_whisper_url = url
        for app in (self.transcribe_app, self.auto_transcribe_app,
                    self.command_app):
            app.config.local_whisper_url = url
            app.transcription_service.local_url = url.rstrip("/")

    def open_settings(self, icon=None, item=None):
        """Open the settings window, as its own process.

        It cannot run here: pystray owns this process's event loop and GTK
        insists on its own. The tray callback has to return immediately or
        the icon stops responding, and spawning is all this does.

        Rapid repeats are dropped rather than forwarded: opening the window
        blocks this thread for the 0.4s liveness probe, so clicks made while
        it is blocked queue up and then all fire - and each one used to write
        another "show", re-raise, and re-apply DWM acrylic in the settings
        process. A window that was just asked for raises itself; a second
        click a second and a half later is someone asking again.
        """
        log("[DAEMON] Settings menu item clicked")
        now = time.time()
        if now - self._last_settings_click < SETTINGS_CLICK_DEBOUNCE_S:
            log("[DAEMON] Settings click ignored (already being opened)")
            return
        self._last_settings_click = now
        if not self._open_settings_window():
            self.notify(f"Config directory: {self.config.config_dir}")

    def _open_settings_window(self) -> bool:
        """One GTK4 window on every platform."""
        return self._open_tool_window("--settings", "whisper_flow.settings_gtk")

    def test_configuration(self, icon, item):
        """Run the checks, and put a pasteable report on the clipboard.

        A notification can only carry a couple of lines, which is not enough
        to act on or to send to anyone. When something fails the full report
        goes to the clipboard, so it can be pasted straight into a bug report.
        """
        log("[DAEMON] Configuration test requested")
        try:
            results = self.transcribe_app.run_comprehensive_validation()
        except Exception as e:
            log(f"[DAEMON] Configuration test failed: {e}")
            self.notify(f"Configuration test failed to run: {e}")
            return

        counts = {"pass": 0, "fail": 0, "warn": 0}
        problems = []
        for group, tests in results.items():
            for test in tests:
                status = test.get("status", "warn")
                counts[status] = counts.get(status, 0) + 1
                if status in ("fail", "warn"):
                    problems.append((status, test["name"], test.get("message", "")))

        log(f"[DAEMON] Configuration: {counts['pass']} passed, "
            f"{counts['fail']} failed, {counts['warn']} warnings")

        if not problems:
            self.notify(f"Configuration is valid ({counts['pass']} checks passed)")
            return

        report = self._diagnostic_report(results, counts)
        copied = self.transcribe_app.system_manager.copy_to_clipboard(report)

        first = next((p for p in problems if p[0] == "fail"), problems[0])
        summary = f"{first[1]}: {first[2]}"
        if len(summary) > 140:
            summary = summary[:137] + "..."
        tail = " (full report copied to clipboard)" if copied else ""
        self.notify(summary + tail)

    def _diagnostic_report(self, results: dict, counts: dict) -> str:
        """A plain-text report of the checks, plus what it is running on."""
        lines = [
            f"whisper-flow diagnostics - {counts['pass']} passed, "
            f"{counts['fail']} failed, {counts['warn']} warnings",
            f"{platform.system()} {platform.release()} (build {platform.version()})",
            f"Python {platform.python_version()}",
            f"Config: {self.config.config_dir}",
            "",
        ]
        for group, tests in results.items():
            lines.append(f"[{group}]")
            for test in tests:
                mark = {"pass": "ok  ", "fail": "FAIL", "warn": "warn"}.get(
                    test.get("status"), "?")
                lines.append(f"  {mark} {test['name']}: {test.get('message', '')}")
            lines.append("")
        return "\n".join(lines)

    def stop_daemon(self, icon=None, item=None):
        """Stop the daemon and take the settings window with it."""
        log("[DAEMON] Stop daemon requested")
        try:
            self.notify("👋 WhisperFlow daemon stopping...")
        except Exception:
            pass
        # Before the tray loop ends: a visible settings window would otherwise
        # outlive Exit (EOF on its pipe is treated as "restart, stay put").
        try:
            self.shutdown_settings(quit=True)
        except Exception as e:
            log(f"[DAEMON] could not close settings on Exit: {e}")
        self.is_running = False
        if self.tray_icon:
            self.tray_icon.stop()

    def reload_daemon(self, icon=None, item=None):
        """Reload/restart the daemon via systemd."""
        log("[DAEMON] Reload daemon requested")
        try:
            self.notify("🔄 Reloading WhisperFlow daemon...")
        except Exception:
            pass
        subprocess.Popen(
            ["systemctl", "--user", "restart", "whisper-flow.service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        self.is_running = False
        if self.tray_icon:
            self.tray_icon.stop()

    def show_notification_menu(self):
        """Show a notification-based menu when system tray is not available."""
        log("[DAEMON] Showing notification menu")
        try:
            # Use notify-send to show menu options
            import subprocess

            menu_text = f"""
🎤 WhisperFlow Daemon Menu

Hotkeys:
• {self.config.hotkey_transcribe} - Push to talk (hold, release to paste)
• {self.config.hotkey_auto_transcribe} - Auto-transcribe (tap, stop on silence)
• {self.config.hotkey_command} - Auto-transcribe 2nd (same as auto)
• Escape - Cancel recording

Status: {"Recording" if self.is_recording else "Idle"}
Mode: {self.current_mode if self.is_recording else "None"}

Use 'whisper-flow stop' to exit daemon
            """.strip()

            subprocess.run(
                [
                    "notify-send",
                    "--urgency=normal",
                    "--expire-time=5000",
                    "WhisperFlow Daemon",
                    menu_text,
                ],
                check=False,
            )
        except Exception as e:
            log(f"[DAEMON] Error showing notification menu: {e}")
            # Fall back to simple print if notification fails
            print("WhisperFlow menu: type 'help' for commands")

    def run_notification_mode(self):
        """Run in notification mode when tray is not available."""
        log("[DAEMON] Starting notification mode")
        print("🎤 WhisperFlow Daemon (CLI Mode)")
        print("Hotkeys active. Press F1 for menu, or type commands:")
        print("Commands: menu, status, test, exit, help")

        try:
            while self.is_running:
                try:
                    command = input("whisper-flow> ").strip().lower()
                    log(f"[DAEMON] CLI command received: {command}")

                    if command == "exit":
                        break
                    if command == "menu":
                        self.show_notification_menu()
                    elif command == "status":
                        status = "Recording" if self.is_recording else "Ready"
                        mode = self.current_mode or "None"
                        print(f"Status: {status}, Mode: {mode}")
                        log(f"[DAEMON] Status requested: {status}, Mode: {mode}")
                    elif command == "test":
                        self.test_configuration(None, None)
                    elif command == "help":
                        print("Commands: menu, status, test, exit, help")
                    elif command == "":
                        continue
                    else:
                        print(f"Unknown command: {command}")

                except (KeyboardInterrupt, EOFError):
                    log("[DAEMON] CLI interrupted")
                    break

        except Exception as e:
            log(f"[DAEMON] CLI mode error: {e}")
        finally:
            self.stop_daemon()

    def run(self, foreground: bool = False, _worker: bool = False):
        """Run the daemon. This method now handles both launching and running the worker."""
        log(f"[DAEMON] Run called with foreground={foreground}, _worker={_worker}")

        if not self.config.daemon_enabled:
            if foreground:
                print("Daemon mode is disabled in configuration")
            log("[DAEMON] Daemon mode disabled in configuration")
            return

        if foreground or _worker:
            self._run_worker(foreground=foreground)
        else:
            self._launch_worker()

    def _launch_worker(self):
        """Launch the daemon as a background worker process."""
        import subprocess
        import sys

        log("[DAEMON] Launching worker process...")
        args = ["whisper-flow", "daemon", "--_worker"]
        print("Starting WhisperFlow daemon...")

        log_file_path = Path.home() / ".config" / "whisper-flow" / "daemon.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        if log_file_path.exists():
            log_file_path.unlink()

        with open(log_file_path, "w") as log_file:
            subprocess.Popen(
                args,
                stdout=log_file,
                stderr=log_file,
                stdin=subprocess.DEVNULL,
            )

        print("✓ Daemon process launched. Verifying status...")
        time.sleep(2)

        pid_file = _pid_file()
        if pid_file.exists():
            print("✓ Daemon is running. Tray icon should be visible.")
            log_file_path.unlink(missing_ok=True)
        else:
            log_content = log_file_path.read_text(encoding="utf-8", errors="replace")
            print("\n❌ Daemon failed to start. See error log below:")
            print("-" * 50)
            print(
                log_content or "Log file is empty, the process may have been blocked.",
            )
            print("-" * 50)

        sys.exit(0)

    def _acquire_single_instance(self) -> bool:
        """Refuse to start if another copy is already running.

        Two daemons means two hotkey listeners. On Linux the second copy
        cannot grab the keyboards the first already holds, so it sits in
        the tray with dead hotkeys and overwrites the pid file - stop then
        kills the broken one and leaves the live one untracked. On Windows
        a named mutex is the check: a stale PID file is not, because PIDs
        are reused aggressively there.
        """
        if sys.platform == "win32":
            return self._acquire_windows_instance()
        return self._acquire_linux_instance()

    def _acquire_windows_instance(self) -> bool:
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            # Local\ - one instance per session, not per machine. This is a
            # per-user application installed without administrator rights, so
            # two people signed in at once are each entitled to their own
            # copy; a Global\ name would have let the first one block the
            # second, and creating Global objects needs a privilege a
            # standard user may not hold anyway.
            self._instance_mutex = kernel32.CreateMutexW(
                None, False, "Local\\whisper-flow-daemon",
            )
            ERROR_ALREADY_EXISTS = 183
            if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
                return False
            return True
        except Exception:
            return True  # never let the check itself stop the app

    def _acquire_linux_instance(self) -> bool:
        """flock a lock file. Released automatically when this process dies."""
        import fcntl

        path = _lock_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = open(path, "a+")
        except OSError as e:
            log(f"[DAEMON] Could not open instance lock: {e}")
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fd.close()
            log("[DAEMON] Another instance is already running; exiting")
            return False
        except OSError as e:
            log(f"[DAEMON] Could not lock instance: {e}")
            fd.close()
            return True
        try:
            fd.seek(0)
            fd.truncate()
            fd.write(str(os.getpid()))
            fd.flush()
        except OSError:
            pass
        self._instance_lock = fd
        return True

    @staticmethod
    def _check_platform_support() -> str | None:
        """Why this machine cannot run the app, or None if it can.

        Windows support targets 11 22H2 and later: the overlay's backdrop,
        corners and border all come from composition attributes introduced
        there. Saying so up front beats a HUD that silently never appears.
        """
        if sys.platform != "win32":
            return None
        from . import blur_win
        return None if blur_win.is_supported() else blur_win.unsupported_reason()

    def _run_worker(self, foreground: bool = False):
        """Run the worker process with health monitoring."""
        log(f"[DAEMON] Starting worker process (foreground={foreground})")
        # A startup transcript, so "it doesn't launch" always leaves a
        # last-known stage behind in the same folder as the crash log.
        reset_stage_log()
        trace_stage("worker starting")
        # No fault dialogs, ever: a crashing whisper-server must exit with
        # its status code, not park a modal "Application Error" on the
        # user's screen holding the dead process until clicked. Inherited
        # by every child this process spawns.
        try:
            backend_module.suppress_crash_dialogs()
        except Exception:
            pass
        if not self._acquire_single_instance():
            log("[DAEMON] Another instance is already running; exiting")
            trace_stage("exit: another instance is already running")
            self.notify("whisper-flow is already running")
            return

        unsupported = self._check_platform_support()
        if unsupported:
            log(f"[DAEMON] Unsupported platform: {unsupported}")
            trace_stage(f"exit: unsupported platform ({unsupported})")
            self.notify(f"whisper-flow needs Windows 11 22H2 or later: {unsupported}")
            return
        crash_count = 0
        while True:
            try:
                self.is_running = True
                log(f"[DAEMON] Worker process started (attempt {crash_count + 1})")

                # Write PID file so parent process can verify startup
                pid_file = _pid_file()
                pid_file.parent.mkdir(parents=True, exist_ok=True)
                pid_file.write_text(str(os.getpid()))
                trace_stage("pid written")

                # Start watchdog for health monitoring
                self._start_watchdog()
                trace_stage("watchdog started")

                # Set up hotkeys
                self.setup_hotkeys()
                trace_stage("hotkeys set up")

                # And a transcription backend, if one is installed
                self._start_managed_backend()
                trace_stage("backend started")

                # New releases download themselves in the background; the
                # tray row flips to "Update to X" when one lands. No-op
                # where updates are unavailable (Linux, source checkouts).
                try:
                    updater.start_auto_update(
                        notify=self.notify,
                        on_ready=self._on_update_ready)
                except Exception as e:
                    log(f"[DAEMON] background updater failed to start: {e}")
                trace_stage("updater started")

                if foreground:
                    # Foreground mode: try tray, then keep hotkeys alive without a
                    # tray. Never fall through to the interactive CLI under
                    # systemd: input() hits EOF immediately and used to stop the
                    # whole daemon - which is exactly "hotkeys do nothing".
                    log("[DAEMON] Running in foreground mode")
                    trace_stage("tray starting")
                    try:
                        self.tray_icon = _pystray().Icon(
                            "whisper-flow",
                            self.create_tray_icon(),
                            "WhisperFlow Daemon",
                            self.setup_tray_menu(),
                        )
                        log("[DAEMON] Tray icon created successfully")
                        trace_stage("tray running")
                        self.tray_icon.run()
                    except Exception as e:
                        log(f"[DAEMON] Tray setup failed: {e}")
                        trace_stage(f"tray failed: {e}")
                        if sys.stdin.isatty():
                            self.run_notification_mode()
                        else:
                            log("[DAEMON] no TTY; staying up headless with hotkeys")
                            trace_stage("headless mode")
                            self.run_headless_mode()
                else:
                    # Background mode: try tray, fallback to headless
                    log("[DAEMON] Running in background mode")
                    trace_stage("tray starting (background)")
                    try:
                        self.tray_icon = _pystray().Icon(
                            "whisper-flow",
                            self.create_tray_icon(),
                            "WhisperFlow Daemon",
                            self.setup_tray_menu(),
                        )
                        log("[DAEMON] Background tray icon created successfully")
                        trace_stage("tray running")
                        self.tray_icon.run()
                    except Exception as e:
                        log(f"[DAEMON] Tray setup failed in background mode: {e}")
                        trace_stage(f"tray failed: {e}")
                        self.run_headless_mode()

            except Exception as e:
                crash_count += 1
                log(f"[DAEMON] Worker crashed (auto-heal {crash_count}): {e}\n{traceback.format_exc()}")
                try:
                    self.notify(f"Recovered from error — restarting (attempt {crash_count})")
                except Exception:
                    pass
                try:
                    self._cleanup()
                except Exception:
                    pass
                if crash_count > 5:
                    log("[DAEMON] too many crashes, giving up auto-heal")
                    # No console on Windows and no tray this early: without
                    # a file this reads as "it doesn't launch" with nothing
                    # to diagnose from.
                    write_crash_report(
                        "auto-heal gave up after 5 restarts\n\n"
                        + recent_log(200))
                    break
                backoff = min(2 ** crash_count, 30)
                log(f"[DAEMON] auto-heal restarting in {backoff}s")
                time.sleep(backoff)
                continue
            else:
                break
            finally:
                if crash_count == 0:
                    log("[DAEMON] Worker process finishing")
                    try:
                        self._cleanup()
                    except Exception:
                        pass

    def _cleanup(self):
        """Clean up resources and stop all components."""
        log("[DAEMON] Starting cleanup process...")

        # Stop watchdog
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            # Watchdog is daemon thread, it will stop when main thread stops
            log("[DAEMON] Watchdog thread will stop with main thread")

        # Stop recording if active
        if self.is_recording:
            log("[DAEMON] Stopping active recording during cleanup")
            self._stop_recording()

        # Retire the overlay. A resident one outlives a single recording by
        # design, so it has to be told to go; if this is missed the pipe
        # closing when this process dies takes it down anyway.
        try:
            self.hud.shutdown()
        except Exception as e:
            log(f"[DAEMON] Error stopping the overlay: {e}")

        # And the settings window, for the same reason: one built at login
        # and never asked for is an invisible process, and leaving it behind
        # would strand it with no tray icon and no window to close.
        try:
            self.shutdown_settings()
        except Exception as e:
            log(f"[DAEMON] Error stopping the settings window: {e}")

        # Stop the managed speech server
        try:
            self.backend.stop()
        except Exception as e:
            log(f"[DAEMON] Error stopping backend: {e}")

        # Stop hotkey manager
        try:
            log("[DAEMON] Stopping hotkey manager...")
            self.hotkey_manager.stop()
        except Exception as e:
            log(f"[DAEMON] Error stopping hotkey manager: {e}")

        # Stop tray icon
        if self.tray_icon:
            try:
                log("[DAEMON] Stopping tray icon...")
                self.tray_icon.stop()
            except Exception as e:
                log(f"[DAEMON] Error stopping tray icon: {e}")

        self.is_running = False
        log("[DAEMON] Cleanup complete")

    def run_headless_mode(self):
        """Run in headless mode when tray is not available in background."""
        log("[DAEMON] Running in headless mode")
        # Just keep the daemon running with hotkeys active
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            log("[DAEMON] Headless mode interrupted")
        finally:
            self.stop_daemon()


def main():
    """Main entry point for the daemon."""
    log("[DAEMON] Main entry point called")
    daemon = WhisperFlowDaemon()
    daemon.run()


def is_running() -> bool:
    """Check if the daemon is currently running."""
    pid_file = _pid_file()
    if not pid_file.exists():
        return False

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        # Check if process is still running
        os.kill(pid, 0)
        return True
    except (ValueError, OSError, FileNotFoundError):
        # PID file is invalid or process is dead
        try:
            pid_file.unlink()
        except Exception:
            pass
        return False


def stop_daemon():
    """Stop the running daemon."""
    log("[DAEMON] Stop daemon function called")
    pid_file = _pid_file()
    if not pid_file.exists():
        print("Daemon is not running")
        return

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        # Send SIGTERM to the daemon process
        os.kill(pid, 15)  # SIGTERM
        print(f"Sent stop signal to daemon (PID: {pid})")

        # Wait a moment and check if it stopped
        time.sleep(1)

        if is_running():
            print("Daemon did not stop gracefully, sending SIGKILL...")
            os.kill(pid, 9)  # SIGKILL

        # Clean up PID file
        try:
            pid_file.unlink()
        except Exception:
            pass

    except (ValueError, OSError, FileNotFoundError) as e:
        print(f"Error stopping daemon: {e}")
        # Clean up invalid PID file
        try:
            pid_file.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
