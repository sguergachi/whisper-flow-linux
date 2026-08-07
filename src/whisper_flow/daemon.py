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
from .hud import HUD
from .logging import log, recent_log, set_logging_enabled
from .paths import pid_file as _pid_file
from .version import build_version

# Modes driven by holding a hotkey down; they cannot be deferred and replayed.
# Only plain push-to-talk. Auto-transcribe and command are single-press:
# speak until silence. Command used to be a held chord of pure modifiers
# (super+shift+alt); releasing them ended the recording in the same gesture
# that started it, so every tap produced a clip too short to keep - which
# looked exactly like "command does nothing".
PUSH_TO_TALK_MODES = ("transcribe",)

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
            pystray.MenuItem(
                f"Transcribe (Push-to-Talk): {self.config.hotkey_transcribe}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                f"Auto-Transcribe (Single Press): {self.config.hotkey_auto_transcribe}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                f"Command (Single Press): {self.config.hotkey_command}",
                None,
                enabled=False,
            ),
            pystray.MenuItem("Settings", self.open_settings),
            pystray.MenuItem("Test setup", self.test_configuration),
            pystray.MenuItem("Copy last error", self.copy_last_error),
            pystray.MenuItem("Copy log", self.copy_log),
            pystray.MenuItem("Check for updates", self.check_for_updates,
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

            # Register transcribe hotkey (push-to-talk)
            self.hotkey_manager.register_hotkey(
                name="transcribe",
                keys=self.config.hotkey_transcribe,
                mode=HotkeyMode.PUSH_TO_TALK,
                callback_press=lambda: self._handle_hotkey_press("transcribe"),
                callback_release=lambda: self._stop_recording_if_active("transcribe"),
                priority=1,
                description="Push-to-talk transcription",
            )

            # Register auto-transcribe hotkey (single press)
            self.hotkey_manager.register_hotkey(
                name="auto_transcribe",
                keys=self.config.hotkey_auto_transcribe,
                mode=HotkeyMode.SINGLE_PRESS,
                callback_press=lambda: self._handle_hotkey_press("auto_transcribe"),
                priority=3,  # Highest priority since it has most keys
                description="Auto-stop transcription",
            )

            # Register command hotkey (single press, stop on silence)
            self.hotkey_manager.register_hotkey(
                name="command",
                keys=self.config.hotkey_command,
                mode=HotkeyMode.SINGLE_PRESS,
                callback_press=lambda: self._handle_hotkey_press("command"),
                priority=2,  # Higher than transcribe since it has more keys
                description="Single-press command dictation (stop on silence)",
            )

            # Set up escape key handling for canceling recordings
            self.hotkey_manager._handle_escape_key = self.cancel_recording

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
                          point=getattr(self, "_hud_point", None))
        except Exception as e:
            log(f"[DAEMON] could not show the overlay: {e}")
        finally:
            if ptl is not None:
                ptl.mark("overlay")
                ptl.report()

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

    def _record_audio_thread(self, mode: str):
        """Handle audio recording in a separate thread with timeout protection."""
        log(f"[DAEMON] Recording thread started for mode: {mode}")
        try:
            app = self._get_app_for_mode(mode)
            log(f"[DAEMON] Using app instance for mode: {mode}")

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
                log(f"[DAEMON] Recording failed for mode: {mode}")
                self._report_failure(f"Recording failed ({mode})")
            else:
                log(f"[DAEMON] Recording completed successfully for mode: {mode}")

        except Exception as e:
            log(f"[DAEMON] Recording thread error for mode {mode}: {e}")
            self._report_failure(f"Recording error ({mode}): {e}",
                                 traceback.format_exc())
        finally:
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
                # It could not be told, so there is no window coming from it.
                # Fall through and build one the slow way rather than report a
                # window that is not going to appear.

            return self._start_tool_window(flag, module, resident=False)

    def shutdown_settings(self, *, quit: bool = False) -> None:
        """Retire the settings window process when the daemon exits.

        Closing stdin is the ordinary path: the child reads end of stream and
        leaves if nobody is looking at it. A window that is already open is
        deliberately kept across a restart (Save restarts the daemon; the
        settings toast must stay put). Tray Exit is different - the whole app
        is going away - so ``quit=True`` tells the child to exit even when
        visible.
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
        try:
            process.wait(timeout=2 if quit else 3)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        if quit and process.poll() is None:
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
            process = self._setup_process = subprocess.Popen(
                cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace")
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
        url = self.backend.start(model)
        if not url:
            self.notify("The new speech model would not start")
            return
        self.config.model_name = model
        self._backend_model = model
        self._backend_engine = engine
        self._use_backend_url(url)
        self.notify(f"Speech model ready ({model.replace('ggml-', '')}, "
                    f"{self.backend.engine_summary()})")

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

        url = self.backend.start(model)
        if not url:
            return
        self._backend_model = model
        self._backend_engine = self.backend.installed_engine()
        self._use_backend_url(url)
        log(f"[BACKEND] {model} on {self.backend.engine_summary()}")

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
            overlay = self.hud.crash_log()
        except Exception as e:
            overlay = f"could not inspect the overlay: {e}"
        if overlay:
            report += ["", "Overlay", overlay]

        if detail:
            report += ["", "Traceback", detail.rstrip()]
        report += ["", "Recent log", recent_log(200) or "(nothing recorded)"]
        return "\n".join(report)

    def check_for_updates(self, icon=None, item=None):
        """Download and restart into a newer version, if there is one.

        On its own thread: this downloads, and a tray callback that does not
        return promptly stops the icon responding.
        """
        threading.Thread(
            target=lambda: updater.apply_now(notify=self.notify),
            daemon=True, name="whisper-flow-update",
        ).start()

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
        """
        log("[DAEMON] Settings menu item clicked")
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
• {self.config.hotkey_transcribe} - Transcribe (push-to-talk)
• {self.config.hotkey_auto_transcribe} - Auto-transcribe
• {self.config.hotkey_command} - Command mode
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

        Two daemons means two hotkey listeners: one press starts two
        recordings, and both type their transcript into the same window. On
        Windows a named mutex is the reliable check - a stale PID file is not,
        because PIDs are reused aggressively there.
        """
        if sys.platform != "win32":
            return True
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
        if not self._acquire_single_instance():
            log("[DAEMON] Another instance is already running; exiting")
            self.notify("whisper-flow is already running")
            return

        unsupported = self._check_platform_support()
        if unsupported:
            log(f"[DAEMON] Unsupported platform: {unsupported}")
            self.notify(f"whisper-flow needs Windows 11 22H2 or later: {unsupported}")
            return
        try:
            self.is_running = True
            log("[DAEMON] Worker process started")

            # Write PID file so parent process can verify startup
            pid_file = _pid_file()
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(os.getpid()))

            # Start watchdog for health monitoring
            self._start_watchdog()

            # Set up hotkeys
            self.setup_hotkeys()

            # And a transcription backend, if one is installed
            self._start_managed_backend()

            if foreground:
                # Foreground mode: try tray, fallback to notification mode
                log("[DAEMON] Running in foreground mode")
                try:
                    self.tray_icon = _pystray().Icon(
                        "whisper-flow",
                        self.create_tray_icon(),
                        "WhisperFlow Daemon",
                        self.setup_tray_menu(),
                    )
                    log("[DAEMON] Tray icon created successfully")
                    self.tray_icon.run()
                except Exception as e:
                    log(f"[DAEMON] Tray setup failed: {e}")
                    self.run_notification_mode()
            else:
                # Background mode: try tray, fallback to headless
                log("[DAEMON] Running in background mode")
                try:
                    self.tray_icon = _pystray().Icon(
                        "whisper-flow",
                        self.create_tray_icon(),
                        "WhisperFlow Daemon",
                        self.setup_tray_menu(),
                    )
                    log("[DAEMON] Background tray icon created successfully")
                    self.tray_icon.run()
                except Exception as e:
                    log(f"[DAEMON] Tray setup failed in background mode: {e}")
                    self.run_headless_mode()

        except Exception as e:
            log(f"[DAEMON] Worker error: {e}")
            self.notify(f"❌ Daemon error: {e}")
        finally:
            log("[DAEMON] Worker process finishing")
            self._cleanup()

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
