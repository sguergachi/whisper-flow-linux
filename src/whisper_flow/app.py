"""Main application class for whisper-flow."""

import os
from pathlib import Path

from .audio import AudioRecorder
from .boost import boost_wav
from .config import Config
from .logging import log, set_logging_enabled
from .streaming import LiveTranscriber
from .system import SystemManager
from .transcription import LIVE_TIMEOUT, TranscriptionService


class WhisperFlow:
    """Main application class for whisper-flow."""

    def __init__(self, config_dir: Path | None = None, mode: str = "default"):
        """Initialize WhisperFlow application.

        Args:
            config_dir: Custom configuration directory
            mode: Processing mode (default, dictation, transcribe, auto_transcribe, command)

        """
        self.config = Config(config_dir=config_dir) if config_dir else Config()
        self.mode = mode

        # Initialize logging based on configuration
        set_logging_enabled(self.config.logging_enabled)

        # Initialize components
        self.system_manager = SystemManager(self.config)
        self.audio_recorder = AudioRecorder(self.config, self.system_manager)
        self.transcription_service = TranscriptionService(self.config)

    def _transcribe_allowing_for_a_whisper(self, audio_file: str) -> str | None:
        """Transcribe, and if nothing comes back, try again louder.

        A whisper reaches the microphone at a peak around 100 out of 32767,
        and whisper.cpp returns nothing at all rather than a poor guess - so
        a recording that plainly contains speech transcribes to silence. The
        audio is there; it is just small.

        The retry costs one extra pass, so it happens only here, on the
        closing transcription, and only when the first attempt produced
        nothing. A recording with no signal in it is not retried at all.
        """
        text = self.transcription_service.transcribe_audio(audio_file)
        if text:
            return text

        louder = f"{audio_file}.louder.wav"
        gain = boost_wav(audio_file, louder)
        if not gain:
            return text
        try:
            retried = self.transcription_service.transcribe_audio(louder)
            if retried:
                log(f"[BOOST] {len(retried)} characters recovered at {gain:.0f}x")
            else:
                log(f"[BOOST] still nothing after {gain:.0f}x")
            return retried
        finally:
            try:
                Path(louder).unlink()
            except Exception:
                pass

    def run_voice_flow_push_to_talk_daemon(self, stop_key: str, stop_event,
                                           level_file: str | None = None,
                                           on_ready=None) -> bool:
        """Run voice flow with daemon-controlled push-to-talk recording.

        Args:
            stop_key: Hotkey combination that stops recording (for display)
            stop_event: Threading event to control recording stop
            level_file: Path to write audio levels for HUD visualization

        Returns:
            True if successful, False otherwise

        """
        try:
            # Record audio with daemon-controlled stop event
            audio_file = self.audio_recorder.record_push_to_talk(
                stop_key, stop_event, level_file=level_file, on_ready=on_ready)

            if not audio_file:
                log("No audio recorded")
                return False

            return self._process_recorded_audio(audio_file)

        except Exception as e:
            log(f"Error in daemon push-to-talk flow: {e}")
            self.system_manager.notify(f"Push-to-talk failed: {e}")
            return False

    def run_voice_flow_push_to_talk_live(self, stop_key: str, stop_event,
                                         level_file: str | None = None,
                                         on_ready=None) -> bool:
        """Push-to-talk that types text while the user is still speaking.

        Words are typed as soon as two consecutive transcription passes agree
        on them, so what lands on screen matches the final transcript instead
        of being a guess that needs correcting. See streaming.LiveTranscriber.

        Args:
            stop_key: Hotkey combination that stops recording (for display)
            stop_event: Threading event to control recording stop
            level_file: Path to write audio levels for HUD visualization

        Returns:
            True if successful, False otherwise

        """
        def live_pass(path):
            # One attempt, short timeout. A live pass is superseded by the
            # next one, so retrying it with backoff only delays the words
            # that are already waiting behind it.
            return self.transcription_service.transcribe_audio(
                path, max_retries=1, timeout=LIVE_TIMEOUT)

        live = LiveTranscriber(
            transcribe=live_pass,
            # Into the window this dictation started in, or not at all. A
            # live pass has another chance every pass and again at the end,
            # so typing it into whatever is in front instead only scatters
            # the dictation across two windows.
            emit=lambda text: self.system_manager.type_text(
                text, only_where_it_started=True),
            sample_rate=self.config.sample_rate,
            interval=self.config.live_interval,
            # Trim + light denoise (HPF, pre-roll gate, peak normalise).
            # Spectral strong-mode is final-pass only; see prepare_frames.
            prepare=self.audio_recorder.prepare_frames,
        )
        live.start()

        audio_file = None
        try:
            audio_file = self.audio_recorder.record_push_to_talk(
                stop_key,
                stop_event,
                level_file=level_file,
                on_tick=live.offer,
                tick_seconds=self.config.live_interval,
                on_ready=on_ready,
            )

            if not audio_file:
                live.stop()
                log("No audio recorded")
                return False

            # Before the closing pass, not after: the live loop would
            # otherwise keep issuing passes over the same audio and queue them
            # in front of the one transcript the user is actually waiting for.
            live.quiesce()

            # One last pass over the complete utterance, then type the tail.
            final_text = self._transcribe_allowing_for_a_whisper(audio_file)
            live.finalize(final_text)

            if not final_text and not live.committed_text:
                log("Transcription produced nothing")
                return False

            if live.delivery_failed:
                # There were words, and none of them landed. Reporting success
                # here is how this looked like "it just does nothing".
                log(f"[LIVE] transcribed {len(final_text or '')} chars but "
                    f"nothing could be typed into the focused window")
                return False

            log("Live transcription complete")
            return True

        except Exception as e:
            live.stop()
            log(f"Error in live push-to-talk flow: {e}")
            self.system_manager.notify(f"Live dictation failed: {e}")
            return False
        finally:
            if audio_file:
                try:
                    Path(audio_file).unlink()
                except Exception:
                    pass

    def run_voice_flow_auto_stop(
        self,
        silence_duration: float = 2.0,
        level_file: str | None = None,
        stop_event=None,
        on_ready=None,
        max_duration: float | None = None,
    ) -> bool:
        """Run voice flow with auto-stop on silence.

        Args:
            silence_duration: Seconds of silence before stopping
            level_file: Path to write audio levels for HUD visualization
            stop_event: Set by the daemon to cancel (Escape / force-stop)
            on_ready: Called once the microphone is capturing (show the HUD)
            max_duration: Hard cap so a silent room cannot hang forever

        Returns:
            True if successful, False otherwise

        """
        try:
            # Record audio until silence detected
            log(f"Recording... Will auto-stop after {silence_duration}s of silence")
            audio_file = self.audio_recorder.record_until_silence(
                silence_duration,
                stop_event=stop_event,
                level_file=level_file,
                on_ready=on_ready,
                max_duration=max_duration,
            )

            if not audio_file:
                # Distinct from a hard failure: the mic opened, nobody spoke
                # (or VAD heard nothing). Without this toast, single-press
                # auto looks exactly like a dead hotkey.
                log("No audio recorded")
                try:
                    self.system_manager.notify(
                        "No speech heard - press the hotkey and speak")
                except Exception:
                    pass
                return False

            return self._process_recorded_audio(audio_file)

        except Exception as e:
            log(f"Error in auto-stop flow: {e}")
            self.system_manager.notify(f"Auto-stop recording failed: {e}")
            return False

    def _process_recorded_audio(self, audio_file: str) -> bool:
        """Process recorded audio file through the full pipeline.

        Args:
            audio_file: Path to the recorded audio file

        Returns:
            True if successful, False otherwise

        """
        try:
            # Transcribe audio
            log("Transcribing...")
            transcript = self._transcribe_allowing_for_a_whisper(audio_file)

            if not transcript:
                log("Transcription failed")
                return False

            log("Transcript completed")

            # Every mode pastes the transcript as it was heard. The local
            # whisper server is the only backend, and it transcribes; there
            # is no cloud step to send the words to.
            final_result = transcript

            log("Final result completed")

            # Paste the result
            if not self.system_manager.paste_text(final_result):
                log("Failed to paste text, copying to clipboard...")
                self.system_manager.copy_to_clipboard(final_result)

            return True

        except Exception as e:
            log(f"Error processing audio: {e}")
            return False
        finally:
            # Cleanup temporary file
            try:
                Path(audio_file).unlink()
            except Exception:
                pass

    def run_comprehensive_validation(self) -> dict:
        """Run comprehensive system validation.

        Returns:
            Dictionary with validation results by category

        """
        results = {}

        # API Configuration validation
        results["api_config"] = self._validate_api_config()

        # System Dependencies validation
        results["system_deps"] = self._validate_system_dependencies()

        # Audio System validation
        results["audio_system"] = self._validate_audio_system()

        # Services validation
        results["services"] = self._validate_services()

        # Hotkeys
        results["hotkeys"] = self._validate_hotkeys()

        # Configuration Files validation
        results["config_files"] = self._validate_config_files()

        # Environment validation
        results["environment"] = self._validate_environment()

        return results

    def _validate_api_config(self) -> list[dict]:
        """Check that a transcription backend is actually configured."""
        local = (self.config.local_whisper_url or "").strip()

        if local:
            return [{
                "name": "Transcription backend",
                "status": "pass",
                "message": f"Local whisper.cpp server at {local}",
            }]
        return [{
            "name": "Transcription backend",
            "status": "fail",
            "message": ("Nothing configured. Set WHISPER_FLOW_LOCAL_WHISPER_URL "
                        "to a whisper.cpp server, or let the app manage one."),
        }]

    def _validate_services(self) -> list[dict]:
        """Check the backend can actually be reached, not merely configured."""
        local = (self.config.local_whisper_url or "").strip()
        if not local:
            return []

        try:
            import requests
            # Any answer at all proves something is listening; the
            # inference endpoint rejects a bare GET, which is still a
            # reachable server.
            requests.get(local, timeout=3)
            reachable, detail = True, "responding"
        except Exception as e:
            reachable, detail = False, f"{type(e).__name__}: {e}"
        return [{
            "name": "Local whisper server",
            "status": "pass" if reachable else "fail",
            "message": (f"{local} {detail}" if reachable else
                        f"Cannot reach {local} - is whisper-server running? ({detail})"),
        }]

    def _validate_system_dependencies(self) -> list[dict]:
        """Check the tools needed to type text on this platform.

        These differ completely per platform, and the previous version checked
        X11 tools everywhere - reporting five warnings on Windows about
        programs that have no business being there.
        """
        import shutil
        import sys

        tests = []
        if sys.platform == "win32":
            tests.append({
                "name": "Text injection",
                "status": "pass",
                "message": "SendInput (built in)",
            })
            return tests

        wayland = bool(os.environ.get("WAYLAND_DISPLAY")) or \
            os.environ.get("XDG_SESSION_TYPE") == "wayland"

        if wayland:
            typing_tools = [("ydotool", True), ("wtype", False)]
            extras = [("wl-copy", False), ("kdotool", False)]
        else:
            typing_tools = [("xdotool", True)]
            extras = [("xclip", False), ("xsel", False)]

        for tool, required in typing_tools + extras:
            present = shutil.which(tool) is not None
            tests.append({
                "name": f"System tool: {tool}",
                "status": "pass" if present else ("fail" if required else "warn"),
                "message": "Available" if present else
                           ("Required for typing text" if required else "Optional, not installed"),
            })
        return tests

    def _validate_audio_system(self) -> list[dict]:
        """Check there is a usable microphone, not merely a working library."""
        tests = []
        try:
            import pyaudio
        except ImportError:
            return [{"name": "Audio library", "status": "fail",
                     "message": "PyAudio is not installed"}]

        try:
            pa = pyaudio.PyAudio()
        except Exception as e:
            return [{"name": "Audio system", "status": "fail",
                     "message": f"Cannot open the audio system: {e}"}]

        try:
            inputs = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    inputs.append(info.get("name", f"device {i}"))

            if inputs:
                chosen = self.config.mic_device_index
                which = (f"device {chosen}" if chosen is not None
                         else f"default ({inputs[0]})")
                tests.append({"name": "Microphone", "status": "pass",
                              "message": f"{len(inputs)} input(s); using {which}"})
            else:
                tests.append({"name": "Microphone", "status": "fail",
                              "message": "No input devices found"})
        finally:
            pa.terminate()
        return tests

    def _validate_config_files(self) -> list[dict]:
        """Report where settings are read from, and whether anything is there."""
        env_file = Path(self.config.config_dir) / ".env"
        return [{
            "name": "Config file",
            "status": "pass" if env_file.exists() else "warn",
            "message": (str(env_file) if env_file.exists()
                        else f"No .env at {env_file}; using defaults"),
        }]

    def _validate_hotkeys(self) -> list[dict]:
        """Report the configured hotkeys, and flag ones the OS will swallow."""
        import sys

        tests = []
        combos = {
            "Push to talk (hold)": self.config.hotkey_transcribe,
            "Auto-transcribe (tap)": self.config.hotkey_auto_transcribe,
            "Auto-transcribe 2nd (tap)": self.config.hotkey_command,
        }
        for label, combo in combos.items():
            status, message = "pass", combo
            if sys.platform == "win32" and any(
                part in {"super", "cmd", "win", "meta"}
                for part in combo.lower().split("+")
            ):
                status = "warn"
                message = (f"{combo} - Windows reserves most Win key "
                           f"combinations; try ctrl+alt")
            tests.append({"name": f"Hotkey: {label}", "status": status,
                          "message": message})
        return tests

    def _validate_environment(self) -> list[dict]:
        """Validate environment setup."""
        tests = []

        # Python version
        import sys

        if sys.version_info >= (3, 11):
            tests.append(
                {
                    "name": "Python Version",
                    "status": "pass",
                    "message": f"Python {sys.version_info.major}.{sys.version_info.minor}",
                },
            )
        else:
            tests.append(
                {
                    "name": "Python Version",
                    "status": "warn",
                    "message": f"Python {sys.version_info.major}.{sys.version_info.minor} (3.11+ recommended)",
                },
            )

        return tests
