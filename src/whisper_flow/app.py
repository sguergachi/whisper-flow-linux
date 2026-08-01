"""Main application class for whisper-flow."""

import os
from pathlib import Path

from .audio import AudioRecorder
from .completion import CompletionService
from .config import Config
from .logging import log, set_logging_enabled
from .prompts import PromptManager
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
        self.completion_service = CompletionService(self.config)
        self.prompt_manager = PromptManager(self.config, self.system_manager)

    def run_voice_flow_push_to_talk_daemon(self, stop_key: str, stop_event, level_file: str | None = None) -> bool:
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
            audio_file = self.audio_recorder.record_push_to_talk(stop_key, stop_event, level_file=level_file)

            if not audio_file:
                log("No audio recorded")
                return False

            return self._process_recorded_audio(audio_file)

        except Exception as e:
            log(f"Error in daemon push-to-talk flow: {e}")
            self.system_manager.notify(f"Push-to-talk failed: {e}")
            return False

    def run_voice_flow_push_to_talk_live(self, stop_key: str, stop_event, level_file: str | None = None) -> bool:
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
            emit=self.system_manager.type_text,
            sample_rate=self.config.sample_rate,
            interval=self.config.live_interval,
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
            )

            if not audio_file:
                live.stop()
                log("No audio recorded")
                return False

            # One last pass over the complete utterance, then type the tail.
            final_text = self.transcription_service.transcribe_audio(audio_file)
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

    def run_voice_flow_auto_stop(self, silence_duration: float = 2.0, level_file: str | None = None) -> bool:
        """Run voice flow with auto-stop on silence.

        Args:
            silence_duration: Seconds of silence before stopping
            level_file: Path to write audio levels for HUD visualization

        Returns:
            True if successful, False otherwise

        """
        try:
            # Record audio until silence detected
            log(f"Recording... Will auto-stop after {silence_duration}s of silence")
            audio_file = self.audio_recorder.record_until_silence(silence_duration, level_file=level_file)

            if not audio_file:
                log("No audio recorded")
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
            transcript = self.transcription_service.transcribe_audio(audio_file)

            if not transcript:
                log("Transcription failed")
                return False

            log("Transcript completed")

            # For transcribe and auto_transcribe modes, return transcript as-is
            if self.mode in ["transcribe", "auto_transcribe"]:
                final_result = transcript
            # Use the new simplified prompt system
            elif self.prompt_manager.should_use_completion():
                log("Processing with AI...")
                messages = self.prompt_manager.get_messages(transcript)

                final_result = self.completion_service.complete_text(messages)

                if not final_result:
                    log("AI processing failed, using raw transcript")
                    final_result = transcript
            else:
                final_result = transcript

            log("Final result completed")

            # Paste the result
            if not self.system_manager.paste_text(final_result):
                log("Failed to paste text, copying to clipboard...")
                self.system_manager._copy_to_clipboard(final_result)

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
        """Check that some transcription backend is actually configured."""
        tests = []
        local = (self.config.local_whisper_url or "").strip()
        key = self.config.openai_api_key

        if local:
            tests.append({
                "name": "Transcription backend",
                "status": "pass",
                "message": f"Local whisper.cpp server at {local}",
            })
        elif key:
            tests.append({
                "name": "Transcription backend",
                "status": "pass",
                "message": f"OpenAI API ({self.config.transcription_model})",
            })
        else:
            tests.append({
                "name": "Transcription backend",
                "status": "fail",
                "message": ("Nothing configured. Set WHISPER_FLOW_LOCAL_WHISPER_URL "
                            "to a whisper.cpp server, or WHISPER_FLOW_OPENAI_API_KEY."),
            })
        return tests

    def _validate_services(self) -> list[dict]:
        """Check the backend can actually be reached, not merely configured."""
        tests = []
        local = (self.config.local_whisper_url or "").strip()

        if local:
            try:
                import requests
                # Any answer at all proves something is listening; the
                # inference endpoint rejects a bare GET, which is still a
                # reachable server.
                requests.get(local, timeout=3)
                reachable, detail = True, "responding"
            except Exception as e:
                reachable, detail = False, f"{type(e).__name__}: {e}"
            tests.append({
                "name": "Local whisper server",
                "status": "pass" if reachable else "fail",
                "message": (f"{local} {detail}" if reachable else
                            f"Cannot reach {local} - is whisper-server running? ({detail})"),
            })
        elif self.config.openai_api_key:
            tests.append({
                "name": "OpenAI API",
                "status": "pass",
                "message": "Key present (not verified until first use)",
            })

        return tests

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
            "Transcribe": self.config.hotkey_transcribe,
            "Auto-transcribe": self.config.hotkey_auto_transcribe,
            "Command": self.config.hotkey_command,
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
