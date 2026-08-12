"""Main application class for whisper-flow."""

import os
from pathlib import Path

from . import audio_debug
from .audio import AudioRecorder
from .boost import boost_wav, rescue_wav
from .config import Config
from .logging import log, set_logging_enabled
from .streaming import LiveTranscriber
from .system import SystemManager
from .transcription import (
    LIVE_TIMEOUT,
    TranscriptionService,
    is_hallucination,
)

# The music-under-speech re-decode. A continuous music bed keeps the room
# level up, so the clip looks usable to every SNR metric and rescue never
# engages; Whisper then transcribes the music's vocals instead of the
# speaker (a 2.6s "What does a control point mean?" came back as the four
# words "This is the show."). Speech is dense: real dictation runs at
# roughly 10-14 chars per second, while a music lyric taken for speech is
# sparser. Below this pace, with a music-like clip, the first transcript is
# not to be trusted and gets one steered re-decode.
MUSIC_STEER_MAX_CHARS_PER_SEC = 8.5
MUSIC_STEER_TEMPERATURE = 0.6
MUSIC_STEER_PROMPT = (
    "Transcribe only the speech spoken by the user, "
    "ignoring any background music or singing."
)


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
        # Tag captures so audio-debug reports show which mode produced them.
        self.audio_recorder._debug_mode = mode
        self.transcription_service = TranscriptionService(self.config)

    def _transcribe_allowing_for_a_whisper(self, audio_file: str) -> str | None:
        """Transcribe, and if nothing comes back, retry louder / café-rescued.

        Two blank-transcript causes share this path:

        1. Quiet-room whisper (peak ~100): amplify the whole clip.
        2. Café music + whisper (peak can be high): speech-band rescue on the
           raw trimmed capture so music peaks no longer starve the voice.

        Retries cost one extra pass each and only run on the closing
        transcription.
        """
        text = self.transcription_service.transcribe_audio(audio_file)
        if text:
            # A confident transcript is not proof it is the speech: under a
            # continuous music bed Whisper can lock onto the music's vocals
            # ("This is the show." for "What does a control point mean?").
            # When the clip is music-like and the transcript is shorter than
            # speech would be, one steered re-decode can recover the words.
            steered = self._retry_steered_under_music(audio_file, text)
            if steered:
                self._finalize_audio_debug(transcript=text,
                                           boost_transcript=steered)
                return steered
            self._finalize_audio_debug(transcript=text)
            return text

        # Still nothing: a plain peak-normalised boost of the *raw* capture -
        # not of audio_file, which is already the smart-voice output and so
        # reads as loud enough for needs_boost to refuse - with the cold-open
        # leading digital silence skipped. That silence makes the floor read
        # as a dead-silent room, sends the clip down the whisper-lift path,
        # and in a room with music the 40x speech-frame gain then amplifies
        # the music bed into the transcript.
        boosted = self._retry_boost_raw(audio_file)
        if boosted:
            self._finalize_audio_debug(transcript=text, boost_transcript=boosted)
            return boosted

        # Quiet-whisper in a genuinely silent room: gate room → amplify voiced
        # frames before raw boost.
        lifted = self._retry_whisper_lift(audio_file)
        if lifted:
            self._finalize_audio_debug(transcript=text, boost_transcript=lifted)
            return lifted

        boost_gain = None
        boost_text = None
        louder = f"{audio_file}.louder.wav"
        gain = boost_wav(audio_file, louder)
        if gain:
            try:
                boost_text = self.transcription_service.transcribe_audio(louder)
                boost_gain = gain
                if boost_text:
                    log(f"[BOOST] {len(boost_text)} characters recovered "
                        f"at {gain:.0f}x")
                    self._copy_debug_wav(louder, "boosted.wav")
                    self._finalize_audio_debug(
                        transcript=text,
                        boost_gain=gain,
                        boost_transcript=boost_text,
                    )
                    return boost_text
                log(f"[BOOST] still nothing after {gain:.0f}x")
                self._copy_debug_wav(louder, "boosted.wav")
            finally:
                try:
                    Path(louder).unlink()
                except Exception:
                    pass

        rescued_text = self._retry_cafe_rescue(audio_file)
        if rescued_text:
            self._finalize_audio_debug(
                transcript=text,
                boost_gain=boost_gain,
                boost_transcript=rescued_text,
            )
            return rescued_text

        self._finalize_audio_debug(
            transcript=text,
            boost_gain=boost_gain,
            boost_transcript=boost_text or rescued_text,
        )
        return boost_text

    def _retry_boost_raw(self, audio_file: str) -> str | None:
        """Boost the raw capture, skipping the cold-open leading silence.

        A freshly opened capture stream delivers ~0.3-0.5s of exact digital
        silence while the device wakes up. That silence is what made the
        noise floor read as a dead-silent room (``floor_rms: 0.0`` in every
        failed capture, ``> 0`` in every successful one), which then pushed
        the smart-voice plan into whisper-lift - and in a room with music the
        40x speech-frame gain amplifies the music bed into the transcript.

        Boosting the raw audio without that silence keeps the voice/music
        ratio as the mic heard it, and is what actually recovers the words.
        """
        config = getattr(self, "config", None)
        if config is None:
            return None
        try:
            import wave as _wave

            last = audio_debug.last_dir(config.config_dir)
            source = last / "raw_untrimmed.wav"
            if not source.is_file():
                source = last / "raw_trimmed.wav"
            if not source.is_file():
                source = Path(audio_file)
            with _wave.open(str(source), "rb") as wf:
                rate = wf.getframerate()
                channels = wf.getnchannels()
                width = wf.getsampwidth()
                pcm = wf.readframes(wf.getnframes())
            import numpy as np

            samples = np.frombuffer(pcm, dtype=np.int16)
            if samples.size == 0:
                return None
            # Skip the cold-open digital silence: it is not room tone, and
            # boosting it alongside the voice only raises the noise floor.
            frame = max(1, int(rate * 0.02))
            usable = samples.size - (samples.size % frame)
            blocks = samples[:usable].reshape(-1, frame).astype(np.float64)
            levels = np.sqrt((blocks ** 2).mean(axis=1))
            first = np.nonzero(levels > 10.0)[0]
            start = first[0] * frame if first.size else 0
            clean = samples[start:]

            clean_path = f"{audio_file}.clean.wav"
            with _wave.open(clean_path, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(width)
                wf.setframerate(rate)
                wf.writeframes(clean.tobytes())
            louder = f"{audio_file}.boosted.wav"
            try:
                gain = boost_wav(clean_path, louder)
                if not gain:
                    return None
                text = self.transcription_service.transcribe_audio(louder)
                self._copy_debug_wav(louder, "boosted.wav")
                if text:
                    log(f"[BOOST] {len(text)} characters recovered from the "
                        f"raw capture at {gain:.0f}x")
                    return text
                log(f"[BOOST] still nothing after {gain:.0f}x on the raw capture")
                return None
            finally:
                try:
                    Path(clean_path).unlink()
                except Exception:
                    pass
                try:
                    Path(louder).unlink()
                except Exception:
                    pass
        except Exception as e:
            log(f"[BOOST] raw retry failed: {e}")
            return None

    def _retry_whisper_lift(self, audio_file: str) -> str | None:
        """Gate the room, then lift quiet speech — for missed whispers.

        ``*sad music*`` is Whisper labelling the bed because the voice never
        reached a usable level. Raw peak-boost makes that worse; this path
        ducks non-speech first so gain hits the whisper, not the room.
        """
        config = getattr(self, "config", None)
        if config is None:
            return None
        if not bool(getattr(config, "smart_voice_amplification", True)):
            return None
        try:
            import numpy as np
            import wave as _wave
            from . import denoise as denoise_mod

            last = audio_debug.last_dir(config.config_dir)
            source = last / "raw_trimmed.wav"
            if not source.is_file():
                source = Path(audio_file)
            untrimmed = last / "raw_untrimmed.wav"
            with _wave.open(str(source), "rb") as wf:
                rate = wf.getframerate()
                pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            if pcm.size == 0:
                return None
            floor = None
            noise_ref = None
            try:
                raw_path = untrimmed if untrimmed.is_file() else source
                with _wave.open(str(raw_path), "rb") as wf:
                    raw = np.frombuffer(wf.readframes(wf.getnframes()),
                                        dtype=np.int16)
                hp, _ = denoise_mod.high_pass(raw, rate)
                floor = denoise_mod.measure_floor(hp, rate)
                noise_ref = denoise_mod.noise_reference(hp, rate, floor=floor)
            except Exception:
                pass
            lifted = denoise_mod.apply_plan(
                pcm, rate, denoise_mod.whisper_lift_plan(),
                floor=floor, noise_ref=noise_ref,
            )
            out = f"{audio_file}.whisper.wav"
            with _wave.open(out, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes(lifted.tobytes())
            try:
                text = self.transcription_service.transcribe_audio(out)
                self._copy_debug_wav(out, "whisper_lift.wav")
                if text:
                    log(f"[VOICE] whisper-lift recovered {len(text)} chars")
                return text
            finally:
                try:
                    Path(out).unlink()
                except Exception:
                    pass
        except Exception as e:
            log(f"[VOICE] whisper-lift retry failed: {e}")
            return None

    def _copy_debug_wav(self, src: str, name: str) -> None:
        try:
            import shutil
            dest = audio_debug.last_dir(self.config.config_dir)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / name)
        except Exception:
            pass

    def _retry_cafe_rescue(self, audio_file: str) -> str | None:
        """Blank-retry with speech-band rescue on the raw trimmed capture."""
        config = getattr(self, "config", None)
        if config is None:
            return None
        if not bool(getattr(config, "smart_voice_amplification", True)):
            return None
        try:
            last = audio_debug.last_dir(config.config_dir)
            # Prefer raw_trimmed (voice + room, no prior gate). Fall back to
            # the file Whisper already rejected.
            source = last / "raw_trimmed.wav"
            if not source.is_file():
                source = Path(audio_file)
            untrimmed = last / "raw_untrimmed.wav"
            out = f"{audio_file}.rescue.wav"
            # Recompute adaptive floor on the full capture (speak-at-HUD:
            # meta pre-roll floor may have been speech). Second pass is the
            # retroactive correction when live never latched in time.
            floor = None
            try:
                import numpy as np
                from . import denoise as denoise_mod
                import wave as _wave
                with _wave.open(str(
                        untrimmed if untrimmed.is_file() else source),
                        "rb") as wf:
                    pcm = np.frombuffer(
                        wf.readframes(wf.getnframes()), dtype=np.int16)
                    rate = wf.getframerate()
                hp, _ = denoise_mod.high_pass(pcm, rate)
                floor = denoise_mod.measure_floor(hp, rate)
            except Exception:
                pass
            gain = rescue_wav(
                str(source), out,
                noise_ref_path=str(untrimmed) if untrimmed.is_file() else None,
                floor=float(floor) if floor else None,
            )
            if not gain:
                return None
            try:
                text = self.transcription_service.transcribe_audio(out)
                self._copy_debug_wav(out, "rescued.wav")
                if text:
                    log(f"[RESCUE] {len(text)} characters recovered "
                        f"(café/whisper path)")
                else:
                    log("[RESCUE] still nothing after speech-band path")
                return text
            finally:
                try:
                    Path(out).unlink()
                except Exception:
                    pass
        except Exception as e:
            log(f"[RESCUE] retry failed: {e}")
            return None

    def _retry_steered_under_music(self, audio_file: str,
                                   first_text: str) -> str | None:
        """Re-decode a short transcript under background music, steered.

        The case the SNR metrics cannot see: a continuous music bed (with or
        without vocals) holds the room level up, so the clip reads as
        usable, rescue never engages, and Whisper transcribes the music
        instead of the speaker. Two signals point at it: the audio is
        music-like (its level never rests - the gate is open most of the
        clip and frames cluster around the median), and the transcript is
        too short for the clip (real dictation runs ~10+ chars/second; a
        music lyric read for speech is sparser).

        The re-decode pairs the strongest cleanup the pipeline has -
        speech-band rescue on the raw trimmed capture - with a decoder
        prompt that says to ignore the music and a warmer temperature so
        the greedy path that locked onto the vocals can be escaped. The
        steered result is kept when it reads *more* speech-paced than the
        first pass; otherwise the first stands.
        """
        config = getattr(self, "config", None)
        if config is None:
            return None
        if not bool(getattr(config, "smart_voice_amplification", True)):
            return None
        try:
            import numpy as np
            import wave as _wave
            from . import denoise as denoise_mod

            last = audio_debug.last_dir(config.config_dir)
            source = last / "raw_trimmed.wav"
            if not source.is_file():
                source = Path(audio_file)
            untrimmed = last / "raw_untrimmed.wav"
            with _wave.open(str(source), "rb") as wf:
                rate = wf.getframerate()
                pcm = np.frombuffer(wf.readframes(wf.getnframes()),
                                    dtype=np.int16)
            if pcm.size == 0 or rate <= 0:
                return None
            duration = pcm.size / rate

            profile = denoise_mod.profile_signal(pcm, rate)
            if not profile.music_like:
                return None

            # Speech pace. Short real phrases under music trigger this too -
            # they cost one extra decode and their own transcript still wins
            # the keep-rule below; a music lyric cannot pass the same bar.
            pace = len(first_text) / duration
            if pace >= MUSIC_STEER_MAX_CHARS_PER_SEC:
                return None

            # Floor on the full capture, like the café rescue: the pre-roll
            # may have been speech, and the bed is what spectral subtraction
            # should be told about.
            floor = None
            try:
                raw_path = untrimmed if untrimmed.is_file() else source
                with _wave.open(str(raw_path), "rb") as wf:
                    raw = np.frombuffer(wf.readframes(wf.getnframes()),
                                        dtype=np.int16)
                hp, _ = denoise_mod.high_pass(raw, rate)
                floor = denoise_mod.measure_floor(hp, rate)
            except Exception:
                pass

            out = f"{audio_file}.music.wav"
            try:
                gain = rescue_wav(
                    str(source), out,
                    noise_ref_path=str(untrimmed) if untrimmed.is_file()
                    else None,
                    floor=float(floor) if floor else None,
                )
                if not gain:
                    return None
                steered = self.transcription_service.transcribe_audio(
                    out,
                    prompt=MUSIC_STEER_PROMPT,
                    temperature=MUSIC_STEER_TEMPERATURE,
                )
                self._copy_debug_wav(out, "music_steer.wav")
                if not steered or is_hallucination(steered):
                    log("[MUSIC] steered re-decode came back blank or tagged")
                    return None
                if len(steered) / duration < pace:
                    log(f"[MUSIC] steered re-decode ({len(steered)} chars) no "
                        f"better than the first pass ({pace:.1f} ch/s)")
                    return None
                log(f"[MUSIC] steered re-decode recovered {len(steered)} chars "
                    f"(first pass {len(first_text)} chars at {pace:.1f} ch/s)")
                return steered
            finally:
                try:
                    Path(out).unlink()
                except Exception:
                    pass
        except Exception as e:
            log(f"[MUSIC] steered retry failed: {e}")
            return None

    def _finalize_audio_debug(self, **kwargs) -> None:
        """Attach transcript outcome to the last capture folder, if any."""
        config = getattr(self, "config", None)
        if config is None:
            return
        try:
            audio_debug.finalize_capture(
                config.config_dir,
                rate=getattr(config, "sample_rate", 16000),
                keep_sample=bool(
                    getattr(config, "keep_all_captures", False)),
                **kwargs,
            )
        except Exception as e:
            log(f"[AUDIO-DEBUG] finalize skipped: {e}")

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
                # A device that vanished mid-enumeration raises; one missing
                # row must not fail the whole check.
                try:
                    info = pa.get_device_info_by_index(i)
                    if int(info.get("maxInputChannels", 0)) > 0:
                        inputs.append(info.get("name", f"device {i}"))
                except Exception:
                    continue

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
