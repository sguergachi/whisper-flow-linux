"""Audio recording functionality for whisper-flow."""

import collections
import contextlib
import os
import sys
import tempfile
import threading
import time
import warnings
import wave

import numpy as np

# Suppress webrtcvad setuptools warning during import
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated",
        category=UserWarning,
    )
    import webrtcvad


# Suppress ALSA warnings during PyAudio import and usage
@contextlib.contextmanager
def suppress_alsa_warnings():
    """Context manager to suppress ALSA warnings."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


try:
    with suppress_alsa_warnings():
        import pyaudio
except ImportError:
    pyaudio = None

from .config import Config
from .logging import log
from .system import SystemManager

# Recordings shorter than this are dropped rather than transcribed - whisper
# invents text when handed a fraction of a second of audio.
MIN_RECORDING_SECONDS = 0.35

# How long the capture stream is held open after a recording. Long enough
# that a run of dictations pays to open the microphone once, short enough
# that the in-use indicator does not simply stay lit.
MIC_KEEP_WARM_SECONDS = 90.0


class AudioRecorder:
    """Audio recording with Voice Activity Detection."""

    def __init__(self, config: Config, system_manager: SystemManager):
        """Initialize audio recorder.

        Args:
            config: Configuration object
            system_manager: System manager for notifications

        """
        self.config = config
        self.system_manager = system_manager
        self.vad = webrtcvad.Vad(config.vad_mode)

        # Initialize PyAudio with ALSA warning suppression.
        #
        # Never fatal. The import above is already guarded, so pyaudio can be
        # None here, and PyAudio() itself raises on a machine whose audio
        # stack will not initialise. Either way this used to abort the
        # recorder's construction and with it the whole daemon, so a sound
        # problem cost the user their tray icon and hotkeys as well as their
        # microphone. _check_pyaudio reports it per recording instead.
        self.pa = None
        self._warm_stream = None
        self._devices_logged = False
        self._warm_chunk = None
        self._warm_timer = None
        self._stream_lock = threading.Lock()
        if pyaudio is None:
            log("[AUDIO] pyaudio is not installed; recording is unavailable")
            return
        try:
            with suppress_alsa_warnings():
                self.pa = pyaudio.PyAudio()
        except Exception as e:
            log(f"[AUDIO] could not initialise the audio system: {e}")

    def _input_device_index(self):
        """Which input device to open, preferring WASAPI on Windows.

        PortAudio enumerates MME first on Windows, and an open() that names
        no device takes the default host API - so this was recording through
        an interface from 1991, which is slow to open and adds latency of its
        own. WASAPI is the modern path and the lowest one a user-mode
        application can reach; below it is the audio engine and kernel
        streaming, which need a driver.

        A device set explicitly in config always wins.
        """
        if self.config.mic_device_index is not None:
            return self.config.mic_device_index
        if sys.platform != "win32" or self.pa is None:
            return None
        try:
            for index in range(self.pa.get_host_api_count()):
                api = self.pa.get_host_api_info_by_index(index)
                if "wasapi" not in str(api.get("name", "")).lower():
                    continue
                device = api.get("defaultInputDevice", -1)
                if device is None or device < 0:
                    continue
                # WASAPI shared mode does not resample: it plays back only at
                # the rate the device is configured for, and rejects anything
                # else with "invalid sample rate". MME quietly converted, which
                # is why moving to WASAPI broke recording outright. Ask first.
                if not self._device_supports_our_rate(device):
                    log(f"[AUDIO] WASAPI device {device} will not do "
                        f"{self.config.sample_rate}Hz; using the default device")
                    return None
                log(f"[AUDIO] using WASAPI input device {device}")
                return device
        except Exception as e:
            log(f"[AUDIO] could not pick a WASAPI device: {e}")
        return None

    def _device_supports_our_rate(self, device: int) -> bool:
        """Whether the device will capture at the rate whisper needs."""
        try:
            return bool(self.pa.is_format_supported(
                float(self.config.sample_rate),
                input_device=device,
                input_channels=1,
                input_format=pyaudio.paInt16,
            ))
        except Exception:
            return False

    def log_input_devices(self) -> None:
        """List the capture devices once, with the one we would choose marked.

        A recording that returns silence is almost always the wrong device
        rather than a broken microphone, and there was no way to tell which
        device was being used, let alone what else was on offer. Logged once
        per process: it is for reading in a report, not a running commentary.
        """
        if self._devices_logged or self.pa is None:
            return
        self._devices_logged = True
        try:
            chosen = self._input_device_index()
            default = self.pa.get_default_input_device_info()
            log(f"[AUDIO] default input: [{default['index']}] "
                f"{default['name']} @ {int(default['defaultSampleRate'])}Hz")
            for index in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(index)
                if not info.get("maxInputChannels"):
                    continue
                api = self.pa.get_host_api_info_by_index(info["hostApi"])["name"]
                mark = " <- using" if index == chosen else ""
                log(f"[AUDIO]   [{index}] {info['name']} "
                    f"({api}, {int(info['defaultSampleRate'])}Hz){mark}")
        except Exception as e:
            log(f"[AUDIO] could not list input devices: {e}")

    def _open_input_stream(self, chunk: int):
        """Open a capture stream, reusing a warm one when there is one.

        Opening is the whole delay between pressing the hotkey and the
        microphone being live. Keeping the stream around means a second
        dictation starts immediately; it is released after
        MIC_KEEP_WARM_SECONDS so the microphone-in-use indicator does not
        stay lit, and so nothing else is kept out of the device.
        """
        with self._stream_lock:
            warm = self._warm_stream
            if warm is not None and self._warm_chunk == chunk:
                self._warm_stream = None
                try:
                    if not warm.is_active():
                        warm.start_stream()
                    log("[AUDIO] reused a warm capture stream")
                    return warm
                except Exception as e:
                    log(f"[AUDIO] warm stream unusable: {e}")
                    self._close_stream(warm)

        self.log_input_devices()
        device = self._input_device_index()
        try:
            with suppress_alsa_warnings():
                return self.pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.config.sample_rate,
                    input_device_index=device,
                    input=True,
                    frames_per_buffer=chunk,
                )
        except Exception as e:
            if device is None:
                raise
            # Whatever the reason, a chosen device must never be the thing
            # that stops a recording. The platform default is what worked
            # before any of this and is always the fallback.
            log(f"[AUDIO] input device {device} would not open ({e}); "
                f"falling back to the default device")
            with suppress_alsa_warnings():
                return self.pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.config.sample_rate,
                    input_device_index=None,
                    input=True,
                    frames_per_buffer=chunk,
                )

    def _keep_stream_warm(self, stream, chunk: int) -> None:
        """Hold a finished stream open briefly, then let the microphone go."""
        if stream is None:
            return
        try:
            stream.stop_stream()
        except Exception:
            self._close_stream(stream)
            return

        with self._stream_lock:
            previous, self._warm_stream = self._warm_stream, stream
            self._warm_chunk = chunk
            if self._warm_timer is not None:
                self._warm_timer.cancel()
            self._warm_timer = threading.Timer(
                MIC_KEEP_WARM_SECONDS, self._release_warm_stream)
            self._warm_timer.daemon = True
            self._warm_timer.start()
        self._close_stream(previous)

    def _release_warm_stream(self) -> None:
        with self._stream_lock:
            stream, self._warm_stream = self._warm_stream, None
            self._warm_timer = None
        if stream is not None:
            log("[AUDIO] releasing the microphone after idle")
            self._close_stream(stream)

    @staticmethod
    def _close_stream(stream) -> None:
        if stream is None:
            return
        try:
            stream.close()
        except Exception:
            pass

    def _read_audio_with_timeout(self, stream, chunk, timeout=0.1):
        """Read audio data (no timeout, PyAudio does not support timeout argument)."""
        try:
            return stream.read(chunk, exception_on_overflow=False)
        except Exception as e:
            log(f"Audio read error: {e}")
            return None

    def _stop_stream_safely(self, stream):
        """Safely stop and close audio stream with timeout protection.

        Args:
            stream: PyAudio stream to stop

        """
        try:
            # Set a timeout for stream operations
            stream.stop_stream()
            stream.close()
        except Exception as e:
            log(f"Error stopping audio stream: {e}")
            # Force close if normal stop fails
            try:
                stream.close()
            except Exception:
                pass

    def record_with_vad(self, stop_event=None, level_file: str | None = None) -> str | None:
        """Record audio with Voice Activity Detection.

        Args:
            stop_event: Threading event to stop recording
            level_file: Path to write audio levels for HUD visualization

        Returns:
            Path to the recorded audio file, or None if cancelled

        """
        if not self._check_pyaudio():
            return None

        # Create temporary file
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        frame_len = int(self.config.sample_rate * self.config.frame_ms / 1000)
        chunk = frame_len

        try:
            with suppress_alsa_warnings():
                stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.config.sample_rate,
                    input_device_index=self.config.mic_device_index,
                    input=True,
                    frames_per_buffer=chunk,
                )

            ring_buffer = collections.deque(maxlen=20)
            recording = False
            frames = []
            last_voice_time = time.time()

            # Stop flag; the daemon signals termination through stop_event.
            stop_flag = {"stop": False}

            try:
                while not stop_flag["stop"]:
                    # Check if stop event is set (for daemon control)
                    if stop_event and stop_event.is_set():
                        stop_flag["stop"] = True
                        break

                    # Read audio with timeout to prevent blocking
                    buf = self._read_audio_with_timeout(stream, chunk)
                    if buf is None:
                        # Audio read failed, try to continue but log warning
                        log("Warning: Audio read error, continuing...")
                        continue

                    self._write_level(level_file, buf)

                    voiced = self.vad.is_speech(buf, self.config.sample_rate)
                    ring_buffer.append(buf)

                    if voiced:
                        if not recording:
                            # Start recording, include buffered audio
                            frames.extend(ring_buffer)
                            recording = True
                        frames.append(buf)
                        last_voice_time = time.time()
                    elif recording and (
                        time.time() - last_voice_time > self.config.silence_timeout
                    ):
                        # Stop recording after silence timeout
                        stop_flag["stop"] = True
                        break

            finally:
                self._stop_stream_safely(stream)

            # Save the recorded audio
            if frames:
                self._save_wav_file(output_path, frames)
                return output_path
            os.unlink(output_path)
            return None

        except Exception as e:
            log(f"Recording error: {e}")
            try:
                os.unlink(output_path)
            except Exception:
                pass
            return None

    @staticmethod
    def _loudness(frames: list) -> tuple[int, int]:
        """Peak and mean amplitude of a recording, 0-32767.

        Distinguishes "the user said nothing" from "the microphone handed us
        silence", which look identical once whisper reports blank audio and
        are entirely different problems - the first is nothing to fix, the
        second is the wrong capture device.
        """
        if not frames:
            return 0, 0
        try:
            samples = np.frombuffer(b"".join(frames), dtype=np.int16)
            if samples.size == 0:
                return 0, 0
            return int(np.abs(samples).max()), int(np.abs(samples).mean())
        except Exception:
            return 0, 0

    def _write_level(self, level_file: str | None, buf: bytes):
        """Write audio level to the level file for HUD visualization.

        Args:
            level_file: Path to level file, or None
            buf: Audio frame bytes

        """
        if not level_file:
            return
        try:
            samples = np.frombuffer(buf, dtype=np.int16)
            if samples.size < 1:
                return
            import struct
            rms = int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            # Append without creating. The daemon deletes this file when the
            # recording stops, and the capture loop can run one more iteration
            # after that; opening with "ab" would resurrect it as an orphan in
            # /tmp, and the HUD uses its absence to know it has been dropped.
            fd = os.open(level_file, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(fd, struct.pack("<i", rms))
            finally:
                os.close(fd)
        except Exception:
            # Level reporting is cosmetic; never let it break a recording.
            pass

    def record_push_to_talk(
        self,
        stop_key: str,
        stop_event=None,
        level_file: str | None = None,
        on_tick=None,
        tick_seconds: float = 1.0,
        on_ready=None,
    ) -> str | None:
        """Record audio with push-to-talk functionality.

        Args:
            stop_key: Key combination to stop recording (for display only)
            stop_event: Threading event to stop recording
            level_file: Path to write audio levels for HUD visualization
            on_tick: Called with a snapshot of the frames so far, roughly every
                tick_seconds, so live transcription can run alongside. It must
                return immediately - it is on the capture loop.
            tick_seconds: How often to hand out a snapshot
            on_ready: Called once the microphone is actually capturing. The
                overlay is shown from here rather than before, because the
                user treats it as "speak now" - and opening a capture stream
                on Windows takes long enough that words spoken against the
                old, earlier overlay were simply not recorded.

        Returns:
            Path to the recorded audio file, or None if cancelled

        """
        if not self._check_pyaudio():
            return None

        # Create temporary file
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        frame_len = int(self.config.sample_rate * self.config.frame_ms / 1000)
        chunk = frame_len

        try:
            opened_at = time.monotonic()
            stream = self._open_input_stream(chunk)

            log(f"[AUDIO] capture stream open in "
                f"{(time.monotonic() - opened_at) * 1000:.0f}ms")
            if on_ready:
                try:
                    on_ready()
                except Exception as e:
                    log(f"[AUDIO] on_ready failed: {e}")

            frames = []
            stop_flag = {"stop": False}

            tick_frames = max(1, int(tick_seconds * 1000 / self.config.frame_ms))
            frame_count = 0
            try:
                while not stop_flag["stop"]:
                    # Check if stop event is set (for daemon control)
                    if stop_event and stop_event.is_set():
                        log(f"[AUDIO] stop_event set after {frame_count} frames")
                        stop_flag["stop"] = True
                        break

                    # Read audio with timeout to prevent blocking
                    buf = self._read_audio_with_timeout(stream, chunk)
                    if buf is None:
                        # Audio read failed, try to continue but log warning
                        log("Warning: Audio read error, continuing...")
                        continue

                    frames.append(buf)
                    self._write_level(level_file, buf)
                    frame_count += 1

                    if on_tick and frame_count % tick_frames == 0:
                        try:
                            on_tick(list(frames))
                        except Exception as e:
                            log(f"[AUDIO] on_tick error: {e}")

            finally:
                # Keep it open briefly rather than closing: reopening is the
                # whole delay between the hotkey and the microphone being live.
                self._keep_stream_warm(stream, chunk)

            # Discard taps too short to contain speech - whisper hallucinates
            # text from a fraction of a second of audio.
            duration = frame_count * self.config.frame_ms / 1000
            if duration < MIN_RECORDING_SECONDS:
                log(f"[AUDIO] discarding {duration:.2f}s recording (too short)")
                os.unlink(output_path)
                return None

            # Save the recorded audio
            if frames:
                self._save_wav_file(output_path, frames)
                peak, mean = self._loudness(frames)
                quiet = " - SILENT, check the capture device" if peak < 200 else ""
                log(f"[AUDIO] recording stopped after {duration:.2f}s, "
                    f"peak {peak} mean {mean}{quiet}")
                return output_path
            # Zero frames from a stream that opened is the signature of a
            # microphone the OS is refusing to hand over. On Windows 11 that
            # is Settings > Privacy > Microphone > "Let desktop apps access
            # your microphone", which denies audio without failing the open.
            device = self.config.mic_device_index
            log(f"[AUDIO] no frames captured in {duration:.2f}s from device "
                f"{'default' if device is None else device} - the microphone "
                f"gave nothing. Check that the OS allows this app to record.")
            os.unlink(output_path)
            return None

        except Exception as e:
            log(f"Recording error: {e}")
            try:
                os.unlink(output_path)
            except Exception:
                pass
            return None

    def record_until_silence(
        self,
        silence_duration: float,
        stop_event=None,
        level_file: str | None = None,
    ) -> str | None:
        """Record audio until silence is detected for the specified duration.

        Args:
            silence_duration: Duration of silence in seconds before stopping
            stop_event: Threading event to stop recording
            level_file: Path to write audio levels for HUD visualization

        Returns:
            Path to the recorded audio file, or None if cancelled

        """
        if not self._check_pyaudio():
            return None

        # Create temporary file
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        frame_len = int(self.config.sample_rate * self.config.frame_ms / 1000)
        chunk = frame_len

        try:
            with suppress_alsa_warnings():
                stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.config.sample_rate,
                    input_device_index=self.config.mic_device_index,
                    input=True,
                    frames_per_buffer=chunk,
                )

            frames = []
            last_voice_time = time.time()
            stop_flag = {"stop": False}
            recording_started = False

            try:
                while not stop_flag["stop"]:
                    # Check if stop event is set (for daemon control)
                    if stop_event and stop_event.is_set():
                        stop_flag["stop"] = True
                        break

                    # Read audio with timeout to prevent blocking
                    buf = self._read_audio_with_timeout(stream, chunk)
                    if buf is None:
                        # Audio read failed, try to continue but log warning
                        log("Warning: Audio read error, continuing...")
                        continue

                    frames.append(buf)
                    self._write_level(level_file, buf)

                    # Check for voice activity
                    voiced = self.vad.is_speech(buf, self.config.sample_rate)

                    if voiced:
                        last_voice_time = time.time()
                        if not recording_started:
                            recording_started = True
                            log("Voice detected, recording...")
                    elif recording_started and (
                        time.time() - last_voice_time > silence_duration
                    ):
                        # Stop recording after silence duration
                        log(
                            f"Silence detected for {silence_duration}s, stopping...",
                        )
                        stop_flag["stop"] = True
                        break

            finally:
                self._stop_stream_safely(stream)

            # Save the recorded audio
            if frames and recording_started:
                self._save_wav_file(output_path, frames)
                return output_path
            os.unlink(output_path)
            return None

        except Exception as e:
            log(f"Recording error: {e}")
            try:
                os.unlink(output_path)
            except Exception:
                pass
            return None

    def _save_wav_file(self, output_path: str, frames: list):
        """Save recorded frames to a WAV file.

        Args:
            output_path: Path to save the WAV file
            frames: List of audio frames to save

        """
        # Apply speedup if enabled (not 1.0)
        if self.config.speedup_audio != 1.0:
            frames = self._speedup_audio_frames(frames, self.config.speedup_audio)

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(b"".join(frames))

    def _speedup_audio_frames(self, frames: list, speed_multiplier: float) -> list:
        """Speed up audio frames by 1.5x using linear interpolation.

        Args:
            frames: List of audio frame bytes
            speed_multiplier: Speed multiplier (1.5 = 1.5x speed, 2.0 = 2x speed, etc.)

        Returns:
            List of speeded up audio frame bytes

        """
        if not frames or speed_multiplier == 1.0:
            return frames

        # Convert frames to numpy array
        audio_data = b"".join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate new length based on speed multiplier
        original_length = len(audio_array)
        new_length = int(original_length / speed_multiplier)

        # Create new time indices for interpolation
        original_indices = np.arange(original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)

        # Interpolate audio data
        speeded_audio = np.interp(new_indices, original_indices, audio_array)

        # Convert back to int16 and then to bytes
        speeded_audio = speeded_audio.astype(np.int16)
        return [speeded_audio.tobytes()]

    def _check_pyaudio(self) -> bool:
        """Check if PyAudio is available.

        Returns:
            True if PyAudio is available, False otherwise

        """
        if pyaudio is None:
            self.system_manager.notify("PyAudio not available")
            return False
        if self.pa is None:
            # Construction is not allowed to fail, so the failure surfaces
            # here, on the recording that needs it.
            log("[AUDIO] the audio system never initialised; retrying")
            try:
                with suppress_alsa_warnings():
                    self.pa = pyaudio.PyAudio()
            except Exception as e:
                log(f"[AUDIO] audio system still unavailable: {e}")
                self.system_manager.notify(f"No audio system: {e}")
                return False
        return True
