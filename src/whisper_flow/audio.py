"""Audio recording functionality for whisper-flow."""

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

from . import denoise
from .boost import DEAD_PEAK, needs_boost
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

# Kept either side of the speech when silence is trimmed. A voice detector
# marks the frame a word becomes audible, not the frame it began, so cutting
# flush against it clips the opening consonant.
TRIM_PAD_MS = 150

# The HUD's level file is a stream of packed int32s, so it has to be opened as
# bytes. os.open on Windows defaults to text mode, where the CRT puts a \r in
# front of every \n it is handed - and one byte of a packed level is 0x0A about
# once in every 256 frames. That extra byte shifted the whole rest of the file
# by one, so from roughly eight seconds into a recording the overlay unpacked
# halves of neighbouring samples as single values: millions instead of
# hundreds. Its gain is adaptive, so one of those flattened the waveform to
# nothing until the peak decayed back down - the bars stopping mid-sentence
# and returning a few seconds later. O_BINARY does not exist off Windows,
# where there is no translation to turn off.
LEVEL_O_BINARY = getattr(os, "O_BINARY", 0)


# MME device names are cut to MAXPNAMELEN, which is 32 bytes including the
# terminator - so 31 characters, and no warning that anything was lost.
MME_NAME_LIMIT = 31


def _same_device(a: str, b: str) -> bool:
    """Whether two PortAudio names refer to the same physical device.

    Not string equality. PortAudio's default input device is reported
    through MME, which truncates, while WASAPI reports the full name - so
    the same microphone has two spellings and matching exactly finds
    neither. It is visible in a real device list:

        [2]  Microphone (2- Realtek(R) Audio     (MME)
        [17] Microphone (2- Realtek(R) Audio)    (WASAPI)

    The closing bracket is the 32nd character. Comparing the part that
    survives truncation matches the two, and cannot confuse devices that
    MME could tell apart in the first place - if two names share their
    first 31 characters, MME reports them identically anyway.
    """
    if not a or not b:
        return False
    return a[:MME_NAME_LIMIT] == b[:MME_NAME_LIMIT]


def trim_silence(frames: list, vad, sample_rate: int, frame_ms: int,
                 pad_ms: int = TRIM_PAD_MS) -> list:
    """Drop leading and trailing frames that hold no speech.

    The microphone is held open between dictations, and a push-to-talk key is
    pressed before the sentence starts and released after it ends, so a clip
    routinely opens and closes on room tone.

    Under thirty seconds this buys nothing - whisper pads every clip out to a
    full window anyway, so the silence was going to be encoded either way.
    Past thirty seconds it stops being free, because the clip is then cut into
    chunks and every chunk is encoded whether anyone spoke in it or not. A
    sixty second recording holding eight seconds of speech took 2133ms
    untrimmed and 918ms trimmed, and the untrimmed pass was the one that
    misheard "ask not" as "asked not": silence is not just slow to encode,
    it is something for the decoder to invent against.

    `frames` comes back untouched when nothing in it is voiced. A recording
    the detector hears nothing in is something to report - the wrong capture
    device, a muted microphone - and truncating it to nothing would turn a
    diagnosable fault into an empty transcript.
    """
    expected = int(sample_rate * frame_ms / 1000) * 2      # 16-bit mono
    first = last = None
    for index, frame in enumerate(frames):
        if len(frame) != expected:      # webrtcvad rejects odd-sized frames
            continue
        try:
            voiced = vad.is_speech(frame, sample_rate)
        except Exception:
            return frames
        if voiced:
            if first is None:
                first = index
            last = index

    if first is None:
        return frames
    pad = int(pad_ms / frame_ms) if frame_ms else 0
    return frames[max(0, first - pad):last + pad + 1]


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
        # What the device is actually running at. Equal to the configured
        # rate until a stream opens at something else, which is the usual
        # case on Windows: the arrays run at 48kHz and will not do 16.
        self._capture_rate = config.sample_rate
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

    def _new_vad(self):
        """A voice detector that has never heard anything before.

        webrtcvad adapts as it goes: each is_speech call folds the frame into
        a running estimate of what the room sounds like, and the answer it
        gives depends on everything it has been handed since it was built. One
        detector kept on the recorder meant every use inherited every earlier
        one - the same audio trimmed to 79 frames the first time a process
        saw it and 54 frames every time after, because by then the estimate
        had settled somewhere else. So the first dictation after launch kept
        noticeably more silence than the rest of the session, and a trim ran
        against whatever the auto-stop detector had made of the last
        recording.

        Everything that needs a detector makes its own here, so a recording
        is judged on its own audio and nothing else.
        """
        return webrtcvad.Vad(self.config.vad_mode)

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
                # Trust the answer only if the device agrees it belongs to
                # this API, and go looking ourselves when it does not.
                #
                # PortAudio reports defaultInputDevice as a global index, and
                # on at least one machine the WASAPI entry named a WDM-KS
                # device - kernel streaming, exclusive to whatever already
                # holds the endpoint. It failed to open with "Unanticipated
                # host error" and capture fell back to the platform default,
                # which is MME: the same microphone several times quieter.
                #
                # Merely refusing the bad answer lands in that same fallback,
                # so it has to be replaced rather than dropped. The machine
                # that produced this had the right microphone sitting at
                # index 18 on the very API that had just misreported it.
                if device is None or device < 0 or not self._device_belongs_to(
                        device, index):
                    if device is not None and device >= 0:
                        log(f"[AUDIO] host API {api.get('name')} names device "
                            f"{device}, which is not one of its own")
                    replacement = self._first_input_on(index)
                    if replacement is None:
                        continue
                    log(f"[AUDIO] using WASAPI input device {replacement} "
                        f"instead")
                    device = replacement
                # WASAPI shared mode does not resample: it plays back only at
                # the rate the device is configured for, and rejects anything
                # else with "invalid sample rate". MME quietly converted, which
                # is why moving to WASAPI broke recording outright. Ask first.
                if not self._device_supports_our_rate(device):
                    # Kept, not abandoned. It runs at its own rate and we
                    # convert; handing it back meant the platform default
                    # through MME, which is the same microphone several
                    # times quieter.
                    log(f"[AUDIO] WASAPI device {device} will not do "
                        f"{self.config.sample_rate}Hz; capturing at "
                        f"{self._native_rate(device)}Hz and converting")
                    return device
                log(f"[AUDIO] using WASAPI input device {device}")
                return device
        except Exception as e:
            log(f"[AUDIO] could not pick a WASAPI device: {e}")
        return None

    def _first_input_on(self, host_api: int) -> int | None:
        """An input device served by this host API, or None.

        Prefers the one carrying the same name as the system default input,
        so the microphone Windows was told to use is the one reached - just
        through WASAPI rather than through MME. Falling back to whichever
        input comes first would otherwise be a coin toss between a headset
        and a webcam.
        """
        try:
            default_name = str(
                self.pa.get_default_input_device_info().get("name", ""))
        except Exception:
            default_name = ""

        first = None
        try:
            for index in range(self.pa.get_device_count()):
                try:
                    info = self.pa.get_device_info_by_index(index)
                    if int(info.get("hostApi", -1)) != host_api:
                        continue
                    if int(info.get("maxInputChannels", 0)) <= 0:
                        continue
                except Exception:
                    continue
                if first is None:
                    first = index
                if _same_device(str(info.get("name", "")), default_name):
                    return index
        except Exception:
            return first
        return first

    def _device_belongs_to(self, device: int, host_api: int) -> bool:
        """Whether `device` is really served by `host_api`.

        Fails open: only a device that positively names a different host API
        is rejected. This is a sanity check on one implausible answer, not a
        licence to discard a working microphone because PortAudio declined to
        describe it.
        """
        try:
            reported = int(self.pa.get_device_info_by_index(device)["hostApi"])
        except Exception:
            return True
        return reported == host_api

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

    def _native_rate(self, device: int | None) -> int:
        """The rate this device actually runs at, for capturing at it.

        Asking a device for 16kHz it does not have meant falling back to the
        platform default through MME, and MME's conversion of the same
        microphone is markedly quieter: measured on an AMD array, 429 peak
        against 1579 at its native 48kHz - the same room, seconds apart.
        That gap is the difference between whisper hearing a sentence and
        returning nothing.
        """
        if device is None:
            return self.config.sample_rate
        try:
            info = self.pa.get_device_info_by_index(device)
            rate = int(info.get("defaultSampleRate") or 0)
            return rate if rate > 0 else self.config.sample_rate
        except Exception:
            return self.config.sample_rate

    def _resample(self, data: bytes, source_rate: int, samples_out: int) -> bytes:
        """Convert captured audio to the rate whisper is given.

        Averaging over the decimation window before interpolating is what
        keeps this honest: dropping samples outright folds everything above
        8kHz back into the speech as aliasing, and sibilance is exactly
        where a microphone puts its energy.

        The output length is given rather than derived, because the frames
        downstream have to line up: webrtcvad accepts only whole 10, 20 or
        30ms frames and rejects anything a sample short.
        """
        target = self.config.sample_rate
        if source_rate == target or not data:
            return data
        samples = np.frombuffer(data, dtype=np.int16)
        if samples.size == 0:
            return data

        window = max(1, int(round(source_rate / target)))
        if window > 1:
            padding = (-samples.size) % window
            if padding:
                samples = np.concatenate(
                    [samples, np.zeros(padding, dtype=np.int16)])
            smoothed = samples.reshape(-1, window).mean(axis=1)
        else:
            smoothed = samples.astype(np.float32)

        if smoothed.size < 2:
            return np.zeros(samples_out, dtype=np.int16).tobytes()
        positions = np.linspace(0, smoothed.size - 1, samples_out)
        converted = np.interp(positions, np.arange(smoothed.size), smoothed)
        return np.clip(converted, -32768, 32767).astype(np.int16).tobytes()

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
        capture_rate = self._native_rate(device)
        try:
            with suppress_alsa_warnings():
                stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=capture_rate,
                    input_device_index=device,
                    input=True,
                    frames_per_buffer=self._capture_frames(chunk, capture_rate),
                )
            self._capture_rate = capture_rate
            if capture_rate != self.config.sample_rate:
                log(f"[AUDIO] capturing at {capture_rate}Hz, converting to "
                    f"{self.config.sample_rate}Hz")
            return stream
        except Exception as e:
            if device is None:
                self._capture_rate = self.config.sample_rate
                raise
            # Whatever the reason, a chosen device must never be the thing
            # that stops a recording. The platform default is what worked
            # before any of this and is always the fallback.
            log(f"[AUDIO] input device {device} would not open ({e}); "
                f"falling back to the default device")
            self._capture_rate = self.config.sample_rate
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
            # The timer carries the stream it was scheduled for. Cancelling a
            # timer that has already begun firing does nothing, and without
            # the identity check that stale firing would close the stream a
            # newer recording just stored here, and clobber the new timer's
            # bookkeeping with it.
            self._warm_timer = threading.Timer(
                MIC_KEEP_WARM_SECONDS, self._release_warm_stream, args=(stream,))
            self._warm_timer.daemon = True
            self._warm_timer.start()
        self._close_stream(previous)

    def _release_warm_stream(self, expected=None) -> None:
        with self._stream_lock:
            if expected is not None and self._warm_stream is not expected:
                return  # a newer recording has already taken the warm slot
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

    def _capture_frames(self, chunk: int, capture_rate: int) -> int:
        """How many frames to read to yield `chunk` at whisper's rate."""
        if capture_rate == self.config.sample_rate:
            return chunk
        return max(1, int(round(chunk * capture_rate / self.config.sample_rate)))

    def _read_audio_with_timeout(self, stream, chunk, timeout=0.1):
        """Read one chunk, at whisper's rate whatever the device runs at.

        The conversion belongs here so that everything downstream - the
        voice detector, the level file, the wav - keeps seeing the rate it
        was written for.
        """
        capture_rate = getattr(self, "_capture_rate", self.config.sample_rate)
        try:
            data = stream.read(self._capture_frames(chunk, capture_rate),
                               exception_on_overflow=False)
        except Exception as e:
            log(f"Audio read error: {e}")
            return None
        try:
            return self._resample(data, capture_rate, chunk)
        except Exception as e:
            # Never let conversion be the thing that ends a recording; the
            # raw audio is wrong-rated but present.
            log(f"[AUDIO] could not convert {capture_rate}Hz audio: {e}")
            return data

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
            fd = os.open(level_file, os.O_WRONLY | os.O_APPEND | LEVEL_O_BINARY)
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
                # A whisper peaks around 100, and calling that a dead device
                # was wrong: the microphone was working, the voice was small.
                # Only a genuine noise floor gets the warning now.
                if peak < DEAD_PEAK:
                    quiet = " - SILENT, check the capture device"
                elif needs_boost(peak):
                    quiet = " - very quiet, will retry amplified if it comes back empty"
                else:
                    quiet = ""
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
        on_ready=None,
        max_duration: float | None = None,
    ) -> str | None:
        """Record audio until silence is detected for the specified duration.

        Args:
            silence_duration: Duration of silence in seconds before stopping
            stop_event: Threading event to stop recording
            level_file: Path to write audio levels for HUD visualization
            on_ready: Called once the microphone is actually capturing
            max_duration: Hard cap on how long to listen (seconds). Without
                this, a silent room never trips VAD and the loop never ends,
                leaving the daemon "busy" so every later hotkey does nothing.

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
        # Fall back to the configured ceiling so a missing max never means
        # "wait forever".
        if max_duration is None:
            max_duration = float(getattr(self.config, "max_recording_duration", 120.0)
                                 or 120.0)

        try:
            stream = self._open_input_stream(chunk)
            if on_ready:
                try:
                    on_ready()
                except Exception as e:
                    log(f"[AUDIO] on_ready failed: {e}")

            frames = []
            # Per recording, not per recorder: this one adapts to the room as
            # it listens, and the room it adapted to last time is not this one.
            vad = self._new_vad()
            started_at = time.time()
            last_voice_time = started_at
            stop_flag = {"stop": False}
            recording_started = False

            try:
                while not stop_flag["stop"]:
                    # Check if stop event is set (for daemon control)
                    if stop_event and stop_event.is_set():
                        stop_flag["stop"] = True
                        break

                    if time.time() - started_at > max_duration:
                        log(f"[AUDIO] auto-stop hit max duration "
                            f"({max_duration:g}s)")
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

                    # Check for voice activity. webrtcvad raises on a bad
                    # frame size; that must not kill the whole recording.
                    try:
                        voiced = vad.is_speech(buf, self.config.sample_rate)
                    except Exception as e:
                        log(f"[AUDIO] VAD skipped a frame: {e}")
                        voiced = False

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
                    elif (
                        not recording_started
                        and time.time() - started_at
                        > max(silence_duration + 2.0, 5.0)
                    ):
                        # No speech after the HUD has been up long enough to
                        # speak: stop rather than sit open until max_duration.
                        # Was silence+1s (~3s), which was too short once the
                        # mic open + overlay lag ate a second of the window.
                        log("[AUDIO] auto-stop: no voice heard, giving up")
                        stop_flag["stop"] = True
                        break

            finally:
                # Same lifecycle as push-to-talk: hold the stream warm briefly
                # rather than paying to reopen the microphone next time.
                self._keep_stream_warm(stream, chunk)

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

    def trim_frames(self, frames: list) -> list:
        """Frames with the silence either side of the speech removed.

        Refuses to hand back anything shorter than the length below which a
        recording is discarded as too short to hold speech: a trim that lands
        there has found almost nothing, and the untrimmed audio is the more
        honest thing to transcribe.
        """
        if not self.config.trim_silence or not frames:
            return frames
        trimmed = trim_silence(
            frames, self._new_vad(), self.config.sample_rate,
            self.config.frame_ms)
        if len(trimmed) * self.config.frame_ms / 1000 < MIN_RECORDING_SECONDS:
            return frames
        if len(trimmed) < len(frames):
            log(f"[AUDIO] trimmed {len(frames) - len(trimmed)} silent frames "
                f"({(len(frames) - len(trimmed)) * self.config.frame_ms / 1000:.2f}s)")
        return trimmed

    def _save_wav_file(self, output_path: str, frames: list):
        """Save recorded frames to a WAV file.

        Args:
            output_path: Path to save the WAV file
            frames: List of audio frames to save

        """
        frames = self.trim_frames(frames)

        # Apply speedup if enabled (not 1.0). After the trim, which counts on
        # frames still being the length the voice detector expects.
        if self.config.speedup_audio != 1.0:
            frames = self._speedup_audio_frames(frames, self.config.speedup_audio)

        audio = b"".join(frames)
        if self.config.noise_filter and audio:
            # Here rather than per chunk: the gate measures the room from the
            # recording it is given, and a 30ms chunk has no idea whether it
            # is quiet because nobody is speaking or because the whole room
            # is quiet.
            try:
                samples = np.frombuffer(audio, dtype=np.int16)
                audio = denoise.clean(
                    samples, self.config.sample_rate).tobytes()
            except Exception as e:
                log(f"[AUDIO] noise filter skipped: {e}")

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(audio)

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
