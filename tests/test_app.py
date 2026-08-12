"""Unit tests for the WhisperFlow application class."""

import tempfile
from unittest.mock import Mock, patch

from whisper_flow.app import WhisperFlow


class TestWhisperFlow:
    """Test the WhisperFlow application class."""

    def test_init_default(self, temp_config_dir):
        """Test WhisperFlow initialization with default parameters."""
        with (
            patch("whisper_flow.app.Config") as mock_config_class,
            patch("whisper_flow.app.AudioRecorder") as mock_audio_recorder_class,
        ):
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_audio_recorder_class.return_value = Mock()
            app = WhisperFlow()
            assert app.mode == "default"
            mock_config_class.assert_called_once_with()

    def test_init_with_config_dir(self, temp_config_dir):
        """Test WhisperFlow initialization with custom config directory."""
        with (
            patch("whisper_flow.app.Config") as mock_config_class,
            patch("whisper_flow.app.AudioRecorder") as mock_audio_recorder_class,
        ):
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_audio_recorder_class.return_value = Mock()
            app = WhisperFlow(config_dir=temp_config_dir, mode="transcribe")
            assert app.mode == "transcribe"
            mock_config_class.assert_called_once_with(config_dir=temp_config_dir)

    def test_run_voice_flow_push_to_talk_daemon_success(self, mock_config):
        """Test successful push-to-talk voice flow."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager") as mock_system_class,
            patch("whisper_flow.app.AudioRecorder") as mock_audio_class,
            patch("whisper_flow.app.TranscriptionService") as mock_transcription_class,
        ):
            # Setup mocks
            mock_system = Mock()
            mock_system_class.return_value = mock_system

            mock_audio = Mock()
            mock_audio.record_push_to_talk.return_value = "/tmp/test.wav"
            mock_audio_class.return_value = mock_audio

            mock_transcription = Mock()
            mock_transcription.transcribe_audio.return_value = "Test transcript"
            mock_transcription_class.return_value = mock_transcription

            # Create app and test
            app = WhisperFlow(mode="command")
            stop_event = Mock()

            result = app.run_voice_flow_push_to_talk_daemon("ctrl+shift+t", stop_event)

            assert result is True
            # on_ready is how the overlay learns the microphone is live; it
            # must reach the recorder or the overlay goes back to appearing
            # before anything is being captured.
            mock_audio.record_push_to_talk.assert_called_once_with(
                "ctrl+shift+t",
                stop_event,
                level_file=None,
                on_ready=None,
            )
            mock_transcription.transcribe_audio.assert_called_once_with("/tmp/test.wav")

    def test_run_voice_flow_push_to_talk_daemon_no_audio(self, mock_config):
        """Test push-to-talk voice flow when no audio is recorded."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager") as mock_system_class,
            patch("whisper_flow.app.AudioRecorder") as mock_audio_class,
            patch("whisper_flow.app.TranscriptionService"),
        ):
            mock_system = Mock()
            mock_system_class.return_value = mock_system

            mock_audio = Mock()
            mock_audio.record_push_to_talk.return_value = None
            mock_audio_class.return_value = mock_audio

            app = WhisperFlow()
            stop_event = Mock()

            result = app.run_voice_flow_push_to_talk_daemon("ctrl+shift+t", stop_event)

            assert result is False

    def test_run_voice_flow_auto_stop_success(self, mock_config):
        """Test successful auto-stop voice flow."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager") as mock_system_class,
            patch("whisper_flow.app.AudioRecorder") as mock_audio_class,
            patch("whisper_flow.app.TranscriptionService") as mock_transcription_class,
        ):
            mock_system = Mock()
            mock_system_class.return_value = mock_system

            mock_audio = Mock()
            mock_audio.record_until_silence.return_value = "/tmp/test.wav"
            mock_audio_class.return_value = mock_audio

            mock_transcription = Mock()
            mock_transcription.transcribe_audio.return_value = "Test transcript"
            mock_transcription_class.return_value = mock_transcription

            app = WhisperFlow(mode="transcribe")

            result = app.run_voice_flow_auto_stop(silence_duration=3.0)

            assert result is True
            mock_audio.record_until_silence.assert_called_once_with(
                3.0,
                stop_event=None,
                level_file=None,
                on_ready=None,
                max_duration=None,
            )

    def test_run_comprehensive_validation(self, mock_config):
        """Test comprehensive validation method."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService"),
        ):
            app = WhisperFlow()

            results = app.run_comprehensive_validation()

            assert isinstance(results, dict)
            assert "api_config" in results
            assert "system_deps" in results
            assert "audio_system" in results
            assert "services" in results
            assert "config_files" in results
            assert "environment" in results

    def test_process_recorded_audio_transcribe_mode(self, mock_config):
        """Test processing recorded audio in transcribe mode."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager") as mock_system_class,
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService") as mock_transcription_class,
        ):
            mock_system = Mock()
            mock_system.paste_text.return_value = True
            mock_system_class.return_value = mock_system
            mock_transcription = Mock()
            mock_transcription.transcribe_audio.return_value = "Test transcript"
            mock_transcription_class.return_value = mock_transcription
            app = WhisperFlow(mode="transcribe")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(b"fake audio data")
                tmp_file.flush()
                result = app._process_recorded_audio(tmp_file.name)
                assert result is True
                mock_transcription.transcribe_audio.assert_called_once_with(
                    tmp_file.name,
                )
                mock_system.paste_text.assert_called_once_with("Test transcript")

    def test_command_mode_pastes_the_transcript_verbatim(self, mock_config):
        """No mode rewrites the words: every one pastes what was heard.

        Command mode used to send the transcript to a hosted model and paste
        its answer. That backend is gone, so it must not silently paste
        something other than the dictation.
        """
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager") as mock_system_class,
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService") as mock_transcription_class,
        ):
            mock_system = Mock()
            mock_system.get_active_window_title.return_value = "Test Window"
            mock_system.paste_text.return_value = True
            mock_system_class.return_value = mock_system
            mock_transcription = Mock()
            mock_transcription.transcribe_audio.return_value = "Test transcript"
            mock_transcription_class.return_value = mock_transcription

            app = WhisperFlow(mode="command")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(b"fake audio data")
                tmp_file.flush()
                result = app._process_recorded_audio(tmp_file.name)
                assert result is True
                mock_transcription.transcribe_audio.assert_called_once_with(
                    tmp_file.name,
                )
                mock_system.paste_text.assert_called_once_with("Test transcript")

    def test_process_recorded_audio_transcription_failure(self, mock_config):
        """Test processing recorded audio when transcription fails."""
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService") as mock_transcription_class,
        ):
            mock_transcription = Mock()
            mock_transcription.transcribe_audio.return_value = None
            mock_transcription_class.return_value = mock_transcription
            app = WhisperFlow()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(b"fake audio data")
                tmp_file.flush()
                result = app._process_recorded_audio(tmp_file.name)
                assert result is False

    def test_validate_api_config_with_a_server(self, mock_config):
        """A whisper.cpp server is the only backend, and it is enough."""
        mock_config.local_whisper_url = "http://127.0.0.1:8082"

        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService"),
        ):
            app = WhisperFlow()
            results = app._validate_api_config()

            assert len(results) == 1
            assert results[0]["name"] == "Transcription backend"
            assert results[0]["status"] == "pass"
            assert "http://127.0.0.1:8082" in results[0]["message"]

    def test_validate_api_config_without_a_server(self, mock_config):
        """No server anywhere is the one real failure."""
        mock_config.local_whisper_url = ""

        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder"),
            patch("whisper_flow.app.TranscriptionService"),
        ):
            app = WhisperFlow()
            results = app._validate_api_config()

            assert len(results) == 1
            assert results[0]["status"] == "fail"
            assert "Nothing configured" in results[0]["message"]

    def test_audio_speedup_configuration(self, mock_config):
        """Test that audio speedup configuration is properly handled."""
        # Test with speedup disabled (1.0 = normal speed)
        mock_config.speedup_audio = 1.0
        with patch("whisper_flow.app.Config", return_value=mock_config):
            app = WhisperFlow()
            assert app.config.speedup_audio == 1.0

        # Test with speedup enabled (1.5x speed)
        mock_config.speedup_audio = 1.5
        with patch("whisper_flow.app.Config", return_value=mock_config):
            app = WhisperFlow()
            assert app.config.speedup_audio == 1.5

    def test_audio_speedup_processing(self, mock_config):
        """Test that audio speedup processing works correctly."""
        # Test with speedup enabled (1.5x speed)
        mock_config.speedup_audio = 1.5
        with (
            patch("whisper_flow.app.Config", return_value=mock_config),
            patch("whisper_flow.app.SystemManager"),
            patch("whisper_flow.app.AudioRecorder") as mock_audio_class,
            patch("whisper_flow.app.TranscriptionService"),
        ):
            mock_audio = Mock()
            mock_audio.config = mock_config
            mock_audio_class.return_value = mock_audio

            app = WhisperFlow()

            # Verify that the audio recorder has the speedup configuration
            assert app.audio_recorder.config.speedup_audio == 1.5

    def test_logging_configuration(self, mock_config):
        """Test that logging configuration is properly handled."""
        # Test with logging disabled (default)
        mock_config.logging_enabled = False
        with patch("whisper_flow.app.Config", return_value=mock_config):
            app = WhisperFlow()
            assert app.config.logging_enabled is False

        # Test with logging enabled
        mock_config.logging_enabled = True
        with patch("whisper_flow.app.Config", return_value=mock_config):
            app = WhisperFlow()
            assert app.config.logging_enabled is True

    def test_logging_function(self):
        """Test that the logging function works correctly."""
        from whisper_flow.logging import log, set_logging_enabled

        # Test with logging disabled
        set_logging_enabled(False)
        # This should not print anything
        log("This should not appear")

        # Test with logging enabled
        set_logging_enabled(True)
        # This should print
        log("This should appear")


def test_the_live_flow_really_does_shorten_its_passes(monkeypatch):
    """Pins the wiring, not a copy of it: catches the closure being changed."""
    from unittest.mock import Mock

    from whisper_flow import app as app_module

    seen = {}

    class Recorder:
        def record_push_to_talk(self, *a, **kw):
            return None                              # stop after the wiring

        def trim_frames(self, frames):
            return frames

        def prepare_frames(self, frames):
            return frames

    class Service:
        def transcribe_audio(self, path, max_retries=3, timeout=None):
            seen["max_retries"] = max_retries
            seen["timeout"] = timeout
            return "text"

    captured = {}

    class FakeLive:
        def __init__(self, transcribe, emit, sample_rate, interval,
                     prepare=None):
            captured["transcribe"] = transcribe

        def start(self): pass
        def stop(self): pass

    monkeypatch.setattr(app_module, "LiveTranscriber", FakeLive)

    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.transcription_service = Service()
    flow.system_manager = Mock()
    flow.audio_recorder = Recorder()
    flow.config = Mock(sample_rate=16000, live_interval=0.9)

    flow.run_voice_flow_push_to_talk_live("super+alt", Mock())
    captured["transcribe"]("/tmp/a.wav")
    assert seen["max_retries"] == 1
    assert seen["timeout"] == app_module.LIVE_TIMEOUT


def test_the_overlay_is_shown_only_once_capture_has_started(monkeypatch):
    """It reads as "speak now", so it must not appear while the mic opens."""
    from unittest.mock import Mock

    from whisper_flow import app as app_module

    order = []

    class Recorder:
        def record_push_to_talk(self, *a, on_ready=None, **kw):
            order.append("stream opened")
            if on_ready:
                on_ready()
            return None

        def trim_frames(self, frames):
            return frames

        def prepare_frames(self, frames):
            return frames

    class FakeLive:
        def __init__(self, **kw): pass
        def start(self): pass
        def stop(self): pass
        def offer(self, frames): pass       # handed to the capture loop

    monkeypatch.setattr(app_module, "LiveTranscriber", FakeLive)
    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.transcription_service = Mock()
    flow.system_manager = Mock()
    flow.audio_recorder = Recorder()
    flow.config = Mock(sample_rate=16000, live_interval=0.9)

    flow.run_voice_flow_push_to_talk_live(
        "super+alt", Mock(), on_ready=lambda: order.append("overlay shown"))
    assert order == ["stream opened", "overlay shown"]


def test_a_quiet_recording_is_retried_louder_before_giving_up(monkeypatch, tmp_path):
    """From a real session: peak 89-126 transcribed to nothing, peak 281
    worked. The speech was captured, it was just small."""
    import wave

    import numpy as np

    from whisper_flow import app as app_module

    quiet = tmp_path / "quiet.wav"
    t = np.linspace(0, 1.0, 16000, False)
    sig = (np.sin(2 * np.pi * 200 * t) * 110).astype(np.int16)
    with wave.open(str(quiet), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(sig.tobytes())

    attempts = []

    class Service:
        def transcribe_audio(self, path, **kw):
            attempts.append(path)
            # Blank for the original, text once it has been amplified.
            return "hello there" if path.endswith(".louder.wav") else None

    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.transcription_service = Service()

    result = flow._transcribe_allowing_for_a_whisper(str(quiet))
    assert result == "hello there"
    assert len(attempts) == 2
    assert attempts[1].endswith(".louder.wav")
    assert not (tmp_path / "quiet.wav.louder.wav").exists()   # cleaned up


def test_a_recording_that_transcribes_is_not_retried(tmp_path):
    from whisper_flow import app as app_module

    attempts = []

    class Service:
        def transcribe_audio(self, path, **kw):
            attempts.append(path)
            return "first time"

    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.transcription_service = Service()
    assert flow._transcribe_allowing_for_a_whisper(
        str(tmp_path / "x.wav")) == "first time"
    assert len(attempts) == 1        # the retry costs a pass; do not spend it


def _write_music_capture(tmp_path):
    """A continuous (music-like) raw capture, as the recorder leaves it."""
    import wave

    import numpy as np

    from whisper_flow import audio_debug

    rng = np.random.default_rng(11)
    rate = 16000
    seconds = 2.6
    t = np.arange(int(rate * seconds)) / rate
    bed = (np.sin(2 * np.pi * 120 * t) * 0.6
           + np.sin(2 * np.pi * 340 * t) * 0.4)
    tremolo = 0.75 + 0.25 * np.sin(2 * np.pi * 1.7 * t)
    sig = (bed * tremolo * 2200
           + rng.normal(0, 180, t.size)).astype(np.int16)
    last = audio_debug.last_dir(tmp_path)
    last.mkdir(parents=True, exist_ok=True)
    with wave.open(str(last / "raw_trimmed.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(sig.tobytes())
    return rate, seconds


def test_a_music_short_transcript_is_redecoded_steered(monkeypatch, tmp_path):
    """The real failure: 'What does a control point mean?' came back as the
    four words 'This is the show.' - whisper transcribed the music's vocals.
    The clip is music-like and the transcript is too short for the speech,
    so one steered re-decode must recover the words."""
    from whisper_flow import app as app_module
    from whisper_flow import audio_debug

    rate, seconds = _write_music_capture(tmp_path)
    attempts = []

    class Service:
        def transcribe_audio(self, path, **kw):
            attempts.append((path, kw))
            if len(attempts) == 1:
                return "This is the show."
            return "What does a control point mean?"

    config = Mock()
    config.config_dir = tmp_path
    config.smart_voice_amplification = True
    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.transcription_service = Service()
    flow.config = config

    source = tmp_path / "x.wav"
    result = flow._transcribe_allowing_for_a_whisper(str(source))
    assert result == "What does a control point mean?"
    assert len(attempts) == 2
    _, kwargs = attempts[1]
    assert kwargs.get("temperature") == 0.6
    assert "music" in (kwargs.get("prompt") or "")
    # The steered audio is written beside the source and cleaned up.
    assert not (tmp_path / "x.wav.music.wav").exists()
    # And the rescue copy is kept for the audio-debug folder.
    assert (audio_debug.last_dir(tmp_path) / "music_steer.wav").exists()


def test_speech_paced_transcripts_never_pay_for_a_redecode(tmp_path):
    """A normal-length transcript is speech whatever the bed sounds like."""
    from whisper_flow import app as app_module

    _write_music_capture(tmp_path)
    attempts = []

    class Service:
        def transcribe_audio(self, path, **kw):
            attempts.append(path)
            return "I'm not going to go to the beach."

    config = Mock()
    config.config_dir = tmp_path
    config.smart_voice_amplification = True
    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.transcription_service = Service()
    flow.config = config

    result = flow._transcribe_allowing_for_a_whisper(str(tmp_path / "x.wav"))
    assert result == "I'm not going to go to the beach."
    assert len(attempts) == 1


def test_no_redecode_for_a_short_transcript_without_a_music_bed(tmp_path):
    """'This is the show.' in a quiet room is a short phrase, not music:
    the bursty level profile must not trigger the steered pass."""
    import wave

    import numpy as np

    from whisper_flow import app as app_module
    from whisper_flow import audio_debug

    rate = 16000
    seconds = 2.6
    rng = np.random.default_rng(13)
    out = np.zeros(int(rate * seconds), dtype=np.int16)
    t = np.arange(int(rate * seconds)) / rate
    for start, length in ((0.1, 0.5), (0.9, 0.6), (1.7, 0.5)):
        span = (t >= start) & (t < start + length)
        formants = sum(np.sin(2 * np.pi * hz * t[span])
                       for hz in (140, 420, 900, 1800)) / 4
        out[span] = (formants * np.hanning(span.sum()) * 3000).astype(np.int16)
    out += rng.normal(0, 60, out.size).astype(np.int16)
    last = audio_debug.last_dir(tmp_path)
    last.mkdir(parents=True, exist_ok=True)
    with wave.open(str(last / "raw_trimmed.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(out.tobytes())

    attempts = []

    class Service:
        def transcribe_audio(self, path, **kw):
            attempts.append(path)
            return "This is the show."

    config = Mock()
    config.config_dir = tmp_path
    config.smart_voice_amplification = True
    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.transcription_service = Service()
    flow.config = config

    result = flow._transcribe_allowing_for_a_whisper(str(tmp_path / "x.wav"))
    assert result == "This is the show."
    assert len(attempts) == 1


def test_a_weaker_steered_result_does_not_replace_the_first(tmp_path):
    """The steered pass only wins when it reads more speech-paced; otherwise
    the first transcript stands rather than being swapped for a guess."""
    from whisper_flow import app as app_module

    rate, seconds = _write_music_capture(tmp_path)
    attempts = []

    class Service:
        def transcribe_audio(self, path, **kw):
            attempts.append(path)
            if len(attempts) == 1:
                return "This is the show."
            return "Yes."

    config = Mock()
    config.config_dir = tmp_path
    config.smart_voice_amplification = True
    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.transcription_service = Service()
    flow.config = config

    result = flow._transcribe_allowing_for_a_whisper(str(tmp_path / "x.wav"))
    assert result == "This is the show."
    assert len(attempts) == 2        # it did try, and it chose to keep


def test_the_music_steer_skips_when_smart_voice_is_off(tmp_path):
    """The re-decode runs on rescue-processed audio, which is smart voice."""
    from whisper_flow import app as app_module

    _write_music_capture(tmp_path)
    attempts = []

    class Service:
        def transcribe_audio(self, path, **kw):
            attempts.append(path)
            return "This is the show."

    config = Mock()
    config.config_dir = tmp_path
    config.smart_voice_amplification = False
    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.transcription_service = Service()
    flow.config = config

    result = flow._transcribe_allowing_for_a_whisper(str(tmp_path / "x.wav"))
    assert result == "This is the show."
    assert len(attempts) == 1


def test_cold_open_silence_is_skipped_before_boosting_the_raw(tmp_path):
    """A cold-opened stream starts with ~0.4s of exact digital silence. The
    raw boost retry must cut that silence before amplifying: boosting it
    alongside the voice only raises the noise floor, and the whisper-lift
    path the silence previously forced amplifies a music bed into the
    transcript instead of the words."""
    import wave

    import numpy as np

    from whisper_flow import app as app_module
    from whisper_flow import audio_debug

    # A capture that opens with digital silence, then holds a quiet voice.
    rng = np.random.default_rng(23)
    n = int(16000 * 2.5)
    audio = np.zeros(n)
    audio[int(0.4 * 16000):] = rng.normal(0, 150, n - int(0.4 * 16000))
    t = np.arange(int(1.2 * 16000)) / 16000
    start = int(0.8 * 16000)
    audio[start:start + t.size] += np.sin(2 * np.pi * 220 * t) * 600
    samples = np.clip(audio, -32768, 32767).astype(np.int16)

    last = audio_debug.last_dir(tmp_path)
    last.mkdir(parents=True, exist_ok=True)
    for name in ("raw_untrimmed.wav", "raw_trimmed.wav"):
        with wave.open(str(last / name), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(samples.tobytes())

    boosted = tmp_path / "audio_file.wav"
    with wave.open(str(boosted), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(samples.tobytes())

    seen = {}

    class Service:
        def transcribe_audio(self, path, **kw):
            with wave.open(path, "rb") as f:
                got = np.frombuffer(f.readframes(f.getnframes()),
                                    dtype=np.int16)
            seen["data"] = got
            return "recovered words"

    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.config = Mock(config_dir=tmp_path, smart_voice_amplification=True)
    flow.transcription_service = Service()

    assert flow._retry_boost_raw(str(boosted)) == "recovered words"
    # The boosted wav is a plain peak-normalised copy of the capture with
    # the leading digital silence removed - not the smart-voice output.
    got = seen["data"]
    assert int(np.abs(got).max()) > 10000
    # 2.5s of capture minus the 0.4s of cold-open digital silence.
    assert got.size < int(16000 * 2.2)


def test_a_silent_recording_is_not_retried(tmp_path):
    """Amplifying a noise floor wastes a pass to produce louder noise."""
    import wave

    import numpy as np

    from whisper_flow import app as app_module

    dead = tmp_path / "dead.wav"
    with wave.open(str(dead), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

    attempts = []

    class Service:
        def transcribe_audio(self, path, **kw):
            attempts.append(path)
            return None

    flow = app_module.WhisperFlow.__new__(app_module.WhisperFlow)
    flow.transcription_service = Service()
    assert flow._transcribe_allowing_for_a_whisper(str(dead)) is None
    assert len(attempts) == 1
