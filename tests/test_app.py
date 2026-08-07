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
    assert flow._transcribe_allowing_for_a_whisper("/tmp/x.wav") == "first time"
    assert len(attempts) == 1        # the retry costs a pass; do not spend it


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
