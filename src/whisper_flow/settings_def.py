"""The settings schema: which knobs exist, and which .env key feeds each one.

Kept free of tkinter so the rules can be tested without a display. The
critical property this module owns is the mapping from a Config attribute to
the environment variable that actually sets it - the fields do not all use
the WHISPER_FLOW_ prefix (VAD_MODE, MIC_DEVICE_INDEX, SILENCE_TIMEOUT and
SAMPLE_RATE predate it), and writing the wrong name would save a setting
that silently never applies.

Clearing a field resets it: an empty value removes the key from the file, so
the default comes back. That is also the only safe way to unset keys that
must parse as numbers - an empty MIC_DEVICE_INDEX= line would fail to parse
and take the whole config down with it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    key: str            # Config attribute
    env: str            # .env variable that feeds it
    label: str
    section: str
    kind: str = "str"   # str | password | bool | int | float | choice
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple = ()
    help: str = ""


SECTIONS = ("Speech", "Hotkeys", "Dictation", "General")

FIELDS = (
    # Speech: the local engine, then the hosted fallback
    Field("manage_local_server", "WHISPER_FLOW_MANAGE_LOCAL_SERVER",
          "Run the speech engine locally", "Speech", "bool",
          help="Download and supervise a whisper.cpp server on this machine"),
    Field("local_server_port", "WHISPER_FLOW_LOCAL_SERVER_PORT",
          "Local server port", "Speech", "int", 1024, 65535),
    Field("fast_encoder", "WHISPER_FLOW_FAST_ENCODER",
          "Match encoder window to clip length", "Speech", "bool",
          help="Up to 1.9x quicker, and it does change some words"),
    Field("local_whisper_url", "WHISPER_FLOW_LOCAL_WHISPER_URL",
          "Whisper server URL", "Speech",
          help="Point at an existing server instead; empty uses the managed one"),
    Field("openai_api_key", "WHISPER_FLOW_OPENAI_API_KEY",
          "OpenAI API key", "Speech", "password",
          help="Used when no local server is configured"),
    Field("transcription_model", "WHISPER_FLOW_TRANSCRIPTION_MODEL",
          "OpenAI transcription model", "Speech"),
    Field("completion_model", "WHISPER_FLOW_COMPLETION_MODEL",
          "OpenAI completion model", "Speech"),
    Field("temperature", "WHISPER_FLOW_TEMPERATURE",
          "Completion temperature", "Speech", "float", 0.0, 2.0),
    # Hotkeys
    Field("hotkey_transcribe", "WHISPER_FLOW_HOTKEY_TRANSCRIBE",
          "Transcribe (push-to-talk)", "Hotkeys"),
    Field("hotkey_auto_transcribe", "WHISPER_FLOW_HOTKEY_AUTO_TRANSCRIBE",
          "Auto-transcribe (single press)", "Hotkeys"),
    Field("hotkey_command", "WHISPER_FLOW_HOTKEY_COMMAND",
          "Command (push-to-talk)", "Hotkeys"),
    # Dictation
    Field("live_transcription", "WHISPER_FLOW_LIVE_TRANSCRIPTION",
          "Type while you speak", "Dictation", "bool"),
    Field("live_interval", "WHISPER_FLOW_LIVE_INTERVAL",
          "Live pass interval (s)", "Dictation", "float", 0.3, 5.0),
    Field("vad_mode", "VAD_MODE", "Voice detection aggressiveness",
          "Dictation", "choice", choices=("0", "1", "2", "3")),
    Field("mic_device_index", "MIC_DEVICE_INDEX", "Microphone",
          "Dictation", "choice"),   # choices filled in from the device list
    Field("sample_rate", "SAMPLE_RATE", "Sample rate (Hz)", "Dictation",
          "choice", choices=("8000", "16000", "32000", "48000")),
    Field("frame_ms", "WHISPER_FLOW_FRAME_MS", "Audio frame (ms)", "Dictation",
          "choice", choices=("10", "20", "30")),
    Field("silence_timeout", "SILENCE_TIMEOUT", "Silence timeout (s)",
          "Dictation", "float", 0.1, 30.0),
    Field("auto_stop_silence_duration", "WHISPER_FLOW_AUTO_STOP_SILENCE",
          "Auto-stop after silence (s)", "Dictation", "float", 0.5, 10.0),
    Field("speedup_audio", "WHISPER_FLOW_SPEEDUP_AUDIO",
          "Audio speed multiplier", "Dictation", "float", 0.5, 3.0,
          help="1 leaves audio untouched"),
    Field("trim_silence", "WHISPER_FLOW_TRIM_SILENCE",
          "Trim silence before transcribing", "Dictation", "bool",
          help="Much quicker on long recordings with pauses in them"),
    Field("max_recording_duration", "WHISPER_FLOW_MAX_RECORDING_DURATION",
          "Maximum recording (s)", "Dictation", "float", 60.0, 1800.0),
    # General
    Field("notifications_enabled", "WHISPER_FLOW_NOTIFICATIONS_ENABLED",
          "Desktop notifications", "General", "bool"),
    Field("notification_min_interval", "WHISPER_FLOW_NOTIFICATION_MIN_INTERVAL",
          "Notification repeat delay (s)", "General", "float", 0.0, 3600.0),
    Field("notification_timeout", "WHISPER_FLOW_NOTIFICATION_TIMEOUT",
          "Notification timeout (ms)", "General", "int", 1000, 10000),
    Field("logging_enabled", "WHISPER_FLOW_LOGGING_ENABLED",
          "Debug logging", "General", "bool"),
    Field("daemon_enabled", "WHISPER_FLOW_DAEMON_ENABLED",
          "Daemon mode", "General", "bool"),
    Field("pystray_backend", "PYSTRAY_BACKEND", "Tray backend", "General",
          "choice", choices=("gtk", "appindicator", "xorg")),
    Field("processing_lock_timeout", "WHISPER_FLOW_PROCESSING_LOCK_TIMEOUT",
          "Processing lock timeout (s)", "General", "float", 1.0, 30.0),
    Field("watchdog_interval", "WHISPER_FLOW_WATCHDOG_INTERVAL",
          "Watchdog interval (s)", "General", "float", 0.5, 10.0),
    Field("queue_request_timeout", "WHISPER_FLOW_QUEUE_REQUEST_TIMEOUT",
          "Queued request expiry (s)", "General", "float", 10.0, 300.0),
)


def serialize(value) -> str:
    """A Python value as its .env representation."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def updates_from(values: dict, current: dict) -> dict[str, str | None]:
    """The .env writes needed to take `current` to `values`.

    Only changed keys are written. A field cleared to empty removes its key
    from the file, resetting it to the default - never an empty value, which
    for numeric keys would be a parse error at the next launch.
    """
    updates: dict[str, str | None] = {}
    for field in FIELDS:
        if field.key not in values:
            continue
        raw, old = values[field.key], current.get(field.key)
        if field.kind == "bool":
            if bool(raw) != bool(old):
                updates[field.env] = serialize(bool(raw))
            continue
        raw = str(raw).strip()
        if raw == "":
            if old is not None and str(old) != "":
                updates[field.env] = None
            continue
        if field.kind in ("int", "float"):
            try:
                if old is not None and float(raw) == float(old):
                    continue
            except (TypeError, ValueError):
                pass
            number = float(raw)
            updates[field.env] = serialize(
                number if field.kind == "float" else int(number))
        elif old is None or raw != str(old):
            updates[field.env] = raw
    return updates


def validate_hotkey(value: str) -> str | None:
    """Why a hotkey combination is not usable, or None if it is."""
    for part in value.split("+"):
        if not part.strip():
            return f"'{value}' has an empty key name"
    return None


def validate(values: dict) -> str | None:
    """The first thing wrong with these values, or None if they can be saved."""
    for field in FIELDS:
        if field.key not in values or field.kind == "bool":
            continue
        raw = str(values[field.key]).strip()
        if raw == "":
            continue                # cleared: resets to the default
        if field.kind in ("int", "float"):
            try:
                number = float(raw)
            except ValueError:
                return f"{field.label}: '{raw}' is not a number"
            if field.minimum is not None and number < field.minimum:
                return f"{field.label}: must be at least {field.minimum:g}"
            if field.maximum is not None and number > field.maximum:
                return f"{field.label}: must be at most {field.maximum:g}"
        if field.key.startswith("hotkey_"):
            problem = validate_hotkey(raw)
            if problem:
                return f"{field.label}: {problem}"
    return None
