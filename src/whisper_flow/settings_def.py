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
    # Which titled group inside the page this belongs to. Layout lives here
    # rather than in the window because this module is the part that can be
    # tested without a display, and because a page that lists every field of
    # a section in one undivided column - which is what there was - stops
    # being readable at about six rows. Dictation had ten.
    group: str = ""
    # Real but rarely touched: watchdog intervals, frame sizes, lock
    # timeouts. Shown behind one expander so the page opens on the settings
    # people actually came for, without hiding anything from those who did.
    advanced: bool = False


SECTIONS = ("Speech", "Hotkeys", "Dictation", "General")

# One line of orientation per group, where the title alone leaves a question.
GROUP_HELP = {
    ("Speech", "Engine"):
        "All on this machine.",
    ("Hotkeys", "Shortcuts"):
        "Two ways to dictate. Push to talk: hold, speak, release — words paste. "
        "Auto-transcribe: tap once, speak; recording ends after silence and pastes. "
        "The second auto key is the same flow (spare shortcut). "
        "Escape cancels. Click a row, then press the keys. Applies after Save.",
    ("Dictation", "Microphone"):
        "What is recorded.",
    ("Dictation", "While you speak"):
        "As you talk, or at the end.",
    ("Dictation", "Stopping"):
        "For Auto-transcribe only: how long quiet must last before it ends, "
        "and the hard cap so a silent room cannot hang forever.",
    ("General", "Notifications"): "",
    ("General", "Daemon"):
        "Owns hotkeys and tray.",
}

ADVANCED_HELP = "Rarely needed."

FIELDS = (
    # ---------------------------------------------------------------- Speech
    Field("manage_local_server", "WHISPER_FLOW_MANAGE_LOCAL_SERVER",
          "Run engine locally", "Speech", "bool",
          help="Runs whisper.cpp here",
          group="Engine"),
    Field("local_whisper_url", "WHISPER_FLOW_LOCAL_WHISPER_URL",
          "Server URL", "Speech",
          help="Empty uses the managed one",
          group="Engine"),
    Field("local_server_port", "WHISPER_FLOW_LOCAL_SERVER_PORT",
          "Port", "Speech", "int", 1024, 65535,
          group="Engine", advanced=True),
    Field("fast_encoder", "WHISPER_FLOW_FAST_ENCODER",
          "Fast encoder", "Speech", "bool",
          help="Up to 1.9x quicker, less exact",
          group="Engine", advanced=True),

    # --------------------------------------------------------------- Hotkeys
    # Three bindings, two behaviours. Push-to-talk is hold-to-speak.
    # Auto-transcribe and the spare "command" binding are both single-press
    # hands-free: tap → speak → silence ends → paste. "Command" used to call
    # a cloud model; that path is gone, so the second key is an extra auto
    # shortcut — the label says so rather than inventing a third mode.
    Field("hotkey_transcribe", "WHISPER_FLOW_HOTKEY_TRANSCRIBE",
          "Push to talk", "Hotkeys", group="Shortcuts",
          help="Hold while speaking, release to paste"),
    Field("hotkey_auto_transcribe", "WHISPER_FLOW_HOTKEY_AUTO_TRANSCRIBE",
          "Auto-transcribe", "Hotkeys", group="Shortcuts",
          help="Tap once, then speak — stops after silence and pastes"),
    Field("hotkey_command", "WHISPER_FLOW_HOTKEY_COMMAND",
          "Auto-transcribe (2nd)", "Hotkeys", group="Shortcuts",
          help="Same as Auto-transcribe — spare hands-free key"),

    # ------------------------------------------------------------- Dictation
    Field("mic_device_index", "MIC_DEVICE_INDEX", "Device",
          "Dictation", "choice",   # choices filled in from the device list
          group="Microphone"),
    Field("noise_filter", "WHISPER_FLOW_NOISE_FILTER",
          "Noise filter", "Dictation", "bool",
          help="High-pass, soft gate, peak normalise",
          group="Microphone"),
    Field("noise_floor", "WHISPER_FLOW_NOISE_FLOOR",
          "Noise floor", "Dictation", "float", 1.2, 5.0,
          help="How far above room tone counts as speech. Higher = more room ignored",
          group="Microphone"),
    Field("spectral_denoise", "WHISPER_FLOW_SPECTRAL_DENOISE",
          "Strong noise filter", "Dictation", "bool",
          help="Also subtracts steady mid-band noise (fans, AC)",
          group="Microphone"),
    Field("sample_rate", "SAMPLE_RATE", "Sample rate (Hz)", "Dictation",
          "choice", choices=("8000", "16000", "32000", "48000"),
          group="Microphone", advanced=True),
    Field("frame_ms", "WHISPER_FLOW_FRAME_MS", "Audio frame (ms)", "Dictation",
          "choice", choices=("10", "20", "30"),
          group="Microphone", advanced=True),
    Field("vad_mode", "VAD_MODE", "Voice detection",
          "Dictation", "choice", choices=("0", "1", "2", "3"),
          group="Microphone", advanced=True),

    Field("live_transcription", "WHISPER_FLOW_LIVE_TRANSCRIPTION",
          "Live typing", "Dictation", "bool",
          group="While you speak"),
    Field("live_interval", "WHISPER_FLOW_LIVE_INTERVAL",
          "Pass interval (s)", "Dictation", "float", 0.3, 5.0,
          group="While you speak"),
    Field("speedup_audio", "WHISPER_FLOW_SPEEDUP_AUDIO",
          "Speed multiplier", "Dictation", "float", 0.5, 3.0,
          help="1 is untouched",
          group="While you speak", advanced=True),
    Field("trim_silence", "WHISPER_FLOW_TRIM_SILENCE",
          "Trim silence", "Dictation", "bool",
          help="Quicker on long recordings",
          group="While you speak", advanced=True),

    Field("auto_stop_silence_duration", "WHISPER_FLOW_AUTO_STOP_SILENCE",
          "Silence before stop (s)", "Dictation", "float", 0.5, 10.0,
          help="Quiet time before Auto-transcribe ends and pastes",
          group="Stopping"),
    Field("silence_timeout", "SILENCE_TIMEOUT", "Silence timeout (s)",
          "Dictation", "float", 0.1, 30.0,
          help="Legacy; Auto-transcribe uses Silence before stop",
          group="Stopping", advanced=True),
    Field("max_recording_duration", "WHISPER_FLOW_MAX_RECORDING_DURATION",
          "Max recording (s)", "Dictation", "float", 60.0, 1800.0,
          help="Hard cap for any recording (Auto-transcribe included)",
          group="Stopping"),

    # --------------------------------------------------------------- General
    Field("notifications_enabled", "WHISPER_FLOW_NOTIFICATIONS_ENABLED",
          "Notifications", "General", "bool", group="Notifications"),
    Field("notification_min_interval", "WHISPER_FLOW_NOTIFICATION_MIN_INTERVAL",
          "Repeat delay (s)", "General", "float", 0.0, 3600.0,
          group="Notifications"),
    Field("notification_timeout", "WHISPER_FLOW_NOTIFICATION_TIMEOUT",
          "Duration (ms)", "General", "int", 1000, 10000,
          group="Notifications"),

    Field("daemon_enabled", "WHISPER_FLOW_DAEMON_ENABLED",
          "Daemon", "General", "bool", group="Daemon"),
    Field("logging_enabled", "WHISPER_FLOW_LOGGING_ENABLED",
          "Debug logging", "General", "bool",
          help="Feeds Copy log",
          group="Daemon"),
    Field("pystray_backend", "PYSTRAY_BACKEND", "Tray backend", "General",
          "choice", choices=("gtk", "appindicator", "xorg"),
          group="Daemon", advanced=True),
    Field("processing_lock_timeout", "WHISPER_FLOW_PROCESSING_LOCK_TIMEOUT",
          "Lock timeout (s)", "General", "float", 1.0, 30.0,
          group="Daemon", advanced=True),
    Field("watchdog_interval", "WHISPER_FLOW_WATCHDOG_INTERVAL",
          "Watchdog interval (s)", "General", "float", 0.5, 10.0,
          group="Daemon", advanced=True),
    Field("queue_request_timeout", "WHISPER_FLOW_QUEUE_REQUEST_TIMEOUT",
          "Queue expiry (s)", "General", "float", 10.0, 300.0,
          group="Daemon", advanced=True),
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


# Canonical order for modifiers in a saved hotkey string. Parsing treats the
# combination as a set, but the UI and the .env file stay readable if every
# capture writes them in the same order.
# Super before alt so a Linux default chord still reads "super+alt", not
# "alt+super", when re-recorded from the settings window.
_HOTKEY_MOD_ORDER = ("ctrl", "super", "alt", "shift")


def format_hotkey(parts) -> str:
    """Join key names into the config form: ``ctrl+alt+k``, ``super+alt``.

    Modifiers first in a fixed order, then the rest alphabetically. Empty
    input becomes an empty string (cleared shortcut).
    """
    names = {str(p).strip().lower() for p in parts if str(p).strip()}
    # Config and both platform listeners treat these as the same key.
    if "cmd" in names or "win" in names or "meta" in names:
        names.discard("cmd")
        names.discard("win")
        names.discard("meta")
        names.add("super")
    if "control" in names:
        names.discard("control")
        names.add("ctrl")
    mods = [m for m in _HOTKEY_MOD_ORDER if m in names]
    rest = sorted(names - set(_HOTKEY_MOD_ORDER))
    return "+".join(mods + rest)


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
