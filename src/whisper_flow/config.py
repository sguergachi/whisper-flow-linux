"""Configuration management for whisper-flow using Pydantic Settings."""

import os
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"


def default_config_dir() -> Path:
    """Where settings live, per platform convention.

    Windows puts per-user application data under LOCALAPPDATA; a dotfile
    directory in the profile root is a Unix habit and looks misplaced there.
    """
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(base) / "whisper-flow"
    return Path.home() / ".config" / "whisper-flow"


def default_hotkeys() -> dict[str, str]:
    """The same physical keys on every platform.

    Super+Alt, which is Win+Alt on a Windows keyboard. Windows briefly had
    Ctrl+Alt instead, on the grounds that releasing Win alone opens the Start
    menu - but muscle memory does not change at the OS boundary, and one
    binding to remember beats two. Alt is held with it, so the Start menu
    does not open on release.
    """
    return {
        "transcribe": "super+alt",
        "auto_transcribe": "ctrl+alt+space",
        "command": "cmd+shift+alt",
    }


_HOTKEYS = default_hotkeys()

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_env_file() -> str | os.PathLike[str]:
    """Resolve a stable .env location.

    Prefer ~/.config/whisper-flow/.env, fall back to project root,
    then CWD-relative ".env".
    """
    config_env = default_config_dir() / ".env"
    if config_env.exists():
        return config_env
    try:
        current_path = Path(__file__).resolve().parent
        max_levels = 8
        level = 0
        while level < max_levels:
            candidate = current_path / ".env"
            if candidate.exists():
                return candidate
            current_path = current_path.parent
            level += 1
    except Exception:
        pass
    return ".env"


class Config(BaseSettings):
    """Configuration manager for whisper-flow using Pydantic Settings."""

    model_config = SettingsConfigDict(
        env_prefix="WHISPER_FLOW_",
        env_file=_resolve_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # File paths
    config_dir: Path = Field(
        default_factory=default_config_dir,
        description="Configuration directory path",
    )

    # API configuration
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )

    # Audio configuration
    vad_mode: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Voice Activity Detection mode (0-3)",
        env="VAD_MODE",
    )
    mic_device_index: int | None = Field(
        default=None,
        description="Microphone device index",
        env="MIC_DEVICE_INDEX",
    )
    frame_ms: int = Field(
        default=30,
        gt=0,
        description="Audio frame duration in milliseconds",
    )
    silence_timeout: float = Field(
        default=1.5,
        gt=0.0,
        description="Silence timeout in seconds",
        env="SILENCE_TIMEOUT",
    )
    sample_rate: int = Field(
        default=16000,
        gt=0,
        description="Audio sample rate in Hz",
        env="SAMPLE_RATE",
    )
    speedup_audio: float = Field(
        default=1,
        description="Audio speed multiplier (1.0 = normal speed, 1.5 = 1.5x speed, etc.)",
        env="WHISPER_FLOW_SPEEDUP_AUDIO",
    )

    # AI model configuration
    transcription_model: str = Field(
        default="gpt-4o-mini-transcribe",
        description="OpenAI model for transcription",
        env="WHISPER_FLOW_TRANSCRIPTION_MODEL",
    )
    completion_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for completion",
        env="WHISPER_FLOW_COMPLETION_MODEL",
    )
    temperature: float = Field(
        default=0.4,
        description="Temperature for completion",
        ge=0.0,
        le=2.0,
        env="WHISPER_FLOW_TEMPERATURE",
    )

    # UI configuration
    notification_timeout: int = Field(
        default=3000,
        description="Notification timeout in milliseconds",
        ge=1000,
        le=10000,
        env="WHISPER_FLOW_NOTIFICATION_TIMEOUT",
    )

    # Daemon Configuration
    daemon_enabled: bool = Field(
        default=True,
        description="Enable daemon mode with system tray",
        env="WHISPER_FLOW_DAEMON_ENABLED",
    )
    auto_stop_silence_duration: float = Field(
        default=2.0,
        description="Seconds of silence before auto-stopping recording",
        ge=0.5,
        le=10.0,
        env="WHISPER_FLOW_AUTO_STOP_SILENCE",
    )
    pystray_backend: str = Field(
        default="gtk",
        description="Pystray backend for system tray (gtk, appindicator, xorg)",
        env="PYSTRAY_BACKEND",
    )

    # Stability and timeout configuration
    max_recording_duration: float = Field(
        default=300.0,  # 5 minutes
        description="Maximum recording duration in seconds before forced stop",
        ge=60.0,
        le=1800.0,  # 30 minutes max
        env="WHISPER_FLOW_MAX_RECORDING_DURATION",
    )
    processing_lock_timeout: float = Field(
        default=5.0,
        description="Timeout for acquiring processing lock in seconds",
        ge=1.0,
        le=30.0,
        env="WHISPER_FLOW_PROCESSING_LOCK_TIMEOUT",
    )
    watchdog_interval: float = Field(
        default=2.0,
        description="Watchdog check interval in seconds",
        ge=0.5,
        le=10.0,
        env="WHISPER_FLOW_WATCHDOG_INTERVAL",
    )
    queue_request_timeout: float = Field(
        default=30.0,
        description="Maximum age of queued requests in seconds before dropping",
        ge=10.0,
        le=300.0,
        env="WHISPER_FLOW_QUEUE_REQUEST_TIMEOUT",
    )

    notifications_enabled: bool = Field(
        default=True,
        description="Show desktop notifications at all",
        env="WHISPER_FLOW_NOTIFICATIONS_ENABLED",
    )
    notification_min_interval: float = Field(
        default=5.0,
        description="Seconds before the same notification may repeat",
        ge=0.0,
        le=3600.0,
        env="WHISPER_FLOW_NOTIFICATION_MIN_INTERVAL",
    )

    # Live (streaming) transcription
    live_transcription: bool = Field(
        default=True,
        description="Type text as you speak instead of all at once on release",
        env="WHISPER_FLOW_LIVE_TRANSCRIPTION",
    )
    live_interval: float = Field(
        default=0.9,
        description="Seconds between live transcription passes",
        ge=0.3,
        le=5.0,
        env="WHISPER_FLOW_LIVE_INTERVAL",
    )

    # Managed local speech engine
    manage_local_server: bool = Field(
        default=True,
        description="Download and run a local whisper.cpp server automatically",
        env="WHISPER_FLOW_MANAGE_LOCAL_SERVER",
    )
    model_name: str = Field(
        # What ships with the build: the one model that keeps up with speech
        # on every machine, from a two-core laptop upwards. The setup window
        # offers better where the hardware allows it.
        default="ggml-base.en",
        description="whisper.cpp model to download and run",
        env="WHISPER_FLOW_MODEL_NAME",
    )
    local_server_port: int = Field(
        default=8082,
        description="Port for the managed local server",
        ge=1024,
        le=65535,
        env="WHISPER_FLOW_LOCAL_SERVER_PORT",
    )

    # Local whisper.cpp configuration
    local_whisper_url: str | None = Field(
        default=None,
        description="URL for local whisper.cpp server (e.g. http://localhost:8082)",
        env="WHISPER_FLOW_LOCAL_WHISPER_URL",
    )

    # Hotkey Configuration
    hotkey_transcribe: str = Field(
        default_factory=lambda: _HOTKEYS["transcribe"],
        description="Hotkey for push-to-talk transcription",
        env="WHISPER_FLOW_HOTKEY_TRANSCRIBE",
    )
    hotkey_auto_transcribe: str = Field(
        default_factory=lambda: _HOTKEYS["auto_transcribe"],
        description="Hotkey for auto-stop transcription",
        env="WHISPER_FLOW_HOTKEY_AUTO_TRANSCRIBE",
    )
    hotkey_command: str = Field(
        default_factory=lambda: _HOTKEYS["command"],
        description="Hotkey for push-to-talk command mode",
        env="WHISPER_FLOW_HOTKEY_COMMAND",
    )

    # Logging configuration
    logging_enabled: bool = Field(
        default=False,  # Disable debug logging now that hotkeys work
        description="Enable debug logging and print statements",
        env="WHISPER_FLOW_LOGGING_ENABLED",
    )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def get_openai_api_key(cls, v):
        """Get OpenAI API key from multiple possible environment variables."""
        if v is not None:
            return v
        # Check standard OPENAI_API_KEY first, then our prefixed version
        return os.getenv("OPENAI_API_KEY") or os.getenv("WHISPER_FLOW_OPENAI_API_KEY")

    @field_validator("config_dir", mode="before")
    @classmethod
    def expand_config_dir(cls, v):
        """Expand config directory path and ensure it exists."""
        if isinstance(v, str):
            path = Path(os.path.expanduser(v))
        else:
            path = v
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_config_files(self):
        """Ensure configuration files exist with default content."""
        # No longer need to create prompt configuration files
        # The system now uses a single template approach
