"""The settings schema, and its contract with Config.

The env names in the schema are the whole point: Config does not use the
WHISPER_FLOW_ prefix for every field (VAD_MODE, MIC_DEVICE_INDEX,
SILENCE_TIMEOUT, SAMPLE_RATE predate it), so a wrong name here saves a
setting that silently never applies. The first test feeds every single
schema key through a real Config to prove the mapping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from whisper_flow import settings_def
from whisper_flow.config import Config

# A distinctive, valid value per constrained field. Anything not listed gets
# a plain string, which every unconstrained field accepts.
_PROBE = {
    "vad_mode": ("3", 3),
    "mic_device_index": ("5", 5),
    "sample_rate": ("48000", 48000),
    "frame_ms": ("20", 20),
    "pystray_backend": ("xorg", "xorg"),
    "local_server_port": ("18081", 18081),
    "notification_timeout": ("5000", 5000),
    "live_interval": ("1.7", 1.7),
    "silence_timeout": ("2.5", 2.5),
    "auto_stop_silence_duration": ("3.5", 3.5),
    "speedup_audio": ("1.5", 1.5),
    "max_recording_duration": ("120.0", 120.0),
    "notification_min_interval": ("7.5", 7.5),
    "processing_lock_timeout": ("7.0", 7.0),
    "watchdog_interval": ("1.5", 1.5),
    "queue_request_timeout": ("60.0", 60.0),
}


@pytest.mark.parametrize("field", settings_def.FIELDS,
                         ids=lambda f: f.key)
def test_every_schema_env_name_actually_feeds_config(field, monkeypatch):
    for other in settings_def.FIELDS:
        monkeypatch.delenv(other.env, raising=False)
    if field.kind == "bool":
        raw, expected = "true", True
    elif field.key in _PROBE:
        raw, expected = _PROBE[field.key]
    elif field.kind in ("int", "float"):
        raw, expected = "42", 42.0
    else:
        raw, expected = "probe-value", "probe-value"
    monkeypatch.setenv(field.env, raw)

    config = Config(_env_file=None)

    assert getattr(config, field.key) == expected


# ------------------------------------------------------------------ updates
def test_updates_only_cover_what_changed():
    current = {"live_interval": "0.9", "logging_enabled": False}
    values = {"live_interval": "1.4", "logging_enabled": False}

    assert settings_def.updates_from(values, current) == {
        "WHISPER_FLOW_LIVE_INTERVAL": "1.4",
    }


def test_equivalent_numbers_are_not_a_change():
    """0.90 and 0.9 are the same setting; rewriting it would be noise."""
    assert settings_def.updates_from(
        {"live_interval": "0.90"}, {"live_interval": "0.9"}) == {}


def test_a_cleared_field_removes_the_key():
    """Empty means reset to default - the key goes, no empty value stays."""
    updates = settings_def.updates_from(
        {"mic_device_index": ""}, {"mic_device_index": "3"})
    assert updates == {"MIC_DEVICE_INDEX": None}


def test_bools_serialize_the_way_pydantic_reads_them():
    assert settings_def.updates_from(
        {"logging_enabled": True}, {"logging_enabled": False}) == {
        "WHISPER_FLOW_LOGGING_ENABLED": "true",
    }


# ---------------------------------------------------------------- validation
def test_validate_accepts_good_values():
    assert settings_def.validate({
        "live_interval": "1.2",
        "hotkey_transcribe": "super+alt",
        "mic_device_index": "",          # cleared: back to the default device
    }) is None


def test_validate_rejects_non_numbers_and_out_of_range():
    assert "not a number" in settings_def.validate({"live_interval": "fast"})
    assert "at most" in settings_def.validate({"live_interval": "99"})
    assert "at least" in settings_def.validate({"live_interval": "0.1"})


def test_validate_rejects_a_hotkey_with_an_empty_part():
    problem = settings_def.validate({"hotkey_transcribe": "super++alt"})
    assert "empty key name" in problem
