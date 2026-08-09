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
    "beam_size": ("5", 5),
    "best_of": ("4", 4),
    "no_speech_thold": ("0.5", 0.5),
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
        # Stay inside any min/max the field declares (noise floor is 1.2–5).
        number = 42.0
        if field.maximum is not None:
            number = min(number, float(field.maximum))
        if field.minimum is not None:
            number = max(number, float(field.minimum))
        if field.kind == "int":
            number = int(number)
        raw, expected = str(number), float(number) if field.kind == "float" else number
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


def test_validate_rejects_a_two_modifier_single_press_hotkey():
    """ctrl+shift is held by dozens of shortcuts; it must never be a
    single-press binding (the usual cause is the layout-switch shortcut
    eating the Space out of Ctrl+Shift+Space during capture)."""
    problem = settings_def.validate({"hotkey_auto_transcribe": "ctrl+shift"})
    assert "beyond the modifiers" in problem
    problem = settings_def.validate({"hotkey_command": "alt+shift"})
    assert "beyond the modifiers" in problem
    problem = settings_def.validate({"hotkey_command": "ctrl"})
    assert "beyond the modifiers" in problem


def test_validate_allows_three_modifier_single_press_hotkeys():
    """ctrl+super+alt is the shipped command hotkey: three modifiers are
    distinctive enough to be a real shortcut, and it must stay saveable."""
    assert settings_def.validate({"hotkey_command": "ctrl+super+alt"}) is None


def test_validate_allows_the_modifiers_only_push_to_talk_default():
    """super+alt push-to-talk is the shipped default; it must stay saveable."""
    assert settings_def.validate({"hotkey_transcribe": "super+alt"}) is None
    assert settings_def.validate(
        {"hotkey_auto_transcribe": "ctrl+shift+space"}) is None


def test_format_hotkey_orders_modifiers_then_keys():
    assert settings_def.format_hotkey({"k", "alt", "ctrl"}) == "ctrl+alt+k"
    assert settings_def.format_hotkey(["super", "alt"]) == "super+alt"
    assert settings_def.format_hotkey([]) == ""


def test_format_hotkey_normalises_cmd_and_control_aliases():
    assert settings_def.format_hotkey({"cmd", "alt"}) == "super+alt"
    assert settings_def.format_hotkey({"control", "shift", "a"}) == "ctrl+shift+a"


# ------------------------------------------------------------------- layout
def test_every_field_names_a_group():
    """A field with no group renders in an untitled block above the rest.

    That is how the pages looked before they were grouped: one undivided
    column per section, ten rows deep on Dictation. A new field must not be
    able to reintroduce it by omission.
    """
    ungrouped = [f.key for f in settings_def.FIELDS if not f.group]
    assert not ungrouped, f"no group set for: {', '.join(ungrouped)}"


def test_each_group_keeps_a_readable_number_of_ordinary_rows():
    """The reason for grouping in the first place, held to a number.

    Expert rows sit behind an expander and do not count against this: they
    are one row until someone opens them.
    """
    from collections import Counter

    counts = Counter((f.section, f.group)
                     for f in settings_def.FIELDS if not f.advanced)
    too_long = {key: n for key, n in counts.items() if n > 5}
    assert not too_long, f"groups needing a split: {too_long}"


def test_advanced_fields_sit_in_a_group_that_also_has_ordinary_ones():
    """Otherwise a group renders as a title over a lone Advanced expander."""
    from collections import defaultdict

    kinds = defaultdict(set)
    for field in settings_def.FIELDS:
        kinds[(field.section, field.group)].add(field.advanced)
    orphans = [key for key, seen in kinds.items() if seen == {True}]
    assert not orphans, f"only-advanced groups: {orphans}"


def test_group_help_refers_to_groups_that_exist():
    """A typo in a heading would silently drop its description."""
    real = {(f.section, f.group) for f in settings_def.FIELDS}
    assert set(settings_def.GROUP_HELP) <= real


def test_hotkey_fields_explain_hold_vs_tap():
    """Settings used to list Auto-transcribe and Command with the same blank help.

    The labels and help must say how each mode works: hold-to-speak vs
    tap-then-silence, and that the second auto key is not a third behaviour.
    """
    by_key = {f.key: f for f in settings_def.FIELDS}
    ptt = by_key["hotkey_transcribe"]
    auto = by_key["hotkey_auto_transcribe"]
    spare = by_key["hotkey_command"]

    assert "hold" in ptt.help.lower()
    assert "release" in ptt.help.lower() or "paste" in ptt.help.lower()

    assert "tap" in auto.help.lower()
    assert "silence" in auto.help.lower()

    assert "same" in spare.help.lower() or "spare" in spare.help.lower()
    assert "2nd" in spare.label.lower() or "second" in spare.label.lower()

    hotkeys_help = settings_def.GROUP_HELP[("Hotkeys", "Shortcuts")].lower()
    assert "hold" in hotkeys_help and "tap" in hotkeys_help

