"""The .env writer: surgical edits, never a rewrite of what it did not touch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_flow import envfile


def test_set_values_replaces_in_place_and_keeps_everything_else(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# my settings\n"
        "WHISPER_FLOW_MODEL_NAME=ggml-base.en\n"
        "\n"
        "WHISPER_FLOW_OPENAI_API_KEY=sk-old  # rotated\n",
        encoding="utf-8",
    )

    envfile.set_values(env, {"WHISPER_FLOW_MODEL_NAME": "ggml-small.en"})

    assert env.read_text(encoding="utf-8") == (
        "# my settings\n"
        "WHISPER_FLOW_MODEL_NAME=ggml-small.en\n"
        "\n"
        "WHISPER_FLOW_OPENAI_API_KEY=sk-old  # rotated\n"
    )


def test_set_values_appends_new_keys_and_creates_the_file(tmp_path):
    env = tmp_path / "nested" / ".env"

    envfile.set_values(env, {"A": "1"})
    envfile.set_values(env, {"B": "2"})

    assert env.read_text(encoding="utf-8") == "A=1\nB=2\n"


def test_set_values_removes_a_key_entirely(tmp_path):
    """Unset means gone, not empty: MIC_DEVICE_INDEX= would not parse."""
    env = tmp_path / ".env"
    env.write_text("MIC_DEVICE_INDEX=3\nSAMPLE_RATE=16000\n", encoding="utf-8")

    envfile.set_values(env, {"MIC_DEVICE_INDEX": None})

    assert env.read_text(encoding="utf-8") == "SAMPLE_RATE=16000\n"
    assert envfile.get(env, "MIC_DEVICE_INDEX") is None
    assert envfile.get(env, "SAMPLE_RATE") == "16000"


def test_commented_lines_are_not_touched(tmp_path):
    env = tmp_path / ".env"
    env.write_text("#MIC_DEVICE_INDEX=9\nMIC_DEVICE_INDEX=3\n", encoding="utf-8")

    envfile.set_values(env, {"MIC_DEVICE_INDEX": None})

    assert env.read_text(encoding="utf-8") == "#MIC_DEVICE_INDEX=9\n"
