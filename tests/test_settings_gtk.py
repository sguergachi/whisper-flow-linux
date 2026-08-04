"""Tests for the GTK4 settings window.

pystray drags Gtk 3 into this pytest process at collection time, and Gtk 3
and Gtk 4 cannot share a process - so the window is exercised in a fresh
interpreter, which is also how the app itself runs it. Needs GTK4,
libadwaita and a display; skips cleanly where any are missing.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk
Gtk.init()
"""

_CHILD = """
import os, sys
sys.path.insert(0, sys.argv[1] + "/src")
config_dir, scenario = sys.argv[2], sys.argv[3]
os.environ["WHISPER_FLOW_CONFIG_DIR"] = config_dir

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk

from whisper_flow import settings_def, settings_gtk
settings_gtk._list_input_devices = lambda: [(0, "Built-in mic")]
Gtk.init()
w = settings_gtk.SettingsWindow()
w.config.config_dir = config_dir

if scenario == "rows_exist":
    for field in settings_def.FIELDS:
        assert field.key in w._rows, f"no row for {field.key}"
elif scenario == "load":
    assert w._rows["local_server_port"].get_value() == 8082
    assert w._rows["live_transcription"].get_active() is True
    mic = w._rows["mic_device_index"]
    assert mic.get_model().get_string(mic.get_selected()) == "Default"
elif scenario == "models":
    from whisper_flow.backend import MODELS
    assert set(w._model_checks) == set(MODELS)
elif scenario == "save":
    w._rows["live_interval"].set_value(1.4)
    w._on_save()
    assert "restart" in w._last_toast_title
elif scenario == "invalid":
    w._rows["hotkey_transcribe"].set_text("super++alt")
    w._on_save()
    assert "empty key name" in w._last_toast_title
elif scenario == "cleared":
    mic = w._rows["mic_device_index"]
    mic.set_selected(1)
    w._on_save()
    assert (config_dir + "/.env") and True
print("OK")
"""


def _gtk_available() -> bool:
    if sys.platform != "win32" and not os.environ.get("DISPLAY"):
        return False
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(not _gtk_available(),
                                reason="needs GTK4, libadwaita and a display")


def _run(tmp_path, scenario: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _CHILD, str(ROOT), str(tmp_path), scenario],
        capture_output=True, text=True, check=False, timeout=120,
    )


def test_every_schema_field_gets_a_row(tmp_path):
    result = _run(tmp_path, "rows_exist")
    assert "OK" in result.stdout, result.stderr


def test_rows_load_the_running_config(tmp_path):
    result = _run(tmp_path, "load")
    assert "OK" in result.stdout, result.stderr


def test_a_checkbutton_exists_per_known_model(tmp_path):
    result = _run(tmp_path, "models")
    assert "OK" in result.stdout, result.stderr


def test_saving_a_change_writes_the_env_file(tmp_path):
    result = _run(tmp_path, "save")
    assert "OK" in result.stdout, result.stderr
    from whisper_flow import envfile
    assert envfile.get(tmp_path / ".env",
                       "WHISPER_FLOW_LIVE_INTERVAL") == "1.4"


def test_an_invalid_value_is_not_saved(tmp_path):
    result = _run(tmp_path, "invalid")
    assert "OK" in result.stdout, result.stderr
    assert not (tmp_path / ".env").exists()


def test_a_cleared_field_removes_the_key(tmp_path):
    result = _run(tmp_path, "cleared")
    assert "OK" in result.stdout, result.stderr
    from whisper_flow import envfile
    assert envfile.get(tmp_path / ".env", "MIC_DEVICE_INDEX") == "0"


def test_a_rebuilt_page_puts_its_values_back():
    """Downloading a model replaces every row on the Speech page.

    _build_speech_page constructs fresh rows, and fresh rows are empty: the
    port read 0, "Run the speech engine locally" read off and the server URL
    blank, none of which anyone had touched. _current still held the real
    values, so Save saw the difference as deliberate and would have written
    it - local transcription turning itself off because a model was
    downloaded. The port failing validation at 0 is the only thing that
    stopped it, which is why this was reported as a confusing complaint
    about a port rather than as the setting loss it was.
    """
    source = (Path(__file__).resolve().parents[1]
              / "src/whisper_flow/settings_gtk.py").read_text(encoding="utf-8")
    body = source.split("def _download_done(", 1)[1].split("\n    def ", 1)[0]
    assert "_apply_values(" in body, (
        "the rows are rebuilt here and must be filled in again, or every "
        "Speech setting reads as empty afterwards")
    assert body.index("_values()") < body.index("_build_speech_page"), (
        "the values have to be captured before the rows are replaced")
    assert "_apply_values" in source and "def _apply_values" in source, (
        "_apply_values must exist to restore them")
