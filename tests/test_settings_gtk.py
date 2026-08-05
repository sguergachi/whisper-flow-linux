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
if scenario == "mic_race":
    # A configured device, so the placeholder dropdown has two entries and
    # there is something to change the selection *to* before the real list
    # lands. Without one the placeholder holds only "Default" and the window
    # this tests cannot be entered.
    os.environ["MIC_DEVICE_INDEX"] = "0"    # the name settings_def writes

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import GLib, Gtk

from whisper_flow import settings_def, settings_gtk

import threading
_gate = threading.Event()
if scenario != "mic_race":
    _gate.set()

def _devices():
    # Held shut for mic_race, so "before the list arrives" is a state the
    # test controls rather than a race it hopes to win. This is what the
    # real enumeration does anyway: it blocks, for as long as PortAudio
    # takes to walk four host APIs.
    _gate.wait(10)
    return [(0, "Built-in mic")]

settings_gtk._list_input_devices = _devices
Gtk.init()
w = settings_gtk.SettingsWindow()
w.config.config_dir = config_dir


def pump(until, seconds=10.0):
    \"\"\"Drive the main loop, as the real window has one driving it.

    The microphone list is enumerated on a thread and applied from an idle
    callback - opening PortAudio is far too slow to do before the window can
    be shown. Nothing here runs a main loop, so without this the dropdown
    still holds only the placeholder.
    \"\"\"
    import time
    context = GLib.MainContext.default()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        context.iteration(False)
        if until():
            return True
        time.sleep(0.005)
    return False


def mic_row():
    return w._rows["mic_device_index"]


def mics_listed():
    # By name, not by count. The placeholder already holds two entries when
    # a device is configured - "Default" and the bare index - so counting
    # them says the list has arrived before it has.
    model = mic_row().get_model()
    return any("Built-in mic" in (model.get_string(i) or "")
               for i in range(model.get_n_items()))

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
    assert pump(mics_listed), "the microphone list never arrived"
    mic = mic_row()
    mic.set_selected(1)
    w._on_save()
    assert (config_dir + "/.env") and True
elif scenario == "prewarm_waits":
    # Built, but nobody has asked for it: it must not appear on its own, and
    # it must not sit there polling rows nobody can see.
    assert not w.get_visible(), "a window built in advance showed itself"
    w._banner.set_revealed(True)
    assert w._refresh_dirty() is True        # the timer stays alive
    assert w._banner.get_revealed(), "an unseen window was diffed anyway"
elif scenario == "prewarm_shows":
    assert not w.get_visible()
    w.show_for_click()
    assert pump(lambda: w.get_visible()), "the window never came up"
elif scenario == "prewarm_rereads":
    # The .env is written after this window was built, which is the ordinary
    # case for one built at login: the daemon downloads a model or writes a
    # setting hours before anyone opens this.
    assert w._rows["local_server_port"].get_value() == 8082
    with open(config_dir + "/.env", "w") as fh:
        fh.write("WHISPER_FLOW_LOCAL_SERVER_PORT=8099\n")
    w.show_for_click()
    assert pump(lambda: w._rows["local_server_port"].get_value() == 8099), (
        f"the window showed the config as it was when it was built "
        f"({w._rows['local_server_port'].get_value()}), not as it is now")
elif scenario == "mic_race":
    # Change the selection before the enumeration lands. That window is
    # exactly as long as PortAudio takes to walk four host APIs, which is
    # the whole reason the enumeration was moved off this thread.
    mic = mic_row()
    assert not mics_listed(), "the list was not deferred at all"
    assert mic.get_model().get_n_items() == 2, "no configured device to leave"
    mic.set_selected(0)                     # back to Default, unsaved
    assert mic.get_model().get_string(mic.get_selected()) == "Default"

    _gate.set()                             # now let the enumeration finish
    assert pump(mics_listed), "the microphone list never arrived"
    after = mic.get_model().get_string(mic.get_selected())
    assert after == "Default", (
        f"the pending selection was reverted to {after!r} when the device "
        f"list arrived; the choice was taken from the saved baseline rather "
        f"than from the row")
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


def test_a_choice_made_before_the_mic_list_arrives_survives_it(tmp_path):
    """Enumerating PortAudio is slow enough for someone to get there first.

    The dropdown is seeded with a placeholder and filled from a thread. The
    swap restored the selection from `_current` - the saved baseline the
    dirty poll diffs against, not live widget state - so a choice made while
    the enumeration was still running was silently undone.
    """
    result = _run(tmp_path, "mic_race")
    assert "OK" in result.stdout, result.stderr


def test_a_window_built_in_advance_stays_off_screen(tmp_path):
    """It is built at login, and nobody asked for a window at login."""
    result = _run(tmp_path, "prewarm_waits")
    assert "OK" in result.stdout, result.stderr


def test_the_click_puts_the_prewarmed_window_up(tmp_path):
    result = _run(tmp_path, "prewarm_shows")
    assert "OK" in result.stdout, result.stderr


def test_a_prewarmed_window_re_reads_the_config_when_it_is_shown(tmp_path):
    """Built at login, opened hours later, and the config moved in between.

    Whatever the daemon wrote meanwhile - a downloaded model, a saved
    setting - has to be on screen when the window appears. Showing what was
    true at login would show stale settings and then save them back over the
    real ones.
    """
    result = _run(tmp_path, "prewarm_rereads")
    assert "OK" in result.stdout, result.stderr


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
    # Wherever the page is rebuilt, not wherever a download happens to end.
    # Installing the GPU engine rebuilds it too, so the two callers share one
    # method and this follows it there.
    body = source.split("def _rebuild_speech_page(", 1)[1].split("\n    def ", 1)[0]
    assert "_apply_values(" in body, (
        "the rows are rebuilt here and must be filled in again, or every "
        "Speech setting reads as empty afterwards")
    assert body.index("_values()") < body.index("_build_speech_page"), (
        "the values have to be captured before the rows are replaced")
    assert "_apply_values" in source and "def _apply_values" in source, (
        "_apply_values must exist to restore them")
    for caller in ("_download_done", "_engine_download_done"):
        called = source.split(f"def {caller}(", 1)[1].split("\n    def ", 1)[0]
        assert "_rebuild_speech_page()" in called, (
            f"{caller} replaces rows without going through the rebuild, so "
            f"the values it captured are never put back")
