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
elif scenario == "daemon_moved_on":
    # The daemon restarts and comes up on a different model. The page exists
    # to say which model is in use, so it has to stop saying the old one.
    w.present()
    assert pump(lambda: w.get_visible())
    before = w._current_model
    w._rows["hotkey_transcribe"].set_text("ctrl+alt+k")   # an unsaved edit
    w._stack.set_visible_child_name("hotkeys")
    settings_gtk.restart.daemon_pid = lambda: 4242
    w.backend.working_model = lambda: "ggml-small.en-q8_0"
    assert before != "ggml-small.en-q8_0", "nothing would change"

    assert w._watch_the_daemon() is True
    assert w._current_model == "ggml-small.en-q8_0"
    assert w._model_checks["ggml-small.en-q8_0"].get_active(), (
        "the radio stayed on the model the daemon has stopped using")
    # Without dragging the user off the page they were on, and without
    # throwing away what they had typed on it.
    assert w._stack.get_visible_child_name() == "hotkeys"
    assert w._rows["hotkey_transcribe"].get_text() == "ctrl+alt+k"
    assert "4242" in w._status_row.get_subtitle()
elif scenario == "unsaved_model_choice":
    # Same restart, but this time the user has picked a model and not saved
    # it. That choice is theirs and outranks what the daemon is doing.
    #
    # Two models on disk, because an uninstalled one refuses to be chosen -
    # the radio snaps straight back off - so there would be no choice to
    # keep. The larger is what working_model() settles on.
    import pathlib
    models = pathlib.Path(config_dir) / "models"
    models.mkdir(parents=True, exist_ok=True)
    for name in ("ggml-large-v3-turbo", "ggml-base.en-q8_0"):
        (models / f"{name}.bin").write_bytes(b"not a model")
    w.present()
    assert pump(lambda: w.get_visible())
    w._current_model = w.backend.working_model()
    w._rebuild_speech_page()

    # Whichever of the two the backend settles on, the user picks the other.
    # Not a fixed name: which one wins is working_model()'s business - the
    # configured one where it is present, the largest otherwise - and this is
    # not the test that pins that down. It needs a model that can be chosen
    # (installed) and that is not already in use (a real choice).
    installed = ("ggml-large-v3-turbo", "ggml-base.en-q8_0")
    assert w._current_model in installed, w._current_model
    other = next(name for name in installed if name != w._current_model)

    w._model_checks[other].set_active(True)
    assert w._selected_model() == other, "the choice never took"

    w.backend.working_model = lambda: "ggml-small.en-q8_0"
    w._watch_the_daemon()
    assert w._current_model == "ggml-small.en-q8_0"
    assert w._model_checks[other].get_active(), (
        "a daemon restart took the model choice away from the user")
elif scenario == "nothing_to_see":
    # Nobody is looking, so nothing is read and nothing is rebuilt.
    called = []
    w.backend.working_model = lambda: called.append("read") or "x"
    assert not w.get_visible()
    assert w._watch_the_daemon() is True
    assert called == [], "an unseen window went to the disk anyway"
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
    #
    # Where the .env lives is not WHISPER_FLOW_CONFIG_DIR's business - that
    # names the config_dir field, while the file is found under the real
    # LOCALAPPDATA or ~/.config - so the lookup is pointed at the temporary
    # one here. What is under test is that showing the window re-reads
    # whatever that lookup answers, not the answer it gave at import.
    from whisper_flow import config as config_module
    assert w._rows["local_server_port"].get_value() == 8082
    with open(config_dir + "/.env", "w") as fh:
        # Escaped: this source is a string in the test that runs it, so a bare
        # newline here ends that string rather than reaching the child.
        fh.write("WHISPER_FLOW_LOCAL_SERVER_PORT=8099\\n")
    config_module._resolve_env_file = lambda: config_dir + "/.env"
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
elif scenario == "badge":
    # The recommended badge is a real widget beside the name, not markup in
    # the title. Markup did reach that spot, and could not be a pill there:
    # it paints its background to the text's logical extents, with neither
    # padding nor a corner radius, so it read as clipped. Walk the rows and
    # find it.
    def walk(widget):
        yield widget
        child = widget.get_first_child()
        while child is not None:
            yield from walk(child)
            child = child.get_next_sibling()

    labels = [x for x in walk(w) if isinstance(x, Gtk.Label)]
    badges = [x for x in labels if x.get_label() == "recommended"]
    assert badges, "no recommended badge anywhere in the window"
    badge = badges[0]
    assert badge.has_css_class("model-pill"), (
        "the badge is not a pill; it cannot be padded or rounded without one")
    assert badge.has_css_class("pill-recommended")

    # Beside a name, not off in the suffix box: its nearest label sibling has
    # to be one of the model names.
    names = {n.replace("ggml-", "") for n in w._model_checks}
    sibling = badge.get_prev_sibling()
    assert isinstance(sibling, Gtk.Label) and sibling.get_label() in names, (
        f"the badge follows {sibling!r}, not a model name")

    # And no leftover markup in any row title.
    assert not any("<span" in (x.get_label() or "") for x in labels), (
        "a title is still carrying Pango markup for the badge")
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


def test_the_model_in_use_follows_a_daemon_restart(tmp_path):
    """The page answered "which model is in use" once, when it was built.

    Saving a different model and pressing Restart left the pill naming the
    old one against a daemon that had already moved on - the page
    contradicting the app it describes, on the one question it exists to
    settle. Now that the window outlives the click that opened it, that
    window is open across the whole restart and has to keep up.
    """
    result = _run(tmp_path, "daemon_moved_on")
    assert "OK" in result.stdout, result.stderr


def test_an_unsaved_model_choice_survives_a_daemon_restart(tmp_path):
    result = _run(tmp_path, "unsaved_model_choice")
    assert "OK" in result.stdout, result.stderr


def test_an_unseen_window_does_not_watch_the_daemon(tmp_path):
    """A window built at login waits for hours; it must wait cheaply."""
    result = _run(tmp_path, "nothing_to_see")
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


def test_the_recommended_badge_is_a_pill_beside_the_name(tmp_path):
    """It was markup for a while, which cannot be padded or rounded.

    Pango paints a span's background to the logical extents of the text and
    no further, so the badge sat with its word hard against its own edges
    however much the run was padded out - it read as clipped. A widget has
    real padding and a real corner radius; the cost is that the row has to
    own its title, because AdwActionRow has no slot beside the name.
    """
    assert _run(tmp_path, "badge").returncode == 0
