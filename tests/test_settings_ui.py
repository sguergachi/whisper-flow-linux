"""Tests for the settings window.

These build a real Tk window, so they need a display and are skipped without
one - the headless logic (schema, env file, restart) is covered elsewhere.
"""

import sys

import pytest

tk = pytest.importorskip("tkinter")

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" and not __import__("os").environ.get("DISPLAY"),
    reason="needs a display",
)


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_FLOW_CONFIG_DIR", str(tmp_path))
    from whisper_flow import settings_ui

    monkeypatch.setattr(settings_ui, "_list_input_devices",
                        lambda: [(0, "Built-in mic")])
    try:
        w = settings_ui.SettingsWindow()
    except tk.TclError as e:
        pytest.skip(f"no usable display: {e}")
    w.config.config_dir = tmp_path
    yield w
    try:
        w.root.destroy()
    except tk.TclError:
        pass


def test_every_schema_field_gets_a_widget(window):
    from whisper_flow import settings_def

    for field in settings_def.FIELDS:
        assert field.key in window._vars, f"no widget for {field.key}"


def test_widgets_load_the_running_config(window):
    assert window._vars["local_server_port"].get() == "8082"
    assert window._vars["live_transcription"].get() is True
    assert window._vars["mic_device_index"].get() == "Default"


def test_a_model_row_exists_per_known_model(window):
    from whisper_flow.backend import MODELS

    def labels(widget):
        found = [widget.cget("text")] if isinstance(widget, tk.Label) else []
        for child in widget.winfo_children():
            found += labels(child)
        return found

    texts = [
        text
        for row in window._models_frame.winfo_children()
        for text in labels(row)
    ]
    for name in MODELS:
        assert any(text.startswith(name.replace("ggml-", ""))
                   for text in texts), f"no row for {name}"


def test_saving_a_change_writes_the_env_file(window, tmp_path):
    from whisper_flow import envfile

    window._vars["live_interval"].set("1.4")
    window._on_save()
    window.root.update()

    assert envfile.get(tmp_path / ".env",
                       "WHISPER_FLOW_LIVE_INTERVAL") == "1.4"
    assert "restart" in window.status.cget("text")


def test_an_invalid_value_is_not_saved(window, tmp_path):
    window._vars["live_interval"].set("99")
    window._on_save()
    window.root.update()

    assert not (tmp_path / ".env").exists()
    assert "at most" in window.status.cget("text")


def test_a_cleared_field_removes_the_key(window, tmp_path):
    from whisper_flow import envfile

    window._vars["mic_device_index"].set("0: Built-in mic")
    window._on_save()
    assert envfile.get(tmp_path / ".env", "MIC_DEVICE_INDEX") == "0"

    window._vars["mic_device_index"].set("Default")
    window._on_save()
    assert envfile.get(tmp_path / ".env", "MIC_DEVICE_INDEX") is None
