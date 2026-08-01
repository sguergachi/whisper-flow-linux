"""Tests for the first-run setup window.

These build a real Tk window, so they need a display and are skipped without
one. The download itself is stubbed: what matters here is that the window
reflects the machine correctly, that progress reaches the bar in the order
the worker emits it, and that a failure leaves a retry rather than a dead end.
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
    from whisper_flow import backend as backend_module
    from whisper_flow import setup_win

    monkeypatch.setattr(backend_module, "detect_accelerator", lambda: "cuda12")
    monkeypatch.setattr(setup_win, "detect_accelerator", lambda: "cuda12")

    try:
        w = setup_win.SetupWindow()
    except tk.TclError as e:
        pytest.skip(f"no usable display: {e}")
    w.config.config_dir = tmp_path
    w.backend.config = w.config
    yield w
    try:
        w.root.destroy()
    except tk.TclError:
        pass


def test_a_gpu_machine_is_offered_the_large_model(window):
    assert window.has_gpu
    assert window.model == "ggml-large-v3-turbo"
    assert "1624 MB" in window.action.cget("text")


def test_progress_shows_the_bar_and_tracks_the_fraction(window):
    assert not window.progress.winfo_ismapped()
    window._on_progress("model", 0.5)
    window.root.update()
    assert window.progress.winfo_ismapped()
    assert window.progress["value"] == pytest.approx(50.0)
    assert "50%" in window.status.cget("text")


def test_the_result_survives_a_late_progress_tick(window):
    """The bar must never talk over the outcome."""
    window._finished()
    window._on_progress("model", 0.9)
    window.root.update()
    assert "Ready" in window.status.cget("text")
    assert window.action.cget("text") == "Done"


def test_a_finished_download_records_the_model_choice(window, tmp_path):
    window._save_choice()
    env = (tmp_path / ".env").read_text()
    assert "WHISPER_FLOW_MODEL_NAME=ggml-large-v3-turbo" in env


def test_saving_the_choice_does_not_discard_other_settings(window, tmp_path):
    (tmp_path / ".env").write_text(
        "WHISPER_FLOW_MODEL_NAME=ggml-base.en\nWHISPER_FLOW_LOCAL_SERVER_PORT=9999\n")
    window._save_choice()
    env = (tmp_path / ".env").read_text()
    assert "WHISPER_FLOW_LOCAL_SERVER_PORT=9999" in env
    assert "ggml-base.en" not in env          # replaced, not duplicated
    assert env.count("WHISPER_FLOW_MODEL_NAME") == 1


def test_a_failure_offers_a_retry_rather_than_a_dead_end(window):
    window._working = True
    window.action.configure(state="disabled")
    window.dismiss.configure(state="disabled")
    window._failed()
    assert window.action.cget("text") == "Try again"
    assert str(window.action.cget("state")) == "normal"
    assert str(window.dismiss.cget("state")) == "normal"


def test_closing_mid_download_is_refused(window):
    """A half-finished 1.6GB download helps nobody."""
    window._working = True
    window._on_close()
    assert window.root.winfo_exists()
    assert not window.backend.setup_seen()


def test_dismissing_is_remembered_so_it_does_not_nag(window):
    window._on_close()
    assert window.backend.setup_seen()
