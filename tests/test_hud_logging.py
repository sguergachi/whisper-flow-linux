"""Logging must work in a windowed build with no console streams."""

import sys
from unittest.mock import Mock

import pytest

pytest.importorskip("gi")
pytest.importorskip("cairo")

from whisper_flow import hud_app


def test_windowed_hud_starts_without_console_streams(tmp_path, monkeypatch):
    monkeypatch.setattr(hud_app.os.path, "expanduser", lambda _: str(tmp_path))
    monkeypatch.setattr(hud_app.Gtk, "init_check", lambda: False)
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", None)
        context.setattr(sys, "stderr", None)
        assert hud_app.main() == 1  # no display, rather than a logging exception
        sys.stderr.write("stderr still works\n")
        sys.stderr.flush()
        sys.stdout._file.close()
        sys.stderr._file.close()
    output = (tmp_path / ".config/whisper-flow/hud.log").read_text()
    assert "[HUD] starting" in output
    assert "no display" in output
    assert "stderr still works" in output


def test_broken_log_file_cannot_kill_the_hud(tmp_path, monkeypatch):
    monkeypatch.setattr(hud_app.os.path, "expanduser", lambda _: str(tmp_path))
    broken_file = Mock()
    broken_file.write.side_effect = OSError("disk full")
    broken_file.flush.side_effect = OSError("disk full")
    monkeypatch.setattr("builtins.open", lambda *a, **kw: broken_file)
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", None)
        context.setattr(sys, "stderr", None)
        hud_app._tee_hud_output()
        print("still alive", flush=True)
