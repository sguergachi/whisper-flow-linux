"""Tests for the Windows typing, clipboard and notification helpers.

Imported directly rather than through the package so they run on Linux: the
module only touches ctypes.windll inside functions, so everything below can
be exercised anywhere.
"""

import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(scope="module")
def system_win():
    """Load the module with the Win32 loader stubbed, so this runs anywhere."""
    import ctypes
    from unittest.mock import Mock

    original = getattr(ctypes, "WinDLL", None)
    ctypes.WinDLL = Mock()              # resolved at import on Windows only
    try:
        return importlib.import_module("whisper_flow.system_win")
    finally:
        if original is None:
            del ctypes.WinDLL
        else:
            ctypes.WinDLL = original


# ------------------------------------------------------------ powershell quoting
def test_an_apostrophe_cannot_end_the_powershell_string(system_win):
    """'can't open device' used to terminate the literal and run as code."""
    quoted = system_win._ps_literal("can't open device")
    assert quoted == "can''t open device"


def test_quotes_are_doubled_not_dropped(system_win):
    assert system_win._ps_literal("'; Remove-Item C:\\ ;'") == (
        "''; Remove-Item C:\\ ;''")


def test_newlines_are_flattened(system_win):
    """A newline would end the command line mid-script."""
    assert "\n" not in system_win._ps_literal("first\nsecond\r\nthird")
    assert system_win._ps_literal("first\nsecond") == "first second"


def test_quoting_survives_non_string_input(system_win):
    assert system_win._ps_literal(42) == "42"


def test_a_quoted_message_still_reads_as_itself(system_win):
    """Escaping must not mangle ordinary text."""
    assert system_win._ps_literal("Recording failed (transcribe)") == (
        "Recording failed (transcribe)")


# ------------------------------------------------------------------- clipboard
def test_the_clip_exe_fallback_sends_a_byte_order_mark(system_win, monkeypatch):
    """Without a BOM clip.exe reads the bytes as the console code page."""
    sent = {}

    def fake_run(cmd, input=None, **kwargs):
        sent["cmd"], sent["input"] = cmd, input
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(system_win.subprocess, "run", fake_run)
    assert system_win._copy_via_clip_exe("héllo") is True
    assert sent["cmd"] == ["clip.exe"]
    assert sent["input"].startswith(b"\xff\xfe")          # UTF-16 LE BOM
    assert sent["input"].decode("utf-16") == "héllo"


def test_the_fallback_reports_failure_rather_than_raising(system_win, monkeypatch):
    monkeypatch.setattr(system_win.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no clip")))
    assert system_win._copy_via_clip_exe("x") is False


def test_utf16_round_trips_through_the_fallback(system_win, monkeypatch):
    """Failure reports carry arrows and emoji; they must survive intact."""
    sent = {}
    monkeypatch.setattr(
        system_win.subprocess, "run",
        lambda cmd, input=None, **k: sent.update(input=input) or
        type("P", (), {"returncode": 0})())
    text = "❌ Recording failed → check the microphone"
    system_win._copy_via_clip_exe(text)
    assert sent["input"].decode("utf-16") == text
