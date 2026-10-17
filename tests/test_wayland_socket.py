"""The HUD must address the session's Wayland socket, not wayland-0.

Sessions name their socket whatever they like - KWin hands out wayland-1 on
some setups - and a window pointed at a name that does not exist simply
never appears, with nothing in any log to say why. The runtime directory is
the registry, and an inherited WAYLAND_DISPLAY stands only while its own
socket actually exists there.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_flow import paths  # noqa: E402


def test_a_live_session_keeps_its_own_answer(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    assert paths.find_wayland_socket() is None


def test_kwin_sessions_on_wayland_1_are_found(monkeypatch, tmp_path):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "wayland-1").touch()
    (tmp_path / "wayland-1.lock").touch()

    assert paths.find_wayland_socket() == "wayland-1"


def test_the_first_sorted_socket_wins(monkeypatch, tmp_path):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "wayland-0").touch()
    (tmp_path / "wayland-1").touch()

    assert paths.find_wayland_socket() == "wayland-0"


def test_stale_name_files_do_not_count(monkeypatch, tmp_path):
    """A .lock file is not the socket; only the socket name itself is."""
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "wayland-1.lock").touch()

    assert paths.find_wayland_socket() is None


def test_no_runtime_dir_is_no_socket(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    assert paths.find_wayland_socket() is None


def test_a_dead_wayland_display_claim_is_replaced(monkeypatch, tmp_path):
    """Pinned-on-write env names have outlived their socket (systemd).

    A unit written under one session says wayland-0; KWin comes back as
    wayland-1. The claim must not read as healthy - nothing can connect
    to it - so the directory's real socket is the answer instead.
    """
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "wayland-1").touch()

    assert paths.find_wayland_socket() == "wayland-1"


def test_a_live_wayland_display_claim_is_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "wayland-1").touch()

    assert paths.find_wayland_socket() is None


def test_daemon_counts_a_dead_claim_as_no_display(monkeypatch, tmp_path):
    """_has_display must re-adopt when the inherited name never existed.

    A tray survives without a display, so a healthy-looking environment
    whose only display variable points at an absent socket reads exactly
    like the login-race the adoption path exists for. The correction has
    to happen here, not at window start: the settings window inherits the
    daemon's environment too.
    """
    import whisper_flow.daemon as daemon_module

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "wayland-1").touch()
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")

    assert daemon_module._wayland_claim_is_live() is False


def test_daemon_counts_a_live_claim_as_a_display(monkeypatch, tmp_path):
    import whisper_flow.daemon as daemon_module

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "wayland-1").touch()
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    assert daemon_module._wayland_claim_is_live() is True
