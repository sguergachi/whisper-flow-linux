"""Blur must never be asked twice for one surface.

GTK 4.23.3 began creating ext_background_effect_surface_v1 for every
wl_surface the compositor advertises a manager for. The protocol allows one
effect object per surface, and a second get_background_effect answers
background_effect_exists - a fatal protocol error that took the whole
connection with it. The overlay died within a frame of enabling blur, so on
Wayland no HUD ever appeared at all: the crash erased the entire window, not
just the frosted backdrop.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

cairo = pytest.importorskip("cairo")
gi = pytest.importorskip("gi")

from whisper_flow import wayland_blur  # noqa: E402
import whisper_flow.hud_app as hud_app_module  # noqa: E402

# ------------------------------------------------------- who owns the effect
def test_gtk_below_4_23_3_leaves_the_effect_to_the_client():
    """Older runtimes never bind the manager: our requests are the only ones."""
    for version in ((4, 22, 4), (4, 23, 0), (4, 23, 2)):
        assert not wayland_blur.gtk_versions_claim_background_effects(*version), (
            version)


def test_gtk_from_4_23_3_claims_every_surface():
    """Exactly where the regression starts, snapshots numbered along."""
    for version in ((4, 23, 3), (4, 23, 90), (4, 24, 1), (4, 25, 0)):
        assert wayland_blur.gtk_versions_claim_background_effects(*version), (
            version)


def test_the_runtime_answer_matches_the_arithmetic():
    """Whatever GTK is actually loaded, the probe agrees with the threshold."""
    from gi.repository import Gtk

    actual = wayland_blur.gtk_claims_background_effects()
    expected = wayland_blur.gtk_versions_claim_background_effects(
        Gtk.get_major_version(), Gtk.get_minor_version(),
        Gtk.get_micro_version())
    assert actual is expected


# ------------------------------------------------- the overlay stands down
def _realize_only_window(monkeypatch):
    """A HudWindow able to run exactly as far as the blur gate."""
    monkeypatch.setattr(sys, "platform", "linux")
    win = hud_app_module.HudWindow.__new__(hud_app_module.HudWindow)
    # A pointer pre-set stops _setup_relative_pointer at its guard, so the
    # test never builds one.
    win._relative_pointer = Mock()
    blur = Mock()
    blur.gtk_claims_background_effects.return_value = True
    monkeypatch.setattr(hud_app_module, "_wayland_blur", blur)
    return win, blur


def test_realize_skips_the_blur_request_when_gtk_holds_the_effect(
        monkeypatch, capsys):
    win, blur = _realize_only_window(monkeypatch)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    ask_for_blur = Mock()
    monkeypatch.setattr(hud_app_module, "enable_blur_rows", ask_for_blur)

    win._on_realize()

    ask_for_blur.assert_not_called()
    out = capsys.readouterr().out
    assert "blur request skipped" in out


def test_realize_without_the_claim_goes_on_to_ask(monkeypatch):
    """GTK 4.22-era stacks keep the hand-marshalled route.

    The GTK bits are stubbed so the dance reaches the blur request: a bare
    __new__ window fails at get_surface first, which would prove nothing.
    """
    win, blur = _realize_only_window(monkeypatch)
    blur.gtk_claims_background_effects.return_value = False
    # enable_blur_rows is a module-level alias captured from
    # _wayland_blur at import, so it is the name the realise dance calls.
    ask_for_blur = Mock()
    ask_for_blur.return_value = Mock()
    monkeypatch.setattr(hud_app_module, "enable_blur_rows", ask_for_blur)
    fake_lib = Mock()
    fake_lib.gdk_wayland_display_get_wl_display.return_value = 1234
    fake_lib.gdk_wayland_surface_get_wl_surface.return_value = 5678
    monkeypatch.setattr(hud_app_module.ctypes, "CDLL", lambda *a, **k: fake_lib)
    monkeypatch.setattr(hud_app_module.Gdk.Display, "get_default", Mock)
    monkeypatch.setattr(hud_app_module, "_gobject_pointer", lambda o: 1)
    win.get_surface = Mock()
    win.stop_button = False
    win.processing = False

    win._on_realize()

    ask_for_blur.assert_called_once()


def test_realize_still_honours_the_env_off_switch(monkeypatch, capsys):
    """The env switch predates GTK's claim and stays first in line."""
    win, blur = _realize_only_window(monkeypatch)
    monkeypatch.setenv("WHISPER_FLOW_HUD_NO_BLUR", "1")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    win._on_realize()

    blur.gtk_claims_background_effects.assert_not_called()
    assert "blur disabled by env" in capsys.readouterr().out


# ------------------------------------------------- the settings stand down
def test_settings_window_skips_the_blur_request_when_gtk_holds_the_effect(
        monkeypatch):
    from whisper_flow import settings_gtk

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setattr(settings_gtk.wayland_blur,
                        "gtk_claims_background_effects", lambda: True)
    ask_for_blur = Mock()
    monkeypatch.setattr(settings_gtk.wayland_blur, "enable_window_blur",
                        ask_for_blur)
    win = settings_gtk.SettingsWindow.__new__(settings_gtk.SettingsWindow)
    lines = []
    monkeypatch.setattr(settings_gtk, "log", lines.append)

    win._on_realize()

    ask_for_blur.assert_not_called()
    assert any("blur request skipped" in line for line in lines)
