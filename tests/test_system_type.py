"""Linux typing path: ydotool budget and clipboard fallback."""

from unittest.mock import Mock

import pytest

from whisper_flow.config import Config
from whisper_flow.system import SystemManager

_REAL_LAYOUT_PROBE = SystemManager._us_layout_active


@pytest.fixture(autouse=True)
def known_us_layout(monkeypatch):
    monkeypatch.setattr(SystemManager, "_us_layout_active", staticmethod(lambda: True))


def test_ydotool_type_timeout_scales_with_text_length():
    # Short text keeps the historical 15s floor.
    assert SystemManager._ydotool_type_timeout("hi") == 15.0
    # ~500 chars used to blow a fixed 15s budget on KDE.
    long_timeout = SystemManager._ydotool_type_timeout("x" * 500)
    assert long_timeout >= 25.0
    assert long_timeout <= 120.0
    # Cap so a stuck helper cannot hang the recording thread forever.
    assert SystemManager._ydotool_type_timeout("x" * 100_000) == 120.0


def test_type_text_falls_back_to_clipboard_when_direct_type_fails(
        tmp_path, monkeypatch):
    """Live finalize used to drop the whole tail when ydotool timed out."""
    # This is the Linux path. On the Windows runner IS_WINDOWS is True at
    # import, so type_text would go to SendInput and never hit the
    # clipboard fallback this is pinning.
    monkeypatch.setattr("whisper_flow.system.IS_WINDOWS", False)
    manager = SystemManager(Config(config_dir=tmp_path))
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)
    monkeypatch.setattr(manager, "_ydotool_type", lambda text: False)
    monkeypatch.setattr(manager, "_wtype_type", lambda text: False)
    # The app's own injector is tried first; on a working host it would
    # really type, and this test is about what happens when nothing types.
    monkeypatch.setattr("whisper_flow.system.type_inject.type_text",
                        lambda text: False)

    copied = []
    monkeypatch.setattr(
        manager, "copy_to_clipboard",
        lambda text: copied.append(text) or True)
    monkeypatch.setattr(manager, "_send_paste_keystroke", lambda: True)

    assert manager.type_text("the whole closing transcript") is True
    assert copied == ["the whole closing transcript"]


def test_type_text_does_not_touch_clipboard_when_ydotool_works(
        tmp_path, monkeypatch):
    monkeypatch.setattr("whisper_flow.system.IS_WINDOWS", False)
    manager = SystemManager(Config(config_dir=tmp_path))
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)
    monkeypatch.setattr("whisper_flow.system.type_inject.type_text",
                        lambda text: False)
    monkeypatch.setattr(manager, "_ydotool_type", lambda text: True)
    clipboard = Mock(return_value=True)
    monkeypatch.setattr(manager, "copy_to_clipboard", clipboard)

    assert manager.type_text("short") is True
    clipboard.assert_not_called()


def test_the_apps_own_injector_types_without_ydotool(tmp_path, monkeypatch):
    """The device the app creates itself is preferred over the host daemon.

    ydotool can fail silently - KWin discarded every keystroke from its
    device here while ydotool still exited 0 - so the app's own injector
    runs first and a success must not reach for ydotool or the clipboard.
    """
    from whisper_flow import system as system_module

    monkeypatch.setattr(system_module, "IS_WINDOWS", False)
    manager = SystemManager(Config(config_dir=tmp_path))
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)

    typed = []
    monkeypatch.setattr(system_module.type_inject, "type_text",
                        lambda text: typed.append(text) or True)
    ydotool = Mock(return_value=True)
    monkeypatch.setattr(manager, "_ydotool_type", ydotool)
    monkeypatch.setattr(manager, "copy_to_clipboard", Mock(return_value=True))

    assert manager.type_text("dictated words") is True
    assert typed == ["dictated words"]
    assert not ydotool.called


def test_ydotool_cannot_claim_unicode_success(tmp_path, monkeypatch):
    from whisper_flow import system

    manager = SystemManager(Config(config_dir=tmp_path))
    run = Mock()
    monkeypatch.setattr(system.subprocess, "run", run)
    assert manager._ydotool_type("café — مرحبا") is False
    run.assert_not_called()


def test_unicode_falls_back_with_the_complete_sentence(tmp_path, monkeypatch):
    from whisper_flow import system

    monkeypatch.setattr(system, "IS_WINDOWS", False)
    manager = SystemManager(Config(config_dir=tmp_path))
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)
    monkeypatch.setattr(manager, "_wtype_type", lambda _: False)
    monkeypatch.setattr(system.type_inject, "HAVE_EVDEV", True)
    # The unsupported em dash must be detected before a device is opened.
    device = Mock(side_effect=AssertionError("should not open an input device"))
    monkeypatch.setattr(system.type_inject, "_device", device)
    clipboard = Mock(return_value=True)
    monkeypatch.setattr(manager, "copy_to_clipboard", clipboard)
    monkeypatch.setattr(manager, "_send_paste_keystroke", lambda: True)

    assert manager.type_text("a—b") is True
    clipboard.assert_called_once_with("a—b")
    device.assert_not_called()


def test_partial_typing_only_retries_the_remaining_suffix(tmp_path, monkeypatch):
    from whisper_flow import system

    manager = SystemManager(Config(config_dir=tmp_path))
    monkeypatch.setattr(system.type_inject, "type_text", Mock(
        side_effect=system.type_inject.PartialTypingError(3)))
    fallback = Mock(return_value=True)
    monkeypatch.setattr(manager, "_ydotool_type", fallback)
    assert manager._type_wayland("abcdef")
    fallback.assert_called_once_with("def")


def test_unrecoverable_partial_typing_is_reported(tmp_path, monkeypatch):
    import pytest
    from whisper_flow import system

    manager = SystemManager(Config(config_dir=tmp_path))
    monkeypatch.setattr(system.type_inject, "type_text", Mock(
        side_effect=system.type_inject.PartialTypingError(3)))
    monkeypatch.setattr(manager, "_ydotool_type", lambda _: False)
    monkeypatch.setattr(manager, "_wtype_type", lambda _: False)
    clipboard = Mock(return_value=True)
    monkeypatch.setattr(manager, "copy_to_clipboard", clipboard)
    monkeypatch.setattr(manager, "_send_paste_keystroke", lambda: False)
    with pytest.raises(system.type_inject.PartialTypingError):
        manager._type_wayland("abcdef")
    clipboard.assert_called_once_with("def")


@pytest.mark.parametrize("layout,expected", [
    ("([('us', '', 'English (US)')],)", True),
    ("([('fr', '', 'French')],)", False),
    ("([('us', 'dvorak', 'English (Dvorak)')],)", False),
    ("malformed response", False),
])
def test_scan_codes_require_a_confirmed_plain_us_layout(layout, expected, monkeypatch):
    from whisper_flow import system

    monkeypatch.setattr(system.shutil, "which", lambda _: "/usr/bin/gdbus")
    monkeypatch.setattr(system.subprocess, "run", Mock(side_effect=[
        Mock(returncode=0, stdout="(uint32 0,)"),
        Mock(returncode=0, stdout=layout),
    ]))
    assert _REAL_LAYOUT_PROBE() is expected


def test_unconfirmed_layout_skips_scan_code_injectors(tmp_path, monkeypatch):
    from whisper_flow import system

    manager = SystemManager(Config(config_dir=tmp_path))
    monkeypatch.setattr(manager, "_us_layout_active", lambda: False)
    own = Mock()
    ydotool = Mock()
    clipboard = Mock(return_value=True)
    monkeypatch.setattr(system.type_inject, "type_text", own)
    monkeypatch.setattr(manager, "_ydotool_type", ydotool)
    monkeypatch.setattr(manager, "_wtype_type", lambda _: False)
    monkeypatch.setattr(manager, "_wtype_paste", lambda: False)
    monkeypatch.setattr(manager, "copy_to_clipboard", clipboard)
    assert manager._type_wayland("azerty") is False
    clipboard.assert_called_once_with("azerty")
    own.assert_not_called()
    ydotool.assert_not_called()
