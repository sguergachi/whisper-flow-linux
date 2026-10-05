"""Linux typing path: ydotool budget and clipboard fallback."""

from unittest.mock import Mock

from whisper_flow.config import Config
from whisper_flow.system import SystemManager


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
    monkeypatch.setattr(manager, "_ydotool_type", lambda text: True)
    clipboard = Mock(return_value=True)
    monkeypatch.setattr(manager, "copy_to_clipboard", clipboard)

    assert manager.type_text("short") is True
    clipboard.assert_not_called()


def test_wayland_paste_waits_until_saved_window_is_really_active(
        tmp_path, monkeypatch):
    monkeypatch.setattr("whisper_flow.system.IS_WINDOWS", False)
    manager = SystemManager(Config(config_dir=tmp_path))
    manager._saved_window = "target"
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)
    monkeypatch.setattr(manager, "_kdotool_available", lambda: True)
    active = iter(("hud", "hud", "target"))
    monkeypatch.setattr(manager, "_get_active_window", lambda: next(active))
    activate = Mock(return_value=True)
    monkeypatch.setattr(manager, "_activate_window", activate)
    copied = Mock(return_value=True)
    pasted = Mock(return_value=True)
    typed = Mock(return_value=True)
    monkeypatch.setattr(manager, "copy_to_clipboard", copied)
    monkeypatch.setattr(manager, "_send_paste_keystroke", pasted)
    monkeypatch.setattr(manager, "_ydotool_type", typed)

    assert manager.paste_text("arrives safely") is True
    activate.assert_called_once_with("target")
    copied.assert_called_once_with("arrives safely")
    pasted.assert_called_once_with()
    typed.assert_not_called()


def test_wayland_paste_preserves_text_if_focus_never_reaches_target(
        tmp_path, monkeypatch):
    monkeypatch.setattr("whisper_flow.system.IS_WINDOWS", False)
    manager = SystemManager(Config(config_dir=tmp_path))
    manager._saved_window = "target"
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)
    monkeypatch.setattr(manager, "_kdotool_available", lambda: True)
    monkeypatch.setattr(
        manager, "_activate_window_and_wait", lambda _target: False)
    typed = Mock(return_value=True)
    copied = Mock(return_value=True)
    monkeypatch.setattr(manager, "_ydotool_type", typed)
    monkeypatch.setattr(manager, "copy_to_clipboard", copied)

    assert manager.paste_text("do not lose me") is False
    typed.assert_not_called()
    copied.assert_called_once_with("do not lose me")


def test_wayland_paste_falls_back_to_direct_type_if_clipboard_paste_fails(
        tmp_path, monkeypatch):
    monkeypatch.setattr("whisper_flow.system.IS_WINDOWS", False)
    manager = SystemManager(Config(config_dir=tmp_path))
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)
    monkeypatch.setattr(manager, "_kdotool_available", lambda: False)
    monkeypatch.setattr(manager, "copy_to_clipboard", lambda _text: True)
    monkeypatch.setattr(manager, "_send_paste_keystroke", lambda: False)
    wtype = Mock(return_value=True)
    ydotool = Mock(return_value=True)
    monkeypatch.setattr(manager, "_wtype_type", wtype)
    monkeypatch.setattr(manager, "_ydotool_type", ydotool)

    assert manager.paste_text("fallback") is True
    wtype.assert_called_once_with("fallback")
    ydotool.assert_not_called()
