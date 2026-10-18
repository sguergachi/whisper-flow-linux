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


def test_wayland_paste_waits_until_saved_window_is_really_active(
        tmp_path, monkeypatch):
    monkeypatch.setattr("whisper_flow.system.IS_WINDOWS", False)
    manager = SystemManager(Config(config_dir=tmp_path))
    manager._saved_window = "target"
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)
    monkeypatch.setattr(manager, "_kdotool_available", lambda: True)
    monkeypatch.setattr("whisper_flow.system.type_inject.type_text", lambda text: False)
    active = iter(("hud", "hud", "target"))
    monkeypatch.setattr(manager, "_get_active_window", lambda: next(active))
    activate = Mock(return_value=True)
    monkeypatch.setattr(manager, "_activate_window", activate)
    copied = Mock(return_value=True)
    pasted = Mock(return_value=True)
    typed = Mock(return_value=True)
    monkeypatch.setattr(manager, "copy_to_clipboard", copied)
    monkeypatch.setattr(
        manager, "_xdotool_paste_if_target_is_x11", lambda _target: False)
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
    monkeypatch.setattr("whisper_flow.system.type_inject.type_text", lambda text: False)
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
    monkeypatch.setattr(
        manager, "_xdotool_paste_if_target_is_x11", lambda _target: False)
    monkeypatch.setattr(manager, "_send_paste_keystroke", lambda: False)
    wtype = Mock(return_value=True)
    ydotool = Mock(return_value=True)
    monkeypatch.setattr("whisper_flow.system.type_inject.type_text", lambda text: False)
    monkeypatch.setattr(manager, "_wtype_type", wtype)
    monkeypatch.setattr(manager, "_ydotool_type", ydotool)

    assert manager.paste_text("fallback") is True
    wtype.assert_called_once_with("fallback")
    ydotool.assert_not_called()



def test_modifier_only_alt_hotkey_detection():
    detect = SystemManager._hotkey_is_modifier_only_alt

    assert detect("super+alt") is True
    assert detect("left_super+right_alt") is True
    assert detect("win+option") is True
    assert detect("ctrl+super+alt") is True
    assert detect("super+alt+space") is False
    assert detect("super+shift") is False
    assert detect("right_ctrl") is False
    assert detect("") is False



def test_wayland_modifier_only_alt_repairs_focus_before_copy_and_paste(
        tmp_path, monkeypatch):
    monkeypatch.setattr("whisper_flow.system.IS_WINDOWS", False)
    config = Config(config_dir=tmp_path, hotkey_transcribe="super+alt")
    manager = SystemManager(config)
    manager._saved_window = "electron"
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)
    monkeypatch.setattr(manager, "_kdotool_available", lambda: True)
    monkeypatch.setattr(manager, "_activate_window_and_wait", lambda _target: True)
    monkeypatch.setattr("whisper_flow.system.type_inject.type_text", lambda text: False)

    events = []
    monkeypatch.setattr(
        manager, "_repair_modifier_only_alt_focus",
        lambda: events.append("escape") or True)
    monkeypatch.setattr(
        manager, "copy_to_clipboard",
        lambda _text: events.append("copy") or True)
    monkeypatch.setattr(
        manager, "_xdotool_paste_if_target_is_x11",
        lambda _target: events.append("paste") or True)

    assert manager.paste_text("hello") is True
    assert events == ["escape", "copy", "paste"]



def test_wayland_non_modifier_hotkey_does_not_send_escape(
        tmp_path, monkeypatch):
    monkeypatch.setattr("whisper_flow.system.IS_WINDOWS", False)
    config = Config(config_dir=tmp_path, hotkey_transcribe="super+alt+space")
    manager = SystemManager(config)
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)
    monkeypatch.setattr(manager, "_kdotool_available", lambda: False)
    ydotool_available = Mock(return_value=True)
    monkeypatch.setattr(manager, "_ydotool_available", ydotool_available)

    assert manager._repair_modifier_only_alt_focus() is False
    ydotool_available.assert_not_called()



def test_wayland_x11_target_prefers_xdotool_paste(
        tmp_path, monkeypatch):
    monkeypatch.setattr("whisper_flow.system.IS_WINDOWS", False)
    manager = SystemManager(Config(config_dir=tmp_path))
    manager._saved_window = "electron"
    monkeypatch.setattr(manager, "_is_wayland", lambda: True)
    monkeypatch.setattr(manager, "_kdotool_available", lambda: True)
    monkeypatch.setattr(manager, "_activate_window_and_wait", lambda _target: True)
    monkeypatch.setattr(manager, "_repair_modifier_only_alt_focus", lambda: True)
    monkeypatch.setattr("whisper_flow.system.type_inject.type_text", lambda text: False)
    monkeypatch.setattr(manager, "copy_to_clipboard", lambda _text: True)
    x11_paste = Mock(return_value=True)
    generic_paste = Mock(return_value=True)
    monkeypatch.setattr(manager, "_xdotool_paste_if_target_is_x11", x11_paste)
    monkeypatch.setattr(manager, "_send_paste_keystroke", generic_paste)

    assert manager.paste_text("hello") is True
    x11_paste.assert_called_once_with("electron")
    generic_paste.assert_not_called()



def test_alt_focus_repair_releases_every_modifier_before_escape(
        tmp_path, monkeypatch):
    manager = SystemManager(
        Config(config_dir=tmp_path, hotkey_transcribe="super+alt"))
    monkeypatch.setattr(manager, "_ydotool_available", lambda: True)
    monkeypatch.setattr("whisper_flow.system.time.sleep", lambda _delay: None)
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return Mock(returncode=0)

    monkeypatch.setattr("whisper_flow.system.subprocess.run", run)

    assert manager._repair_modifier_only_alt_focus() is True
    assert calls == [
        ["ydotool", "key", *[
            f"{code}:0" for code in SystemManager.MODIFIER_CODES]],
        ["ydotool", "key", "1:1", "1:0"],
    ]



def test_alt_focus_repair_falls_back_to_wayland_virtual_keyboard(
        tmp_path, monkeypatch):
    manager = SystemManager(
        Config(config_dir=tmp_path, hotkey_transcribe="super+alt"))
    monkeypatch.setattr(manager, "_ydotool_available", lambda: False)
    monkeypatch.setattr("whisper_flow.system.shutil.which", lambda tool: tool)
    monkeypatch.setattr("whisper_flow.system.time.sleep", lambda _delay: None)
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr("whisper_flow.system.subprocess.run", run)

    assert manager._repair_modifier_only_alt_focus() is True
    run.assert_called_once_with(
        ["wtype", "-k", "Escape"], check=False, capture_output=True)



def test_xdotool_paste_requires_active_x11_pid_to_match_target(
        tmp_path, monkeypatch):
    manager = SystemManager(Config(config_dir=tmp_path))
    monkeypatch.setattr(manager, "_kdotool_available", lambda: True)
    monkeypatch.setattr("whisper_flow.system.shutil.which", lambda tool: tool)
    calls = []
    answers = iter((
        Mock(returncode=0, stdout="123\n"),
        Mock(returncode=0, stdout="456\n"),
        Mock(returncode=0, stdout="999\n"),
    ))

    def run(command, **_kwargs):
        calls.append(command)
        return next(answers)

    monkeypatch.setattr("whisper_flow.system.subprocess.run", run)

    assert manager._xdotool_paste_if_target_is_x11("target") is False
    assert calls == [
        ["kdotool", "getwindowpid", "target"],
        ["xdotool", "getactivewindow"],
        ["xdotool", "getwindowpid", "456"],
    ]



def test_xdotool_paste_uses_xtest_only_after_pid_match(
        tmp_path, monkeypatch):
    manager = SystemManager(Config(config_dir=tmp_path))
    monkeypatch.setattr(manager, "_kdotool_available", lambda: True)
    monkeypatch.setattr("whisper_flow.system.shutil.which", lambda tool: tool)
    calls = []
    answers = iter((
        Mock(returncode=0, stdout="123\n"),
        Mock(returncode=0, stdout="456\n"),
        Mock(returncode=0, stdout="123\n"),
        Mock(returncode=0, stdout=""),
    ))

    def run(command, **_kwargs):
        calls.append(command)
        return next(answers)

    monkeypatch.setattr("whisper_flow.system.subprocess.run", run)

    assert manager._xdotool_paste_if_target_is_x11("target") is True
    assert calls[-1] == ["xdotool", "key", "ctrl+v"]



def test_alt_repair_skipped_when_compositor_consumes_the_chord(
        tmp_path, monkeypatch):
    """A portal-consumed chord never reaches apps: no stray Escape."""
    manager = SystemManager(
        Config(config_dir=tmp_path, hotkey_transcribe="super+alt"))
    manager.hotkey_consumer = lambda: True
    monkeypatch.setattr(manager, "_ydotool_available", lambda: True)

    assert manager._repair_modifier_only_alt_focus() is False



def test_alt_repair_still_runs_for_observed_chords(tmp_path, monkeypatch):
    """The fail-open observer leaks Alt: the repair stays mandatory."""
    manager = SystemManager(
        Config(config_dir=tmp_path, hotkey_transcribe="super+alt"))
    manager.hotkey_consumer = lambda: False
    monkeypatch.setattr(manager, "_ydotool_available", lambda: False)
    monkeypatch.setattr("whisper_flow.system.shutil.which", lambda tool: tool)
    monkeypatch.setattr("whisper_flow.system.time.sleep", lambda _delay: None)
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr("whisper_flow.system.subprocess.run", run)

    assert manager._repair_modifier_only_alt_focus() is True
    run.assert_called_once()
