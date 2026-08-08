"""First-run desktop integration for the frozen Linux build."""

import sys

import pytest


@pytest.fixture
def appimage_path(tmp_path):
    path = tmp_path / "WhisperFlow-1.2.3.AppImage"
    path.write_bytes(b"ELF-APPIMAGE")
    return str(path)


@pytest.fixture
def frozen(monkeypatch, appimage_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("APPIMAGE", appimage_path)
    return monkeypatch


@pytest.fixture
def data_home(monkeypatch, tmp_path):
    data = tmp_path / "data"
    config = tmp_path / "config"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    return data, config


@pytest.fixture
def frozen_binary(frozen, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    frozen.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    frozen.setattr(sys, "executable", str(bundle / "whisper-flow"), raising=False)
    return bundle


def test_installs_menu_entry_and_autostart(appimage_path, frozen, data_home, frozen_binary):
    data, config = data_home

    from whisper_flow import desktop_install

    desktop_install.ensure_desktop_integration()

    menu = data / "applications" / "whisper-flow.desktop"
    start = config / "autostart" / "whisper-flow.desktop"
    assert menu.is_file()
    assert start.is_file()
    assert f'Exec="{appimage_path}"' in menu.read_text(encoding="utf-8")
    assert start.read_text(encoding="utf-8").startswith("[Desktop Entry]")
    assert desktop_install.appimage_path() == appimage_path


def test_copies_the_shipped_icon_next_to_the_binary(frozen, data_home, frozen_binary):
    data, _ = data_home
    (frozen_binary / "whisper-flow.png").write_bytes(b"ICON")

    from whisper_flow import desktop_install

    desktop_install.ensure_desktop_integration()
    icon = data / "icons" / "hicolor" / "256x256" / "apps" / "whisper-flow.png"
    assert icon.read_bytes() == b"ICON"


def test_repoints_the_entry_when_the_appimage_moves(frozen, data_home, frozen_binary, tmp_path):
    data, config = data_home

    from whisper_flow import desktop_install

    desktop_install.ensure_desktop_integration()
    menu = data / "applications" / "whisper-flow.desktop"
    assert "WhisperFlow-1.2.3" in menu.read_text(encoding="utf-8")

    moved = tmp_path / "Moved" / "WhisperFlow-9.9.9.AppImage"
    moved.parent.mkdir()
    moved.write_bytes(b"ELF-APPIMAGE")
    frozen.setenv("APPIMAGE", str(moved))
    desktop_install.ensure_desktop_integration()
    second = menu.read_text(encoding="utf-8")
    assert f'Exec="{moved}"' in second
    assert "WhisperFlow-1.2.3" not in second


def test_does_not_rewrite_when_already_pointing_here(frozen, data_home, frozen_binary):
    data, _ = data_home

    from whisper_flow import desktop_install

    desktop_install.ensure_desktop_integration()
    menu = data / "applications" / "whisper-flow.desktop"
    first = menu.stat().st_mtime_ns
    desktop_install.ensure_desktop_integration()
    assert menu.stat().st_mtime_ns == first


def test_skips_when_not_frozen(data_home):
    data, _ = data_home

    from whisper_flow import desktop_install

    desktop_install.ensure_desktop_integration()
    assert not (data / "applications" / "whisper-flow.desktop").exists()


def test_skips_without_an_appimage(frozen, data_home, frozen_binary):
    data, _ = data_home
    frozen.delenv("APPIMAGE")

    from whisper_flow import desktop_install

    desktop_install.ensure_desktop_integration()
    assert not (data / "applications" / "whisper-flow.desktop").exists()


def test_skip_env_override(frozen, data_home, frozen_binary):
    data, _ = data_home
    frozen.setenv("WHISPER_FLOW_SKIP_DESKTOP", "1")

    from whisper_flow import desktop_install

    desktop_install.ensure_desktop_integration()
    assert not (data / "applications" / "whisper-flow.desktop").exists()
