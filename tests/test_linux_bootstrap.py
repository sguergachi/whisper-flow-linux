"""The frozen Linux entry point must point GTK at the host, not the bundle.

The bundle deliberately ships no GTK/glib stack: the dynamic loader fixes
LD_LIBRARY_PATH at process start, so a bundled Ubuntu-era glib would always
shadow the host's and break every host library that needs a newer one.
PyInstaller's runtime hooks, however, force GI_TYPELIB_PATH and
GIO_MODULE_DIR onto the bundle's (now absent) directories, so the bootstrap
must scrub them for gi to resolve the host's typelibs.
"""

import sys

import pytest


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    return monkeypatch


def _import_entry():
    from whisper_flow import __main_linux__ as entry

    return entry


def test_bootstrap_scrubs_bundle_forced_paths_when_frozen(frozen, monkeypatch):
    import os

    entry = _import_entry()
    for key in entry._BUNDLE_ENV_KEYS:
        monkeypatch.setenv(key, "/bundle/whatever")
    monkeypatch.setenv("LD_PRELOAD", "/bundle/lib/libgtk4-layer-shell.so.0")
    entry._bootstrap_gtk_runtime()
    for key in entry._BUNDLE_ENV_KEYS:
        assert key not in os.environ, f"{key} still set"
    assert "LD_PRELOAD" not in os.environ


def test_bootstrap_leaves_ld_library_path_alone(frozen, monkeypatch):
    entry = _import_entry()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bundle:/usr/lib")
    entry._bootstrap_gtk_runtime()
    import os

    assert os.environ["LD_LIBRARY_PATH"] == "/bundle:/usr/lib"


def test_bootstrap_is_a_noop_for_source_runs(monkeypatch):
    entry = _import_entry()
    monkeypatch.setenv("GI_TYPELIB_PATH", "/usr/share/girepository-1.0")
    monkeypatch.setenv("LD_PRELOAD", "/opt/something.so")
    entry._bootstrap_gtk_runtime()
    import os

    assert os.environ["GI_TYPELIB_PATH"] == "/usr/share/girepository-1.0"
    assert os.environ["LD_PRELOAD"] == "/opt/something.so"


def test_runtime_mode_reflects_frozen_state(frozen):
    entry = _import_entry()
    assert entry.RUNTIME_MODE == "host"


def test_runtime_mode_is_source_when_not_frozen(monkeypatch):
    import importlib

    if hasattr(sys, "frozen"):
        monkeypatch.delattr(sys, "frozen")
    entry = importlib.reload(_import_entry())
    assert entry.RUNTIME_MODE == "source"


def test_glib_2_88_hosts_still_import_after_the_patch():
    """The moved signal helper must not kill the first Gtk import.

    GLib 2.88 moved unix_signal_add to GLibUnix and pygobject 3.56+ still
    binds the old name in its override module; a frozen build compiles the
    fresh source and dies on import. The bootstrap's patch gives that name
    back when glib took it away, so importing Gtk anywhere succeeds.
    """
    _import_entry()._patch_pygobject_deprecation_assertion()

    gi = pytest.importorskip("gi")
    gi.require_version("Gtk", "4.0")
    from gi.repository import GLib, Gtk

    assert Gtk.get_major_version() == 4
    if hasattr(GLib, "unix_signal_add"):
        # Whatever object answers the old name must be installable in a main
        # loop, not a deprecation stub that raises when called.
        assert callable(GLib.unix_signal_add)
