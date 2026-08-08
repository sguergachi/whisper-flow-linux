"""Frozen-build entry point for Linux.

One executable, three roles: the tray daemon, the overlay (`--hud`), and the
settings window (`--settings`). Same shape as the Windows freeze, without
Velopack hooks.

The AppImage wraps this binary so double-click runs on most x86_64 Linux
desktops (glibc from the build host onward). The bundle ships the Python
runtime, the app and the engine; GTK, typelibs and system libraries come
from the host, exactly as they do for the venv installer (install.sh).

Bundling the GTK stack instead was tried and broke newer distros. The
dynamic loader fixes LD_LIBRARY_PATH at process start, so a bundled
Ubuntu-era glib always shadows the host's, and every host library that
needs a newer glib (libjson-glib, libnotify, system binaries like `sh`)
then dies with "undefined symbol". The bundle therefore contains none of
those libraries, and PyInstaller's runtime hooks are made to point back at
the host by scrubbing the paths they force.
"""
from __future__ import annotations

import multiprocessing
import os
import sys

# What GTK runtime the current process uses. Frozen builds always run on
# the host's stack; source runs are the venv installer's world.
RUNTIME_MODE = "host" if getattr(sys, "frozen", False) else "source"

# Paths PyInstaller's runtime hooks force onto the bundle, which is thin
# and does not contain them. Removed so gi, GSettings and GdkPixbuf resolve
# the host's typelibs, schemas and loaders instead.
_BUNDLE_ENV_KEYS = (
    "GI_TYPELIB_PATH",
    "GSETTINGS_SCHEMA_DIR",
    "GDK_PIXBUF_MODULEDIR",
    "GIO_MODULE_DIR",
)


def _patch_pygobject_deprecation_assertion() -> None:
    """Work around pygobject's GLib override crash on glib >= 2.88.

    glib 2.88 moved GLib.unix_signal_add to GLibUnix.signal_add. The GLib
    override module in PyGObject 3.56 still registers the old name as a
    deprecated alias without defining it, and gi.overrides.load_overrides
    raises AssertionError when a registered deprecation is not in the
    module's __all__. Surface runtimes hide it with a stale .pyc; a frozen
    build compiles the fresh source and dies on the first Gtk import.
    Dropping deprecations for names the module does not export is exactly
    what the assertion is enforcing anyway.
    """
    try:
        import gi.importer as _importer
        import gi.overrides as _overrides

        _load = _importer.load_overrides
        if getattr(_load, "_wf_patched", False):
            return

        def _load_patched(introspection_module):
            try:
                return _load(introspection_module)
            except AssertionError:
                ns = introspection_module.__name__.rsplit(".", 1)[-1]
                override_mod = sys.modules.get(f"gi.overrides.{ns}")
                if override_mod is None or not hasattr(override_mod, "__all__"):
                    raise
                exported = set(override_mod.__all__)
                _overrides._deprecated_attrs[ns] = [
                    (attr, replacement)
                    for attr, replacement in _overrides._deprecated_attrs.get(ns, [])
                    if attr in exported
                ]
                return _load(introspection_module)

        _load_patched._wf_patched = True
        _importer.load_overrides = _load_patched
    except Exception:
        pass


def _bootstrap_gtk_runtime() -> None:
    """Point PyGObject at the host's GTK stack.

    Must run before anything imports gi. PyInstaller's rthooks set
    GI_TYPELIB_PATH and GIO_MODULE_DIR onto the bundle's (now absent)
    gi_typelibs/gio_modules; scrubbing them here lets gi find the host's
    typelibs. LD_LIBRARY_PATH is left alone on purpose: the loader fixed it
    at startup, and the bundle no longer contains any host-system library,
    so children (sh, notify-send, the engine) cannot pick up conflicts.
    """
    if not getattr(sys, "frozen", False):
        _patch_pygobject_deprecation_assertion()
        return
    for key in _BUNDLE_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.pop("LD_PRELOAD", None)
    os.environ.setdefault("NO_AT_BRIDGE", "1")
    _patch_pygobject_deprecation_assertion()


def _selftest() -> int:
    """Prove the host GTK stack can open a window on this machine."""
    report: list[str] = []
    status = 0
    try:
        import gi

        _dbg = os.environ.get("WHISPER_FLOW_SELFTEST_DEBUG")
        if _dbg:
            import gi.overrides as _ov
            report.append(f"DEBUG _deprecated_attrs={dict(_ov._deprecated_attrs)}")
            from gi.module import get_introspection_module
            raw = get_introspection_module("GLib")
            report.append(f"DEBUG raw unix_signal_add={hasattr(raw, 'unix_signal_add')}")
            import gi._gi as _g
            report.append(f"DEBUG _gi={_g.__file__}")
            try:
                with open("/proc/self/maps") as mf:
                    for line in mf:
                        p = line.strip().split()[-1]
                        if any(s in p for s in ("girepository", "libglib", "libgobject")):
                            report.append("DEBUG MAP " + p)
            except OSError:
                pass

        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")
        from gi.repository import Adw, Gtk  # noqa: F401

        report.append("Gtk 4 + Adw import OK")
        if not Gtk.init_check():
            raise RuntimeError("Gtk.init_check() failed — no usable display")
        Adw.init()
        try:
            gi.require_foreign("cairo")
            report.append("cairo foreign struct converter present")
        except Exception as e:
            report.append(f"cairo foreign: {e}")
        # Layer shell is Wayland-only; missing it on X11 is fine.
        try:
            gi.require_version("Gtk4LayerShell", "1.0")
            from gi.repository import Gtk4LayerShell  # noqa: F401

            report.append("Gtk4LayerShell available")
        except Exception as e:
            report.append(f"Gtk4LayerShell optional: {e}")
        report.append(f"GTK runtime: {RUNTIME_MODE}")
        report.append("selftest OK")
    except Exception:
        import traceback

        report.append(traceback.format_exc())
        report.append(f"GI_TYPELIB_PATH={os.environ.get('GI_TYPELIB_PATH')}")
        report.append(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}")
        status = 1

    text = "\n".join(report)
    out = os.environ.get("WHISPER_FLOW_SELFTEST_OUT")
    if out:
        try:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError:
            pass
    print(text, flush=True)
    return status


def main() -> int:
    multiprocessing.freeze_support()
    _bootstrap_gtk_runtime()

    if "--selftest" in sys.argv:
        return _selftest()

    if "--hud" in sys.argv:
        from whisper_flow.hud_app import main as hud_main

        return hud_main()

    if "--settings" in sys.argv:
        from whisper_flow.settings_gtk import main as settings_main

        return settings_main()

    # First run of the AppImage registers it with the desktop - app menu
    # entry, icon, login autostart - so it installs like a normal app.
    # HUD/settings children above never reach this; source runs are the
    # venv installer's world and are left alone.
    try:
        from whisper_flow.desktop_install import ensure_desktop_integration

        ensure_desktop_integration()
    except Exception:
        pass

    from whisper_flow.daemon import WhisperFlowDaemon

    WhisperFlowDaemon().run(foreground=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
