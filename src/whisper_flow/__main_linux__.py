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


def _trace(label, *detail):
    """Append a line to WHISPER_FLOW_GI_TRACE, when that file is asked for.

    The override-loading dance below runs inside frozen processes whose only
    symptom is a traceback that never reaches a log. Asking for a trace on
    such a host is how this code got written, so it stays cheap and closed.
    """
    path = os.environ.get("WHISPER_FLOW_GI_TRACE")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(" ".join([label, *map(str, detail)]) + "\n")
    except OSError:
        pass


def _patch_pygobject_deprecation_assertion() -> None:
    """Keep gi.repository.GLib importable on hosts past GLib 2.88.

    GLib 2.88 moved unix_signal_add and friends into the GLibUnix namespace.
    Two pygobject gaps then break a plain `from gi.repository import Gtk`:

    The GLib override module still registers unix_signal_add_full as a
    deprecated alias whenever the platform namespace has signal_add, but its
    own export of that name sits behind `hasattr(GLib, "unix_signal_add")` -
    false on a moved host - so load_overrides raises the assertion it wrote
    for exactly this mismatch ("... wasn't added to __all__"). Deregistering
    deprecations the override does not export is what the assertion asks
    for, and pygobject 3.56 kept the registry as a list of pairs where
    3.58 switched to a dict, so both shapes are cleaned as they are found.
    The failed attempt also leaves a half-executed override module in
    sys.modules; the retry re-executes it whole.

    Before any of that, the raw GLib namespace is given the moved
    implementation back, so bindings and deprecation descriptors have a
    real function to hand out rather than an absent symbol; anything that
    says GLib.unix_signal_add then works, calling GLibUnix.signal_add.
    """
    try:
        import gi.importer as _importer
        import gi.overrides as _overrides

        _load = _importer.load_overrides
        if getattr(_load, "_wf_patched", False):
            return

        def _load_patched(introspection_module):
            ns = introspection_module.__name__.rsplit(".", 1)[-1]
            _bridge_moved_unix_signal(introspection_module)
            try:
                return _load(introspection_module)
            except AssertionError:
                override_mod = sys.modules.get(f"gi.overrides.{ns}")
                if override_mod is None or not hasattr(override_mod, "__all__"):
                    raise
                exported = set(override_mod.__all__)
                deprec = _overrides._deprecated_attrs.get(ns)
                if hasattr(deprec, "items"):
                    _overrides._deprecated_attrs[ns] = {
                        attr: replacement
                        for attr, replacement in deprec.items()
                        if attr in exported
                    }
                elif deprec is not None:
                    _overrides._deprecated_attrs[ns] = [
                        (attr, replacement)
                        for attr, replacement in deprec
                        if attr in exported
                    ]
                # The first attempt died mid-execution, leaving a half-built
                # override module cached - the retry would see exactly the
                # broken shape it just claimed. Drop it so the module
                # re-executes whole against the trimmed deprecation list.
                sys.modules.pop(f"gi.overrides.{ns}", None)
                try:
                    return _load(introspection_module)
                except Exception as second:
                    _trace("retry-failed", ns, type(second).__name__, second)
                    raise

        _load_patched._wf_patched = True
        _importer.load_overrides = _load_patched
        _trace("patched", "gi.importer.load_overrides")
    except Exception as why:
        _trace("patch-err", type(why).__name__, why)


def _bridge_moved_unix_signal(introspection_module):
    """Give the raw GLib namespace its moved signal helper back.

    GLib >= 2.88 hosts remove unix_signal_add from the GLib namespace, and
    the override module's export of unix_signal_add_full checks for it
    before binding. Importing GLibUnix here instead would re-enter the
    namespace loader from inside the GLib load it is servicing - a
    recursion the loader does not survive - so the bound object is a
    forwarder that resolves GLibUnix at call time, after the load has
    finished. Everything about the name then behaves as before the move:
    hasattr sees a function, the override exports both spellings, and the
    deprecation machinery hands out a callable rather than a dead import.

    Only a GLib namespace without the symbol is touched; hosts from
    Debian bookworm to Arch load through here, and must never see this
    become the thing that breaks import.
    """
    try:
        name = introspection_module.__name__.rsplit(".", 1)[-1]
        if name != "GLib" or hasattr(introspection_module, "unix_signal_add"):
            return

        def _unix_signal_forwarder(priority, signum, handler, *user_data):
            from gi.repository import GLibUnix

            return GLibUnix.signal_add(priority, signum, handler, *user_data)

        introspection_module.unix_signal_add = _unix_signal_forwarder
        if hasattr(introspection_module, "unix_signal_add"):
            _trace("bridged", "GLib.unix_signal_add -> GLibUnix.signal_add")
        else:
            _trace("bridge-set-failed", "GLib")
    except Exception as why:
        _trace("bridge-err", type(why).__name__, why)


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
    try:
        from whisper_flow.updater import cleanup_replaced_appimage

        cleanup_replaced_appimage()
    except Exception:
        pass

    from whisper_flow.daemon import WhisperFlowDaemon

    WhisperFlowDaemon().run(foreground=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
