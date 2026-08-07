"""Frozen-build entry point for Linux.

One executable, three roles: the tray daemon, the overlay (`--hud`), and the
settings window (`--settings`). Same shape as the Windows freeze, without
Velopack hooks.

The AppImage wraps this binary so double-click runs on most x86_64 Linux
desktops (glibc from the build host onward). GTK, typelibs and shared libs
ship inside; nothing is pip-installed on the user's machine.
"""
from __future__ import annotations

import multiprocessing
import os
import sys


def _bootstrap_gtk_runtime() -> None:
    """Point PyGObject / the dynamic linker at the bundled GTK runtime.

    Must run before anything imports gi. Same assignment rules as Windows:
    PyInstaller's own gi hook may set GI_TYPELIB_PATH first to a wrong set
    (pystray's GTK 3), so ours is assigned and put first.
    """
    if not getattr(sys, "frozen", False):
        return
    base = sys._MEIPASS

    typelibs = os.path.join(base, "girepository-1.0")
    if os.path.isdir(typelibs):
        inherited = os.environ.get("GI_TYPELIB_PATH")
        os.environ["GI_TYPELIB_PATH"] = (
            f"{typelibs}{os.pathsep}{inherited}" if inherited else typelibs
        )

    schemas = os.path.join(base, "share", "glib-2.0", "schemas")
    if os.path.isdir(schemas):
        os.environ["GSETTINGS_SCHEMA_DIR"] = schemas

    # Prefer bundled shared libraries over whatever the host has (or lacks).
    lib_dirs = [
        os.path.join(base, "lib"),
        os.path.join(base, "lib", "x86_64-linux-gnu"),
        base,
    ]
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    lib_path = os.pathsep.join(d for d in lib_dirs if os.path.isdir(d))
    if lib_path:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{lib_path}{os.pathsep}{existing_ld}" if existing_ld else lib_path
        )

    loaders = os.path.join(base, "lib", "gdk-pixbuf-2.0", "2.10.0", "loaders")
    if os.path.isdir(loaders):
        os.environ.setdefault("GDK_PIXBUF_MODULEDIR", loaders)

    # Soft renderer: portable across GPUs and remote/nested displays.
    os.environ.setdefault("GSK_RENDERER", "cairo")
    os.environ.setdefault("NO_AT_BRIDGE", "1")

    # gtk4-layer-shell must preload ahead of libwayland-client. Prefer the
    # copy we shipped; fall back to the host if the user has a newer one.
    for name in (
        "libgtk4-layer-shell.so.0",
        "libgtk4-layer-shell.so",
    ):
        candidate = os.path.join(base, "lib", name)
        if not os.path.exists(candidate):
            candidate = os.path.join(base, name)
        if os.path.exists(candidate):
            existing = os.environ.get("LD_PRELOAD", "")
            os.environ["LD_PRELOAD"] = (
                f"{candidate}:{existing}" if existing else candidate
            )
            break


def _selftest() -> int:
    """Prove the frozen GTK runtime can open a window on this machine."""
    report: list[str] = []
    status = 0
    try:
        import gi

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

    from whisper_flow.daemon import WhisperFlowDaemon

    WhisperFlowDaemon().run(foreground=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
