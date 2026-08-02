"""Frozen-build entry point for Windows.

One executable, two roles. The overlay runs as a separate process so GTK/Tk
cannot stall audio capture, but shipping it as a second .exe beside the first
just raises the question of which one to run. Instead this binary re-launches
itself with --hud.

A folder build makes that cheap: the second launch reuses the same unpacked
directory. A one-file build would re-extract the whole payload on every
recording, which is why this is not one.
"""
import multiprocessing
import os
import sys


def _bootstrap_gtk_runtime() -> None:
    """Point PyGObject at the GTK runtime bundled beside the executable.

    Typelibs, schemas and pixbuf loaders are data, not code: nothing finds
    them by itself inside a frozen build, and every GTK window fails at once
    without them. Must run before anything imports gi.
    """
    if not getattr(sys, "frozen", False):
        return
    base = sys._MEIPASS

    # GI_TYPELIB_PATH is *assigned*, not defaulted, and ours goes first.
    #
    # PyInstaller ships its own gi runtime hook, and runtime hooks run before
    # this script. It does `os.environ['GI_TYPELIB_PATH'] = <base>/gi_typelibs`
    # - a hard assignment - holding whatever its GI hooks inferred from the
    # import graph. That graph also contains pystray's GTK 3 backend, so what
    # it collects is a GTK 3 flavoured guess and need not contain Gtk-4.0 at
    # all. setdefault() therefore did nothing here, the curated GTK 4 typelibs
    # in the spec were never on the search path, and the app died on
    # "Namespace Gtk not available" at the first window - off a green build.
    #
    # PyInstaller's directory stays on the path behind ours, so anything it
    # found that the spec does not enumerate still resolves.
    typelibs = os.path.join(base, "girepository-1.0")
    inherited = os.environ.get("GI_TYPELIB_PATH")
    os.environ["GI_TYPELIB_PATH"] = (
        f"{typelibs}{os.pathsep}{inherited}" if inherited else typelibs)

    os.environ.setdefault(
        "GSETTINGS_SCHEMA_DIR", os.path.join(base, "share", "glib-2.0", "schemas"))
    # A loaders.cache would carry the build machine's absolute paths; a
    # module directory is scanned fresh on the user's machine instead.
    os.environ.setdefault(
        "GDK_PIXBUF_MODULEDIR",
        os.path.join(base, "lib", "gdk-pixbuf-2.0", "2.10.0", "loaders"))


def _selftest() -> int:
    """Resolve every GTK namespace the UI needs, then exit.

    A frozen build with missing typelibs still builds, installs and launches:
    it fails when the first window opens, on the user's machine, which is
    exactly how it shipped. CI runs this against the built folder so that
    failure lands in the build instead.

    The executable is windowed, so there is no console to print to. The
    report goes to the file named by WHISPER_FLOW_SELFTEST_OUT when set.
    """
    report, status = [], 0
    try:
        import gi

        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")
        from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: F401

        report.append(
            f"GTK {Gtk.get_major_version()}.{Gtk.get_minor_version()}, "
            f"GLib {GLib.MAJOR_VERSION}.{GLib.MINOR_VERSION}, "
            f"Adw and Gdk resolved")
        report.append(f"GI_TYPELIB_PATH={os.environ.get('GI_TYPELIB_PATH')}")

        # Resolving the namespaces proved nothing about the runtime: the
        # first build that passed this check still could not open a window,
        # because a missing GSettings schema aborts the process rather than
        # returning an error. Look one up, then actually build a window.
        # No hardcoded schema name: GTK 4 renamed its own settings schemas
        # under org.gtk.gtk4.*, so probing a GTK 3 name reports a missing
        # bundle that is in fact present. Ask what the source holds instead
        # - the bug being guarded against is a directory with no compiled
        # schemas at all, which shows up as an empty list either way.
        schema_dir = os.environ.get("GSETTINGS_SCHEMA_DIR")
        source = Gio.SettingsSchemaSource.new_from_directory(
            schema_dir, None, True)
        listed = source.list_schemas(True)
        schemas = sorted(set(listed[0]) | set(listed[1]))
        gtk_schemas = [name for name in schemas if name.startswith("org.gtk.")]
        if not gtk_schemas:
            raise RuntimeError(
                f"{schema_dir} has {len(schemas)} schemas and none from GTK; "
                f"is gschemas.compiled bundled?")
        report.append(
            f"{len(schemas)} schemas from {schema_dir} "
            f"(GTK: {', '.join(gtk_schemas[:3])})")

        if not Gtk.init_check():
            raise RuntimeError("Gtk.init_check() failed - no usable display")
        Adw.init()
        # Adw.Window, not the Adw.ApplicationWindow the real windows use:
        # binding one to an Adw.Application outside that application's
        # startup signal earns a Gtk-CRITICAL, and a check that prints a
        # warning every run is one whose warnings stop being read. Both
        # realize the same native surface, which is what is under test.
        window = Adw.Window(default_width=200)
        window.set_content(Gtk.Label(label="selftest"))
        window.realize()
        report.append("Adw.Window realized")
        window.destroy()
    except Exception:
        import traceback

        report.append(traceback.format_exc())
        report.append(f"GI_TYPELIB_PATH={os.environ.get('GI_TYPELIB_PATH')}")
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
    # Without this a frozen build re-runs the whole program in every worker
    # process it spawns.
    multiprocessing.freeze_support()
    _bootstrap_gtk_runtime()

    # Before every other branch: it must not depend on config, a device or
    # anything else that could fail for its own reasons and mask the answer.
    if "--selftest" in sys.argv:
        return _selftest()

    if "--hud" in sys.argv:
        from whisper_flow.hud_app import main as hud_main
        return hud_main()

    if "--setup" in sys.argv:
        from whisper_flow.setup_gtk import main as setup_main
        return setup_main()

    if "--settings" in sys.argv:
        from whisper_flow.settings_gtk import main as settings_main
        return settings_main()

    # Velopack's hooks, before anything else in the application starts.
    #
    # The installer runs this executable with --veloapp-* arguments at
    # install, update and uninstall to create shortcuts and tidy up old
    # versions; run() acts on them and exits. Doing anything first - reading
    # config, opening a device - would happen during an install, which is
    # not a moment when this app should be doing work.
    #
    # Deliberately after the two branches above. Those are our own child
    # processes and never receive a Velopack hook, and the overlay is the
    # thing the user waits for, so it is kept clear of anything avoidable.
    try:
        import velopack
        velopack.App().run()
    except ImportError:
        pass                    # a source checkout, not an installed build
    except Exception as e:
        print(f"[whisper-flow] velopack startup failed: {e}", flush=True)

    from whisper_flow.daemon import WhisperFlowDaemon
    WhisperFlowDaemon().run(foreground=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
