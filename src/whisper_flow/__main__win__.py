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
    os.environ.setdefault(
        "GI_TYPELIB_PATH", os.path.join(base, "girepository-1.0"))
    os.environ.setdefault(
        "GSETTINGS_SCHEMA_DIR", os.path.join(base, "share", "glib-2.0", "schemas"))
    # A loaders.cache would carry the build machine's absolute paths; a
    # module directory is scanned fresh on the user's machine instead.
    os.environ.setdefault(
        "GDK_PIXBUF_MODULEDIR",
        os.path.join(base, "lib", "gdk-pixbuf-2.0", "2.10.0", "loaders"))


def main() -> int:
    # Without this a frozen build re-runs the whole program in every worker
    # process it spawns.
    multiprocessing.freeze_support()
    _bootstrap_gtk_runtime()

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
