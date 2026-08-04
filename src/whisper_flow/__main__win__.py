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


def _pump(seconds: float = 5.0, until=None) -> None:
    """Drive the main loop without owning it, so a window can map and draw.

    Bounded by a deadline rather than by a count of iterations: mapping a
    window and getting a frame out of it takes as long as the machine takes,
    and a fixed number of turns is a check that passes on a desktop and fails
    on a loaded runner for no reason anyone can act on.
    """
    import time

    from gi.repository import GLib

    context = GLib.MainContext.default()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        context.iteration(False)
        if until is not None and until():
            return
        time.sleep(0.005)


def _selftest_overlay() -> str:
    """Show the real overlay and confirm it drew.

    Not "did a window object appear": the overlay's whole content is one
    cairo draw callback, and the build that shipped created the window,
    styled it, and threw out of every frame. `_drew` is set by that callback,
    so it is only true if a frame really reached our code.
    """
    from whisper_flow.hud_app import HudWindow

    window = HudWindow("", None, resident=True)
    window.realize()
    window.begin_show("")
    _pump(until=lambda: getattr(window, "_drew", False))
    drew, realized = getattr(window, "_drew", False), window.get_realized()
    window.destroy()
    _pump(0.2)
    if not realized:
        raise RuntimeError("the overlay window never realized")
    if not drew:
        raise RuntimeError(
            "the overlay window realized but never drew a frame - is the "
            "cairo foreign struct converter bundled?")
    return "overlay window drew a frame"


def _selftest_icons() -> str:
    """Confirm the bundled theme holds every icon the UI names.

    GTK does not report a missing icon; it draws the broken-image glyph and
    carries on, so this only ever surfaces as someone looking at the window.
    The bundle shipped 653 SVGs and no index.theme, flattened out of the
    subdirectories the theme declares - which is a theme GTK cannot read at
    all. What still rendered came from libgtk's own compiled-in set, and made
    the bundle look half-working rather than unused.
    """
    from gi.repository import Gdk, Gtk

    from whisper_flow.settings_gtk import ICON_NAMES

    theme = Gtk.IconTheme.get_for_display(Gdk.Display.get_default())
    theme.add_resource_path("/org/gnome/Adwaita/icons")
    missing = [name for name in ICON_NAMES if not theme.has_icon(name)]
    if missing:
        try:
            where = list(theme.get_search_path() or [])
        except Exception as e:          # never let the diagnosis fail first
            where = f"(search path unavailable: {e})"
        raise RuntimeError(
            f"the icon theme is missing {', '.join(missing)} - is "
            f"index.theme bundled, and the symbolic tree unflattened? "
            f"search path: {where}")
    return f"icon theme resolves all {len(ICON_NAMES)} icons the UI names"


def _selftest_settings() -> str:
    """Build the real settings window, inside an application as it runs.

    It builds itself in an activate handler, where GObject swallows whatever
    Python raises: the process kept running and simply never showed anything.
    Re-raise here so a broken settings window is a failed build.
    """
    from gi.repository import Adw

    from whisper_flow.settings_gtk import SettingsWindow

    failure, shown = [], []

    def activate(app):
        try:
            window = SettingsWindow(application=app)
            window.present()
            _pump(until=window.get_realized)
            shown.append(window.get_realized())
            window.destroy()
        except Exception:
            import traceback
            failure.append(traceback.format_exc())
        app.quit()

    app = Adw.Application(application_id="dev.whisperflow.selftest")
    app.connect("activate", activate)
    app.run([sys.argv[0]])
    if failure:
        raise RuntimeError(f"the settings window could not be built:\n{failure[0]}")
    if not shown or not shown[0]:
        raise RuntimeError("the settings window never realized")
    return "settings window realized"


def _selftest() -> int:
    """Open the windows this app shows, then exit.

    A frozen build with missing typelibs still builds, installs and launches:
    it fails when the first window opens, on the user's machine, which is
    exactly how it shipped. CI runs this against the built folder so that
    failure lands in the build instead.

    It has to open the real windows to mean anything. A version of this check
    that resolved the namespaces and realized a bare Adw.Window passed on a
    build where neither the overlay nor the settings window could appear at
    all - the runtime was fine and the calls into it were not.

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

        # The overlay paints itself with cairo, and PyGObject hands GTK's
        # cairo_t to pycairo through an extension it imports by a name built
        # at runtime - so PyInstaller never saw it and never bundled it. The
        # overlay then opened a window and raised on every single frame,
        # drawing nothing. Ask for the converter by name; it is the one check
        # here that does not depend on a window manager.
        gi.require_foreign("cairo")
        report.append("cairo foreign struct converter present")

        # A generic Adw.Window realized fine while both real windows were
        # broken, which is how this check stayed green through the whole
        # thing. Build the windows the app actually shows, the way it shows
        # them, and let them fail here rather than on someone's desktop.
        report.append(_selftest_icons())
        report.append(_selftest_overlay())
        report.append(_selftest_settings())
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


def _stop_running_app() -> None:
    """Take the previous version down, so the installer can replace it.

    Every install today failed on this. The daemon runs out of the very
    directory being replaced, so Velopack could not move it: four retries
    of "Access is denied" and then "Install Partially Succeeded" with the
    old version still in place. It says as much in its own message - close
    the application and try again - and an installer that needs the user to
    know that is not one that installs cleanly.

    The daemon writes its pid, so there is no guessing involved. The speech
    server dies with it, through the job object it was started in.
    """
    try:
        from pathlib import Path
        pid_file = Path.home() / ".config" / "whisper-flow" / "daemon.pid"
        if not pid_file.exists():
            return
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return

    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # PROCESS_TERMINATE | SYNCHRONIZE
        handle = kernel32.OpenProcess(0x1 | 0x00100000, False, pid)
        if not handle:
            return                  # already gone; nothing holding the files
        try:
            kernel32.TerminateProcess(handle, 0)
            kernel32.WaitForSingleObject(handle, 10000)
            print(f"[whisper-flow] stopped the running daemon ({pid}) so the "
                  f"installer can replace it", flush=True)
        finally:
            kernel32.CloseHandle(handle)
    except Exception as e:
        print(f"[whisper-flow] could not stop the running daemon: {e}",
              flush=True)


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

    if "--settings" in sys.argv:
        from whisper_flow.settings_gtk import main as settings_main
        return settings_main()

    # Velopack's hooks, before anything else in the application starts.
    #
    # The installer runs this executable with --veloapp-* arguments at
    # install, update and uninstall to create shortcuts and tidy up old
    # versions. Doing anything first - reading config, opening a device -
    # would happen during an install, which is not a moment when this app
    # should be doing work.
    #
    # Deliberately after the two branches above. Those are our own child
    # processes and never receive a Velopack hook, and the overlay is the
    # thing the user waits for, so it is kept clear of anything avoidable.
    hook = next((arg for arg in sys.argv[1:]
                 if arg.startswith("--veloapp-")), None)
    try:
        import velopack
        velopack.App().run()
    except ImportError:
        pass                    # a source checkout, not an installed build
    except Exception as e:
        print(f"[whisper-flow] velopack startup failed: {e}", flush=True)

    # Exit on a hook rather than trusting run() to have done it.
    #
    # It does not. The installer waits 30 seconds for this process to end,
    # and run() returns instead of exiting, so control reached the daemon
    # below and the hook became a daemon that never returns. Velopack killed
    # it and reported "Install Partially Succeeded" - an install that had in
    # fact done everything, undone by the app refusing to leave.
    #
    # There is nothing for us to do in a hook anyway: shortcuts, the package
    # directory and the uninstall entry are all Velopack's own work. Leaving
    # promptly is the entire contract.
    if hook:
        _stop_running_app()
        return 0

    from whisper_flow.daemon import WhisperFlowDaemon
    WhisperFlowDaemon().run(foreground=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
