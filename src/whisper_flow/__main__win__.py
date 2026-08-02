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
import sys


def main() -> int:
    # Without this a frozen build re-runs the whole program in every worker
    # process it spawns.
    multiprocessing.freeze_support()

    if "--hud" in sys.argv:
        from whisper_flow.hud_win import main as hud_main
        return hud_main()

    if "--setup" in sys.argv:
        from whisper_flow.setup_ui import main as setup_main
        return setup_main()

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
