"""Frozen-build entry point for the Windows tray daemon.

PyInstaller needs a script, not a console_script entry point, and this keeps
the import surface small: pulling in whisper_flow/__init__ would drag the
whole package along, including the Linux-only modules.
"""
import multiprocessing
import sys


def main() -> int:
    # Without this a frozen build re-runs the whole program in every worker
    # process it spawns.
    multiprocessing.freeze_support()
    from whisper_flow.daemon import WhisperFlowDaemon
    WhisperFlowDaemon().run(foreground=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
