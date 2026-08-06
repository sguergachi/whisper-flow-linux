"""Command-line interface for whisper-flow."""

import os
import subprocess
import warnings
from pathlib import Path
from typing import Annotated

import typer
from typer import Option

from . import envfile
from .app import WhisperFlow
from .version import build_version

# Suppress warnings for cleaner CLI output
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress ALSA warnings
os.environ["ALSA_SUPPRESS_WARNINGS"] = "1"
os.environ["ALSA_PCM_CARD"] = "0"
os.environ["ALSA_PCM_DEVICE"] = "0"

app = typer.Typer(
    name="whisper-flow",
    help="AI-powered voice-to-text with context-aware processing.",
    add_completion=False,
    no_args_is_help=True,
)

# Type aliases for common options
ConfigDirOption = Annotated[
    Path,
    Option("--config-dir", help="Custom configuration directory"),
]


def version_callback(value: bool) -> None:
    """Show version and exit.

    From the same place the tray menu and the failure report read it. This
    was a literal "0.1.0" that nothing updated, so the one command whose
    entire job is to say which version this is had been wrong since 0.2.
    """
    if value:
        typer.echo(f"whisper-flow {build_version()}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        Option("--version", callback=version_callback, help="Show version and exit"),
    ] = False,
) -> None:
    """WhisperFlow - AI-powered voice-to-text with context-aware processing."""


@app.command("set-device")
def set_device(
    device_index: Annotated[int, Option(help="Audio input device index")],
    config_dir: ConfigDirOption = None,
):
    """Set the microphone device index."""
    flow_app = WhisperFlow(config_dir)
    try:
        import pyaudio
    except ImportError:
        typer.echo("PyAudio not available")
        raise typer.Exit(1)

    pa = pyaudio.PyAudio()
    try:
        # Global PortAudio indices, matching what the recorder passes to
        # pa.open() - numbering them per host API stored a value that named
        # a different device, or none at all.
        if device_index < 0 or device_index >= pa.get_device_count():
            typer.echo("Invalid device index. Use 'whisper-flow list-devices' to see available devices.")
            raise typer.Exit(1)
        dev = pa.get_device_info_by_index(device_index)
        if dev.get("maxInputChannels", 0) == 0:
            typer.echo(f"Device [{device_index}] is not an input device.")
            raise typer.Exit(1)
    finally:
        pa.terminate()

    # Save to env file for persistence
    envfile.set_values(
        flow_app.config.config_dir / ".env",
        {"WHISPER_FLOW_MIC_DEVICE_INDEX": str(device_index)},
    )
    typer.echo(f"✓ Set microphone to device [{device_index}]")
    typer.echo("Restart the daemon for changes to take effect: systemctl --user restart whisper-flow")


@app.command("list-devices")
def list_devices(config_dir: ConfigDirOption = None):
    """List available audio input devices."""
    flow_app = WhisperFlow(config_dir)
    try:
        import pyaudio
    except ImportError:
        typer.echo("PyAudio not available")
        return

    pa = pyaudio.PyAudio()
    try:
        typer.echo("Available audio input devices:")
        typer.echo("─" * 50)
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if dev.get("maxInputChannels", 0) > 0:
                name = dev.get("name", "Unknown")
                channels = dev.get("maxInputChannels", 0)
                rate = dev.get("defaultSampleRate", 0)
                sel = "  ← current" if i == flow_app.config.mic_device_index else ""
                typer.echo(f"  [{i}] {name}{sel}")
                typer.echo(f"       Channels: {channels}, Sample Rate: {int(rate)} Hz")
    finally:
        pa.terminate()


@app.command("init-config")
def init_config(config_dir: ConfigDirOption = None):
    """Initialize configuration files with defaults."""
    flow_app = WhisperFlow(config_dir)
    flow_app.config.ensure_config_files()
    typer.echo(f"✓ Configuration files initialized in {flow_app.config.config_dir}")
    typer.echo("\nNext steps:")
    typer.echo("1. Run 'whisper-flow validate' to check your setup")
    typer.echo("2. Run 'whisper-flow daemon' to start the background service")
    typer.echo("3. Use the configured hotkeys for voice input")


@app.command()
def daemon(
    config_dir: ConfigDirOption = None,
    foreground: Annotated[
        bool,
        Option("--foreground", "-f", help="Run in foreground (don't daemonize)"),
    ] = False,
    _worker: Annotated[
        bool,
        Option("--_worker", help="Internal flag for background worker.", hidden=True),
    ] = False,
):
    """Start the WhisperFlow daemon with system tray and global hotkeys."""
    from .daemon import WhisperFlowDaemon

    if not _worker and not foreground:
        # This is the initial launch, so we start the background worker.
        WhisperFlowDaemon(config_dir).run(foreground=False, _worker=False)
        return

    # This is the actual worker process (or foreground mode).
    try:
        daemon_instance = WhisperFlowDaemon(config_dir)
        daemon_instance.run(foreground=foreground, _worker=_worker)
    except KeyboardInterrupt:
        if foreground:
            typer.echo("\nDaemon stopped by user")
    except ImportError as e:
        typer.echo(f"Error: Missing dependency for daemon mode: {e}", err=True)
        typer.echo("Install with: pip install 'whisper-flow[daemon]'", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error starting daemon: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def stop(
    config_dir: ConfigDirOption = None,
):
    """Stop the running WhisperFlow daemon."""
    from .daemon import stop_daemon

    try:
        stop_daemon()
    except Exception as e:
        typer.echo(f"Error stopping daemon: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def status(config_dir: ConfigDirOption = None):
    """Show system status and configuration."""
    flow_app = WhisperFlow(config_dir)

    typer.echo("WhisperFlow Status")
    typer.echo("==================")
    typer.echo(f"Mode: {flow_app.mode}")
    typer.echo(f"Config Directory: {flow_app.config.config_dir}")
    typer.echo(f"Speech model: {flow_app.config.model_name}")
    typer.echo(f"Whisper server: {flow_app.config.local_whisper_url or '(managed)'}")
    typer.echo()

    # Daemon configuration
    typer.echo("Daemon Configuration:")
    typer.echo(f"  Daemon Enabled: {'Yes' if flow_app.config.daemon_enabled else 'No'}")
    typer.echo(f"  Auto-stop Silence: {flow_app.config.auto_stop_silence_duration}s")
    typer.echo()

    # Hotkeys
    typer.echo("Hotkeys:")
    typer.echo(f"  🎤 Transcribe: {flow_app.config.hotkey_transcribe}")
    typer.echo(f"  🔴 Auto-Transcribe: {flow_app.config.hotkey_auto_transcribe}")
    typer.echo(f"  🤖 Command: {flow_app.config.hotkey_command}")
    typer.echo()

    # Audio configuration
    typer.echo("Audio:")
    device_text = flow_app.config.mic_device_index or "Default"
    typer.echo(f"  Device Index: {device_text}")
    typer.echo(f"  Sample Rate: {flow_app.config.sample_rate} Hz")
    typer.echo(f"  VAD Mode: {flow_app.config.vad_mode}")
    typer.echo()

    # System dependencies
    typer.echo("System Dependencies:")
    deps = {
        "xdotool": ["xdotool", "--version"],
        "xclip": ["xclip", "-version"],
        "xsel": ["xsel", "--version"],
        "notify-send": ["notify-send", "--version"],
        "wmctrl": ["wmctrl", "-m"],
    }

    for name, cmd in deps.items():
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            typer.echo(f"  {name}: ✓")
        except (subprocess.CalledProcessError, FileNotFoundError):
            typer.echo(f"  {name}: ✗")
    typer.echo()

    # Usage instructions
    typer.echo("Getting Started:")
    typer.echo("  1. Run 'whisper-flow daemon' to start background service")
    typer.echo("  2. Use hotkeys for voice input:")
    typer.echo(
        f"     • {flow_app.config.hotkey_transcribe}: Push-to-talk transcription",
    )
    typer.echo(
        f"     • {flow_app.config.hotkey_auto_transcribe}: Auto-stop transcription",
    )
    typer.echo(f"     • {flow_app.config.hotkey_command}: Command mode")
    typer.echo("  3. Press Escape to cancel any recording")


@app.command()
def validate(config_dir: ConfigDirOption = None):
    """Validate configuration and dependencies."""
    flow_app = WhisperFlow(config_dir)

    try:
        validation_results = flow_app.run_comprehensive_validation()
    except Exception as e:
        # Only a validation that could not run at all lands here. This used
        # to wrap the whole function, including the typer.Exit(1) raised on
        # failure - and since that is an exception too, a run with a failing
        # check reported "Validation error: 1" instead of the failure.
        typer.echo(f"Validation error: {e}", err=True)
        raise typer.Exit(1) from e

    # Every result, not just the count. This said "check the issues above"
    # while printing nothing above it, so the one failing check - the whole
    # reason to run this - was the one thing it would not tell you.
    marks = {"pass": "✅", "warning": "⚠️ ", "fail": "❌"}
    total = passed = 0
    for tests in validation_results.values():
        for test in tests:
            total += 1
            passed += test["status"] == "pass"
            mark = marks.get(test["status"], test["status"])
            typer.echo(f"  {mark} {test['name']}: {test['message']}")

    typer.echo(f"\nValidation Results: {passed}/{total} tests passed")
    if passed == total:
        typer.echo("✅ All validations passed! System is ready.")
        return
    typer.echo("❌ Some validations failed.")
    raise typer.Exit(1)


def main_entry():
    """Main entry point for the whisper-flow command."""
    app()


def dictation_entry():
    """Entry point for legacy dictation command - redirects to daemon."""
    typer.echo("The dictation command has been replaced with the daemon.")
    typer.echo("Use: whisper-flow daemon")
    typer.echo("Then use hotkeys for voice input.")


if __name__ == "__main__":
    main_entry()
