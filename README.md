# WhisperFlow

Hold a key, talk, and the words appear in whatever you were typing into.

![The recording overlay, with a live waveform](docs/hud.gif)

> A fork of [sapountzis/whisper-flow-linux](https://github.com/sapountzis/whisper-flow-linux)
> by Andreas Sapountzis, whose commits open this repository's history.
> This fork adds live transcription, a Wayland layer-shell overlay with
> compositor blur, GPU transcription through whisper.cpp, and a Windows 11
> build. Bugs in any of that belong here, not upstream.

## Features

- 🎤 **Voice Transcription**: Real-time speech-to-text, entirely on your
  machine, through a local whisper.cpp server
- 🔧 **System Tray**: Background daemon with tray icon and global hotkeys
- ⌨️ **Global Hotkeys**: Push-to-talk and single-press voice activation
- 📝 **Multiple Modes**: Transcribe, Auto-Transcribe, and Command modes
- ⚙️ **Configurable**: Customizable models and settings

## Windows 11

Download the installer from the
[latest release](https://github.com/sguergachi/whisper-flow-linux/releases),
run it, and hold `Ctrl+Alt` to dictate. Nothing else is required: the
installer ships a speech engine and model, so it transcribes offline from the
first launch, with no API key and no Python.

It installs per-user without administrator rights, adds itself to Startup so
it runs at login, and updates itself — **Check for updates** in the tray
menu. Updates are deltas, so they fetch only what changed rather than the
whole installer again.

Windows 11 22H2 or newer, because the overlay uses composition attributes
added in that release.

## Which model runs

The app ships `base.en`, which keeps up with speech on any machine from a
two-core laptop upwards. Where the hardware can do better, the setup window
offers better — one button, with a progress bar. Decline and the bundled model
keeps working; switch later from **Settings** in the tray, where every model
can be downloaded and picked.

| Machine | Model | Why |
|---|---|---|
| NVIDIA GPU | `large-v3-turbo` | Far more accurate, still faster than speech |
| Desktop or workstation CPU | `small.en-q8_0` | Keeps up given enough cores and RAM |
| Typical laptop | `base.en-q8_0` | Bundled; the safe default everywhere |
| Thin, old or 4GB machine | `tiny.en-q8_0` | Anything larger falls behind |

The CPU models are the 8-bit builds. On a six-core desktop those transcribe
20–25% faster than the same model in full precision and download at half the
size, for the same transcript on eight of nine test clips. Notably it is
*only* the 8-bit ones worth having: the 5-bit builds are smaller again and
measurably **slower**, because unpacking five-bit weights costs more time
than the memory traffic it saves.

**Integrated graphics are not used, and cannot be.** whisper.cpp publishes
prebuilt binaries only for CPU and NVIDIA cuBLAS — no Vulkan, SYCL or
OpenVINO on any platform — so on an Intel or AMD GPU there is no engine to
point at. Those machines run on the CPU, with the model sized to it and one
thread per physical core — the count where the encoder measured fastest, and
well short of the point where oversubscription collapses it.

Nothing large is ever downloaded without being asked.

## Linux: Quick Setup (One-Click Install)

### Prerequisites

- **Linux** (Ubuntu/Debian/Mint) with desktop environment
- **Python 3.12** (required for system tray support)
- **A [whisper.cpp](https://github.com/ggerganov/whisper.cpp) server** — the
  app starts and manages one for you unless you point it at your own

### 1. Install System Dependencies

```bash
# Install required system packages for tray icon support
sudo apt update && sudo apt install -y \
    python3-gi \
    gir1.2-gtk-3.0 \
    gir1.2-appindicator3-0.1 \
    libappindicator3-1 \
    python3-venv \
    python3-pip
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/sguergachi/whisper-flow-linux.git
cd whisper-flow-linux

# Create virtual environment with system site packages
python3.12 -m venv .venv --system-site-packages

# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 3. Point at a Whisper Server (optional)

The app starts and manages a local whisper.cpp server on its own. To use one
you already run instead:

```bash
export WHISPER_FLOW_LOCAL_WHISPER_URL="http://127.0.0.1:8082"
```

### 4. Start the Daemon

```bash
# Start the daemon with tray icon (GTK backend configured by default)
whisper-flow daemon --foreground

# Or start in background
whisper-flow daemon
```

You should see a microphone icon in your system tray. Right-click it to access the menu!

## Usage

### System Tray

- **Right-click** the tray icon to access:
  - Settings — every option: speech model, hotkeys, dictation, notifications
  - Speech model... — quick one-button model setup
  - Test Configuration  
  - Exit

### Global Hotkeys

- **🎤 Transcribe**: `Ctrl+Cmd` (push-to-talk)
- **🔴 Auto-Transcribe**: `Ctrl+Cmd+Space` (single press)
- **🤖 Command**: `Ctrl+Cmd+Alt` (single press)
- **🛑 Cancel**: `Escape`
- **📋 Menu**: `F1`

### CLI Commands

```bash
# Initialize configuration
whisper-flow init-config

# Start daemon
whisper-flow daemon

# Stop daemon
whisper-flow stop

# Check status
whisper-flow status

# Test configuration
whisper-flow validate
```

## Troubleshooting

### Tray Icon Not Working?

If you see the tray icon but the menu doesn't appear:

1. **Check backend**: The daemon should show "Pystray backend: gtk"
2. **Verify gi module**: `python -c "import gi; print('OK')"`
3. **Restart daemon**: `whisper-flow stop && whisper-flow daemon`

### Common Issues

- **"No module named 'gi'"**: Install system packages from step 1
- **"XOrg backend"**: Ensure Python 3.12 and system packages are installed
- **No tray icon**: Check if your desktop environment supports system tray

### Manual Backend Selection

The GTK backend is configured by default. To use a different backend:

```bash
# Set environment variable
export PYSTRAY_BACKEND=appindicator
whisper-flow daemon

# Or edit config file
# ~/.config/whisper-flow/config.yaml: pystray_backend: "appindicator"
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
isort src/
```

### Project Structure

```
whisper-flow/
├── src/whisper_flow/
│   ├── app.py          # Main application logic
│   ├── daemon.py       # System tray daemon
│   ├── cli.py          # Command-line interface
│   ├── audio.py        # Audio recording
│   ├── transcription.py # whisper.cpp server client
│   ├── backend.py      # Managed local whisper.cpp server
│   └── config.py       # Configuration management
├── pyproject.toml      # Project configuration
└── README.md          # This file
```

## Configuration

Settings live in `~/.config/whisper-flow/.env`, and every one of them can be
set from the environment instead. **Settings** in the tray menu opens a
window onto all of them; `whisper-flow status` prints what is in effect.

## License

The upstream project states MIT in its README but has never included a
LICENSE file, so there is no licence text or copyright line to inherit and
this fork cannot supply one on the original author's behalf. Treat the
licensing as unsettled until upstream adds one.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/sguergachi/whisper-flow-linux/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sguergachi/whisper-flow-linux/discussions)
- **Upstream**: [sapountzis/whisper-flow-linux](https://github.com/sapountzis/whisper-flow-linux)
  — for anything predating this fork

### Manual Testing

Once setup is complete, you can run individual tests:

```bash
# Test tray functionality
python tests/test_tray.py

# Test daemon components
python tests/test_daemon_tray.py
```