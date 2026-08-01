# whisper-flow for Windows

**Requires Windows 11 22H2 (build 22621) or later.**

Unzip anywhere and run `whisper-flow.exe`. It sits in the notification area;
there is no window.

## What you need

* A transcription backend. Either a local
  [whisper.cpp](https://github.com/ggerganov/whisper.cpp) server, or an OpenAI
  API key.
* A microphone.

## Configuration

Settings are read from `%USERPROFILE%\.config\whisper-flow\.env`:

```ini
# Local server, or leave blank and set a key instead
WHISPER_FLOW_LOCAL_WHISPER_URL=http://127.0.0.1:8082
# WHISPER_FLOW_OPENAI_API_KEY=sk-...

WHISPER_FLOW_HOTKEY_TRANSCRIBE=ctrl+alt
WHISPER_FLOW_LIVE_TRANSCRIPTION=true
```

Hold the hotkey, talk, let go. With live transcription on, words are typed as
you speak rather than all at once at the end.

## How it differs from the Linux build

* **The overlay is composited by DWM.** Acrylic backdrop, rounded corners and
  border all come from `DwmSetWindowAttribute`, which is why the build floor
  is 22H2. Earlier builds are refused with a message rather than started into
  a half-working state - the daemon checks before it opens anything.
* **Super/Win as a hotkey is a poor choice here.** The shell claims it, and
  unlike the Linux build this one does not intercept the key. `ctrl+alt` is
  the default instead.

## Known limitations

* The hotkey listener uses a low-level keyboard hook. Windows will silently
  drop the hook if the machine is under heavy load; the daemon notices and
  reinstalls it, but a keypress can be missed while that happens.
* Applications running as administrator will not receive typed text unless
  whisper-flow is also running as administrator. This is a Windows security
  boundary, not something the app can work around.
