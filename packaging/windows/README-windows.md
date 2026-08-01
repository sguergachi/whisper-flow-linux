# whisper-flow for Windows

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

* **Blur comes from DWM.** Windows 11 22H2 and later get acrylic through
  `DWMWA_SYSTEMBACKDROP_TYPE`; Windows 10 1803 and later fall back to the
  undocumented accent policy. On a build with neither, the overlay turns
  itself opaque rather than sitting there as a dim rectangle. The HUD log
  says which route it took.
* **The overlay's rounded corners are not antialiased.** They come from a
  Win32 window region, which is a hard-edged mask.
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
