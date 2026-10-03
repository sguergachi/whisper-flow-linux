# whisper-flow for Windows

**Requires Windows 11 22H2 (build 22621) or later.**

Run the installer, then hold **Ctrl+Alt**, talk, and let go.

There is one executable, `whisper-flow.exe`. It runs the tray app; it also
provides the recording overlay, which it launches for itself. The words are
typed into whatever had focus.

It lives in the notification area — there is no main window.

## What you need

* The installer includes a resident [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
  worker and a starter model. No separate server installation is needed.
* A microphone.

## Configuration

`%LOCALAPPDATA%\whisper-flow\.env`:

```ini
# Point at your own server, or leave blank for the managed one
WHISPER_FLOW_LOCAL_WHISPER_URL=http://127.0.0.1:8082

WHISPER_FLOW_HOTKEY_TRANSCRIBE=ctrl+alt
WHISPER_FLOW_LIVE_TRANSCRIPTION=true

# Set to false if the notifications get in the way
WHISPER_FLOW_NOTIFICATIONS_ENABLED=true
```

With live transcription on, words are typed as you speak rather than all at
once at the end. A word is only typed once two consecutive passes agree on
it, so what lands is what the final transcript says.

## Windows machines without administrator access

The default local engine keeps the model loaded in a private worker process
and exchanges audio over process pipes. It opens no HTTP listening port and
requires no Windows service, firewall change, CUDA toolkit, or administrator
access. On an NVIDIA machine it uses Vulkan through the existing graphics
driver. CPU machines use the same resident worker on the CPU.

**Settings → Speech → Install GPU engine** downloads and selects
`ggml-large-v3-turbo`. The Windows Vulkan worker is already included, so
there is no CUDA engine archive to install. The model download is about
1.6 GB; interrupted downloads retry and resume when the server supplies a
file validator. The model stays loaded between recordings, including live
preview passes. GPU shader compilation can make the first decode slower.

The legacy HTTP engine is still available under **Speech → Advanced → Local
engine → server**, or with `WHISPER_FLOW_LOCAL_ENGINE_TRANSPORT=server`.
Its Visual C++ runtime DLLs ship alongside the application and are copied
beside downloaded engines; there is no separate elevated runtime installer.
An explicit Server URL continues to use that external server.

If the worker fails, the app records the reason in
`%LOCALAPPDATA%\whisper-flow\resident-worker.log` and uses the existing
CLI fallback. It does not repeatedly download or relaunch the failed
worker. The worker remains subject to the machine's application and GPU
security policies; this design does not disable or alter them.

## How it works here, and how that differs from Linux

* **Hotkeys are polled, not hooked.** A low-level keyboard hook is the usual
  approach and is the wrong one in Python: its callback needs the GIL, and
  while another thread holds it Windows blocks *all* system input waiting —
  the mouse and keyboard freeze. Polling key state cannot stall anything and
  cannot swallow a keystroke.
* **The overlay is composited by DWM** — acrylic backdrop, rounded corners,
  border — which is why the build floor is 22H2.
* **Ctrl+Alt is the default, not Super.** Windows reserves most Win
  combinations and a bare tap opens the Start menu.

## Known limitations

* A key press shorter than about 16ms can be missed, since hotkeys are
  sampled at 60Hz. This does not matter for hold-to-talk.
* Applications running as administrator will not accept typed text unless
  whisper-flow also runs as administrator. That is a Windows security
  boundary, not something an application can work around.
* Only one copy runs at a time. A second is refused rather than started,
  because two hotkey listeners would record and type twice.

## When something is wrong

Right-click the tray icon and choose **Test Configuration**. It checks the
transcription backend, whether the server actually answers, the microphone,
and the hotkeys. If anything fails, the full report is copied to the
clipboard ready to paste into a bug report.
