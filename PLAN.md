# Plan

Running list of what we want to accomplish. Checked off when done and
verified — not when written.

## Now

- [ ] Verify the Velopack installer on a real Windows machine: that it
      installs, starts at login, and that "Check for updates" finds and
      applies a newer release. CI proves it builds, not that it updates.
- [ ] Remove `packaging/windows/installer.iss` once Velopack has shipped a
      working build; keeping it until then leaves a fallback.

## Windows CI: what it does and does not prove

The `test (windows)` job runs on a real `windows-latest` runner — build
26100, so `blur_win.is_supported()` is True there and the DWM path is
exercised. 161 passed, 5 skipped, and 10 Windows-only tests collected.

It **does** prove: the clipboard round-trips through Win32 (including a
400-line report), `GetAsyncKeyState` polling works, `RtlGetVersion` returns
a real build, the overlay process starts and stays up, and the package does
not drag in the openai SDK.

It **cannot** prove: anything needing a GPU, a microphone (the runner has no
audio device, so recording itself is untested), or Windows 11 desktop
behaviour like acrylic actually rendering. It is Windows Server, not
Windows 11 — the APIs are the same, the desktop is not.

## Overlay latency

Where the time goes when the hotkey is pressed, worst first:

0. ~~Drawing the tray icon~~ — fixed. `_render_mic_icon` supersamples to
   512px and runs a 41-pixel MaxFilter: **~615ms**, and it ran on every
   recording start *and* stop, ahead of opening the microphone. Roughly 1.2s
   per dictation spent drawing a picture of a microphone. Both colours are
   constants, so they are drawn once at startup now.

1. **Process startup.** Every recording spawns `whisper-flow.exe --hud`. In a
   PyInstaller onedir build that re-runs the bootloader, initialises a fresh
   Python, and re-reads the exe — which Defender inspects. Hundreds of ms
   before any of our code runs. **Not fixable by making our code faster.**
2. ~~Package import cost~~ — fixed. `whisper_flow/__init__` was importing the
   openai SDK, pyaudio, pystray and PIL: 742ms → 2ms (PEP 562 lazy imports).
3. Tk window creation and first paint: ~25ms. Not the problem.

Techniques, in order of expected gain:

- [x] **Keep the overlay process alive, and pre-warm it** — done, and it
      beat the target. Measured time from `show()` to actually on screen:
      **114–159ms** spawning per recording, **6–8ms** resident. Pre-warming
      at daemon start fixes the first press too: **142ms → 6ms**, which was
      the remaining "still slow" report. ~19x, and Windows should
      gain more than Linux because the frozen bootloader it avoids is far
      more expensive than a native interpreter.
      - Commands go down the process's stdin: `show <levels>`, `hide`.
      - End of stream means the daemon is gone, so the overlay cannot be
        stranded on screen even if the daemon is killed.
      - Started on first use, not at daemon start: a machine that never
        dictates never pays for it, and a crash costs one slow press rather
        than a permanently missing overlay.
      - Windows only. The GTK overlay on Linux is a layer-shell surface with
        its own lifecycle and already appears fast enough.
- [ ] Measure it on Windows, where the win should be larger than the 19x
      seen on Linux. Nothing here has been timed on the real target.
- [ ] Trim what the frozen build loads at startup (exclude unused modules
      from the analysis) — helps item 1 a little, not a 10x.
- [ ] Measure on Windows before and after. No more guessing from Linux
      numbers; the CI Windows job can time `--hud` startup.

## Done

- [x] Bundle an engine and model so a fresh install transcribes offline
- [x] One-button setup window with a progress bar, and a first-run flow
- [x] Size the model to the machine (CUDA / workstation / laptop / thin)
- [x] Install the engine on Linux too, not just Windows
- [x] Auto-build and publish on every merge to master
- [x] Failure reports on the clipboard — **was broken until 74d6f01**: the
      daemon called `copy_to_clipboard`, the method was `_copy_to_clipboard`,
      and the tests mocked the name that did not exist
- [x] Overlay no longer imports the whole application (742ms → 2ms)
- [x] Typing failures are no longer silent; SendInput falls back to the
      clipboard when UIPI refuses it
- [x] Keyboard set rebuilds on hotplug instead of dying quietly
- [x] Windows trace pass: overlay draws without blur, right Windows key,
      `Local\` mutex, PowerShell quoting, clipboard encoding, frame loop
- [x] Same hotkey on both platforms (Super+Alt)
- [x] Velopack replaces Inno Setup: delta updates, so a code-only release is
      megabytes rather than the ~160MB full installer, plus in-app "Check for
      updates" and a Startup shortcut so it runs at login
- [x] Windows tests running on a Windows runner in CI, gating the installer
      — first run immediately caught the clipboard being broken on 64-bit
      Windows (undeclared ctypes signatures truncating handles) and logging
      dying on a cp1252 console

## Notes

- Push straight to master, no branch or PR.
- whisper.cpp ships no Vulkan/SYCL/OpenVINO binaries, so integrated GPUs
  cannot be accelerated; non-NVIDIA machines run on CPU by design.
