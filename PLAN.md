# Plan

Running list of what we want to accomplish. Checked off when done and
verified — not when written.

## Now

- [ ] **Make the overlay appear instantly (target 10x)** — still slow on
      Windows after the import fix. See "Overlay latency" below for the
      techniques and which one is the real win.
- [ ] **Windows-specific tests in GitHub Actions** — the Windows code is
      currently only tested on Linux with `ctypes.WinDLL` stubbed, so nothing
      exercises the real Win32 calls. Add a `windows-latest` test job.
- [ ] **Velopack installer** (the maintained successor to Squirrel) — fast
      install, delta updates, in-app auto-update.
- [ ] **Always start on Windows login** — via the installer's Startup
      shortcut, not a manual step.

## Overlay latency

Where the time goes when the hotkey is pressed, worst first:

1. **Process startup.** Every recording spawns `whisper-flow.exe --hud`. In a
   PyInstaller onedir build that re-runs the bootloader, initialises a fresh
   Python, and re-reads the exe — which Defender inspects. Hundreds of ms
   before any of our code runs. **Not fixable by making our code faster.**
2. ~~Package import cost~~ — fixed. `whisper_flow/__init__` was importing the
   openai SDK, pyaudio, pystray and PIL: 742ms → 2ms (PEP 562 lazy imports).
3. Tk window creation and first paint: ~25ms. Not the problem.

Techniques, in order of expected gain:

- [ ] **Keep the overlay process alive** (the 10x). Start it once, hidden,
      when the daemon starts; show and hide it on command instead of
      spawning and killing a process per recording. Removes items 1 and 3
      from the press path entirely — latency becomes "unhide a window",
      single-digit ms. Needs: a control channel (the level file already acts
      as one signal), hidden-window lifecycle, and a guarantee the overlay
      cannot outlive the daemon.
- [ ] **Pre-warm on first hotkey registration** — weaker version of the
      above; still pays process startup, just earlier. Fallback if a
      persistent process proves unstable.
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

## Notes

- Push straight to master, no branch or PR.
- whisper.cpp ships no Vulkan/SYCL/OpenVINO binaries, so integrated GPUs
  cannot be accelerated; non-NVIDIA machines run on CPU by design.
