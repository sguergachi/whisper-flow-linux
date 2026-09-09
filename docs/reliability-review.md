# Windows and Linux reliability review

The September 2026 review followed the recovery changes in `6328df4`
(port fallback), `fbbbcf5` (background warmup), `134f411` (no downloads during
recording), and `31ffe06` (engine working directory), plus the earlier
configuration reload work.

Fixed failure paths:

- A startup timeout left a live child behind and forgot its selected port.
  The next start could falsely succeed at the configured port. Failed starts
  now terminate and reap the child; readiness is tracked separately from
  process liveness. Shutdown also waits after a forced kill.
- Preflight repair could quarantine an engine after its command had already
  been built. Commands, working directories and Linux library paths are now
  selected after repair. CPU thread counts follow the engine actually used,
  including CPU fallback on a machine with a GPU.
- GPU fallback and crash repair could download despite `allow_download=False`.
  Both now honor it. The complete fallback sequence shares the lifecycle lock
  so another starter cannot interleave with engine quarantine.
- Live transcription could synchronously wait for model recovery. Live failures
  now schedule one background warmup; final transcription retains its recovery
  and CLI fallback. Repeated live failures cannot queue multiple warmup workers.
- Explicit configuration directories selected where Save wrote, but did not
  select the file read at startup. Each Config now resolves its own `.env`;
  settings reload retains the same directory, including after the first save.

Readiness probes use `/health` and reject server errors. A 404 is retained for
older engines without that endpoint. The upstream endpoint distinguishes ready
from loading:
https://github.com/ggml-org/whisper.cpp/blob/master/examples/server/server.cpp

Regression coverage includes failed-start retries on an alternate port,
forced-kill reaping, repaired binary selection, readiness transitions,
HTTP 500/503 retries, forbidden downloads, coalesced background recovery,
and configuration save/reload isolation. GTK tests exercise the real settings
window in a separate interpreter. Native Windows tests run in the Windows CI
job; physical microphone, keyboard-hook and GPU-driver behavior still requires
validation on the affected hardware.
