"""Exercise the packaged worker twice: one process, one model load."""
import os
import sys
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from whisper_flow.resident import ResidentWorker


def main():
    config = SimpleNamespace(language="en", beam_size=1, best_of=2,
                             fast_encoder=False, no_speech_thold=0.6)
    exe = Path(sys.argv[1]).resolve()
    if "--isolated-dlls" in sys.argv[4:]:
        assert sys.platform == "win32"
        # Keep the runner's compiler/runtime directories out of the child's
        # search path. The helper must carry its own redistributable DLLs.
        for dll in ("vulkan-1.dll", "libwinpthread-1.dll"):
            assert (exe.parent / dll).is_file(), dll
        os.environ["PATH"] = os.pathsep.join((str(exe.parent),
                                              str(Path(os.environ["SystemRoot"]) / "System32")))
    with tempfile.TemporaryDirectory() as directory:
        model = Path(sys.argv[2]).resolve()
        if "--unicode-path" in sys.argv[4:]:
            target = Path(directory) / "café 漢字" / model.name
            target.parent.mkdir()
            shutil.copyfile(model, target)
            model = target
        worker = ResidentWorker(exe, model,
                                4, "--gpu" in sys.argv[4:], Path(directory) / "worker.log")
        try:
            pid = worker.process.pid
            for i in range(2):
                start = time.monotonic()
                text = worker.transcribe(sys.argv[3], config)
                print(f"pass {i + 1} pid={pid} seconds={time.monotonic() - start:.3f}: {text}")
                assert "country" in text.lower() and "you" in text.lower(), text
                assert worker.process.pid == pid and worker.alive
            log = (Path(directory) / "worker.log").read_text(errors="replace")
            assert log.count("worker: loading model") == 1, log
        finally:
            worker.stop()
        assert worker.process.poll() is not None


if __name__ == "__main__":
    main()
