"""Exercise the packaged worker twice: one process, one model load."""
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from whisper_flow.resident import ResidentWorker


def main():
    config = SimpleNamespace(language="en", beam_size=1, best_of=2,
                             fast_encoder=False, no_speech_thold=0.6)
    with tempfile.TemporaryDirectory() as directory:
        worker = ResidentWorker(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(),
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
            assert log.count("whisper_init_from_file_with_params_no_state: loading model") == 1, log
        finally:
            worker.stop()
        assert worker.process.poll() is not None


if __name__ == "__main__":
    main()
