"""A resident whisper.cpp worker over private process pipes, with no listener.

The native worker owns the model and GPU context for its entire lifetime.
Requests serialize; native crashes and stalled pipe I/O cannot kill or hang
the tray. Audio stays in memory and only the child's stderr reaches disk.
"""
from __future__ import annotations

import queue
import re
import struct
import subprocess
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class ReferenceDecodeError(RuntimeError):
    """The process runs but does not correctly decode known speech."""


REFERENCE_TEXT = ("and so my fellow americans ask not what your country can do for you "
                  "ask what you can do for your country")


def reference_error_rate(text: str) -> float:
    """Word edit distance; punctuation/case are irrelevant to the check."""
    expected = REFERENCE_TEXT.split()
    actual = re.findall(r"[a-z]+", text.lower())
    previous = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        row = [i]
        for j, received in enumerate(actual, 1):
            row.append(min(row[-1] + 1, previous[j] + 1,
                           previous[j - 1] + (word != received)))
        previous = row
    return previous[-1] / len(expected)


class ResidentWorker:
    def __init__(self, exe: Path, model: Path, threads: int, gpu: bool,
                 log_path: Path, creationflags: int = 0, adopt=None,
                 flash_attn: bool = True, env=None):
        self.exe, self.model, self.gpu = Path(exe), Path(model), gpu
        self._lock = threading.Lock()
        self._responses = queue.Queue()
        self.process = None
        self._stderr = None
        self.ready = False
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._stderr = open(log_path, "ab")
            self.process = subprocess.Popen(
                [str(self.exe), str(self.model), str(threads), "1" if gpu else "0",
                 "1" if flash_attn else "0"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                cwd=str(self.exe.parent), creationflags=creationflags, bufsize=0, env=env,
            )
            if adopt:
                adopt(self.process)
            threading.Thread(target=self._read_responses, daemon=True,
                             name="whisper-flow-worker-read").start()
            response = self._response(180)
            if response != ("READY GPU" if gpu else "READY CPU"):
                raise RuntimeError(f"Unexpected worker startup: {response}")
            self.ready = True
        except BaseException:
            self.stop()
            raise

    @property
    def alive(self) -> bool:
        return bool(self.ready and self.process and self.process.poll() is None)

    def verify_reference(self, audio_path: Path) -> list[float]:
        """Check real speech twice, after warmup, in the same GPU context.

        Silence can execute corrupt GPU kernels without exposing incorrect
        output. The bundled public reference bypasses microphones, gain,
        denoising and live prompts so those cannot mask a numerical failure.
        """
        config = SimpleNamespace(language="en", beam_size=1, best_of=2,
                                 fast_encoder=False, no_speech_thold=0.6)
        errors = []
        for _ in range(2):
            text = self.transcribe(audio_path, config, timeout=90)
            error = reference_error_rate(text)
            if error > 0.25:
                raise ReferenceDecodeError(
                    f"Known-speech check failed ({error:.0%} word error): {text[:240]!r}")
            errors.append(error)
        return errors

    @staticmethod
    def _read_exact(stream, length: int) -> bytes:
        parts = []
        while length:
            part = stream.read(length)
            if not part:
                raise RuntimeError("Speech worker closed its response pipe")
            parts.append(part)
            length -= len(part)
        return b"".join(parts)

    def _read_responses(self):
        try:
            while True:
                header = self._read_exact(self.process.stdout, 12)
                magic, size, status = struct.unpack("<4sII", header)
                if magic != b"WFR1" or size > 4 * 1024 * 1024:
                    raise RuntimeError("Invalid speech worker response")
                text = self._read_exact(self.process.stdout, size).decode("utf-8")
                self._responses.put(RuntimeError(text) if status else text)
        except Exception as e:
            self._responses.put(e)

    def _response(self, timeout: float) -> str:
        try:
            result = self._responses.get(timeout=max(0, timeout))
        except queue.Empty:
            raise TimeoutError("Speech worker timed out") from None
        if isinstance(result, Exception):
            raise result
        return result

    @staticmethod
    def _request(audio_path, config, prompt=None, temperature=None) -> bytes:
        with wave.open(str(audio_path), "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("Speech worker requires mono 16 kHz PCM16 WAV audio")
            if wav.getnframes() > 16000 * 600:
                raise ValueError("Speech worker accepts at most ten minutes of audio")
            raw = wav.readframes(wav.getnframes())
        pcm = (np.frombuffer(raw, dtype="<i2").astype("<f4") / 32768.0).astype("<f4")
        if not len(pcm):
            raise ValueError("Empty audio recording")
        language = (getattr(config, "language", None) or "en").encode("utf-8")
        prompt = (prompt or "").encode("utf-8")
        if len(language) > 32 or len(prompt) > 65536:
            raise ValueError("Speech language or prompt is too long")
        context = 0
        if getattr(config, "fast_encoder", False):
            from .transcription import audio_context
            context = audio_context(len(pcm) / 16000) or 0
        header = struct.pack(
            "<4s7I2f", b"WFW1", len(pcm), getattr(config, "beam_size", 1),
            getattr(config, "best_of", 2), bool(getattr(config, "suppress_nst", False)),
            context, len(language), len(prompt), temperature or 0.0,
            getattr(config, "no_speech_thold", 0.6),
        )
        return header + language + prompt + pcm.tobytes()

    def transcribe(self, audio_path, config, timeout=180, prompt=None,
                   temperature=None) -> str:
        deadline = time.monotonic() + timeout
        request = self._request(audio_path, config, prompt, temperature)
        if not self._lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise TimeoutError("Speech worker is busy")
        try:
            if not self.alive:
                raise RuntimeError("Speech worker is not running")
            # A frozen child can block a pipe write. Keep that write off the
            # caller, so the same deadline covers sending PCM and decoding it.
            def write():
                try:
                    data = memoryview(request)
                    while data:
                        n = self.process.stdin.write(data)
                        if not n:
                            raise RuntimeError("Speech worker closed its input pipe")
                        data = data[n:]
                except Exception as e:
                    self._responses.put(e)
            threading.Thread(target=write, daemon=True,
                             name="whisper-flow-worker-write").start()
            return self._response(deadline - time.monotonic()).strip()
        except Exception:
            # Never let a late reply to a timed-out request become the next
            # recording's transcript. A failed exchange invalidates the child.
            self.stop()
            raise
        finally:
            self._lock.release()

    def stop(self):
        self.ready = False
        try:
            if self.process:
                if self.process.poll() is None:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass
                self.process.wait(timeout=5)
        finally:
            if self.process:
                for stream in (self.process.stdin, self.process.stdout):
                    if stream:
                        stream.close()
            if self._stderr:
                self._stderr.close()
