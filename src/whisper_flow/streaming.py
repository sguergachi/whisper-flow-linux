"""Live transcription: emit text while the user is still speaking.

Whisper is not a streaming model. Each pass sees the whole utterance so far and
is free to revise words it produced earlier - "to" becomes "two" once the next
clause arrives. Typing every pass verbatim would therefore type text that later
turns out wrong, and keystrokes cannot be recalled.

So text is committed on agreement (the LocalAgreement-2 rule): a word is typed
only once two consecutive passes have produced it at the same position. A word
that survives seeing more audio is one whisper has stopped revising. The cost is
latency of roughly one pass interval; the benefit is that committed text is as
good as the final transcript, so live mode does not trade away accuracy.
"""

import os
import tempfile
import threading
import time
import wave

from .logging import log

# Trailing words carry no right-hand context yet, so they are the ones whisper
# keeps re-punctuating. Committing everything except the last one costs a little
# latency and removes most mid-sentence punctuation artifacts.
HOLD_BACK_WORDS = 1


def _common_prefix(a: list[str], b: list[str]) -> int:
    """Number of leading words the two hypotheses agree on."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class LiveTranscriber:
    """Runs transcription passes alongside recording and emits stable text."""

    def __init__(self, transcribe, emit, sample_rate: int, interval: float = 0.9):
        """Initialize the live transcriber.

        Args:
            transcribe: Callable taking a wav path and returning text or None
            emit: Callable taking the newly committed text to type
            sample_rate: Sample rate of the incoming frames
            interval: Minimum seconds between passes

        """
        self._transcribe = transcribe
        self._emit = emit
        self._sample_rate = sample_rate
        self._interval = interval

        self._pending = None  # latest frame snapshot awaiting a pass
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = False
        self._thread = None

        self._prev: list[str] = []       # previous pass's full hypothesis
        self._committed: list[str] = []  # words already typed
        self._emitted_any = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="whisper-flow-live",
        )
        self._thread.start()

    def offer(self, frames: list):
        """Hand over the latest audio. Called from the capture loop; returns at once."""
        with self._lock:
            self._pending = frames
        self._wake.set()

    def _loop(self):
        while self._running:
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            if not self._running:
                return

            with self._lock:
                frames, self._pending = self._pending, None
            if not frames:
                continue

            started = time.monotonic()
            text = self._run_pass(frames)
            if text is not None:
                self._commit(text)

            # Never queue up behind ourselves: if a pass took longer than the
            # interval, the next snapshot simply replaces the stale one.
            remaining = self._interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)

    def _run_pass(self, frames: list) -> str | None:
        path = None
        try:
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="whisper-flow-live-")
            os.close(fd)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(b"".join(frames))
            return self._transcribe(path)
        except Exception as e:
            log(f"[LIVE] pass failed: {e}")
            return None
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _commit(self, text: str):
        words = text.split()
        agreed = _common_prefix(self._prev, words)
        self._prev = words

        # Hold the trailing word back even when it agrees. A word at the end of
        # a hypothesis has no right-hand context yet, so it is where whisper
        # still adds or changes punctuation ("you." becoming "you,"). One more
        # word of audio settles it, at the cost of one interval of latency.
        agreed = min(agreed, max(0, len(words) - HOLD_BACK_WORDS))

        if agreed > len(self._committed):
            new = words[len(self._committed):agreed]
            self._committed = words[:agreed]
            self._send(new)

    def _send(self, words: list[str]):
        if not words:
            return
        chunk = " ".join(words)
        if self._emitted_any:
            chunk = " " + chunk
        self._emitted_any = True
        try:
            self._emit(chunk)
        except Exception as e:
            log(f"[LIVE] emit failed: {e}")

    def finalize(self, final_text: str | None) -> None:
        """Emit whatever the final transcript holds beyond what was committed.

        The final pass sees the complete utterance, so it supersedes the
        running hypothesis - only its tail is still unspoken for.
        """
        self._running = False
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        if not final_text:
            return
        words = final_text.split()
        if len(words) > len(self._committed):
            self._send(words[len(self._committed):])

    @property
    def committed_text(self) -> str:
        return " ".join(self._committed)

    def stop(self):
        """Tear down without emitting anything further."""
        self._running = False
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
