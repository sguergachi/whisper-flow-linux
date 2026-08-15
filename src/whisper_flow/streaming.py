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
from .transcription import collapse_repetition

# Trailing words carry no right-hand context yet, so they are the ones whisper
# keeps re-punctuating. Committing everything except the last one costs a little
# latency and removes most mid-sentence punctuation artifacts.
HOLD_BACK_WORDS = 1

# The fastest a live pass will be repeated. Below this the gain in latency
# is smaller than the extra load, and on a local server it would mean
# transcribing the same audio continuously.
MIN_PASS_INTERVAL = 0.35


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

    def __init__(self, transcribe, emit, sample_rate: int, interval: float = 0.9,
                 prepare=None):
        """Initialize the live transcriber.

        Args:
            transcribe: Callable taking a wav path and returning text or None
            emit: Callable taking the newly committed text to type
            sample_rate: Sample rate of the incoming frames
            interval: Minimum seconds between passes
            prepare: Optional callable to condition frames before a pass runs,
                used to strip silence. It must be deterministic on a given
                prefix of audio: two passes over the same words have to agree
                before either is typed, so a filter that decided differently
                from one pass to the next would stop text committing at all.

        """
        self._transcribe = transcribe
        self._emit = emit
        self._sample_rate = sample_rate
        self._interval = interval
        self._prepare = prepare

        self._pending = None  # latest frame snapshot awaiting a pass
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = False
        self._thread = None

        self._prev: list[str] = []       # previous pass's full hypothesis
        self._committed: list[str] = []  # words already typed
        self._emitted_any = False
        self._delivered_any = False
        self._delivery_failures = 0
        # Guards the committed state and typing. A pass can outlive the join in
        # finalize - the transcription call has its own, much longer timeout -
        # and without this it would commit on top of the tail already typed,
        # duplicating words at the end of a dictation.
        self._emit_lock = threading.Lock()
        self._closed = False

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

            # Pace to the machine rather than to a fixed number. A word is
            # only typed once two passes agree on it, so latency is about
            # twice the cadence - and waiting a flat 0.9s when a pass takes
            # 0.25s spends most of that interval doing nothing.
            #
            # The cadence is the pass's own duration, floored so a very fast
            # backend does not transcribe continuously, and capped by the
            # configured interval so it can still be slowed deliberately.
            # A machine where a pass takes longer than the cap simply runs
            # back to back, exactly as before: this cannot overload anything
            # that was previously keeping up.
            elapsed = time.monotonic() - started
            cadence = min(self._interval, max(MIN_PASS_INTERVAL, elapsed))
            remaining = cadence - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _run_pass(self, frames: list) -> str | None:
        path = None
        try:
            if self._prepare:
                try:
                    frames = self._prepare(frames) or frames
                except Exception as e:
                    log(f"[LIVE] prepare failed, using raw audio: {e}")
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
        with self._emit_lock:
            if self._closed:
                return  # finalize already had the last word
            self._commit_locked(text)

    def _commit_locked(self, text: str):
        # Collapse before agreement, not after. Two passes that each grew
        # by another copy of the same sentence agree on the extras, and
        # those extras get typed. The closing pass is often a single copy;
        # by then the earlier copies are already in the window.
        text = collapse_repetition(text)
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
            # Committed only once they have actually been typed. It means
            # "the user has these already", and it is what finalize()
            # subtracts to find the tail - so counting a refused delivery as
            # committed dropped those words from the live pass *and* from the
            # closing one. Left uncommitted they are simply offered again on
            # the next pass, and the closing transcript covers them anyway.
            if self._send(new):
                self._committed = words[:agreed]

    def _send(self, words: list[str]) -> bool:
        """Type words, reporting whether they arrived.

        Caller must hold _emit_lock.
        """
        if not words:
            return True
        chunk = " ".join(words)
        if self._emitted_any:
            chunk = " " + chunk
        try:
            # A False return means the text never reached the application.
            # Ignoring it produced the worst failure this has had: the
            # recording works, the overlay animates, and nothing is typed,
            # with nothing logged to say so.
            if self._emit(chunk) is False:
                self._delivery_failures += 1
                log(f"[LIVE] emit reported failure for {len(chunk)} chars")
                return False
        except Exception as e:
            self._delivery_failures += 1
            log(f"[LIVE] emit failed: {e}")
            return False
        # Only on the way out, and only on success: it decides whether the
        # next chunk is given a leading space, and a refused chunk has not
        # put anything on screen for one to separate from.
        self._emitted_any = True
        self._delivered_any = True
        return True

    def quiesce(self, timeout: float = 1.0) -> None:
        """Stop starting new passes, before the closing one is issued.

        whisper.cpp serves one request at a time. Recording ends, the final
        pass is sent, and the live loop - which knows nothing about that - was
        still free to start another pass over the same audio and get in front
        of it. Each one delays the transcript the user is waiting for by its
        own duration, and a pass was seen starting seven seconds after the key
        was released.

        The wait here is short and does not have to succeed: a pass already
        in flight cannot be recalled, and the point is only that no *further*
        pass begins. Unlike stop(), the transcriber stays open, so finalize
        can still type the tail.
        """
        self._running = False
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=timeout)

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

        with self._emit_lock:
            if self._closed:
                return
            self._closed = True
            if not final_text:
                return
            words = collapse_repetition(final_text).split()
            if len(words) > len(self._committed):
                self._send(words[len(self._committed):])
                self._committed = words

    @property
    def delivery_failed(self) -> bool:
        """True when text was produced but none of it reached the screen.

        Distinguishes "nothing was said" from "everything was said and the
        typing was rejected", which look identical to the user otherwise.
        """
        return bool(self._delivery_failures and not self._delivered_any)

    @property
    def committed_text(self) -> str:
        with self._emit_lock:
            return " ".join(self._committed)

    def stop(self):
        """Tear down without emitting anything further."""
        self._running = False
        self._wake.set()
        with self._emit_lock:
            self._closed = True
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
