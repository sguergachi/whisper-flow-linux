"""Guards on live transcription.

Anything this types cannot be taken back, so the rules it has to keep are:
never type a word two passes have not agreed on, and never type anything
after the final transcript has been emitted.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_flow.streaming import LiveTranscriber  # noqa: E402


def _transcriber(typed, transcribe=lambda p: None):
    return LiveTranscriber(
        transcribe=transcribe, emit=typed.append, sample_rate=16000, interval=0.05,
    )


def test_a_word_is_typed_only_once_two_passes_agree():
    """Whisper revises earlier words; typing a guess cannot be undone."""
    typed = []
    lt = _transcriber(typed)
    lt._commit("I want to")          # "to" is still a guess
    assert "".join(typed) == ""
    lt._commit("I want two")         # revised before it was ever typed
    assert "".join(typed) == "I want"


def test_finalize_emits_the_remaining_tail_once():
    typed = []
    lt = _transcriber(typed)
    for hypothesis in ("a b", "a b c", "a b c d"):
        lt._commit(hypothesis)
    lt.finalize("a b c d e")
    assert "".join(typed) == "a b c d e"


def test_a_pass_finishing_after_finalize_types_nothing():
    """The join in finalize has a timeout; a slow pass can outlive it."""
    typed = []
    lt = _transcriber(typed)
    lt._commit("hello there")
    lt.finalize("hello there world")
    before = list(typed)

    lt._commit("hello there world again entirely different")
    assert typed == before, "typed more text after the final transcript"


def test_stop_prevents_any_further_typing():
    typed = []
    lt = _transcriber(typed)
    lt._commit("one two")          # first pass agrees with nothing yet
    lt._commit("one two three")    # now "one two" is settled
    before = "".join(typed)
    assert before, "expected something typed before stop"

    lt.stop()
    lt._commit("one two three four five")
    assert "".join(typed) == before


def test_concurrent_commit_and_finalize_do_not_interleave():
    """The worker thread and the caller both reach the typing path."""
    typed = []
    lt = _transcriber(typed)
    for hypothesis in ("x y", "x y z"):
        lt._commit(hypothesis)

    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            lt._commit("x y z w v u")

    t = threading.Thread(target=hammer, daemon=True)
    t.start()
    time.sleep(0.02)
    lt.finalize("x y z w v u t")
    settled = list(typed)          # nothing may be appended past this point
    stop.set()
    t.join(timeout=2)

    assert typed == settled, "typed more text after the final transcript"
    assert "".join(typed).startswith("x y")


# ---------------------------------------------------- delivery of the words
def test_a_refused_emit_is_noticed_rather_than_ignored():
    """Recording works, the overlay animates, and nothing is typed."""
    live = LiveTranscriber(transcribe=lambda p: None,
                           emit=lambda text: False,      # SendInput refused
                           sample_rate=16000)
    live._send(["hello", "there"])
    assert live.delivery_failed


def test_a_successful_emit_is_not_reported_as_failure():
    live = LiveTranscriber(transcribe=lambda p: None,
                           emit=lambda text: True, sample_rate=16000)
    live._send(["hello"])
    assert not live.delivery_failed


def test_an_emit_returning_none_counts_as_delivered():
    """Most callables return None; only an explicit False means refused."""
    live = LiveTranscriber(transcribe=lambda p: None,
                           emit=lambda text: None, sample_rate=16000)
    live._send(["hello"])
    assert not live.delivery_failed


def test_partial_delivery_is_not_a_total_failure():
    calls = []

    def emit(text):
        calls.append(text)
        return len(calls) > 1          # first refused, second accepted

    live = LiveTranscriber(transcribe=lambda p: None, emit=emit,
                           sample_rate=16000)
    live._send(["one"])
    live._send(["two"])
    assert not live.delivery_failed    # something reached the screen


def test_a_raising_emit_counts_as_a_delivery_failure():
    def emit(text):
        raise OSError("clipboard on fire")

    live = LiveTranscriber(transcribe=lambda p: None, emit=emit,
                           sample_rate=16000)
    live._send(["hello"])
    assert live.delivery_failed


# ------------------------------------------------------------------ timeouts
def test_a_live_pass_uses_one_attempt_and_a_short_timeout():
    """Retrying a live pass only delays the words queued behind it."""
    from unittest.mock import Mock

    from whisper_flow.app import WhisperFlow
    from whisper_flow.transcription import FINAL_TIMEOUT, LIVE_TIMEOUT

    assert LIVE_TIMEOUT < FINAL_TIMEOUT

    app = WhisperFlow.__new__(WhisperFlow)          # no config or devices
    app.transcription_service = Mock()
    app.transcription_service.transcribe_audio.return_value = "hi"

    # Rebuild the closure the live flow passes to LiveTranscriber.
    def live_pass(path):
        return app.transcription_service.transcribe_audio(
            path, max_retries=1, timeout=LIVE_TIMEOUT)

    live_pass("/tmp/x.wav")
    kwargs = app.transcription_service.transcribe_audio.call_args.kwargs
    assert kwargs["max_retries"] == 1
    assert kwargs["timeout"] == LIVE_TIMEOUT
