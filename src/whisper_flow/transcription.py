"""Audio transcription functionality for whisper-flow."""

import math
import re
import time
import wave
from pathlib import Path

import requests

from .config import Config
from .logging import log


def _normalize(text: str | None) -> str:
    """Collapse whisper's per-segment line breaks into a single dictated line.

    Whisper splits its output on segment boundaries, which land mid-sentence.
    Pasting those raw drops line breaks into the middle of what the user said.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


# Non-speech tags Whisper invents on over-amplified room noise / music beds.
# Treat as blank so the caller can retry milder processing. The greedy pass
# labels a voice-over-music clip "[MUSIC]", and a re-roll at higher
# temperature often recovers the words - so every tag must be caught here or
# the retry pastes "*(laughing)*" as if it were dictation.
_HALLUCINATION = re.compile(
    r"^[\s\*\[\(\{\"']*"
    r"(blank[_\s-]?audio|music|sad music|upbeat music|music playing|"
    r"applause|applaud|applauding|laughter|laughing|laughs|chuckling|"
    r"giggling|mumbling|mumbles|muffled|muffled voices|muffled speaking|"
    r"chattering|chatter|whistling|singing|humming|clapping|clicking|"
    r"beeping|tones|tone|feedback|static|noise|background noise|"
    r"off-mic|off mic|conversation|crowd|audience|cheering|cheers|"
    r"silence|inaudible|inaudible\.|coughing|cough|breathing|sigh|"
    r"sniffing|speaking in foreign language|foreign language|"
    r"thank you\.?|thanks for watching\.?|please subscribe( to my channel)?"
    r"( for more)?\.?!?|subscribe|you$)"
    r"[\s\*\]\)\}\"'.,!?]*$",
    re.IGNORECASE,
)


def is_hallucination(text: str | None) -> bool:
    """True when the transcript is a non-speech / junk decode, not dictation."""
    if not text:
        return True
    t = text.strip()
    if not t:
        return True
    if _HALLUCINATION.match(t):
        return True
    # Words outside the brackets are dictation: "Keeps you going. [INAUDIBLE]"
    # is words plus one trailing tag, and the words must be kept. A transcript
    # that is entirely tags - *(laughing)*, [Music], (muffled voices),
    # [SOUND], [CLICK] [INAUDIBLE] - is never dictation, whatever the tag
    # says. Whisper invents unbounded tag words for nature beds ([birdsong],
    # [wind], [SOUND]...), so no keyword list can close the set.
    outside = re.sub(r"[\*\[\(\{][^\]\)\}]*[\*\]\)\}]", " ", t)
    if re.search(r"[A-Za-z]", outside):
        return False
    return True


# A recording can run to the configured maximum, and a CPU-only machine
# transcribes it slower than real time, so the closing pass gets room.
FINAL_TIMEOUT = 300.0

# A live pass is best-effort and superseded by the next one a second later,
# so it must never be the reason dictation appears to hang. At five minutes
# with three retries, one unresponsive server could stall a pass for a
# quarter of an hour while the overlay sat there animating.
LIVE_TIMEOUT = 30.0

# Whisper pads every clip out to a full 30 second window before encoding it,
# so a three second dictation costs almost exactly what an eleven second one
# does - 785ms against 945ms, measured on a six-core desktop. Nearly all of
# that is spent encoding silence that was never recorded.
#
# `audio_ctx` shortens the window, and the encoder time falls with it: 1.7 to
# 1.9x end to end through the server. The floor below is not about covering
# the speech, and it is not a guarantee. Truncating the window truncates the
# position embeddings with it, and the decoder does notice: measured on
# base.en, "ask not what your country can do for you" came back as "asked
# not" on two clips of five, at 768 and at 1024 and on one clip at 1280 too.
# Other clips went the other way and recovered a word the full window had
# dropped, which is the same instability seen from the useful side.
#
# Hence config.fast_encoder defaults to off, and this runs only for someone
# who has asked for it. 768 is where the speed is; it is not where safety is,
# because on this evidence there is no window short enough to be worth
# setting and long enough to be reliably faithful.
AUDIO_CTX_FULL = 1500                       # the whole 30 seconds
AUDIO_CTX_MIN = 768                         # below this the transcript drifts
_CTX_PER_SECOND = AUDIO_CTX_FULL / 30.0
_CTX_STEP = 256                             # whisper.cpp pads audio_ctx to this
_CTX_SLACK_SECONDS = 1.0


def audio_context(duration: float) -> int:
    """The shortest encoder window that still holds `duration` of speech.

    0 means "use the whole window", which is what anything long enough to be
    split across whisper's 30 second chunks needs anyway: the setting applies
    per chunk, so a short window would make every chunk after the first deaf
    to its own second half.
    """
    if duration <= 0 or duration >= 30.0:
        return 0
    needed = math.ceil((duration + _CTX_SLACK_SECONDS) * _CTX_PER_SECOND)
    # whisper.cpp rounds audio_ctx up to a multiple of 256 internally, so
    # asking for 1000 silently gets 1024. Ask for what will be used.
    needed = math.ceil(needed / _CTX_STEP) * _CTX_STEP
    if needed >= AUDIO_CTX_FULL:
        return 0
    return max(AUDIO_CTX_MIN, needed)


def wav_duration(path: str) -> float:
    """Seconds of audio in a WAV file, or 0.0 if it cannot be read."""
    try:
        with wave.open(path, "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() / rate if rate else 0.0
    except Exception:
        return 0.0


class TranscriptionService:
    """Audio transcription against a local whisper.cpp server."""

    def __init__(self, config: Config):
        """Initialize transcription service.

        Args:
            config: Configuration object

        """
        self.config = config
        self.local_url = None

        if config.local_whisper_url:
            self.local_url = config.local_whisper_url.rstrip("/")

    def transcribe_audio(self, audio_path: str, max_retries: int = 3,
                         timeout: float = FINAL_TIMEOUT,
                         prompt: str | None = None,
                         temperature: float | None = None) -> str | None:
        """Transcribe audio file.

        Args:
            audio_path: Path to the recorded audio file
            max_retries: Maximum number of retry attempts
            prompt: Optional initial prompt handed to Whisper, used to steer
                a re-decode away from background music and towards speech
            temperature: Optional decode temperature for the re-decode; the
                normal path stays at the deterministic default

        Returns:
            Transcribed text or None if failed/blank

        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        for attempt in range(max_retries):
            try:
                if not self.local_url:
                    raise RuntimeError(
                        "No whisper server configured. Point WHISPER_FLOW_LOCAL_WHISPER_URL "
                        "at one, or let the app manage one (it does by default).",
                    )
                text = self._transcribe_local(
                    audio_path, timeout, prompt=prompt, temperature=temperature)

                text = _normalize(text)
                if text and text != "[BLANK_AUDIO]" and not is_hallucination(text):
                    return text
                if text and is_hallucination(text):
                    log(f"Hallucination transcript discarded: {text!r}")
                    return None

                log("Blank audio detected, skipping paste")
                return None
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Transcription failed after {max_retries} attempts: {e}",
                    )
                wait_time = 2**attempt
                log(
                    f"Transcription attempt {attempt + 1} failed, retrying in {wait_time}s...",
                )
                time.sleep(wait_time)

        return None

    def _transcribe_local(self, audio_path: str,
                          timeout: float = FINAL_TIMEOUT,
                          prompt: str | None = None,
                          temperature: float | None = None) -> str:
        """Transcribe audio using local whisper.cpp server.

        Args:
            audio_path: Path to the audio file
            prompt: Optional initial prompt for the decoder (steering)
            temperature: Optional decode temperature override

        Returns:
            Transcribed text

        """
        inference_url = f"{self.local_url}/inference"

        # Sized per request rather than once at startup: the server takes
        # these as form fields, so every clip gets a window cut to its own
        # length without the engine being restarted or even told in advance.
        data = {
            # Quiet café whispers often score as "no speech" at the default
            # threshold; a lower bar makes the decoder try harder instead of
            # returning blank. Temperature 0 keeps the first pass deterministic.
            "temperature": "0.0",
            "temperature_inc": "0.2",
            "no_speech_thold": str(self.config.no_speech_thold),
        }
        # A re-decode steered away from background music: the prompt biases
        # the language model towards speech, and a warmer start lets it
        # escape the greedy path that locked onto the music's vocals.
        if temperature is not None:
            data["temperature"] = str(temperature)
        if prompt:
            data["prompt"] = prompt
        # Decode knobs from settings; the defaults match what the app always
        # sent, so a fresh install behaves exactly as before.
        if getattr(self.config, "beam_size", 1) > 1:
            data["beam_size"] = str(self.config.beam_size)
            data["best_of"] = str(getattr(self.config, "best_of", 2))
        if getattr(self.config, "suppress_nst", False):
            data["suppress_nst"] = "true"
        # English-locked installs (common) stay stable under noise; a free
        # language id on café music hallucinates other languages.
        language = getattr(self.config, "language", None) or "en"
        if language and language != "auto":
            data["language"] = str(language)
        if self.config.fast_encoder:
            context = audio_context(wav_duration(audio_path))
            if context:
                data["audio_ctx"] = str(context)

        with open(audio_path, "rb") as audio_file:
            resp = requests.post(
                inference_url,
                files={"file": audio_file},
                data=data,
                timeout=timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("text", "").strip()
