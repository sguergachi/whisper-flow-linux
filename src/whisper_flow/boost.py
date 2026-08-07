"""Rescuing a recording that was too quiet - or too buried - to transcribe.

Two cases look the same to Whisper (blank text) and different on the meter:

  * A whisper in a quiet room: peak ~100/32767. Amplify the whole clip.
  * A whisper in a café: peak can be thousands (music) while the voice is
    still small. Whole-clip gain does nothing useful; speech-band rescue
    from denoise does.

Nothing here invents signal. It only re-scales and re-filters what was
captured.
"""

import wave
from pathlib import Path

import numpy as np

from . import denoise
from .logging import log

# Peak of a recording that has no useful signal at all - a muted input or a
# dead device. Below this, amplifying only produces louder noise.
DEAD_PEAK = 30

# Amplify towards this fraction of full scale. Short of 1.0 so that a
# transient does not clip flat.
TARGET_PEAK = 0.89

# Ceiling on the gain. A recording whose peak is 100 needs about 290x to
# reach the target; beyond this the noise floor arrives with the speech.
MAX_GAIN = 300.0

# Speech that matters starts well above this. Removing what is below it
# takes out mains hum, desk rumble and any DC offset, all of which would
# otherwise be amplified alongside the voice.
HIGHPASS_HZ = 80.0


def needs_boost(peak: int) -> bool:
    """Whether a recording is quiet enough to be worth retrying louder."""
    return DEAD_PEAK <= peak < int(TARGET_PEAK * 32767 * 0.25)


def _highpass(samples: np.ndarray, rate: int) -> np.ndarray:
    """One-pole high-pass, to stop gain amplifying offset and rumble."""
    filtered, _ = denoise.high_pass(samples, rate, cutoff_hz=HIGHPASS_HZ)
    return filtered


def _write_wav(path: str, samples: np.ndarray, rate: int,
               channels: int = 1, width: int = 2) -> None:
    with wave.open(path, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def boost_wav(path: str, out_path: str) -> float | None:
    """Write an amplified copy of a wav. Returns the gain, or None.

    None means there was nothing worth doing: no signal to amplify, or the
    recording was already loud enough that being quiet is not the reason
    transcription failed.
    """
    try:
        with wave.open(path, "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            frames = source.readframes(source.getnframes())

        if width != 2:
            return None                      # only 16-bit is ever recorded here
        samples = np.frombuffer(frames, dtype=np.int16)
        if samples.size == 0:
            return None

        peak = int(np.abs(samples).max())
        if peak < DEAD_PEAK:
            log(f"[BOOST] peak {peak} is below the noise floor; nothing to lift")
            return None
        # Café music can already be loud while the whisper is buried. Peak
        # boost would only turn the music up; speech-band rescue handles that.
        if not needs_boost(peak):
            return None

        filtered = _highpass(samples, rate)
        filtered_peak = float(np.abs(filtered).max())
        if filtered_peak < 1.0:
            return None

        gain = min(MAX_GAIN, TARGET_PEAK * 32767.0 / filtered_peak)
        if gain <= 1.05:
            return None                      # already as loud as it needs to be

        louder = np.clip(filtered * gain, -32768, 32767).astype(np.int16)
        _write_wav(out_path, louder, rate, channels=channels, width=width)

        log(f"[BOOST] amplified {gain:.0f}x (peak {peak} -> "
            f"{int(np.abs(louder).max())})")
        return gain
    except Exception as e:
        log(f"[BOOST] could not amplify: {e}")
        return None


def rescue_wav(path: str, out_path: str,
               noise_ref_path: str | None = None,
               floor: float | None = None,
               gate_threshold: float | None = None) -> float | None:
    """Café/whisper rescue: speech-band clean + speech-energy normalise.

    Used when a blank transcript came back and simple peak boost either
    refused (music already loud) or still failed. Returns a pseudo-gain
    (speech-frame lift) for logging, or None if nothing changed.
    """
    try:
        with wave.open(path, "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            frames = source.readframes(source.getnframes())

        if width != 2:
            return None
        samples = np.frombuffer(frames, dtype=np.int16)
        if samples.size == 0:
            return None

        peak_before = int(np.abs(samples).max())
        if peak_before < DEAD_PEAK:
            log(f"[RESCUE] peak {peak_before} is dead; nothing to lift")
            return None

        noise_ref = None
        if noise_ref_path and Path(noise_ref_path).is_file():
            try:
                with wave.open(noise_ref_path, "rb") as nf:
                    ref = np.frombuffer(
                        nf.readframes(nf.getnframes()), dtype=np.int16)
                # Use opening pre-roll of untrimmed capture as noise profile.
                pre = min(ref.size, max(1, int(rate * denoise.PRE_ROLL_MS / 1000)))
                noise_ref = ref[:pre]
            except Exception as e:
                log(f"[RESCUE] noise ref skipped: {e}")

        rescued = denoise.rescue_samples(
            samples, rate,
            floor=floor,
            noise_ref=noise_ref,
            gate_threshold=gate_threshold,
        )
        peak_after = int(np.abs(rescued).max())
        # Report speech lift relative to input peak (can be <1 if we cut bass).
        gain = (peak_after / peak_before) if peak_before else 1.0
        _write_wav(out_path, rescued, rate, channels=channels, width=width)
        log(f"[RESCUE] speech-band path "
            f"(peak {peak_before} -> {peak_after}, ratio {gain:.2f})")
        return max(gain, 1.0)
    except Exception as e:
        log(f"[RESCUE] could not process: {e}")
        return None
