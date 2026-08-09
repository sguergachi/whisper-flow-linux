# Neural speech enhancement evaluation (2026-08)

Question: can an advanced noise-suppression model recover whispers that
Whisper.cpp misses, in rooms with music / nature ambience?

## Setup

Six real failed captures (all verified: mic levels normal, whisper-lift
path amplified the bed into `[MUSIC]`/`[SOUND]` tags). Each enhanced with:

  * **DTLN** (Westhausen, MIT) — 16kHz ONNX, ~1M params, DNS-Challenge
    PESQ 3.04. ONNX runtime, frame-wise with carried LSTM state.
  * **DeepFilterNet 3** (Rikorose, MIT/Apache-2.0) — 48kHz, real-time CPU,
    pre-compiled `deep-filter` binary, sweep of `--atten-lim-db` 4..100.
  * **RNNoise** (Xiph, BSD-3) — could not be evaluated: no PyPI wheel and
    no C compiler on the test machine.

Baseline: the app's plain peak boost (HPF + 0.89 peak-normalise) of the
clean trimmed capture.

## Results

| capture      | plain boost (shipped) | DTLN        | DTLN+boost | DeepFilterNet |
|--------------|-----------------------|-------------|------------|---------------|
| 13:06:14     | "Keeps you going."     | "Keeps."    | "Keeps."   | [BLANK_AUDIO] |
| 11:51        | blank                  | blank       | blank      | [BLANK_AUDIO] |
| 15:09:24     | blank (music bed)      | blank       | "Bye. Bye." (filler halluc.) | blank |
| 12:38        | blank                  | blank       | [INAUDIBLE]| blank |
| 11:48        | blank                  | blank       | [INAUDIBLE]| [BLANK_AUDIO] |
| 13:27        | blank                  | blank       | blank      | [BLANK_AUDIO] |

Sanity check: **DeepFilterNet turns clean quiet speech (jfk at peak 1000,
which Whisper transcribes perfectly) into [BLANK_AUDIO].** DTLN shifts the
spectrum to 65% energy above 3kHz, stripping the 300-3000Hz band where
intelligibility lives, and over-suppresses whisper-level input.

## Conclusion

The neural enhancers are trained on normal-level speech. They treat
whisper-level input as noise, gate it out, and (on music beds) leave
filler hallucinations ("Bye. Bye. Bye.") that the app would paste. The
shipped plain boost already outperforms all three on every capture, and
the rescue chain (plain boost → whisper-lift → boost → spectral rescue)
is the right escalation order.

Do not add a neural enhancer as a first-pass or universal step. If one is
ever added, it belongs *last* in the rescue chain, gated by a setting,
and only for the café-music case — with the DTLN "blank output" failure
mode checked before any paste.

## Whisper.cpp server flags: tested, both are dead ends

`--vad` (Silero) and `--suppress-nst` were also measured against the same
captures (server v1.9.1, ggml-silero-v5.1.2 VAD, threshold sweep 0.10-0.50).

* **`--vad`**: Silero VAD does not register whisper-level speech. The one
  recoverable capture (13:06:14, "Keeps you going." via plain boost) comes
  back empty at every threshold, while music beds still leak "Thank you."
  fillers at the low thresholds that would let a whisper through. It
  removes exactly the clips this app is built to recover.
* **`--suppress-nst`**: worse than nothing. It replaces the bracketed
  tags - `[APPLAUSE]`, `[MUSIC PLAYING]`, `[ Inaudible ]` - with word
  fillers: "Thank you.", "I'm sorry, I'm sorry.", "Bye, guys. Bye. Bye."
  The bracketed tags are caught and rejected by the app's
  `is_hallucination`; the word fillers look like dictation and would be
  pasted. 13:06:14 goes from "(muffled voices)" (rejected) to
  "I'm sorry, I'm sorry." (pasted).

Leave both flags off. The app's own webrtc VAD and the boost chain
already do the only thing that works: keep the whisper's energy and gate
on the client, then reject any all-bracket decode instead of pasting it.

## Spectral gating (noisereduce): works, but beats nothing shipped

The one family left untested was classical DSP spectral gating - estimate
the stationary noise spectrum, then soft-mask each time-frequency bin.
Measured with `noisereduce` (stationary=True, prop_decrease 0.70-0.99,
n_fft 256-1024) layered with the shipped HPF and boost, plus a
gate-then-rescue ordering, on all six captures:

* 13:06:14 (the recoverable whisper): gate+boost returns real words
  ("Keeps you, Doctor." 3/3, "Keeps you, cop truckers." through
  gate>rescue) where the *café rescue* returns blank - but the shipped
  plain-boost-of-raw (`_retry_boost_raw`) already returns the correct
  "Keeps you going." earlier in the chain. The gate recovers nothing the
  chain does not.
* Every buried capture (11:48, 11:51, 12:38, 13:27, 15:09:24): gate
  produces only tags or blank at every setting - never words, and never a
  hallucination that would be pasted. The whisper-band (300-3400 Hz)
  energy is 0.0 dB above the rest of the spectrum in these clips: the
  voice's band carries no measurable advantage to extract.

Cost: ~35ms on a 2s clip (2% of real time) in pure numpy - cheap enough
for the final pass. It does no harm (worst case is a tag the app already
rejects), but it is not an improvement over the shipped chain on any
capture, so it is not integrated.

The complete answer is empirical: DTLN, DeepFilterNet, Silero VAD,
--suppress-nst, spectral gating, whisper-lift, and the café rescue have
all been measured against real captures. The shipped chain - plain boost
of raw with cold-open silence skipped, then the escalation rungs - is the
best available for whisper-under-music, and captures where the voice's
band shows no energy advantage over the bed are unrecoverable by any
filtering.
