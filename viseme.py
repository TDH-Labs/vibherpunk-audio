"""Phoneme-timed viseme tracks for the Vibherpunk voice client (round 2).

Round 1 drove the client's mouth overlay from audio amplitude (open amount
proportional to loudness). Round 2 replaces the drive with mouth SHAPES
timed to the phonemes being spoken:

    sentence text -> espeak-ng phonemes -> class-weighted proportional
    allocation over the sentence's exact (post-trim) audio duration ->
    timed viseme cues (Rhubarb/Preston-Blair letters A-H, X) -> Pipecat
    data channel -> client renders open/width/round mouth shapes.

Why this approach (measured 2026-11-05, see delegation
vibherpunk-lipsync-visemes-20261105 research notes):
  * Pocket TTS exposes no timing output at all (source-verified:
    PocketTTSService has no word/phoneme timestamp support), so the track
    must be derived.
  * Rhubarb Lip Sync derives timings from the audio but needs ~1.2x the
    audio's duration in CPU per sentence (x86 binary under Rosetta) —
    cues would land after the sentence finished playing. Rejected.
  * Whisper word timestamps are fast enough warm (~0.25 s) but their
    word-start error vs the text-derived track is the same ~100-200 ms
    as the track's own error — no measured gain for a second live model.
  * espeak-ng runs in ~40 ms from the KNOWN text, before/regardless of
    audio, and the phoneme (hence shape) sequence is exact. Bilabial
    closure timing vs Rhubarb's audio-derived reference: mean |err|
    125 ms across 24 closures in the sample set.

The tap is zero-latency: frames pass through untouched; the cue track is
computed in an executor when the sentence's audio completes and pushed
over the data channel keyed by sentence order. The client joins the
track at its measured playback position; sentences pipelined behind the
currently-playing one are fully covered. If no track arrives for a
sentence, the client keeps the round-1 amplitude drive as fallback.
"""

import asyncio
import subprocess
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

ESPEAK = "/opt/homebrew/bin/espeak-ng"
if not Path(ESPEAK).exists():
    ESPEAK = "espeak-ng"

# espeak-ng mnemonic phonemes, longest first for greedy tokenizing.
_MULTI = ["tS", "dZ", "aI", "aU", "OI", "eI", "@U", "I@", "e@", "U@",
          "i:", "u:", "A:", "O:", "3:"]
_SINGLE = set("pbtdkgfvszhTDmnNlrwjIEeVQ@aiuoUZSRx")


def _weight(ph: str) -> float:
    """Relative duration of a phoneme class (proportional allocation)."""
    if ph in ("p", "b", "t", "d", "k", "g"):
        return 0.70  # stops
    if ph in ("m", "n", "N"):
        return 0.90  # nasals
    if ph in ("f", "v", "s", "z", "S", "Z", "T", "D", "h", "x"):
        return 1.00  # fricatives
    if ph in ("tS", "dZ"):
        return 1.20  # affricates
    if ph in ("r", "l", "w", "j"):
        return 0.90  # approximants
    if ph in ("i:", "u:", "A:", "O:", "3:"):
        return 1.45  # long vowels
    if ph in ("aI", "aU", "OI", "eI", "@U", "I@", "e@", "U@"):
        return 1.50  # diphthongs
    return 0.80  # short vowels / schwa


def _viseme(ph: str) -> str:
    """Phoneme -> Rhubarb/Preston-Blair mouth-shape letter."""
    if ph in ("p", "b", "m"):
        return "A"  # lips closed (bilabials)
    if ph in ("f", "v"):
        return "G"  # teeth on lower lip
    if ph in ("u:", "U", "w", "@U", "aU"):
        return "F"  # tight round (oo)
    if ph in ("O:", "OI"):
        return "E"  # round open (oh)
    if ph in ("A:", "aI"):
        return "D"  # wide open (ah)
    if ph in ("i:", "I", "eI", "E"):
        return "C"  # wide (ee/eh)
    if ph == "l":
        return "H"  # tongue tip up
    return "B"  # slight open: everything else


def phonemes_for(text: str) -> list[list[str]]:
    """Text -> words of espeak-ng phoneme mnemonics (en)."""
    out = subprocess.run(
        [ESPEAK, "-xq", "--pho", "-v", "en", text],
        capture_output=True, text=True, timeout=5,
    ).stdout.strip()
    words = []
    for raw in out.split():
        phs, i, s = [], 0, raw.replace("'", "").replace(",", "")
        while i < len(s):
            if s[i:i + 2] in _MULTI:
                phs.append(s[i:i + 2])
                i += 2
            elif s[i] in _SINGLE:
                phs.append(s[i])
                i += 1
            else:
                i += 1  # stress marks, length debris
        phs = ["3:" if p == "3" else p for p in phs]
        if phs:
            words.append(phs)
    return words


def build_track(text: str, duration: float) -> dict | None:
    """Timed viseme cue track for one sentence's audio.

    Returns {"dur": seconds, "cues": [{"s","e","v"}...]} with consecutive
    identical shapes merged, or None if the text yields no phonemes.
    """
    words = phonemes_for(text)
    flat = [(wi, ph) for wi, w in enumerate(words) for ph in w]
    if not flat or duration <= 0:
        return None
    total_w = sum(_weight(ph) for _, ph in flat) + 0.35 * max(0, len(words) - 1)
    cues, t, prev_wi = [], 0.0, 0
    for wi, ph in flat:
        if wi != prev_wi:
            t += duration * 0.35 / total_w
            # word gap: brief near-rest shape so the mouth settles
            cues.append({"s": round(t - duration * 0.35 / total_w, 3),
                         "e": round(t, 3), "v": "X"})
            prev_wi = wi
        d = duration * _weight(ph) / total_w
        cues.append({"s": round(t, 3), "e": round(t + d, 3), "v": _viseme(ph)})
        t += d
    merged: list[dict] = []
    for c in cues:
        if merged and merged[-1]["v"] == c["v"] and abs(merged[-1]["e"] - c["s"]) < 0.002:
            merged[-1]["e"] = c["e"]
        else:
            merged.append(dict(c))
    return {"dur": round(duration, 3), "cues": merged}


class VisemeTap(FrameProcessor):
    """After the TTS + leading-silence trimmer: pairs each spoken
    sentence's text (TTSTextFrame) with its shipped audio (byte count
    between TTS start/stop), builds the viseme track off-loop, and pushes
    it to the client over the data channel via `notify` (dict) -> None.

    Frames are pushed downstream BEFORE any bookkeeping, and all compute
    happens in an executor: the audio path gains zero latency.
    """

    def __init__(self, notify=None, **kwargs):
        super().__init__(**kwargs)
        self._notify = notify
        self._texts: list[str] = []
        self._audio_bytes = 0
        self._sample_rate = 24000
        self._seq = 0

    def _reset(self):
        self._texts = []
        self._audio_bytes = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, InterruptionFrame):
            self._reset()
        elif isinstance(frame, TTSTextFrame):
            if frame.text and frame.text.strip():
                self._texts.append(frame.text.strip())
        elif isinstance(frame, TTSAudioRawFrame):
            self._audio_bytes += len(frame.audio)
            if frame.sample_rate:
                self._sample_rate = frame.sample_rate
        elif isinstance(frame, TTSStoppedFrame):
            audio_bytes, self._audio_bytes = self._audio_bytes, 0
            texts, self._texts = self._texts, []
            if not texts or audio_bytes <= 0:
                return
            # The sentence frame is the longest text in this utterance
            # window (word-level frames, if any, are fragments of it).
            text = max(texts, key=len)
            duration = audio_bytes / 2 / self._sample_rate  # int16 mono
            if duration < 0.05:
                return
            self._seq += 1
            seq = self._seq
            loop = asyncio.get_running_loop()
            try:
                track = await loop.run_in_executor(None, build_track, text, duration)
            except Exception as e:
                logger.warning(f"viseme track failed (seq {seq}): {e}")
                return
            if track is None or self._notify is None:
                return
            try:
                self._notify({
                    "type": "viseme",
                    "seq": seq,
                    "dur": track["dur"],
                    "text": text[:120],
                    "cues": track["cues"],
                })
                logger.debug(
                    f"viseme track seq={seq} cues={len(track['cues'])} "
                    f"dur={track['dur']:.2f}s text={text[:60]!r}"
                )
            except Exception as e:
                logger.warning(f"viseme notify failed (seq {seq}): {e}")
