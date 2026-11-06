"""Sherpa wake-gate proof: drive the PRODUCTION detector
(hermes_voice.WakeWordGate with HERMES_WAKE_ENGINE=sherpa) with the
Oct 27 eval WAVs in /tmp/wakeeval/wav16, under two stream conditions:

- STREAM-START: utterance at the very start of the stream (the prep the
  original eval/prototype used). Asserts: clanker 3/3 fire with phrase
  "hey clanker"; negatives 0/6.
- LIVE-LIKE: 1 s of silence before the utterance (the stream has been
  running, as on a live mic). Asserts: negatives 0/6 (the false-fire
  direction is the dangerous one) and clanker >= 2/3.

Measured finding behind the split (Oct 27): one synthetic clanker
rendition (pos_clanker_1) fires only at stream start; with any stream
history it stays below detection at every threshold 0.05-0.29 and at
per-keyword boosts 2.0/3.0. "hey viber" is printed, NOT asserted —
known weak spot pending a trained model (see RUNBOOK.md).
"""
import os
import sys

os.environ["HERMES_WAKE_ENGINE"] = "sherpa"
os.environ.setdefault("HERMES_VOICE_LOGDIR", "/tmp/hv-sherpa-gate-logs")
sys.path.insert(0, "/Users/ai/workspace/hermes-voice")

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly

import hermes_voice as hv

WAVDIR = "/tmp/wakeeval/wav16"
CLANKER = ["pos_clanker_1.wav", "pos_clanker_2.wav", "pos_clanker_alex.wav"]
VIBER = ["pos_viber_1.wav", "pos_viber_2.wav", "pos_viber_sam.wav"]
NEGATIVES = [
    "neg_hermes.wav",
    "neg_driver.wav",
    "neg_biker.wav",
    "neg_sentence1.wav",
    "neg_sentence2.wav",
    "neg_sentence3.wav",
]
SIL = np.zeros(16000, dtype=np.int16)


class StubSession:
    def __init__(self):
        self.events = []
        self.last_activity = 0.0

    def record(self, kind, **kw):
        self.events.append({"kind": kind, **kw})

    def note_activity(self):
        pass


class StubBridge:
    def pending_count(self):
        return 0


def load16k(path):
    sr, data = scipy.io.wavfile.read(path)
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    else:
        x = data.astype(np.float64)
    if sr != 16000:
        x = resample_poly(x, 16000, sr)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def run_wav(gate, path, head):
    gate._reset_detector()
    gate.last_wake_phrase = None
    pcm = load16k(path)
    audio = np.concatenate([head, pcm, SIL])
    chunk = 1280
    for i in range(0, len(audio) - chunk + 1, chunk):
        if gate._detect(audio[i : i + chunk].tobytes()):
            return True, gate.last_wake_phrase
    return False, gate.last_wake_phrase


def suite(gate, label, head):
    out = {}
    for group, names in (("clanker", CLANKER), ("viber", VIBER), ("neg", NEGATIVES)):
        for name in names:
            fired, phrase = run_wav(gate, f"{WAVDIR}/{name}", head)
            out[name] = (fired, phrase)
            print(f"[{label}] {name}: fired={fired} phrase={phrase!r}", flush=True)
    c = sum(out[n][0] for n in CLANKER)
    v = sum(out[n][0] for n in VIBER)
    neg = sum(out[n][0] for n in NEGATIVES)
    print(f"[{label}] clanker={c}/3 viber={v}/3 negatives_fired={neg}/6", flush=True)
    return out, c, v, neg


def main():
    gate = hv.WakeWordGate(StubSession(), StubBridge())
    assert gate._detector_ok, "detector failed to load — gate would run UNGATED"
    assert gate._spotter is not None, "sherpa spotter not built"
    registered = sorted(gate._sherpa_display_to_phrase)
    print(f"registered keywords: {registered}", flush=True)
    assert registered == ["HEY_CLANKER", "HEY_VIBAR", "HEY_VIBER"], registered
    print(f"threshold: {gate._sherpa_threshold:.2f}", flush=True)

    out_s, c_s, _v_s, neg_s = suite(gate, "stream-start", np.zeros(0, dtype=np.int16))
    assert c_s == 3, f"stream-start clanker {c_s}/3 (need 3/3)"
    assert all(
        out_s[n][1] == "hey clanker" for n in CLANKER
    ), "stream-start clanker phrase mismatch"
    assert neg_s == 0, f"stream-start negatives fired {neg_s}/6"

    _out_l, c_l, _v_l, neg_l = suite(gate, "live-like", SIL)
    assert neg_l == 0, f"live-like negatives fired {neg_l}/6"
    assert c_l >= 2, f"live-like clanker {c_l}/3 (need >= 2/3)"

    print("SHERPA_GATE_OK", flush=True)


if __name__ == "__main__":
    main()
