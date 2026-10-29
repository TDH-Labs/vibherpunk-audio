"""Wake detector unit proof: stream test WAVs through pyopen-wakeword with
the bundled hey_hermes.tflite; print max score + whether Hermes' own firing
rule (threshold 0.6, 3 consecutive over-threshold windows) fires."""
import sys

import numpy as np
import scipy.io.wavfile
from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures
from scipy.signal import resample_poly

MODEL = "/path/to/.hermes/hermes-agent/tools/wakewords/hey_hermes.tflite"
THRESHOLD = 0.6
CONFIRM = 3


def load16k(path):
    sr, data = scipy.io.wavfile.read(path)
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    else:
        x = data.astype(np.float64)
    if sr != 16000:
        x = resample_poly(x, 16000, sr)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def run(path):
    feats = OpenWakeWordFeatures.from_builtin()
    model = OpenWakeWord.from_model(MODEL)
    pcm = load16k(path)
    # pad 1s silence head/tail like a real room
    pcm = np.concatenate([np.zeros(16000, dtype=np.int16), pcm, np.zeros(16000, dtype=np.int16)])
    max_score = 0.0
    streak = 0
    fired = False
    chunk = 1280
    for i in range(0, len(pcm) - chunk + 1, chunk):
        frame = pcm[i : i + chunk]
        for emb in feats.process_streaming(frame.tobytes()):
            for score in model.process_streaming(emb):
                max_score = max(max_score, float(score))
                if score >= THRESHOLD:
                    streak += 1
                    if streak >= CONFIRM:
                        fired = True
                else:
                    streak = 0
    print(f"OWW {path}: max={max_score:.3f} fired={fired}", flush=True)


for p in sys.argv[1:]:
    run(p)
print("OWW_DONE", flush=True)
