"""Offline Silero internals: max confidence + max smoothed volume per file."""
import asyncio
import sys

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams


def load16k(path):
    sr, data = scipy.io.wavfile.read(path)
    x = data.astype(np.float64) / 32768.0 if data.dtype == np.int16 else data.astype(np.float64)
    if sr != 16000:
        x = resample_poly(x, 16000, sr)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


async def run(path):
    pcm = load16k(path)
    an = SileroVADAnalyzer(sample_rate=16000)
    an.set_sample_rate(16000)
    an.set_params(VADParams(confidence=0.7, min_volume=0.6, start_secs=0.2, stop_secs=0.6))
    stats = {"conf": 0.0, "vol": 0.0, "speaking_states": 0, "calls": 0}
    orig_conf = an.voice_confidence
    orig_vol = an._get_smoothed_volume

    def conf(audio):
        v = orig_conf(audio)
        stats["conf"] = max(stats["conf"], float(np.max(np.asarray(v, dtype=np.float64))))
        return v

    def vol(audio):
        v = orig_vol(audio)
        stats["vol"] = max(stats["vol"], float(np.max(np.asarray(v, dtype=np.float64))))
        return v

    an.voice_confidence = conf
    an._get_smoothed_volume = vol
    raw = pcm.tobytes()
    chunk = 1280
    for i in range(0, len(raw), chunk):
        st = await an.analyze_audio(raw[i : i + chunk])
        stats["calls"] += 1
        if "SPEAKING" in str(st) or "STARTING" in str(st):
            stats["speaking_states"] += 1
    print(f"{path}: max_conf={stats['conf']:.3f} max_vol={stats['vol']:.3f} "
          f"nonquiet_states={stats['speaking_states']}/{stats['calls']}")


async def main():
    for p in sys.argv[1:]:
        await run(p)


if __name__ == "__main__":
    asyncio.run(main())
