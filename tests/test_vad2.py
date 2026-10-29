"""Offline Silero VAD (params initialized) on two speech files."""
import asyncio
import sys

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState


def load16k(path):
    sr, data = scipy.io.wavfile.read(path)
    x = data.astype(np.float64) / 32768.0 if data.dtype == np.int16 else data.astype(np.float64)
    if sr != 16000:
        x = resample_poly(x, 16000, sr)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


async def run(path):
    pcm = load16k(path)
    an = SileroVADAnalyzer(sample_rate=16000)
    an.set_params(VADParams(confidence=0.7, min_volume=0.6, start_secs=0.2, stop_secs=0.6))
    counts = {}
    chunk = 640 * 2  # bytes per 640-sample chunk, as the controller feeds
    raw = pcm.tobytes()
    for i in range(0, len(raw), chunk):
        st = await an.analyze_audio(raw[i : i + chunk])
        counts[str(st)] = counts.get(str(st), 0) + 1
    print(f"{path}: {counts}")


async def main():
    await run(sys.argv[1])
    if len(sys.argv) > 2:
        await run(sys.argv[2])


if __name__ == "__main__":
    asyncio.run(main())
