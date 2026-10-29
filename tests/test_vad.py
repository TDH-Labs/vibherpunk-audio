"""Offline Silero VAD check on tests/known.wav (16 kHz) at given params."""
import asyncio
import os
import sys

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState


async def run(label, params):
    sr, data = scipy.io.wavfile.read("tests/known.wav")
    x = data.astype(np.float64) / 32768.0
    x = resample_poly(x, 16000, sr)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    an = SileroVADAnalyzer(sample_rate=16000, params=params)
    state = VADState.QUIET
    speaking_ms = 0
    transitions = []
    frame = 512
    for i in range(0, len(pcm) - frame, frame):
        chunk = pcm[i : i + frame].tobytes()
        new = await an.analyze_audio(chunk)
        if new != state:
            transitions.append((round(i / 16), str(new)))
            state = new
        if new == VADState.SPEAKING:
            speaking_ms += frame / 16
    print(f"{label}: transitions={transitions[:6]} speaking_total={speaking_ms:.0f}ms")


async def main():
    await run("defaults(conf=.7,minvol=.6,start=.2,stop=.6)",
              VADParams(confidence=0.7, min_volume=0.6, start_secs=0.2, stop_secs=0.6))
    await run("relaxed(conf=.5,minvol=.3,start=.2,stop=.6)",
              VADParams(confidence=0.5, min_volume=0.3, start_secs=0.2, stop_secs=0.6))


if __name__ == "__main__":
    asyncio.run(main())
