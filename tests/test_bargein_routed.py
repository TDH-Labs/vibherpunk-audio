"""Simulated barge-in regression on the CURRENT pipeline assembly.

Same protocol as the Phase 0 measurement (tests/test_bargein.py): ask for a
long story; once TTS audio is flowing, inject real speech (tests/known.wav)
as InputAudioRawFrames at 8x real-time feed; measure speech-start -> last
TTS audio frame out, and whether an InterruptionFrame propagated. The
pipeline is hv.build_pipeline (Feeder/Sink ends), wake gate off. Lane
backend from HERMES_LANE_BACKEND. SIMULATED — no transport/speaker in the
loop; comparable to the ~148 ms baseline measured the same way.
"""
import asyncio
import json
import os
import sys
import time

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly

sys.path.insert(0, "/Users/ai/workspace/hermes-voice")

from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import hermes_voice as hv

STATE = {
    "audio_flowing": asyncio.Event(),
    "done": asyncio.Event(),
    "speech_start": None,
    "last_audio": None,
    "audio_secs_before": 0.0,
    "interruption_seen": False,
}


def pcm16k(path):
    sr, data = scipy.io.wavfile.read(path)
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    else:
        x = data.astype(np.float64)
    if sr != 16000:
        x = resample_poly(x, 16000, sr)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


class Feeder(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            asyncio.create_task(self._drive())

    async def _feed(self, pcm, chunk=320, pace=1.0):
        for i in range(0, len(pcm), chunk):
            await self.push_frame(
                InputAudioRawFrame(
                    audio=pcm[i : i + chunk].tobytes(),
                    sample_rate=16000,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )
            await asyncio.sleep((chunk / 16000) * pace)

    async def _drive(self):
        await asyncio.sleep(6)  # warm-up
        await self._feed(pcm16k("tests/utt_story.wav"))
        # Trailing silence so the turn closes and the lane starts its story.
        await self._feed(np.zeros(int(1.2 * 16000), dtype=np.int16))
        try:
            await asyncio.wait_for(STATE["audio_flowing"].wait(), timeout=40)
        except asyncio.TimeoutError:
            STATE["done"].set()
            return
        await asyncio.sleep(0.3)
        pcm = pcm16k("tests/known.wav")
        STATE["speech_start"] = time.monotonic()
        chunk = 640  # 40 ms at 16 kHz, fed at 8x (5 ms sleeps)
        for i in range(0, len(pcm), chunk):
            await self.push_frame(
                InputAudioRawFrame(
                    audio=pcm[i : i + chunk].tobytes(),
                    sample_rate=16000,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )
            await asyncio.sleep(0.005)
        STATE["done"].set()


class Sink(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            STATE["interruption_seen"] = True
        if isinstance(frame, TTSAudioRawFrame):
            STATE["last_audio"] = time.monotonic()
            if STATE["speech_start"] is None:
                STATE["audio_secs_before"] += len(frame.audio) / 2 / frame.sample_rate
            STATE["audio_flowing"].set()
        await self.push_frame(frame, direction)


async def main():
    feeder = Feeder()
    sink = Sink()
    _pipeline, task, _bridge = hv.build_pipeline(feeder, sink)

    async def finisher():
        try:
            await asyncio.wait_for(STATE["done"].wait(), timeout=120)
        except asyncio.TimeoutError:
            pass
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if STATE["last_audio"] and time.monotonic() - STATE["last_audio"] > 1.2:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)
        await task.queue_frame(EndFrame())

    await asyncio.gather(PipelineRunner().run(task), finisher())
    res = {
        "simulated": True,
        "backend": hv.LANE_BACKEND,
        "bot_audio_secs_before_speech": round(STATE["audio_secs_before"], 2),
        "interruption_frame_seen": STATE["interruption_seen"],
        "speech_start_to_last_audio_ms": (
            round((STATE["last_audio"] - STATE["speech_start"]) * 1000)
            if STATE["last_audio"] and STATE["speech_start"]
            else None
        ),
    }
    print("BARGE_JSON " + json.dumps(res), flush=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
