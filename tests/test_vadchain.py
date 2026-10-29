"""Isolate the VAD break: (A) feeder -> user aggregator(VAD) direct;
(B) feeder -> WhisperSTT -> recorder (what sample_rate do frames carry?)."""
import asyncio
import json
import os
import sys
import time

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from hermes_voice import build_stt

OUT = {"A_vad_start": 0, "A_vad_stop": 0, "B_rates": {}, "B_frames": 0}
AN = {"calls": 0, "speaking": 0, "states": []}


class LoggingVAD(SileroVADAnalyzer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._tee = open("tests/vad_tee.raw", "wb")
        self._conf_max = 0.0
        _orig = self.voice_confidence

        def conf(buffer, _orig=_orig):
            v = _orig(buffer)
            self._conf_max = max(self._conf_max, float(np.max(np.asarray(v, dtype=np.float64))))
            return v

        self.voice_confidence = conf

    async def analyze_audio(self, buffer):
        if not getattr(self, "_dumped", False):
            self._dumped = True
            AN["sample_rate"] = self.sample_rate
            AN["params"] = str(self._params)
            AN["vad_frames"] = self._vad_frames
        self._tee.write(buffer)
        st = await super().analyze_audio(buffer)
        AN["calls"] += 1
        AN["conf_max"] = self._conf_max
        if str(st) not in AN["states"]:
            AN["states"].append(str(st))
        if "SPEAKING" in str(st):
            AN["speaking"] += 1
        return st


def pcm16k():
    sr, data = scipy.io.wavfile.read("tests/known.wav")
    x = data.astype(np.float64) / 32768.0
    x = resample_poly(x, 16000, sr)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


class Feeder(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            asyncio.create_task(self._drive())

    async def _drive(self):
        await asyncio.sleep(0.8)
        pcm = pcm16k()
        chunk = 640
        for i in range(0, len(pcm), chunk):
            await self.push_frame(
                InputAudioRawFrame(
                    audio=pcm[i : i + chunk].tobytes(), sample_rate=16000, num_channels=1
                ),
                FrameDirection.DOWNSTREAM,
            )
            await asyncio.sleep(0.005)
        await asyncio.sleep(1.2)


class RecorderA(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            OUT["A_vad_start"] += 1
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            OUT["A_vad_stop"] += 1
        await self.push_frame(frame, direction)


class RecorderB(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            OUT["B_frames"] += 1
            OUT["B_rates"][str(frame.sample_rate)] = OUT["B_rates"].get(str(frame.sample_rate), 0) + 1
        await self.push_frame(frame, direction)


async def main():
    # Probe A: direct to aggregator
    context = LLMContext(messages=[])
    aggs = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=LoggingVAD(sample_rate=16000, params=VADParams(stop_secs=0.6))
        ),
    )
    task_a = PipelineTask(
        Pipeline([Feeder(), aggs.user(), RecorderA(), aggs.assistant()]),
        params=PipelineParams(enable_metrics=True),
    )

    async def finish(task, delay):
        await asyncio.sleep(delay)
        await task.queue_frame(EndFrame())

    await asyncio.gather(PipelineRunner().run(task_a), finish(task_a, 9))

    # Probe B: through STT
    task_b = PipelineTask(
        Pipeline([Feeder(), build_stt(), RecorderB()]),
        params=PipelineParams(enable_metrics=True),
    )
    await asyncio.gather(PipelineRunner().run(task_b), finish(task_b, 25))

    OUT["analyzer"] = AN
    print("RESULT_JSON " + json.dumps(OUT))


if __name__ == "__main__":
    asyncio.run(main())
