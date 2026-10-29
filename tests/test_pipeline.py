"""Headless full pass: text-in -> Hermes gateway LLM -> Pocket TTS -> audio file.

Usage: python tests/test_pipeline.py "<prompt>" <out.wav>
Gateway URL comes from HERMES_GATEWAY_URL (set to a dead port for the
gateway-down test). Prints one RESULT_JSON line.
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, "/path/to/hermes-voice")

import numpy as np
import scipy.io.wavfile
from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hermes_voice import GatewayDownAnnouncer, LeadingSilenceTrimmer, build_llm, build_tts

PROMPT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "tests/out_reply.wav"
STATE = {"t0": None, "first_audio": None, "done": asyncio.Event(), "error": False}


class Injector(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, ErrorFrame):
            STATE["error"] = True
            STATE["done"].set()
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame) and STATE["t0"] is None:
            STATE["t0"] = time.time()
            ctx = LLMContext(messages=[{"role": "user", "content": PROMPT}])
            await self.push_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)


class AudioSink(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buf = bytearray()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            if STATE["first_audio"] is None:
                STATE["first_audio"] = time.time()
            self._buf.extend(frame.audio)
        elif isinstance(frame, LLMFullResponseEndFrame):
            STATE["done"].set()
        elif isinstance(frame, ErrorFrame):
            STATE["error"] = True
            STATE["done"].set()
        await self.push_frame(frame, direction)

    def save(self, path):
        if not self._buf:
            return 0.0
        arr = np.frombuffer(bytes(self._buf), dtype=np.int16)
        scipy.io.wavfile.write(path, 24000, arr)
        return len(arr) / 24000.0


async def main():
    sink = AudioSink()
    chain = [Injector(), GatewayDownAnnouncer(), build_llm(), build_tts()]
    if os.environ.get("TRIM"):
        chain.append(LeadingSilenceTrimmer())
    chain.append(sink)
    pipeline = Pipeline(chain)
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    runner = PipelineRunner()

    async def finisher():
        try:
            await asyncio.wait_for(STATE["done"].wait(), timeout=75)
        except asyncio.TimeoutError:
            logger.warning("finisher: timeout waiting for response end")
        await asyncio.sleep(3.0)  # let TTS tail drain
        await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(task), finisher())

    dur = sink.save(OUT)
    result = {
        "prompt": PROMPT,
        "gateway": os.environ.get("HERMES_GATEWAY_URL", "http://127.0.0.1:8642/v1"),
        "first_audio_ms": (
            round((STATE["first_audio"] - STATE["t0"]) * 1000)
            if STATE["first_audio"] and STATE["t0"]
            else None
        ),
        "audio_secs": round(dur, 2),
        "error_seen": STATE["error"],
        "out": OUT,
    }
    print("RESULT_JSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
