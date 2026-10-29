"""Warm multi-turn measurement: models load once (as in the live bot), then
several text-in turns are timed per stage:
  inject -> first LLM text frame -> first audio frame -> first NON-SILENT audio.
Turn 0 is warm-up and excluded from the medians. Prints TURN_JSON lines.
"""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/path/to/hermes-voice")

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from hermes_voice import GatewayDownAnnouncer, build_llm, build_tts

PROMPTS = [
    "Say the word ready.",  # warm-up, excluded
    "In one short sentence, what is the capital of France?",
    "In one short sentence, how many legs does a spider have?",
    "Tell me one short interesting fact about the moon.",
    "In one short sentence, what color do you get mixing blue and yellow paint?",
]
TURNS = []  # per-turn dicts
DONE = asyncio.Event()


class Injector(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.idx = -1

    async def send_next(self):
        self.idx += 1
        if self.idx >= len(PROMPTS):
            DONE.set()
            return
        TURNS.append({"turn": self.idx, "t_inject": time.time(), "first_text": None,
                      "first_audio": None, "first_sound": None})
        ctx = LLMContext(messages=[{"role": "user", "content": PROMPTS[self.idx]}])
        await self.push_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.send_next()


class Probe(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and TURNS and TURNS[-1]["first_text"] is None:
            TURNS[-1]["first_text"] = time.time()
        await self.push_frame(frame, direction)


class Sink(FrameProcessor):
    def __init__(self, injector, **kwargs):
        super().__init__(**kwargs)
        self._injector = injector

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and TURNS:
            now = time.time()
            cur = TURNS[-1]
            if cur["first_audio"] is None:
                cur["first_audio"] = now
            if cur["first_sound"] is None:
                arr = np.frombuffer(frame.audio, dtype=np.int16)
                if len(arr) and float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))) > 300:
                    cur["first_sound"] = now
        elif isinstance(frame, LLMFullResponseEndFrame):
            asyncio.get_event_loop().create_task(self._advance())
        await self.push_frame(frame, direction)

    async def _advance(self):
        await asyncio.sleep(1.2)
        await self._injector.send_next()


async def main():
    injector = Injector()
    pipeline = Pipeline([injector, build_llm(), Probe(), GatewayDownAnnouncer(), build_tts(), Sink(injector)])
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
            await asyncio.wait_for(DONE.wait(), timeout=120)
        except asyncio.TimeoutError:
            logger.warning("warm test timeout")
        await asyncio.sleep(0.5)
        await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(task), finisher())

    for t in TURNS:
        row = {"turn": t["turn"], "warmup": t["turn"] == 0}
        for k in ("first_text", "first_audio", "first_sound"):
            row[k + "_ms"] = round((t[k] - t["t_inject"]) * 1000) if t[k] else None
        print("TURN_JSON " + json.dumps(row), flush=True)
    print("WARM_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
