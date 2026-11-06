"""Fallback proof (standalone): point the cloud lane at the
selective-failure proxy so every routed call is rejected with a 500; each
turn must degrade to the relay configuration and still be served (spoken
answer + lane_fallback event) — never dead air."""
import asyncio
import json
import os
import sys
import time

os.environ.setdefault("HERMES_WAKE", "off")
os.environ["HERMES_LANE_BACKEND"] = "cloud"
os.environ["HERMES_FUSED_LOOKUP"] = "on"
os.environ.setdefault(
    "HERMES_VOICE_LOGDIR",
    "/Users/ai/workspace/hermes-voice/logs-cloud-fallback",
)
sys.path.insert(0, "/Users/ai/workspace/hermes-voice")
os.chdir("/Users/ai/workspace/hermes-voice")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: F401,E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402

import hermes_voice as hv  # noqa: E402

hv.LANE_BASE_URL = "http://127.0.0.1:8919/v1"


def pcm16k(path):
    data, sr = sf.read(path, dtype="int16", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data[:, 0]
    if sr != 16000:
        n = int(len(data) * 16000 / sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, n), np.arange(len(data)), data
        ).astype(np.int16)
    return data


class Feeder(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.started.set()

    async def feed_pcm(self, pcm, chunk=320):
        for i in range(0, len(pcm), chunk):
            await self.push_frame(
                InputAudioRawFrame(
                    audio=pcm[i : i + chunk].tobytes(),
                    sample_rate=16000,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )
            await asyncio.sleep(chunk / 16000)

    async def say(self, wav, tail=2.0):
        await self.feed_pcm(pcm16k(wav))
        await self.feed_pcm(np.zeros(int(tail * 16000), dtype=np.int16))


class Sink(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.texts = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and frame.text.strip():
            self.texts.append(frame.text)
        await self.push_frame(frame, direction)


def turn_count():
    return len([e for e in hv.SESSION.events if e["kind"] == "turn"])


async def wait_turn(prev, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if turn_count() > prev:
            return True
        await asyncio.sleep(0.25)
    return False


async def main():
    feeder = Feeder()
    sink = Sink()
    _pipeline, task, _bridge = hv.build_pipeline(feeder, sink)

    async def script():
        await feeder.started.wait()
        await asyncio.sleep(6)
        n = turn_count()
        await feeder.say("tests/utt_chat1.wav")
        print("chat turn done:", await wait_turn(n), flush=True)
        n = turn_count()
        await feeder.say("tests/utt_lk_wife.wav")
        print("wife turn done:", await wait_turn(n), flush=True)
        await asyncio.sleep(2)
        await task.queue_frame(EndFrame())

    await asyncio.gather(PipelineRunner().run(task), script())
    ev = hv.SESSION.events
    print(
        "FALLBACK_JSON "
        + json.dumps(
            {
                "turns": turn_count(),
                "fallback_events": len(
                    [e for e in ev if e["kind"] == "lane_fallback"]
                ),
                "lane_errors": len([e for e in ev if e["kind"] == "lane_error"]),
                "spoken": [t for t in sink.texts if t.strip()],
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
