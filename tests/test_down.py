"""Failure-mode acceptance, two variants selected by MODE env:

MODE=worker (HERMES_WORKER_URL -> dead port): a work turn dispatches, the
worker call fails, the failure is SPOKEN (WORKER_DOWN_MESSAGE injected),
and the conversation survives (a later chat turn still gets a lane answer).

MODE=lane (LANE_BASE_URL -> dead port): a chat turn errors at the lane
LLM; the announcer speaks LANE_DOWN_MESSAGE instead of dead air.
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

from pipecat.frames.frames import EndFrame, InputAudioRawFrame, StartFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import hermes_voice as hv

SESSION = hv.SESSION
MODE = os.environ.get("MODE", "worker")
STATE = {"ready": asyncio.Event(), "done": None}
RESULTS = {"asserts": {}, "mode": MODE}


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
            asyncio.create_task(drive(self))

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

    async def say(self, wav, tail_silence=2.0):
        await self.feed_pcm(pcm16k(wav))
        await self.feed_pcm(np.zeros(int(tail_silence * 16000), dtype=np.int16))


class Sink(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def events(kind):
    return [e for e in SESSION.events if e["kind"] == kind]


def spoken(source=None):
    if source is None:
        return SESSION.spoken
    return [s for s in SESSION.spoken if s["source"] == source]


async def wait_for(pred, timeout, desc):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        await asyncio.sleep(0.25)
    raise TimeoutError(f"wait_for timed out: {desc}")


async def drive(feeder):
    await STATE["ready"].wait()
    await asyncio.sleep(6)
    a = RESULTS["asserts"]
    try:
        if MODE == "worker":
            before = len(spoken("lane"))
            await feeder.say("tests/utt_chat1.wav")
            await wait_for(lambda b=before: len(spoken("lane")) > b or None, 30, "warmup reply")
            await feeder.say("tests/utt_work.wav")
            await wait_for(lambda: events("dispatch") or None, 25, "dispatch")
            inj = await wait_for(lambda: spoken("injected") or None, 60, "worker failure speech")
            RESULTS["injected_text"] = inj[-1]["text"]
            a["worker_error_event"] = bool(events("worker_error"))
            a["failure_spoken_exact"] = inj[-1]["text"] == hv.WORKER_DOWN_MESSAGE
            # conversation survives
            before = len(spoken("lane"))
            await feeder.say("tests/utt_chat2.wav")
            await wait_for(lambda b=before: len(spoken("lane")) > b or None, 30, "post-failure chat")
            a["conversation_survives"] = True
            RESULTS["post_failure_text"] = spoken("lane")[-1]["text"]
        else:  # lane down
            await feeder.say("tests/utt_chat1.wav")
            inj = await wait_for(lambda: spoken("injected") or None, 60, "lane failure speech")
            RESULTS["injected_text"] = inj[-1]["text"]
            a["lane_error_event"] = bool(events("lane_error"))
            a["failure_spoken_exact"] = inj[-1]["text"] == hv.LANE_DOWN_MESSAGE
    except Exception as e:
        RESULTS["driver_error"] = f"{type(e).__name__}: {e}"
    finally:
        STATE["done"].set()


async def main():
    feeder = Feeder()
    _p, task, _b = hv.build_pipeline(feeder, Sink())
    STATE["done"] = asyncio.Event()
    STATE["ready"].set()

    async def finisher():
        try:
            await asyncio.wait_for(STATE["done"].wait(), timeout=420)
        except asyncio.TimeoutError:
            RESULTS["driver_error"] = "global timeout"
        await asyncio.sleep(0.5)
        await task.queue_frame(EndFrame())

    await asyncio.gather(PipelineRunner().run(task), finisher())
    print("DOWN_JSON " + json.dumps(RESULTS, default=str), flush=True)
    ok = all(v is True for v in RESULTS["asserts"].values()) and len(RESULTS["asserts"]) >= 3
    print("DOWN_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
