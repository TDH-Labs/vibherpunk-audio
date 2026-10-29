"""Phase 1b — content-verifiable work completion.

The Phase 1 run proved the completion mechanism (worker's SPOKEN SUMMARY
injected verbatim) but the fact I asserted (the user's city) is not in Hermes'
memory — the worker honestly reported no record. This re-run uses a fact
verifiable from disk: the first line of ~/.hermes/SOUL.md contains
"built by Nous Research". The worker must read the file with its tools;
the injected summary must contain that token and equal the job summary.
"""
import asyncio
import json
import sys
import time

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample_poly

sys.path.insert(0, "/path/to/hermes-voice")

from pipecat.frames.frames import EndFrame, InputAudioRawFrame, StartFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import hermes_voice as hv

SESSION = hv.SESSION
STATE = {"ready": asyncio.Event(), "bridge": None, "done": None}
RESULTS = {"asserts": {}}


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
        n = int(tail_silence * 16000)
        await self.feed_pcm(np.zeros(n, dtype=np.int16))


class Sink(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def events(kind):
    return [e for e in SESSION.events if e["kind"] == kind]


def spoken(source):
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
    bridge = STATE["bridge"]
    try:
        before = len(spoken("lane"))
        await feeder.say("tests/utt_chat1.wav")  # warm-up chat turn
        await wait_for(lambda b=before: len(spoken("lane")) > b or None, 30, "warmup reply")
        await asyncio.sleep(0.5)

        await feeder.say("tests/utt_work4.wav")
        disp = await wait_for(lambda: events("dispatch") or None, 25, "dispatch")
        RESULTS["dispatch_task"] = disp[-1]["task"]
        before = len(spoken("lane"))
        await wait_for(lambda b=before: len(spoken("lane")) > b or None, 25, "ack")
        RESULTS["ack_text"] = spoken("lane")[-1]["text"]

        done = await wait_for(lambda: events("worker_done") or None, 300, "worker_done")
        RESULTS["worker_done"] = done[-1]
        inj = await wait_for(lambda: spoken("injected") or None, 20, "injection")
        RESULTS["injected_text"] = inj[-1]["text"]
        job = list(bridge.jobs.values())[-1]
        RESULTS["job_summary"] = job["summary"]
        RESULTS["marker_used"] = "SPOKEN SUMMARY:" in (job["result"] or "")
        await asyncio.sleep(1.0)
    except Exception as e:
        RESULTS["driver_error"] = f"{type(e).__name__}: {e}"
    finally:
        STATE["done"].set()


async def main():
    feeder = Feeder()
    _p, task, bridge = hv.build_pipeline(feeder, Sink())
    STATE["bridge"] = bridge
    STATE["done"] = asyncio.Event()
    STATE["ready"].set()

    async def finisher():
        try:
            await asyncio.wait_for(STATE["done"].wait(), timeout=600)
        except asyncio.TimeoutError:
            RESULTS["driver_error"] = "global timeout"
        await asyncio.sleep(0.5)
        await task.queue_frame(EndFrame())

    await asyncio.gather(PipelineRunner().run(task), finisher())
    inj = (RESULTS.get("injected_text") or "").lower()
    a = RESULTS["asserts"]
    a["dispatched"] = bool(RESULTS.get("dispatch_task"))
    a["summary_verbatim"] = RESULTS.get("injected_text") == RESULTS.get("job_summary")
    a["content_verified_os_version"] = "macos" in inj and "27" in inj
    ok = all(a.values())
    print("PHASE1D_JSON " + json.dumps(RESULTS, default=str), flush=True)
    print("PHASE1D_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
