"""Wake-gate acceptance (HERMES_WAKE=on, HERMES_FOLLOWUP_SECS=6).

1. Speech without the wake phrase -> nothing transcribed, nothing spoken.
2. "Hey Hermes" + question -> wake fires, question answered.
3. After the follow-up window expires idle -> gate re-arms (wake_rearm
   event), and speech without the wake phrase is swallowed again.
"""
import asyncio
import json
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
STATE = {"ready": asyncio.Event(), "done": None}
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

    async def feed_silence(self, secs):
        await self.feed_pcm(np.zeros(int(secs * 16000), dtype=np.int16))

    async def say(self, wav, tail_silence=2.0):
        await self.feed_pcm(pcm16k(wav))
        await self.feed_silence(tail_silence)


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
        # 1. No wake phrase -> swallowed.
        await feeder.say("tests/utt_nowake_chat.wav", tail_silence=2.5)
        await asyncio.sleep(4)
        a["nowake_no_transcription"] = len(events("transcription")) == 0
        a["nowake_no_speech"] = len(spoken()) == 0

        # 2. Wake phrase + question -> fires and answers.
        await feeder.feed_pcm(pcm16k("tests/utt_wake.wav"))
        await feeder.feed_silence(0.3)
        await feeder.say("tests/utt_chat1.wav")
        await wait_for(lambda: events("wake_fire") or None, 20, "wake_fire")
        a["wake_fired"] = True
        await wait_for(lambda: spoken("lane") or None, 30, "post-wake answer")
        a["answered_after_wake"] = True
        RESULTS["answer_text"] = spoken("lane")[-1]["text"]
        RESULTS["transcripts"] = [e["text"] for e in events("transcription")]

        # 3. Idle past the follow-up window -> re-arm; speech swallowed again.
        await wait_for(lambda: events("wake_rearm") or None, 30, "wake_rearm")
        a["rearmed"] = True
        n_tr = len(events("transcription"))
        await feeder.say("tests/utt_nowake_chat.wav", tail_silence=2.5)
        await asyncio.sleep(4)
        a["rearmed_swallows_again"] = len(events("transcription")) == n_tr
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
    print("GATE_JSON " + json.dumps(RESULTS, default=str), flush=True)
    ok = all(v is True for v in RESULTS["asserts"].values()) and len(RESULTS["asserts"]) >= 6
    print("GATE_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
