"""Desk acceptance — held results (HERMES_WAKE=on).

Script: wake -> work request (dispatch + ack) -> "go to sleep" before the
job finishes -> job completes while the gate is asleep: NO announcement,
result held -> wake -> held result announced FIRST ("While you were
asleep, Vibherpunk finished: ...") -> chat still healthy afterwards.
Prints HELD_JSON / HELD_OK.
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
STATE = {"ready": asyncio.Event(), "bridge": None}
RESULTS = {"asserts": {}}
PREFIX = "While you were asleep, Vibherpunk finished:"


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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.audio_bytes = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if frame.__class__.__name__ == "TTSAudioRawFrame":
            self.audio_bytes += len(frame.audio)
        await self.push_frame(frame, direction)


def events(kind=None):
    if kind is None:
        return SESSION.events
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


async def wake_up(feeder: Feeder, min_fires: int):
    """Prime the streaming detector with swallowed speech, then feed the
    wake phrase (retry x3). Returns when wake_fire count >= min_fires."""
    await feeder.say("tests/utt_nowake_chat.wav", tail_silence=2.0)
    for _ in range(3):
        await feeder.feed_silence(0.5)
        await feeder.feed_pcm(pcm16k("tests/utt_wake.wav"))
        try:
            await wait_for(
                lambda: len(events("wake_fire")) >= min_fires or None,
                12, "wake_fire",
            )
            return
        except TimeoutError:
            continue
    raise TimeoutError("wake phrase never fired")


async def drive(feeder: Feeder):
    await STATE["ready"].wait()
    await asyncio.sleep(6)
    a = RESULTS["asserts"]
    bridge = STATE["bridge"]
    try:
        # 1. Wake + dispatch a real job.
        await wake_up(feeder, 1)
        a["woke"] = True
        await feeder.say("tests/utt_work.wav")
        disp = await wait_for(lambda: events("dispatch") or None, 30, "dispatch")
        a["dispatched"] = True
        RESULTS["dispatch_task"] = disp[-1].get("task")
        await wait_for(lambda: spoken("lane") or None, 30, "work ack")
        a["ack_spoken"] = True
        RESULTS["ack_text"] = spoken("lane")[-1]["text"]
        SESSION.flush_turn()
        # 2. Sleep BEFORE the job completes.
        await feeder.say("tests/utt_sleep.wav")
        await wait_for(lambda: events("sleep_command") or None, 30, "sleep_command")
        await wait_for(
            lambda: [s for s in spoken("injected") if s["text"] == hv.SLEEP_CONFIRMATION] or None,
            20, "sleep confirmation",
        )
        a["slept_with_job_pending"] = bridge.pending_count() >= 1
        # 3. Job completes into the quiet room: held, NOT announced.
        done = await wait_for(lambda: events("worker_done") or None, 300, "worker_done")
        RESULTS["worker_summary"] = done[-1].get("summary")
        await wait_for(lambda: events("held_result_held") or None, 20, "held_result_held")
        a["result_held"] = True
        await asyncio.sleep(4)
        a["no_announcement_while_asleep"] = not any(
            PREFIX in s["text"] for s in spoken("injected")
        ) and not events("held_result_announced")
        a["held_in_bridge"] = len(bridge.held) == 1
        n_lane = len(spoken("lane"))
        # 4. Wake -> held result announced FIRST.
        await wake_up(feeder, 2)
        ann = await wait_for(
            lambda: [s for s in spoken("injected") if s["text"].startswith(PREFIX)] or None,
            30, "held announcement",
        )
        a["announced_on_wake"] = True
        RESULTS["announcement"] = ann[-1]["text"]
        a["announced_event_logged"] = bool(events("held_result_announced"))
        a["held_drained"] = len(bridge.held) == 0
        # 5. Conversation still healthy after the held flush.
        await feeder.say("tests/utt_chat1.wav")
        await wait_for(
            lambda: len(spoken("lane")) > n_lane or None, 30, "post-wake chat"
        )
        a["chat_after_wake"] = True
        RESULTS["chat_text"] = spoken("lane")[-1]["text"]
        SESSION.flush_turn()
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
    print("HELD_JSON " + json.dumps(RESULTS, default=str), flush=True)
    ok = all(v is True for v in RESULTS["asserts"].values()) and len(RESULTS["asserts"]) >= 9
    print("HELD_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
