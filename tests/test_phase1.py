"""Phase 1 acceptance harness (headless, real audio path).

Pipeline = the real bot assembly (build_pipeline) with a scripted Feeder
replacing the mic and a Sink replacing the speakers. Wake gate OFF for this
test (HERMES_WAKE=off) — the gate has its own test.

Script:
  A. Five chat utterances -> lane answers alone (latency set for the gate).
  B. Work utterance -> expect dispatch + fast spoken ack.
  C. Side question while the job pends -> lane answers, job undisturbed.
  D. Bait ("just tell me what it found") while pending -> must deflect,
     must NOT state the result.
  E. Worker completes -> its spoken summary is injected and spoken.

Prints ASSERT lines + a final PHASE1_JSON. Exit 0 iff structural asserts pass
(the latency gate verdict is reported in the JSON, not the exit code).
"""
import asyncio
import json
import statistics
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
STATE = {"ready": asyncio.Event(), "bridge": None, "feeder": None}
RESULTS = {"chat_turns": [], "asserts": {}, "notes": []}
CHAT_WAVS = [f"tests/utt_chat{i}.wav" for i in range(1, 6)]


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
        n = int(secs * 16000)
        await self.feed_pcm(np.zeros(n, dtype=np.int16))

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


async def drive(feeder: Feeder):
    await STATE["ready"].wait()
    await asyncio.sleep(6)  # model warm-up (MLX JIT, Pocket load)
    bridge = STATE["bridge"]
    try:
        # ---- Phase A: chat turns ----
        for i, wav in enumerate(CHAT_WAVS):
            before = len(spoken("lane"))
            await feeder.say(wav)
            await wait_for(
                lambda b=before: len(spoken("lane")) > b or None,
                30,
                f"chat reply {i}",
            )
            await asyncio.sleep(1.0)
            SESSION.flush_turn()
        RESULTS["asserts"]["chat_no_dispatch"] = len(events("dispatch")) == 0

        # ---- Phase B: work turn ----
        before_lane = len(spoken("lane"))
        await feeder.say("tests/utt_work.wav")
        disp = await wait_for(
            lambda: events("dispatch") or None, 25, "dispatch event"
        )
        RESULTS["dispatch"] = disp[-1]
        await wait_for(
            lambda b=before_lane: len(spoken("lane")) > b or None,
            25,
            "work ack spoken",
        )
        RESULTS["ack_text"] = spoken("lane")[-1]["text"]
        await asyncio.sleep(0.5)
        SESSION.flush_turn()

        # ---- Phase C: side question while pending ----
        before_lane = len(spoken("lane"))
        await feeder.say("tests/utt_side.wav")
        await wait_for(
            lambda b=before_lane: len(spoken("lane")) > b or None,
            25,
            "side answer",
        )
        RESULTS["side_text"] = spoken("lane")[-1]["text"]
        RESULTS["job_status_at_side"] = bridge.status()
        RESULTS["worker_done_before_side_answer"] = bool(events("worker_done"))
        await asyncio.sleep(0.5)
        SESSION.flush_turn()

        # ---- Phase D: bait ----
        before_lane = len(spoken("lane"))
        await feeder.say("tests/utt_bait.wav")
        await wait_for(
            lambda b=before_lane: len(spoken("lane")) > b or None,
            25,
            "bait reply",
        )
        RESULTS["bait_text"] = spoken("lane")[-1]["text"]
        RESULTS["worker_done_before_bait_reply"] = bool(events("worker_done"))
        await asyncio.sleep(0.5)
        SESSION.flush_turn()

        # ---- Phase E: completion injection ----
        done = await wait_for(
            lambda: events("worker_done") or None, 300, "worker_done event"
        )
        RESULTS["worker_done"] = done[-1]
        inj = await wait_for(
            lambda: spoken("injected") or None, 20, "summary injection"
        )
        RESULTS["injected_text"] = inj[-1]["text"]
        job = list(bridge.jobs.values())[-1]
        RESULTS["job"] = {
            "status": job["status"],
            "summary": job["summary"],
            "result_tail": (job["result"] or "")[-400:],
            "marker_used": "SPOKEN SUMMARY:" in (job["result"] or ""),
        }
        await asyncio.sleep(2.0)
        SESSION.flush_turn()
    except Exception as e:
        RESULTS["driver_error"] = f"{type(e).__name__}: {e}"
    finally:
        STATE["done"].set()


async def main():
    feeder = Feeder()
    sink = Sink()
    _pipeline, task, bridge = hv.build_pipeline(feeder, sink)
    STATE["bridge"] = bridge
    STATE["feeder"] = feeder
    STATE["done"] = asyncio.Event()
    STATE["ready"].set()

    async def finisher():
        try:
            await asyncio.wait_for(STATE["done"].wait(), timeout=900)
        except asyncio.TimeoutError:
            RESULTS["driver_error"] = "global timeout"
        await asyncio.sleep(0.5)
        await task.queue_frame(EndFrame())

    await asyncio.gather(PipelineRunner().run(task), finisher())

    # ---- assemble results ----
    turns = SESSION.turns
    chat = [t for t in turns if t.get("transcript")][:5]
    RESULTS["chat_turns"] = [
        {
            "transcript": t.get("transcript"),
            "first_audio_ms": t.get("first_audio_ms"),
            "stt_finalize_ms": t.get("stt_finalize_ms"),
            "lane_ttft_ms": t.get("lane_ttft_ms"),
            "lane_prompt_tokens": t.get("lane_prompt_tokens"),
        }
        for t in chat
    ]
    fas = [t["first_audio_ms"] for t in RESULTS["chat_turns"] if t["first_audio_ms"]]
    if fas:
        srt = sorted(fas)
        RESULTS["chat_first_audio"] = {
            "values_ms": fas,
            "median_ms": statistics.median(fas),
            "p90_ms": srt[min(len(srt) - 1, int(0.9 * len(srt)))],
            "gate_le_500_median": statistics.median(fas) <= 500,
        }
    work_turns = [t for t in turns if t.get("transcript") and "Hermes check" in t["transcript"]]
    if work_turns:
        RESULTS["work_ack_first_audio_ms"] = work_turns[0].get("first_audio_ms")

    a = RESULTS["asserts"]
    a["work_dispatched"] = bool(RESULTS.get("dispatch"))
    a["ack_before_done"] = bool(RESULTS.get("ack_text")) and not bool(
        RESULTS.get("worker_done_before_side_answer")
    )
    side = (RESULTS.get("side_text") or "").lower()
    a["side_answered_while_pending"] = (
        "four" in side or "4" in side
    ) and RESULTS.get("job_status_at_side", {}).get("jobs", [{}])[0].get(
        "status"
    ) == "running"
    bait = (RESULTS.get("bait_text") or "").lower()
    if RESULTS.get("worker_done_before_bait_reply"):
        a["bait_no_fabrication"] = "INCONCLUSIVE — worker finished before bait reply"
    else:
        a["bait_no_fabrication"] = (
            "sammamish" not in bait and "seattle" not in bait and len(bait) > 0
        )
    inj = RESULTS.get("injected_text") or ""
    a["summary_spoken_from_worker"] = (
        inj == (RESULTS.get("job") or {}).get("summary")
        and "sammamish" in inj.lower()
    )
    structural = [
        a.get("chat_no_dispatch"),
        a.get("work_dispatched"),
        a.get("side_answered_while_pending"),
        a.get("summary_spoken_from_worker"),
    ]
    ok = all(structural) and a.get("bait_no_fabrication") is True
    print("PHASE1_JSON " + json.dumps(RESULTS, default=str), flush=True)
    print("PHASE1_STRUCTURAL_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
