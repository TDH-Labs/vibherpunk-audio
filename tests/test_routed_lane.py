"""Routed-lane bench (headless, real audio path) — Phase 1 harness pattern.

Pipeline = the real bot assembly (hv.build_pipeline) with a scripted Feeder
replacing the mic and a Sink replacing the speakers. Wake gate OFF
(HERMES_WAKE=off). The lane backend comes from HERMES_LANE_BACKEND
(relay|local); BENCH_MODE=full runs chat + discipline + ack + in-flow,
BENCH_MODE=baseline runs chat + ack only (relay comparator, same method).

Prints one ROUTED_JSON with per-step facts (transcripts, first-audio,
exits in order, spoken texts, dispatch task texts, lookup events) for
scoring. Latency gates are judged from this JSON, not from exit codes.
"""
import asyncio
import json
import os
import statistics
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
MODE = os.environ.get("BENCH_MODE", "full")
STATE = {"ready": asyncio.Event(), "bridge": None}
RESULTS = {"mode": MODE, "backend": hv.LANE_BACKEND, "steps": [], "notes": []}

CHAT_WAVS = [f"tests/utt_chat{i}.wav" for i in range(1, 21)]


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


def events(kind=None):
    if kind is None:
        return SESSION.events
    return [e for e in SESSION.events if e["kind"] == kind]


async def wait_for(pred, timeout, desc):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        await asyncio.sleep(0.2)
    RESULTS["notes"].append(f"TIMEOUT waiting: {desc}")
    return None


class Step:
    def __init__(self, name, expect):
        self.name = name
        self.expect = expect
        self.e0 = len(SESSION.events)
        self.turn = None

    def finish(self):
        evs = SESSION.events[self.e0 :]
        self.window = evs
        self.spoken = [e.get("text", "") for e in evs if e["kind"] == "spoken_lane"]
        self.dispatches = [e.get("task", "") for e in evs if e["kind"] == "dispatch"]
        self.lookups = [
            {
                "query": e.get("query"),
                "hit": e.get("hit"),
                "ms": e.get("ms"),
                "speculative": e.get("speculative"),
                "source": e.get("source"),
            }
            for e in evs
            if e["kind"] == "lookup"
        ]
        self.anticipations = [
            e.get("what") for e in evs if e["kind"] == "anticipation"
        ]
        exits = []
        for e in evs:
            # Speculative (anticipator) lookups are preparation, not an
            # exit — only the lane's own tool calls count as exits.
            if (
                e["kind"] == "lookup"
                and not e.get("speculative")
                and "lookup" not in exits
            ):
                exits.append("lookup")
            elif e["kind"] == "dispatch" and "delegate" not in exits:
                exits.append("delegate")
            elif e["kind"] == "spoken_lane" and "answer" not in exits:
                exits.append("answer")
        self.exits = exits
        self.first_exit = exits[0] if exits else "none"
        raw = dict(SESSION.turn) if SESSION.turn else {}
        SESSION.flush_turn()
        # Derived timings (first_audio_ms etc.) exist only on the flushed
        # turn; accept it only if it is THIS step's turn (transcript match).
        derived = {}
        if SESSION.turns and SESSION.turns[-1].get("transcript") == raw.get(
            "transcript"
        ):
            derived = SESSION.turns[-1]
        self.turn = {
            "transcript": raw.get("transcript"),
            "first_audio_ms": derived.get("first_audio_ms"),
            "lane_ttft_ms": derived.get("lane_ttft_ms"),
            "stt_finalize_ms": derived.get("stt_finalize_ms"),
            "lane_prompt_tokens": raw.get("lane_prompt_tokens"),
        }
        RESULTS["steps"].append(
            {
                "name": self.name,
                "expect": self.expect,
                "first_exit": self.first_exit,
                "exits": self.exits,
                "spoken": self.spoken,
                "dispatches": self.dispatches,
                "lookups": self.lookups,
                "anticipations": self.anticipations,
                **self.turn,
            }
        )
        return self


async def run_step(feeder, name, wav, expect, wait="lane", timeout=35):
    st = Step(name, expect)
    n_lane = len(events("spoken_lane"))
    n_disp = len(events("dispatch"))
    await feeder.say(wav)
    if wait in ("lane", "dispatch"):
        if wait == "dispatch":
            # Quiet probe: give a dispatch a head start so the lane-reply
            # wait below measures the ack, not the tool round trip. No
            # note on timeout — finish() records what actually happened.
            t0 = time.time()
            while time.time() - t0 < 10 and len(events("dispatch")) <= n_disp:
                await asyncio.sleep(0.2)
        await wait_for(
            lambda: len(events("spoken_lane")) > n_lane or None,
            timeout,
            f"{name} lane reply",
        )
    await asyncio.sleep(1.0)
    return st.finish()


async def wait_job_done(feeder, timeout=280):
    # Drain ALL pending jobs (including stragglers from earlier phases) so a
    # completion injection can never land inside a later measured turn.
    await wait_for(
        lambda: STATE["bridge"].pending_count() == 0 or None,
        timeout,
        "worker drain",
    )
    await asyncio.sleep(2.5)  # let the injection speak


async def drive(feeder: Feeder):
    await STATE["ready"].wait()
    await asyncio.sleep(6)  # model warm-up (MLX JIT, Pocket load)
    try:
        # ---- Phase A: 20 chat turns ----
        for i, wav in enumerate(CHAT_WAVS):
            await run_step(feeder, f"chat{i+1}", wav, "answer")

        if MODE == "full":
            # ---- Phase B: discipline ----
            await run_step(feeder, "lk_city", "tests/utt_lk_city.wav", "lookup")
            await run_step(feeder, "lk_tz", "tests/utt_lk_tz.wav", "lookup")
            await run_step(feeder, "lk_mia", "tests/utt_lk_mia.wav", "lookup")
            await run_step(feeder, "lk_brain", "tests/utt_lk_brain.wav", "lookup")
            await run_step(feeder, "lk_wife", "tests/utt_lk_wife.wav", "escalate")
            await run_step(feeder, "lk_blood", "tests/utt_lk_blood.wav", "escalate")
            await run_step(feeder, "dlg_soul", "tests/utt_work2.wav", "delegate", wait="dispatch")
            await run_step(feeder, "dlg_os", "tests/utt_work4.wav", "delegate", wait="dispatch")

            # ---- Phase C: ack turns, serialized on completion ----
            await run_step(feeder, "ack_profiles", "tests/utt_work3.wav", "delegate", wait="dispatch")
            await wait_job_done(feeder)
            await run_step(feeder, "ack_email", "tests/utt_wk_email.wav", "delegate", wait="dispatch")
            await wait_job_done(feeder)
            await run_step(feeder, "ack_citytz", "tests/utt_work.wav", "delegate-or-lookup", wait="lane")
            await wait_job_done(feeder)

            # ---- Phase D: in-flow while a job pends ----
            await run_step(feeder, "flow_dispatch", "tests/utt_wk_research.wav", "delegate", wait="dispatch")
            await run_step(feeder, "flow_side", "tests/utt_side2.wav", "answer")
            await run_step(feeder, "flow_bait", "tests/utt_bait2.wav", "answer-hold")
            await run_step(feeder, "flow_check", "tests/utt_check.wav", "answer-status")
            await wait_job_done(feeder)
        else:
            # ---- Baseline: ack turns only, serialized ----
            for name, wav in [
                ("ack_work", "tests/utt_work.wav"),
                ("ack_soul", "tests/utt_work2.wav"),
                ("ack_profiles", "tests/utt_work3.wav"),
                ("ack_os", "tests/utt_work4.wav"),
                ("ack_research", "tests/utt_wk_research.wav"),
            ]:
                await run_step(feeder, name, wav, "delegate", wait="dispatch")
                await wait_job_done(feeder)
    except Exception as e:
        RESULTS["driver_error"] = f"{type(e).__name__}: {e}"
    finally:
        STATE["done"].set()


async def main():
    feeder = Feeder()
    sink = Sink()
    _pipeline, task, bridge = hv.build_pipeline(feeder, sink)
    STATE["bridge"] = bridge
    STATE["done"] = asyncio.Event()
    STATE["ready"].set()

    async def finisher():
        try:
            await asyncio.wait_for(STATE["done"].wait(), timeout=1800)
        except asyncio.TimeoutError:
            RESULTS["driver_error"] = "global timeout"
        await asyncio.sleep(0.5)
        await task.queue_frame(EndFrame())

    await asyncio.gather(PipelineRunner().run(task), finisher())

    chat = [s for s in RESULTS["steps"] if s["name"].startswith("chat")]
    fas = [s["first_audio_ms"] for s in chat if s.get("first_audio_ms")]
    if fas:
        srt = sorted(fas)
        RESULTS["chat_first_audio"] = {
            "values_ms": fas,
            "median_ms": statistics.median(fas),
            "median_excl_first_ms": statistics.median(fas[1:]) if len(fas) > 1 else None,
            "p90_ms": srt[min(len(srt) - 1, int(0.9 * len(srt)))],
            "n": len(fas),
        }
    acks = [
        s
        for s in RESULTS["steps"]
        if (s["name"].startswith("ack") or s["name"] in ("dlg_soul", "dlg_os", "flow_dispatch"))
        and "delegate" in s["exits"]
        and s.get("first_audio_ms")
    ]
    if acks:
        RESULTS["ack_first_audio"] = {
            "turns": {s["name"]: s["first_audio_ms"] for s in acks},
            "median_ms": statistics.median([s["first_audio_ms"] for s in acks]),
            "n": len(acks),
        }
    print("ROUTED_JSON " + json.dumps(RESULTS, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
