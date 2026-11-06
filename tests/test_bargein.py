"""Simulated barge-in measurement (headless).

Pipeline mirrors hermes_voice.py: Feeder -> WhisperSTT -> user aggregator
(Silero VAD + default Smart Turn strategies) -> announcer -> OpenAI LLM ->
Pocket TTS -> silence trimmer -> Sink.

Protocol: ask for a LONG story; once TTS audio is flowing, inject real
speech (tests/known.wav resampled to 16 kHz) as InputAudioRawFrames.
Measure: speech-start -> last TTS audio frame out, and whether an
InterruptionFrame propagated. SIMULATED — no transport/speaker in the loop;
the live perceived figure is Adam's test.
"""
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
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
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

from hermes_voice import (
    GATEWAY_BASE_URL,
    GatewayDownAnnouncer,
    LeadingSilenceTrimmer,
    build_llm,
    build_stt,
    build_tts,
)

STATE = {
    "audio_flowing": asyncio.Event(),
    "done": asyncio.Event(),
    "t0": None,
    "speech_start": None,
    "last_audio": None,
    "audio_frames_after_speech": 0,
    "interruption_seen": False,
    "audio_secs_before": 0.0,
}


def speech_pcm_16k() -> np.ndarray:
    sr, data = scipy.io.wavfile.read("tests/known.wav")
    # known.wav is float32 in [-1, 1]; int16 files need /32768. (An earlier
    # version divided unconditionally and injected digital silence — the
    # first two barge runs measured VAD/STT behavior on zeros, not speech.)
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    else:
        x = data.astype(np.float64)
    if sr != 16000:
        x = resample_poly(x, 16000, sr)
    gain = float(os.environ.get("SPEECH_GAIN", "1"))
    if gain != 1.0:
        x = np.clip(x * gain, -0.98, 0.98)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


class Feeder(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame) and STATE["t0"] is None:
            STATE["t0"] = time.monotonic()
            asyncio.create_task(self._drive())

    async def _drive(self):
        await asyncio.sleep(1.0)
        ctx = LLMContext(
            messages=[
                {
                    "role": "user",
                    "content": "Tell me a long story about a dragon and a lighthouse "
                    "keeper. At least ten sentences, please.",
                }
            ]
        )
        await self.push_frame(LLMContextFrame(context=ctx), FrameDirection.DOWNSTREAM)
        # Wait for the bot to be speaking, then talk over it.
        await STATE["audio_flowing"].wait()
        await asyncio.sleep(0.3)
        pcm = speech_pcm_16k()
        STATE["speech_start"] = time.monotonic()
        chunk = 640  # 40 ms at 16 kHz
        for i in range(0, len(pcm), chunk):
            await self.push_frame(
                InputAudioRawFrame(
                    audio=pcm[i : i + chunk].tobytes(), sample_rate=16000, num_channels=1
                ),
                FrameDirection.DOWNSTREAM,
            )
            await asyncio.sleep(0.005)


TAPS = {"pre_stt": {"audio": 0, "vad_start": 0}, "post_stt": {"audio": 0, "vad_start": 0}}


class Tap(FrameProcessor):
    def __init__(self, tap_name, **kwargs):
        super().__init__(**kwargs)
        self._tap = tap_name

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, InputAudioRawFrame):
                TAPS[self._tap]["audio"] += 1
            elif type(frame).__name__ == "VADUserStartedSpeakingFrame":
                TAPS[self._tap]["vad_start"] += 1
        await self.push_frame(frame, direction)


class Sink(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            STATE["interruption_seen"] = True
        if isinstance(frame, TTSAudioRawFrame):
            now = time.monotonic()
            STATE["last_audio"] = now
            STATE["audio_flowing"].set()
            if STATE["speech_start"] is None:
                STATE["audio_secs_before"] += len(frame.audio) / 2 / frame.sample_rate
            else:
                STATE["audio_frames_after_speech"] += 1
        await self.push_frame(frame, direction)


async def main():
    feeder = Feeder()
    sink = Sink()
    context = LLMContext(messages=[])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=16000,
                params=VADParams(
                    confidence=float(os.environ.get("VAD_CONF", "0.7")),
                    min_volume=float(os.environ.get("VAD_MINVOL", "0.6")),
                    start_secs=0.2,
                    stop_secs=0.6,
                ),
            )
        ),
    )
    pipeline = Pipeline(
        [
            feeder,
            Tap("pre_stt"),
            build_stt(),
            Tap("post_stt"),
            aggregators.user(),
            GatewayDownAnnouncer(),
            build_llm(),
            build_tts(),
            LeadingSilenceTrimmer(),
            sink,
            aggregators.assistant(),
        ]
    )
    task = PipelineTask(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    )

    async def finisher():
        # Wait for speech injection, then for audio to stop (or a 6 s cap).
        while STATE["speech_start"] is None:
            await asyncio.sleep(0.05)
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if STATE["last_audio"] and time.monotonic() - STATE["last_audio"] > 1.2:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)
        await task.queue_frame(__import__("pipecat.frames.frames", fromlist=["EndFrame"]).EndFrame())

    runner = PipelineRunner()
    await asyncio.gather(runner.run(task), finisher())

    res = {
        "simulated": True,
        "gateway": GATEWAY_BASE_URL,
        "bot_audio_secs_before_speech": round(STATE["audio_secs_before"], 2),
        "interruption_frame_seen": STATE["interruption_seen"],
        "speech_start_to_last_audio_ms": (
            round((STATE["last_audio"] - STATE["speech_start"]) * 1000)
            if STATE["last_audio"] and STATE["speech_start"]
            else None
        ),
        "audio_frames_after_speech": STATE["audio_frames_after_speech"],
        "taps": TAPS,
    }
    print("RESULT_JSON " + json.dumps(res))


if __name__ == "__main__":
    asyncio.run(main())
