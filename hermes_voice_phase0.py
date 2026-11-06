#!/usr/bin/env python3
"""Hermes live voice — Pipecat front-end (Phase 0 spike).

Pipeline: LocalAudioTransport (mic/speakers) -> Silero VAD + Smart Turn (in the
user aggregator) -> faster-whisper STT -> Hermes gateway (OpenAI-compatible,
model "hermes-agent") -> Pocket TTS (streaming, catalog voice) -> speakers.

Run (from /Users/ai/workspace/hermes-voice):
    API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-) \
        .venv/bin/python hermes_voice.py

Env overrides: HERMES_GATEWAY_URL, HERMES_VOICE (default eve), HERMES_WHISPER
(default small).
"""
import os

import numpy as np
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    ErrorFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
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
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.pocket_tts.tts import PocketTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

# Default route = Antigravity bridge on http://127.0.0.1:8011/v1
# Override with AGY_GATEWAY_URL or HERMES_GATEWAY_URL.
GATEWAY_BASE_URL = (
    os.environ.get("AGY_GATEWAY_URL")
    or os.environ.get("HERMES_GATEWAY_URL")
    or "http://127.0.0.1:8011/v1"
)
GATEWAY_MODEL = (
    os.environ.get("AGY_WORKER_MODEL")
    or os.environ.get("HERMES_WORKER_MODEL")
    or "gemini-3.8-flash-low"
)
VOICE = os.environ.get("HERMES_VOICE", "eve")
WHISPER_MODEL = os.environ.get("HERMES_WHISPER", "small")

SYSTEM_INSTRUCTION = (
    "You are Vibherpunk (backed by Antigravity) in a live voice conversation. Reply in short, natural spoken "
    "sentences — one or two at a time. No lists, no markdown, no code unless asked. "
    "If a task will take a while, first say what you are doing in one short sentence."
)


class GatewayDownAnnouncer(FrameProcessor):
    """Speak the failure when the LLM service errors, instead of dead air.

    Placement matters: pipecat services push ErrorFrames UPSTREAM, so this
    processor sits BEFORE the LLM service and watches for ErrorFrames flowing
    upstream. On the first one it injects a spoken message downstream (the
    TTS service is after the LLM; unhandled frames pass through).
    """

    MESSAGE = "Sorry, I can't reach Hermes right now. The gateway may be down."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._announced = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, ErrorFrame)
            and direction == FrameDirection.UPSTREAM
            and not self._announced
        ):
            self._announced = True
            logger.warning("LLM error detected — speaking gateway-down message")
            await self.push_frame(
                TTSSpeakFrame(text=self.MESSAGE), FrameDirection.DOWNSTREAM
            )
        await self.push_frame(frame, direction)


class LeadingSilenceTrimmer(FrameProcessor):
    """Drop silent frames at the head of each TTS utterance.

    Pocket streams carry 0.5–0.85 s of digital silence before the first
    speech energy (measured on our own generated WAVs, 2026-10-27). With a
    realtime-paced transport that silence is dead air the user hears. This
    processor swallows sub-threshold audio frames from each TTS start until
    the first frame with real energy, then passes everything through until
    the next utterance.
    """

    THRESHOLD = 300.0  # int16 RMS (~0.01 full scale — the measured floor)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._armed = True
        self.dropped_ms = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (TTSStartedFrame, InterruptionFrame)):
            self._armed = True
        if (
            isinstance(frame, TTSAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
            and self._armed
        ):
            arr = np.frombuffer(frame.audio, dtype=np.int16)
            rms = (
                float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
                if len(arr)
                else 0.0
            )
            if rms < self.THRESHOLD:
                self.dropped_ms += int(len(arr) / frame.sample_rate * 1000)
                return  # swallow the silent frame
            self._armed = False
            logger.debug(f"trimmer: dropped {self.dropped_ms} ms leading silence")
        await self.push_frame(frame, direction)


def build_llm(base_url: str | None = None) -> OpenAILLMService:
    return OpenAILLMService(
        api_key=os.environ.get("API_SERVER_KEY", ""),
        base_url=base_url or GATEWAY_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=GATEWAY_MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )


def build_tts() -> PocketTTSService:
    return PocketTTSService(
        settings=PocketTTSService.Settings(voice=VOICE, language=Language.EN)
    )


def build_stt() -> WhisperSTTService:
    return WhisperSTTService(
        model=WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        settings=WhisperSTTService.Settings(language=Language.EN),
    )


async def main():
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        )
    )

    context = LLMContext(messages=[])
    user_params = LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=16000, params=VADParams(stop_secs=0.6)
        )
        # Default turn strategies apply: VAD + transcription start,
        # TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3) stop.
    )
    aggregators = LLMContextAggregatorPair(context, user_params=user_params)

    pipeline = Pipeline(
        [
            transport.input(),
            build_stt(),
            aggregators.user(),
            GatewayDownAnnouncer(),
            build_llm(),
            build_tts(),
            LeadingSilenceTrimmer(),
            transport.output(),
            aggregators.assistant(),
        ]
    )

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
    await runner.run(task)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
