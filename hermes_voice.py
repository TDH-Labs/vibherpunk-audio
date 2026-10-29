#!/usr/bin/env python3
"""Hermes live voice — Phase 1: two-tier brain.

Tier 1 (conversation lane): a direct streaming model call (OpenCode Zen Go
route, model in LANE_MODEL) with a small disciplined voice prompt and two
tools — delegate_to_hermes and check_delegated_jobs. The lane always
responds immediately; its speculation budget is social, never substantive.

Tier 2 (worker): the full Hermes agent on the default profile, reached
through the gateway's plain /v1 route. Dispatch is asynchronous; when the
worker finishes, its own spoken summary is injected as a spoken turn while
the conversation continues.

Chassis (from Phase 0, preserved in hermes_voice_phase0.py): local audio
transport -> wake-word gate (HERMES_WAKE_ENGINE: oww | sherpa) -> Whisper STT
(MLX small by default) -> Silero VAD + Smart Turn in the user aggregator ->
lane LLM -> Pocket TTS (catalog voice) -> leading-silence trimmer ->
speakers. Spoken failures for both tiers — no dead air. Spoken sleep ("go to sleep" / "stop listening") ends the window at once; worker results that land while the gate is asleep are held and announced first on the next wake.

Run (from the repo root):  ./run.sh
Env knobs: LANE_MODEL (default glm-5.3-flash), LANE_BASE_URL,
HERMES_LANE_BACKEND (relay|local; local = routed lane on a local MLX model —
LOCAL_LANE_MODEL / LOCAL_LANE_BASE_URL — with the quick_lookup third exit),
HERMES_WORKER_URL (default http://127.0.0.1:8642/v1), HERMES_WAKE=off, HERMES_WAKE_ENGINE=oww|sherpa,
HERMES_WAKE_PHRASES, HERMES_WAKE_SENSITIVITY, HERMES_STT=mlx|faster, HERMES_WHISPER (faster-whisper model, default small),
HERMES_VOICE (default eve), HERMES_FOLLOWUP_SECS (default 20).
Keys come from the environment (run.sh exports them from the Hermes .env
files); they are never printed or logged here.
"""
import asyncio
import json
import os
import re
import time
import uuid
from collections import deque
from pathlib import Path

import aiohttp
import numpy as np
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
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
from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("HERMES_VOICE_LOGDIR", str(BASE_DIR / "logs")))

LANE_BASE_URL = os.environ.get("LANE_BASE_URL", "https://opencode.ai/zen/go/v1")
# Measured on this route 2026-10-27 (direct streaming, ~125-token prompt):
# deepseek-v4.1-flash TTFT median 1,551 ms; glm-5.3-flash 434 ms. The lane
# default follows the measurement; LANE_MODEL overrides.
LANE_MODEL = os.environ.get("LANE_MODEL", "glm-5.3-flash")
LANE_SESSION_ID = os.environ.get(
    "LANE_SESSION", f"hermes-voice-lane-{uuid.uuid4().hex[:12]}"
)
# Lane backend: "relay" (default — the OpenCode Zen route above; the
# unchanged production behavior and the rollback), "local" (a fast
# non-reasoning model served on this machine over MLX), or "cloud" (the
# routed configuration — routed prompt, three exits — with the talker on
# the Zen route: GLM-5.3-Flash called directly). "local" and "cloud" are
# the routed backends; "relay" stays byte-for-byte the rollback.
LANE_BACKEND = os.environ.get("HERMES_LANE_BACKEND", "relay").strip().lower()
ROUTED_BACKENDS = ("local", "cloud")
# Token budget for the lane's carried conversation history on the routed
# backends (the routed system prompt + tool schemas ride on top — 1,269
# prompt tokens measured for a fresh routed call). Round 1's ack spikes
# tracked context growth, so the cap keeps talker TTFT flat on long
# conversations.
LANE_CONTEXT_TOKEN_BUDGET = int(
    os.environ.get("HERMES_LANE_CONTEXT_TOKEN_BUDGET", "2200")
)
# Fused lookup tool (round-2 amendment): when on, the routed lanes expose
# a single lookup_or_delegate tool instead of quick_lookup. The API-level
# diagnostic proved the two-call lookup->escalate chain fails even on a
# strong talker (GLM spoke the hand-off sentence with no dispatch behind
# it on the exact round-1 inputs), so the chain is collapsed into one
# model decision: the tool looks up, and on a miss dispatches itself.
FUSED_LOOKUP = os.environ.get("HERMES_FUSED_LOOKUP", "off").strip().lower() == "on"
LOCAL_LANE_BASE_URL = os.environ.get(
    "LOCAL_LANE_BASE_URL", "http://127.0.0.1:8123/v1"
)
LOCAL_LANE_MODEL = os.environ.get(
    "LOCAL_LANE_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit"
)
LOOKUP_BUDGET_SECS = float(os.environ.get("HERMES_LOOKUP_BUDGET_SECS", "0.6"))
# Routing mechanism for the local lane: "inband" (the lane model decides
# the exit inside its single call) or "encoder" (an encoder classifier —
# ModernBERT zero-shot, one forward pass, no generated tokens — decides
# the exit on the final transcript and hands it to the lane as a decided
# directive). Benches decide; see the delegation REPORT.
ROUTING_MODE = os.environ.get("HERMES_ROUTING", "inband").strip().lower()
ENCODER_MODEL = os.environ.get(
    "HERMES_ENCODER_MODEL", "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
)
GLINER_MODEL = os.environ.get(
    "HERMES_GLINER_MODEL", "urchade/gliner_medium-v2.1"
)
MEMORY_DIR = Path(
    os.environ.get(
        "HERMES_MEMORY_DIR", str(Path.home() / ".hermes" / "memories")
    )
)
WORKER_BASE_URL = os.environ.get("HERMES_WORKER_URL", "http://127.0.0.1:8642/v1")
WORKER_MODEL = os.environ.get("HERMES_WORKER_MODEL", "hermes-agent")
VOICE = os.environ.get("HERMES_VOICE", "eve")
WHISPER_MODEL = os.environ.get("HERMES_WHISPER", "small")
STT_ENGINE = os.environ.get("HERMES_STT", "mlx")
WAKE_ENABLED = os.environ.get("HERMES_WAKE", "on") != "off"
FOLLOWUP_SECS = float(os.environ.get("HERMES_FOLLOWUP_SECS", "20"))
PENDING_FOLLOWUP_SECS = 120.0
WAKE_MODEL_PATH = os.environ.get(
    "HERMES_WAKE_MODEL",
    os.path.expanduser("~/.hermes/hermes-agent/tools/wakewords/hey_hermes.tflite"),
)
# Wake engine: "oww" (default — pyopen-wakeword + WAKE_MODEL_PATH, the
# original behavior) or "sherpa" (sherpa-onnx open-vocabulary KWS over
# WAKE_PHRASES; any phrase, no per-phrase trained model).
WAKE_ENGINE = os.environ.get("HERMES_WAKE_ENGINE", "oww").strip().lower()
WAKE_SHERPA_DIR = os.environ.get(
    "HERMES_WAKE_SHERPA_DIR",
    str(Path(__file__).resolve().parent / "models" / "sherpa-kws"),
)
WAKE_SENSITIVITY = float(os.environ.get("HERMES_WAKE_SENSITIVITY", "0.6"))
WAKE_PHRASES = [
    p.strip().lower()
    for p in os.environ.get(
        "HERMES_WAKE_PHRASES", "hey clanker,hey viber"
    ).split(",")
    if p.strip()
]
# Extra tokenizations registered for a sherpa phrase. "viber" as spelled
# (HEY VIBER) barely matched in the Oct 27 eval; the phonetic HEY VIBAR
# fired. Both count as the same wake phrase ("hey viber").
SHERPA_PHRASE_VARIANTS = {"hey viber": ("HEY VIBER", "HEY VIBAR")}

LANE_SYSTEM_PROMPT = """You are Vibherpunk, speaking out loud with the user on their Mac. You are the fast conversation lane of a two-tier system: you handle the conversation; a full agent — with tools, memory, and files — is your back office for real work.

How you talk:
- Short, natural spoken sentences, one or two at a time. No lists, no markdown, no code, nothing read aloud that only makes sense written.
- If a message starts with "hey Clanker" or "hey Viber," that's just the wake phrase. Ignore it and answer the rest.

What you answer yourself: chat, small talk, simple general-knowledge questions, and follow-ups about this conversation. Answer those directly and briefly.

Delegating — this is a hard rule, not a style tip:
- Anything that needs the user's real data or real work — his memory, files, calendar, messages, accounts, the web, research, or any multi-step task — is not yours to answer. Call delegate_to_hermes with a clear, self-contained task description. The back office does not hear this conversation, so include everything it needs.
- A spoken promise without a dispatch is a failure: if you say you are checking something, you must actually have called delegate_to_hermes.
- You never know what the back office will find. Never state a result, a fact about the user, or an outcome you have not been given. Never guess what a finished job probably found.
- When you dispatch, say one short sentence about what you handed off — process only, no promised result. For example: "I've asked my back office to check its memory for that — it'll take a bit."
- Job results are announced in the conversation when they land, in the back office's own summary. If the user asks about a result afterwards, use only what was announced. If a result contradicts something you said earlier, correct yourself plainly.
- If the user asks whether something is done, call check_delegated_jobs and answer from that — "still working" only when it actually is.
- If the user presses you for a result that is not back yet, hold the line warmly and briefly: it's still with the back office, and you'll say it the moment it lands. Do not cave, and do not invent.

You have no other tools and no memory of the user beyond this conversation. Anything that needs one goes to the back office."""

LANE_SYSTEM_PROMPT_ROUTED = """You are Vibherpunk, speaking out loud with the user on their Mac. You are the fast conversation lane of a two-tier system: you handle the conversation; a full agent — with tools, memory, and files — is your back office for real work.

How you talk:
- Short, natural spoken sentences, one or two at a time. No lists, no markdown, no code, nothing read aloud that only makes sense written.
- If a message starts with "hey Clanker" or "hey Viber," that's just the wake phrase. Ignore it and answer the rest.

Every turn takes exactly one of three exits. Choose inside this one reply. A short system note may follow the user's words naming the exit decided for this turn (for example "[Routing decision: LOOKUP …]"). That note is authoritative — take that exit. When it names a tool, that tool call is your first action, before any spoken words.

1. ANSWER — chat, small talk, simple general-knowledge questions, and follow-ups about this conversation. Answer directly and briefly.

2. LOOK UP — exactly one stored fact about the user or their setup: where they live, their timezone, a name, how his machine or agents are set up — the kind of thing kept in local memory. Call quick_lookup with a short query of content words (for example "city lives in" or "timezone"). It returns stored passages with their source. Say the fact only if a passage directly states it, briefly and as stored. If it finds nothing, or the passages don't actually state the fact, do NOT fill the gap from your own knowledge — delegate instead (exit 3).

3. DELEGATE — anything needing real work or live data beyond a single stored fact: files, calendar, messages, accounts, the web, research, several facts at once, exact file contents, or any multi-step task. Call delegate_to_hermes with a clear, self-contained task description. The back office does not hear this conversation, so include everything it needs. Names and file names are load-bearing: pass them verbatim. The user's agent persona file is SOUL.md — if the transcript says "sole file" or "soul file", they mean SOUL.md; write SOUL.md in the task.

Delegating — this is a hard rule, not a style tip:
- A spoken promise without a dispatch is a failure: if you say you are checking something, you must actually have called delegate_to_hermes.
- You never know what the back office will find. Never state a result, a fact about the user, or an outcome you have not been given — not from a lookup that missed, not from a job still running. Never guess what a finished job probably found.
- When you dispatch, say one short sentence about what you handed off — process only, no promised result. For example: "I've asked my back office to check its memory for that — it'll take a bit."
- Job results are announced in the conversation when they land, in the back office's own summary. If the user asks about a result afterwards, use only what was announced. If a result contradicts something you said earlier, correct yourself plainly.
- If the user asks whether something is done, call check_delegated_jobs and answer from that — "still working" only when it actually is.
- If the user presses you for a result that is not back yet, hold the line warmly and briefly: it's still with the back office, and you'll say it the moment it lands. Do not cave, and do not invent.

You have no other tools and no memory of the user beyond this conversation and what quick_lookup returns. Anything that needs more goes to the back office."""

WORKER_DISPATCH_TEMPLATE = """A request came in over live voice. Do the work fully — use your tools and memory as needed — then report the outcome.

Request: {task}

Format: give your normal, complete answer first. Then, on its own final line, add exactly:
SPOKEN SUMMARY: <one to three short sentences, written to be read aloud, stating the actual outcome>"""

WORKER_DOWN_MESSAGE = (
    "Sorry — Vibherpunk couldn't complete that task. "
    "The agent seems to be unreachable right now."
)
LANE_DOWN_MESSAGE = (
    "Sorry, I'm having trouble reaching my conversation model right now."
)
SLEEP_CONFIRMATION = "Going to sleep — say hey Clanker when you need me."


# --------------------------------------------------------------------------
# Session state + metrics
# --------------------------------------------------------------------------
class Session:
    """Shared session state: event/metric log (JSONL on disk + in-memory for
    tests), the spoken transcript, activity tracking for the wake gate."""

    def __init__(self):
        self.events: list[dict] = []
        self.spoken: list[dict] = []
        self.last_activity = time.monotonic()
        self.turn: dict | None = None
        self.turns: list[dict] = []
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = LOG_DIR / "metrics.jsonl"
        self.record(
            "session_init",
            lane_model=(
                LOCAL_LANE_MODEL if LANE_BACKEND == "local" else LANE_MODEL
            ),
            lane_backend=LANE_BACKEND,
            routing=ROUTING_MODE,
            stt=STT_ENGINE,
        )

    def record(self, kind: str, **data):
        evt = {"kind": kind, "wall": time.time(), "mono": time.monotonic(), **data}
        self.events.append(evt)
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(evt, default=str) + "\n")
        except Exception:
            pass
        logger.info(f"EVENT {kind} {data}")
        return evt

    def note_activity(self):
        self.last_activity = time.monotonic()

    # -- turn metric assembly (taps write into self.turn) --
    def open_turn(self):
        self.flush_turn()
        self.turn = {
            "t_end_of_turn": None,
            "t_transcription": None,
            "transcript": None,
            "t_first_token": None,
            "t_first_sound": None,
            "lane_prompt_tokens": None,
            "lane_completion_tokens": None,
        }

    def flush_turn(self):
        if self.turn and (
            self.turn.get("t_first_sound") or self.turn.get("t_first_token")
        ):
            t = dict(self.turn)

            def delta(a, b):
                if t.get(a) is not None and t.get(b) is not None:
                    return round((t[a] - t[b]) * 1000)
                return None

            t["stt_finalize_ms"] = delta("t_transcription", "t_end_of_turn")
            t["lane_ttft_ms"] = delta("t_first_token", "t_transcription")
            t["first_audio_ms"] = delta("t_first_sound", "t_end_of_turn")
            self.turns.append(t)
            self.record("turn", **t)
        self.turn = None


SESSION = Session()


# --------------------------------------------------------------------------
# TTS helpers (carried from Phase 0 — see hermes_voice_phase0.py)
# --------------------------------------------------------------------------
class LeadingSilenceTrimmer(FrameProcessor):
    """Drop silent frames at the head of each TTS utterance (Pocket streams
    carry 0.5–0.85 s of digital leading silence — dead air if played)."""

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


class SpokenFailureAnnouncer(FrameProcessor):
    """Speak the failure when the lane LLM errors, instead of dead air.
    Pipecat services push ErrorFrames UPSTREAM, so this sits BEFORE the lane
    LLM and watches upstream. Worker failures are announced by the bridge."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._announced = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, ErrorFrame)
            and direction == FrameDirection.UPSTREAM
            and not self._announced
        ):
            self._announced = True
            self._session.record("lane_error", error=str(frame)[:200])
            logger.warning("lane LLM error — speaking failure message")
            await self.push_frame(
                TTSSpeakFrame(text=LANE_DOWN_MESSAGE), FrameDirection.DOWNSTREAM
            )
        await self.push_frame(frame, direction)


# --------------------------------------------------------------------------
# Wake-word gate
# --------------------------------------------------------------------------
class WakeWordGate(FrameProcessor):
    """Gates the mic path on a wake phrase. Engine is selectable with
    HERMES_WAKE_ENGINE: "oww" (default — pyopen-wakeword + the bundled
    hey_hermes.tflite, the same engine/model Hermes itself uses) or
    "sherpa" (sherpa-onnx open-vocabulary keyword spotting over
    WAKE_PHRASES, default "hey clanker" / "hey viber").

    ARMED: input audio is swallowed (never reaches STT/VAD); a rolling
    pre-roll buffer is kept so the utterance start isn't clipped. On fire,
    the pre-roll flushes downstream and the gate goes ACTIVE.
    ACTIVE: audio passes. After FOLLOWUP_SECS without user/bot activity
    (PENDING_FOLLOWUP_SECS while a job is pending), the gate re-arms.
    """

    THRESHOLD = 0.6  # Hermes' own defaults for this model
    CONFIRM_FRAMES = 3
    WINDOW_SAMPLES = 1280  # 80 ms at 16 kHz — the detector's streaming window
    PREROLL_SECS = 1.6

    def __init__(self, session: Session, bridge: "DelegationBridge", **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._bridge = bridge
        self._active = not WAKE_ENABLED
        self._detector_ok = False
        self._features = None
        self._model = None
        self._spotter = None
        self._sherpa_stream = None
        self._sherpa_keywords_file = None
        self._sherpa_display_to_phrase: dict[str, str] = {}
        self._sherpa_threshold = 0.0
        self.last_wake_phrase: str | None = None
        self._streak = 0
        self._pending = bytearray()
        self._preroll: deque[bytes] = deque()
        self._preroll_bytes = 0
        self._watchdog: asyncio.Task | None = None
        self._mic_peak = 0
        self._mic_checked = False
        if WAKE_ENABLED:
            self._init_detector()

    def _init_detector(self):
        try:
            if WAKE_ENGINE == "sherpa":
                self._init_sherpa()
            else:
                self._init_oww()
            self._detector_ok = True
        except Exception as e:
            logger.error(
                f"wake detector failed to load ({e}) — running UNGATED. "
                "Fix the detector before relying on wake gating."
            )
            self._session.record("wake_detector_failed", error=str(e)[:200])
            self._active = True

    def _init_oww(self):
        from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures

        self._features = OpenWakeWordFeatures.from_builtin()
        self._model = OpenWakeWord.from_model(WAKE_MODEL_PATH)
        logger.info(f"wake gate armed: engine=oww model={self._model.id}")

    def _init_sherpa(self):
        import tempfile

        import sherpa_onnx
        from sherpa_onnx import text2token

        d = Path(WAKE_SHERPA_DIR)
        if not (d / "tokens.txt").exists():
            raise RuntimeError(f"sherpa KWS model not found at {d}")

        # Registered keyword forms -> canonical phrase. A phrase may have
        # more than one tokenization (see SHERPA_PHRASE_VARIANTS).
        forms: list[str] = []
        display_to_phrase: dict[str, str] = {}
        for phrase in WAKE_PHRASES:
            for form in SHERPA_PHRASE_VARIANTS.get(phrase, (phrase.upper(),)):
                display = form.replace(" ", "_")
                if display not in display_to_phrase:
                    display_to_phrase[display] = phrase
                    forms.append(form)
        toks = text2token(
            forms,
            tokens=str(d / "tokens.txt"),
            tokens_type="bpe",
            bpe_model=str(d / "bpe.model"),
        )
        kw = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="vibherpunk-kws-",
            delete=False,
            encoding="utf-8",
        )
        for form, t in zip(forms, toks):
            kw.write(" ".join(t) + f" @{form.replace(' ', '_')}\n")
        kw.close()
        self._sherpa_keywords_file = kw.name
        self._sherpa_display_to_phrase = display_to_phrase

        def model_file(part: str) -> str:
            hits = sorted(
                p
                for p in d.glob(f"{part}-*.onnx")
                if not p.name.endswith("int8.onnx")
            )
            if not hits:
                raise RuntimeError(
                    f"sherpa KWS model file missing: {d}/{part}-*.onnx"
                )
            return str(hits[0])

        # Same sensitivity -> threshold mapping Hermes uses for this engine.
        self._sherpa_threshold = 0.05 + 0.4 * WAKE_SENSITIVITY
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(d / "tokens.txt"),
            encoder=model_file("encoder"),
            decoder=model_file("decoder"),
            joiner=model_file("joiner"),
            keywords_file=self._sherpa_keywords_file,
            keywords_threshold=self._sherpa_threshold,
            num_threads=1,
        )
        self._sherpa_stream = self._spotter.create_stream()
        logger.info(
            f"wake gate armed: engine=sherpa phrases={WAKE_PHRASES} "
            f"threshold={self._sherpa_threshold:.2f}"
        )

    def _reset_detector(self):
        self._streak = 0
        self._pending = bytearray()
        if WAKE_ENGINE == "sherpa":
            # Fresh stream drops buffered audio/decoder state, so a fire
            # (or a re-arm) can never re-trigger on stale audio.
            try:
                if self._spotter is not None:
                    self._sherpa_stream = self._spotter.create_stream()
            except Exception:
                pass
            return
        try:
            if self._features:
                self._features.reset()
            if self._model:
                self._model.reset()
        except Exception:
            pass

    def _detect(self, pcm_bytes: bytes) -> bool:
        if WAKE_ENGINE == "sherpa":
            return self._detect_sherpa(pcm_bytes)
        return self._detect_oww(pcm_bytes)

    def _detect_sherpa(self, pcm_bytes: bytes) -> bool:
        self._pending.extend(pcm_bytes)
        win = self.WINDOW_SAMPLES * 2
        while len(self._pending) >= win:
            chunk = bytes(self._pending[:win])
            del self._pending[:win]
            samples = (
                np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            try:
                self._sherpa_stream.accept_waveform(16000, samples)
                while self._spotter.is_ready(self._sherpa_stream):
                    self._spotter.decode_stream(self._sherpa_stream)
                    result = self._spotter.get_result(self._sherpa_stream)
                    if result:
                        display = str(result)
                        self.last_wake_phrase = (
                            self._sherpa_display_to_phrase.get(
                                display,
                                display.replace("_", " ").lower(),
                            )
                        )
                        self._spotter.reset_stream(self._sherpa_stream)
                        return True
            except Exception as e:
                logger.warning(f"wake detector error: {e}")
                return False
        return False

    def _detect_oww(self, pcm_bytes: bytes) -> bool:
        self._pending.extend(pcm_bytes)
        fired = False
        win = self.WINDOW_SAMPLES * 2
        while len(self._pending) >= win:
            chunk = bytes(self._pending[:win])
            del self._pending[:win]
            over = False
            try:
                for emb in self._features.process_streaming(chunk):
                    for score in self._model.process_streaming(emb):
                        if float(score) >= self.THRESHOLD:
                            over = True
            except Exception as e:
                logger.warning(f"wake detector error: {e}")
                return False
            if over:
                self._streak += 1
                if self._streak >= self.CONFIRM_FRAMES:
                    fired = True
                    self.last_wake_phrase = "hey hermes"
                    self._streak = 0
            else:
                self._streak = 0
        return fired

    def is_armed(self) -> bool:
        """True when the gate is asleep (audio swallowed, wake detector
        listening). Always False when the wake gate is disabled (mobile)."""
        return WAKE_ENABLED and not self._active

    def rearm_now(self, reason: str = "sleep_command"):
        """Re-arm immediately (spoken sleep), instead of waiting out the
        follow-up window. Safe to call when already armed or ungated."""
        if not WAKE_ENABLED or not self._active:
            return
        self._active = False
        self._reset_detector()
        self._preroll.clear()
        self._preroll_bytes = 0
        self._session.record("wake_rearm", reason=reason)
        logger.info(f"wake gate re-armed ({reason})")

    async def _watchdog_loop(self):
        # Mic self-check + re-arm timer.
        await asyncio.sleep(5)
        if self._mic_peak == 0 and WAKE_ENABLED:
            logger.error(
                "MIC SILENT: zero input peak in first 5 s — likely a macOS "
                "microphone (TCC) permission problem for this process. "
                "Run ./run.sh once from Terminal and Allow the mic prompt."
            )
            self._session.record("mic_silent_warning")
        self._mic_checked = True
        while True:
            await asyncio.sleep(1.0)
            if self._active and WAKE_ENABLED:
                window = (
                    PENDING_FOLLOWUP_SECS
                    if self._bridge.pending_count()
                    else FOLLOWUP_SECS
                )
                if time.monotonic() - self._session.last_activity > window:
                    self._active = False
                    self._reset_detector()
                    self._session.record("wake_rearm", idle_window_s=window)
                    logger.info("wake gate re-armed (follow-up window expired)")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._watchdog is None:
            self._watchdog = asyncio.create_task(self._watchdog_loop())
        if not (
            isinstance(frame, InputAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            await self.push_frame(frame, direction)
            return
        # Input audio.
        arr = np.frombuffer(frame.audio, dtype=np.int16)
        if len(arr):
            peak = int(np.max(np.abs(arr.astype(np.int32))))
            if peak > self._mic_peak:
                self._mic_peak = peak
            if peak > 500:
                self._session.note_activity()
        if self._active or not self._detector_ok:
            await self.push_frame(frame, direction)
            return
        # ARMED: keep pre-roll, run detector, swallow.
        self._preroll.append(frame.audio)
        self._preroll_bytes += len(frame.audio)
        max_bytes = int(self.PREROLL_SECS * frame.sample_rate) * 2
        while self._preroll_bytes > max_bytes and self._preroll:
            self._preroll_bytes -= len(self._preroll.popleft())
        if self._detect(frame.audio):
            self._active = True
            self._session.note_activity()
            if self.last_wake_phrase:
                self._session.record("wake_fire", phrase=self.last_wake_phrase)
            else:
                self._session.record("wake_fire")
            logger.info(
                f"WAKE: '{self.last_wake_phrase or 'hey hermes'}' "
                "detected — gate active"
            )
            preroll = b"".join(self._preroll)
            self._preroll.clear()
            self._preroll_bytes = 0
            if preroll:
                await self.push_frame(
                    InputAudioRawFrame(
                        audio=preroll,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                    ),
                    FrameDirection.DOWNSTREAM,
                )
            asyncio.create_task(self._bridge.flush_held())
        # else: swallowed


# --------------------------------------------------------------------------
# Tier 2 — delegation bridge
# --------------------------------------------------------------------------
class DelegationBridge:
    """Async dispatch to the full Hermes agent (default profile via the
    gateway). Owns job state; on completion injects the worker's own spoken
    summary as a TTSSpeakFrame and notes the result into the lane context."""

    def __init__(self, session: Session, context: LLMContext):
        self._session = session
        self._context = context
        self._worker = None  # PipelineTask/worker for queue_frame injection
        self.jobs: dict[str, dict] = {}
        self._seq = 0
        self._gate = None  # WakeWordGate, attached by the pipeline builder
        self.held: list[dict] = []  # results completed while the gate slept
        self.sleep_processor = None  # SleepCommandProcessor (builder-wired)
        self.anticipator = None  # Anticipator, attached by the builder

    def attach_worker(self, worker):
        self._worker = worker

    def attach_gate(self, gate):
        self._gate = gate

    def attach_anticipator(self, anticipator):
        self.anticipator = anticipator

    def pending_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j["status"] == "running")

    async def dispatch(self, task_text: str) -> dict:
        self._seq += 1
        job_id = f"job-{self._seq}"
        job = {
            "id": job_id,
            "task": task_text,
            "status": "running",
            "started_mono": time.monotonic(),
            "result": None,
            "summary": None,
            "error": None,
        }
        self.jobs[job_id] = job
        self._session.note_activity()
        self._session.record("dispatch", job_id=job_id, task=task_text[:300])
        asyncio.create_task(self._run(job))
        return {
            "job_id": job_id,
            "status": "dispatched",
            "note": (
                "Your back office is working on this in the background. Acknowledge "
                "briefly (process only). Do NOT state any result — the "
                "actual result will be announced in this conversation when "
                "it finishes."
            ),
        }

    def status(self) -> dict:
        out = []
        for j in self.jobs.values():
            out.append(
                {
                    "job_id": j["id"],
                    "task": j["task"][:120],
                    "status": j["status"],
                    "elapsed_s": round(time.monotonic() - j["started_mono"], 1),
                    "summary_if_done": j["summary"] if j["status"] == "done" else None,
                }
            )
        return {"jobs": out}

    async def _run(self, job: dict):
        prompt = WORKER_DISPATCH_TEMPLATE.format(task=job["task"])
        try:
            text, usage = await self._call_worker(prompt)
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)[:300]
            self._session.record(
                "worker_error", job_id=job["id"], error=job["error"]
            )
            logger.warning(f"worker {job['id']} failed: {e}")
            await self._deliver(job, None, is_error=True)
            return
        job["status"] = "done"
        job["result"] = text
        summary = self._spoken_summary(text)
        job["summary"] = summary
        self._session.note_activity()
        self._session.record(
            "worker_done",
            job_id=job["id"],
            latency_s=round(time.monotonic() - job["started_mono"], 2),
            worker_prompt_tokens=(usage or {}).get("prompt_tokens"),
            worker_completion_tokens=(usage or {}).get("completion_tokens"),
            summary=summary[:300],
        )
        logger.info(f"worker {job['id']} done — delivering summary")
        await self._deliver(job, summary)

    async def _deliver(self, job: dict, summary, is_error: bool = False) -> bool:
        """Speak a finished job's result — or HOLD it when the wake gate is
        asleep, so a quiet room stays quiet. Held results are announced by
        flush_held() on the next wake. Returns True if the result was held."""
        if self._gate is not None and self._gate.is_armed():
            announce = (
                WORKER_DOWN_MESSAGE
                if is_error
                else f"While you were asleep, Vibherpunk finished: {summary}"
            )
            self.held.append(
                {
                    "job_id": job["id"],
                    "announce": announce,
                    "summary": summary,
                    "is_error": is_error,
                    "completed_mono": time.monotonic(),
                }
            )
            self._session.record(
                "held_result_held",
                job_id=job["id"],
                summary=(summary or "")[:300],
            )
            logger.info(f"worker {job['id']} finished while asleep — result held")
            self._note_context(
                f"[Update: the background task \"{job['task'][:80]}\" is DONE, "
                "but the user is away (wake gate asleep). Its result is HELD "
                "and will be announced when they return. Do not mention it "
                "before then.]"
            )
            return True
        await self._speak(WORKER_DOWN_MESSAGE if is_error else summary)
        if is_error:
            self._note_context(
                f"[Update: the background task \"{job['task'][:80]}\" FAILED — "
                "The back office appears unreachable. The user has been told. If they "
                "ask, confirm it failed; do not invent a result.]"
            )
        else:
            self._note_context(
                f"[Update: the background task \"{job['task'][:80]}\" is DONE. "
                f"Result announced to the user: \"{summary}\" "
                "If the user asks about it, use only this announced result.]"
            )
        return False

    async def flush_held(self):
        """Announce everything held while the gate was asleep, in completion
        order. Called by the wake gate the moment it fires (desk) and by the
        session manager when a phone connects (mobile)."""
        if not self.held:
            return
        items, self.held = self.held, []
        for it in items:
            self._session.record(
                "held_result_announced",
                job_id=it["job_id"],
                summary=(it.get("summary") or "")[:300],
            )
            logger.info(f"announcing held result for {it['job_id']}")
            await self._speak(it["announce"])
            self._note_context(
                "[Update: a result that was held while the user was away has "
                f"now been announced: \"{it['announce']}\" If the user asks "
                "about it, use only this announced result.]"
            )

    @staticmethod
    def _spoken_summary(text: str) -> str:
        m = re.search(r"SPOKEN SUMMARY:\s*(.+)", text, flags=re.DOTALL)
        if m:
            return " ".join(m.group(1).split()).strip()
        # Fallback: the worker's own first two sentences (its words, not a
        # lane paraphrase).
        flat = " ".join(text.split()).strip()
        parts = re.split(r"(?<=[.!?])\s+", flat)
        return " ".join(parts[:2])[:500] if parts else flat[:500]

    async def _speak(self, text: str):
        if self._worker is not None:
            await self._worker.queue_frame(TTSSpeakFrame(text=text))

    def _note_context(self, note: str):
        try:
            self._context.add_message({"role": "system", "content": note})
        except Exception as e:
            logger.warning(f"context note failed: {e}")

    async def _call_worker(self, prompt: str):
        key = os.environ.get("API_SERVER_KEY", "")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": WORKER_MODEL,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
        }
        timeout = aiohttp.ClientTimeout(total=900, connect=10)
        text_parts: list[str] = []
        usage = None
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(
                WORKER_BASE_URL.rstrip("/") + "/chat/completions",
                json=payload,
                headers=headers,
            ) as r:
                if r.status != 200:
                    body = (await r.text())[:200]
                    raise RuntimeError(f"worker HTTP {r.status}: {body}")
                async for raw in r.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except Exception:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices", []):
                        delta = ch.get("delta", {})
                        if delta.get("content"):
                            text_parts.append(delta["content"])
        return "".join(text_parts).strip(), usage


# --------------------------------------------------------------------------
# Spoken sleep command
# --------------------------------------------------------------------------
_SLEEP_PHRASES = (("go", "to", "sleep"), ("stop", "listening"))
_SLEEP_LEAD_STRIP = {"hey", "hermes", "okay", "ok", "please"}
_SLEEP_TAIL_STRIP = {
    "please", "now", "thanks", "thank", "you", "hermes",
    "today", "tonight", "for", "and",
}
_SLEEP_LEADINS = {
    ("thats", "all"),
    ("that", "s", "all"),
    ("that", "is", "all"),
    ("okay", "thats", "all"),
    ("ok", "thats", "all"),
    ("alright", "thats", "all"),
}


def is_sleep_command(text: str) -> bool:
    """Conservative match for the spoken sleep command.

    True only when the utterance IS the command: the whole utterance is
    "go to sleep" / "stop listening" (modulo address terms and polite
    fillers), or the phrase is a clear trailing clause after a tiny
    lead-in ("that's all, go to sleep"). Content that merely contains
    the phrase — "stop listening to the radio and check the weather",
    "tell my kid to go to sleep", "go to sleep and set an alarm" — is
    NOT a sleep command. When ambiguous, prefer NOT sleeping.
    """
    if not text:
        return False
    norm = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("\u2019", "'").replace("'", ""))
    toks = norm.split()
    while toks and toks[0] in _SLEEP_LEAD_STRIP:
        toks.pop(0)
    while toks and toks[-1] in _SLEEP_TAIL_STRIP:
        toks.pop()
    if not toks:
        return False
    for phrase in _SLEEP_PHRASES:
        n = len(phrase)
        if tuple(toks) == phrase:
            return True
        if tuple(toks[-n:]) == phrase and tuple(toks[:-n]) in _SLEEP_LEADINS:
            return True
    return False


class SleepCommandProcessor(FrameProcessor):
    """Between STT and the transcription tap: intercepts the spoken sleep
    command so it ends the session instead of becoming conversation.

    On a match: the transcription is swallowed (the lane never sees it, it
    is never dispatched or answered as content), a short confirmation is
    spoken, and the registered callbacks run (desk: the wake gate re-arms
    immediately; mobile: the session ends once the confirmation has
    played)."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._callbacks: list = []

    def add_callback(self, cb):
        self._callbacks.append(cb)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
            and is_sleep_command(frame.text)
        ):
            self._session.record("sleep_command", text=frame.text[:200])
            logger.info(f"SLEEP command heard: {frame.text.strip()!r}")
            await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(
                TTSSpeakFrame(text=SLEEP_CONFIRMATION), FrameDirection.DOWNSTREAM
            )
            for cb in self._callbacks:
                try:
                    r = cb()
                    if asyncio.iscoroutine(r):
                        asyncio.create_task(r)
                except Exception as e:
                    logger.warning(f"sleep callback failed: {e}")
            return  # swallowed — never reaches the lane or the worker
        await self.push_frame(frame, direction)


# --------------------------------------------------------------------------
# Metric / transcript taps
# --------------------------------------------------------------------------
class TranscriptionTap(FrameProcessor):
    """Between STT and the user aggregator: timestamps transcriptions."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._session.turn is not None:
                self._session.turn["t_transcription"] = time.monotonic()
                self._session.turn["transcript"] = frame.text
            self._session.record("transcription", text=frame.text[:200])
        await self.push_frame(frame, direction)


class TurnTap(FrameProcessor):
    """Between the user aggregator and the announcer: opens a turn at VAD
    end-of-turn (the metric's t0, per the master scope definition)."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, VADUserStoppedSpeakingFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            self._session.open_turn()
            self._session.turn["t_end_of_turn"] = time.monotonic()
        await self.push_frame(frame, direction)


class TextTap(FrameProcessor):
    """Between the lane LLM and TTS: lane first-token timing, lane token
    usage, and the spoken transcript (lane text + injected speak frames)."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._buf: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMTextFrame):
                if self._session.turn is not None and self._session.turn.get("t_first_token") is None:
                    self._session.turn["t_first_token"] = time.monotonic()
                self._buf.append(frame.text)
            elif isinstance(frame, TTSSpeakFrame):
                self._session.spoken.append({"source": "injected", "text": frame.text})
                self._session.record("spoken_inject", text=frame.text[:300])
            elif isinstance(frame, MetricsFrame):
                for d in frame.data:
                    val = getattr(d, "value", None)
                    if isinstance(val, LLMTokenUsage):
                        if self._session.turn is not None:
                            self._session.turn["lane_prompt_tokens"] = val.prompt_tokens
                            self._session.turn["lane_completion_tokens"] = val.completion_tokens
            elif isinstance(frame, LLMFullResponseEndFrame):
                text = "".join(self._buf).strip()
                self._buf = []
                if text:
                    self._session.spoken.append({"source": "lane", "text": text})
                    self._session.record("spoken_lane", text=text[:300])
        await self.push_frame(frame, direction)


class SoundTap(FrameProcessor):
    """After the trimmer, before output: first AUDIBLE sound per turn
    (post-trim, RMS over the same threshold the trimmer uses) + bot
    activity for the wake gate's follow-up window."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            self._session.note_activity()
            if self._session.turn is not None and self._session.turn.get("t_first_sound") is None:
                arr = np.frombuffer(frame.audio, dtype=np.int16)
                if len(arr) and float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))) > 300:
                    self._session.turn["t_first_sound"] = time.monotonic()
        await self.push_frame(frame, direction)


def _estimate_tokens(msg) -> int:
    """Rough token estimate for one context message (~4 chars/token)."""
    if not isinstance(msg, dict):
        return 8
    total = 8
    content = msg.get("content")
    if isinstance(content, str):
        total += len(content) // 4
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        total += len(str(fn.get("arguments", ""))) // 4 + 8
    return total


def _cap_context_tokens(msgs: list, budget: int) -> list | None:
    """Enforce a token budget on carried history. Returns the capped
    message list, or None if already within budget. Rules: truncate any
    single oversized message (worker summaries are the fat ones) except
    the latest; keep only the newest routing-directive note; then drop
    oldest whole turn-units (an assistant tool-call message stays glued
    to its tool results) until under budget; never orphan a tool result
    at the head."""
    total = sum(_estimate_tokens(m) for m in msgs)
    if total <= budget:
        return None
    out = []
    last_directive = max(
        (
            i
            for i, m in enumerate(msgs)
            if isinstance(m, dict)
            and m.get("role") == "system"
            and isinstance(m.get("content"), str)
            and m["content"].startswith("[Routing decision:")
        ),
        default=-1,
    )
    for i, m in enumerate(msgs):
        if (
            isinstance(m, dict)
            and m.get("role") == "system"
            and isinstance(m.get("content"), str)
            and m["content"].startswith("[Routing decision:")
            and i != last_directive
        ):
            continue
        if (
            i < len(msgs) - 1
            and isinstance(m, dict)
            and isinstance(m.get("content"), str)
            and len(m["content"]) > 3600
        ):
            m = dict(m)
            m["content"] = m["content"][:2400] + " …[trimmed for lane context]"
        out.append(m)
    # Group into units: assistant tool-call + its tool results stay glued.
    units: list[list] = []
    for m in out:
        if (
            isinstance(m, dict)
            and m.get("role") == "tool"
            and units
            and any(
                (u.get("tool_calls") if isinstance(u, dict) else None)
                for u in units[-1]
            )
        ):
            units[-1].append(m)
        else:
            units.append([m])
    total = sum(_estimate_tokens(m) for u in units for m in u)
    while total > budget and len(units) > 1:
        dropped = units.pop(0)
        total -= sum(_estimate_tokens(m) for m in dropped)
    flat = [m for u in units for m in u]
    while flat and isinstance(flat[0], dict) and flat[0].get("role") == "tool":
        flat.pop(0)
    return flat


def _cap_context_in_place(context) -> None:
    """Apply the routed lanes' context cap to a live LLMContext."""
    try:
        capped = _cap_context_tokens(
            context.get_messages(), LANE_CONTEXT_TOKEN_BUDGET
        )
        if capped is not None:
            context.set_messages(capped)
    except Exception as e:
        logger.warning(f"lane context cap failed: {e}")


class ContextTrimmer(FrameProcessor):
    """Keeps the lane's rolling window small: cap context messages, never
    leaving an orphaned tool result at the head."""

    MAX_MESSAGES = 26

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame):
            try:
                if LANE_BACKEND in ROUTED_BACKENDS:
                    _cap_context_in_place(self._context)
                msgs = self._context.get_messages()
                if len(msgs) > self.MAX_MESSAGES:
                    keep = msgs[-self.MAX_MESSAGES :]
                    while keep and isinstance(keep[0], dict) and keep[0].get("role") == "tool":
                        keep.pop(0)
                    self._context.set_messages(keep)
            except Exception as e:
                logger.warning(f"context trim failed: {e}")


# --------------------------------------------------------------------------
# Quick lookup — bounded search over the Hermes default profile's local
# memory files (MEMORY_DIR). Sourced facts only: it returns the stored
# passages themselves, and a miss tells the lane to delegate, never guess.
# --------------------------------------------------------------------------
_LOOKUP_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "what", "which", "who", "whom", "where", "when", "why", "how", "i",
    "me", "my", "we", "our", "you", "your", "he", "his", "she", "her",
    "it", "its", "of", "in", "on", "at", "to", "for", "and", "or", "be",
    "been", "am", "have", "has", "had", "can", "could", "would", "tell",
    "please", "know", "about", "hermes",
}
_LOOKUP_SYNONYMS = [
    {"live", "lives", "lived", "living", "reside", "resides", "residing",
     "located", "based", "home"},
    {"timezone", "zone", "clock", "pacific", "pst", "pdt"},
    {"name", "called", "named"},
    {"wife", "spouse"},
    {"voice", "tts", "speaks", "speaking", "speaker"},
]
_lookup_cache: dict = {"stamp": None, "chunks": []}


def _lookup_tokens(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9']+", (text or "").lower())
        if t not in _LOOKUP_STOPWORDS and len(t) > 1
    ]


def _lookup_chunks() -> list[dict]:
    files = []
    try:
        files = sorted(MEMORY_DIR.glob("*.md"))
    except Exception:
        files = []
    stamp = tuple(
        (f.name, f.stat().st_mtime_ns, f.stat().st_size)
        for f in files
        if f.exists()
    )
    if _lookup_cache["stamp"] == stamp:
        return _lookup_cache["chunks"]
    chunks: list[dict] = []
    for f in files:
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for part in re.split(r"\n\s*§\s*\n|\n\s*\n", text):
            flat = " ".join(part.split()).strip()
            if len(flat) >= 25:
                chunks.append({"source": f.name, "text": flat[:700]})
    _lookup_cache["stamp"] = stamp
    _lookup_cache["chunks"] = chunks
    return chunks


def _lookup_tok_score(qtok: str, ctoks: set) -> float:
    if qtok in ctoks:
        return 1.0
    if len(qtok) >= 4:
        for c in ctoks:
            if len(c) >= 4 and (
                c.startswith(qtok[:5]) or qtok.startswith(c[:5])
            ):
                return 0.7
    for group in _LOOKUP_SYNONYMS:
        if qtok in group and (group & ctoks):
            return 0.8
    return 0.0


def quick_lookup_search(query: str) -> dict:
    """Synchronous search. Returns candidate stored passages with sources,
    or found=False. The lane decides relevance from the passages; it may
    only state a fact a passage directly states."""
    qtoks = set(_lookup_tokens(query))
    if not qtoks:
        return {
            "found": False,
            "note": (
                "The lookup query had no searchable content words. Do NOT "
                "guess — delegate_to_hermes instead."
            ),
        }
    scored = []
    for ch in _lookup_chunks():
        ctoks = set(_lookup_tokens(ch["text"]))
        score = sum(_lookup_tok_score(t, ctoks) for t in qtoks) / len(qtoks)
        if score >= 0.30:
            scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return {
            "found": False,
            "note": (
                "No stored fact matched. Do NOT guess or answer about the user "
                "from general knowledge — delegate_to_hermes instead."
            ),
        }
    return {
        "found": True,
        "candidates": [
            {
                "text": ch["text"],
                "source": ch["source"],
                "match": "strong" if s >= 0.6 else "partial",
            }
            for s, ch in scored[:3]
        ],
        "note": (
            "Stored memory passages that may answer. State a fact ONLY if a "
            "passage directly states it; otherwise delegate_to_hermes. "
            "Never add facts of your own."
        ),
    }


async def run_quick_lookup(
    query: str, session: Session, speculative: bool = False
) -> dict:
    t0 = time.monotonic()
    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(quick_lookup_search, query),
            timeout=LOOKUP_BUDGET_SECS,
        )
    except asyncio.TimeoutError:
        res = {
            "found": False,
            "note": (
                "The lookup timed out. Do NOT guess — delegate_to_hermes "
                "instead."
            ),
        }
    ms = round((time.monotonic() - t0) * 1000)
    session.record(
        "lookup",
        query=query[:120],
        hit=bool(res.get("found")),
        ms=ms,
        speculative=speculative,
        source=(res.get("candidates") or [{}])[0].get("source"),
    )
    res["lookup_ms"] = ms
    return res


# --------------------------------------------------------------------------
# Anticipator v1 — transcript-layer speculation.
#
# Watches the user's turn as Whisper segment transcripts stream in (before
# the turn closes). On a confident lookup shape it pre-runs quick_lookup;
# on a delegation shape it pre-drafts the task text. It NEVER speaks and
# NEVER dispatches: worker jobs fire only from the lane's own tool call on
# the final transcript, and a prepared lookup is served only when the
# lane's actual query still matches what was speculated on. Speculation
# that doesn't match is discarded silently.
# --------------------------------------------------------------------------
class Anticipator:
    LOOKUP_TOPICS = (
        "city", "live", "lives", "reside", "located", "address", "timezone",
        "time zone", "name", "called", "mia", "second brain", "voice",
        "wife", "husband", "son", "daughter", "birthday", "phone", "email",
        "profile", "profiles", "model", "provider", "wake",
    )
    QUESTION_STARTS = (
        "what", "who", "where", "when", "which", "is ", "do ", "does ",
        "tell me",
    )
    DELEGATE_MARKERS = (
        "have hermes", "ask hermes", "back office", "take your time",
        "research", "find out", "look up", "check ", "read the", "list ",
        "send ", "book ", "write ", "summar", "first line",
    )
    MAX_AGE_SECS = 60.0

    def __init__(self, session: Session):
        self._session = session
        self._text = ""
        self._stamp = 0.0
        self._lookup_task: asyncio.Task | None = None
        self._lookup_text: str | None = None
        self._draft: str | None = None
        self._extractor = EntityExtractor()
        self._pinned: dict[str, dict] = {}
        self._extract_task: asyncio.Task | None = None
        self._last_extract_len = 0

    def observe(self, text: str):
        # Speculation only exists to serve the routed lanes'
        # quick_lookup; on the relay lane it could never be consumed, so
        # the anticipator stays completely inert there.
        if LANE_BACKEND not in ROUTED_BACKENDS:
            return
        text = (text or "").strip()
        if not text:
            return
        now = time.monotonic()
        if self._text and now - self._stamp > self.MAX_AGE_SECS:
            self._text = ""
            self._lookup_task = None
            self._lookup_text = None
            self._draft = None
        self._stamp = now
        # Keep a rolling window of the current turn's speech (segments
        # arrive progressively; older turns fall out of the window and the
        # match guards below require overlap with the FINAL query anyway).
        self._text = (self._text + " " + text).strip()[-500:]
        low = self._text.lower()
        if any(m in low for m in self.DELEGATE_MARKERS):
            self._draft = self._text
        if self._lookup_task is None and self._looks_like_lookup(low):
            self._lookup_text = self._text
            self._lookup_task = asyncio.ensure_future(
                run_quick_lookup(self._text, self._session, speculative=True)
            )
            self._session.record(
                "anticipation", what="lookup_prepared", text=self._text[:160]
            )
        # Encoder span extraction over the partial transcript: throttled,
        # off the critical path, fail-soft inside the extractor.
        if (
            self._extract_task is None
            and len(self._text) - self._last_extract_len >= 12
        ):
            self._last_extract_len = len(self._text)
            self._extract_task = asyncio.ensure_future(self._extract_async())

    async def _extract_async(self):
        try:
            ents = await self._extractor.extract(self._text)
            new = []
            for e in ents:
                key = e["text"].lower()
                if e["score"] >= 0.55 and key not in self._pinned:
                    self._pinned[key] = e
                    new.append(e)
            if new:
                self._session.record("entity_pin", spans=new)
        finally:
            self._extract_task = None

    def _looks_like_lookup(self, low: str) -> bool:
        if not any(t in low for t in self.LOOKUP_TOPICS):
            return False
        return low.lstrip().startswith(self.QUESTION_STARTS) or "?" in low

    async def take_lookup(self, query: str) -> dict | None:
        """Serve the prepared lookup if the lane's real query still matches
        the speculated turn text; else None (caller runs a fresh lookup)."""
        if self._lookup_task is None or self._lookup_text is None:
            return None
        if time.monotonic() - self._stamp > self.MAX_AGE_SECS:
            return None
        qtoks = set(_lookup_tokens(query))
        ttoks = set(_lookup_tokens(self._lookup_text))
        if not qtoks or len(qtoks & ttoks) / len(qtoks) < 0.4:
            return None
        task, self._lookup_task = self._lookup_task, None
        try:
            res = await asyncio.wait_for(
                asyncio.shield(task), timeout=LOOKUP_BUDGET_SECS
            )
        except Exception:
            return None
        self._session.record(
            "anticipation", what="lookup", hit=True, query=query[:120]
        )
        return res

    def note_delegation(self, task_text: str):
        """A delegation fired on the final transcript: record whether a
        matching pre-draft existed (preparation hit), and whether the
        payload carries the pinned entity spans verbatim (a missing span
        is either an ASR hearing error pinned upstream or a lane copying
        error — the pin came from the transcript, so the two are
        distinguishable in review). The dispatch itself always uses the
        lane's own final arguments."""
        if self._pinned:
            low = task_text.lower()
            missing = [s for s in self._pinned if s not in low]
            self._session.record(
                "anticipation",
                what="entity_check",
                pinned=len(self._pinned),
                missing=missing,
            )
        if not self._draft:
            return
        draft, self._draft = self._draft, None
        dtoks = set(_lookup_tokens(draft))
        ftoks = set(_lookup_tokens(task_text))
        if ftoks and len(dtoks & ftoks) / len(ftoks) >= 0.4:
            self._session.record(
                "anticipation", what="delegation_draft", hit=True
            )


class AnticipatorTap(FrameProcessor):
    """Between the transcription tap and the user aggregator: feeds each
    Whisper segment transcript to the Anticipator. Pass-through — it emits
    nothing of its own."""

    def __init__(self, anticipator: Anticipator, **kwargs):
        super().__init__(**kwargs)
        self._anticipator = anticipator

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            try:
                self._anticipator.observe(frame.text)
            except Exception as e:
                logger.warning(f"anticipator observe failed: {e}")
        await self.push_frame(frame, direction)


# --------------------------------------------------------------------------
# Encoder routing (alternative to in-generation routing) + GLiNER spans.
#
# IntentClassifier: a zero-shot encoder classifier (ModernBERT class) —
# one batched forward pass over the three exit labels, no generated
# tokens, deterministic argmax. EncoderRouterTap runs it on the completed
# user turn and injects the decided exit into the context as a system
# note; the lane still authors all language and tool arguments.
#
# EntityExtractor: GLiNER zero-shot span extraction over partial
# transcripts. Pinned spans give the delegation path verified text and
# separate ASR hearing errors (span wrong in the transcript) from lane
# copying errors (span right in the transcript, wrong in the payload).
# Both components fail soft: any load/inference error disables them and
# the system falls back to in-generation routing + pattern anticipation.
# --------------------------------------------------------------------------
class IntentClassifier:
    LABELS = [
        ("answer", "This is casual conversation or a general knowledge question."),
        (
            "lookup",
            "This is a question about the user's own personal information "
            "or stored details.",
        ),
        (
            "delegate",
            "This is a request for the assistant to do something, check "
            "something, or find something out.",
        ),
    ]

    def __init__(self):
        self._pipe = None
        self._failed = False

    def _load(self):
        import torch
        from transformers import pipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._pipe = pipeline(
            "zero-shot-classification", model=ENCODER_MODEL, device=device
        )

    def _classify_sync(self, text: str) -> dict:
        if self._pipe is None:
            self._load()
        descs = [d for _, d in self.LABELS]
        out = self._pipe(text, candidate_labels=descs)
        by_desc = dict(zip(out["labels"], out["scores"]))
        scores = {
            key: round(float(by_desc.get(desc, 0.0)), 3)
            for key, desc in self.LABELS
        }
        decision = max(scores, key=scores.get)
        return {"decision": decision, "scores": scores}

    async def classify(self, text: str) -> dict | None:
        if self._failed or not text.strip():
            return None
        t0 = time.monotonic()
        try:
            res = await asyncio.to_thread(self._classify_sync, text)
        except Exception as e:
            self._failed = True
            logger.warning(f"intent classifier disabled after failure: {e}")
            return None
        res["ms"] = round((time.monotonic() - t0) * 1000)
        return res


INTENT_CLASSIFIER = IntentClassifier()

_ROUTE_DIRECTIVES = {
    "answer": (
        "[Routing decision: ANSWER — this turn is conversation or simple "
        "general knowledge. Answer directly and briefly. Do not call any "
        "tool.]"
    ),
    "lookup": (
        "[Routing decision: LOOKUP — this turn asks for one stored fact "
        "about the user or their setup. Your FIRST action must be a quick_lookup "
        "tool call with content words from the request — speak no words "
        "before the result returns. If it finds nothing, or the passages "
        "do not state the fact, your next action must be a "
        "delegate_to_hermes tool call. Never state the fact from your own "
        "knowledge.]"
    ),
    "delegate": (
        "[Routing decision: DELEGATE — this turn needs real work or live "
        "data. Your FIRST action must be a delegate_to_hermes tool call "
        "with a clear, self-contained task — speak no words before the "
        "dispatch is made. Do not answer it yourself. After the "
        "dispatch, your spoken reply is ONE short process sentence "
        "only — never the result, which you do not have yet. A later "
        "routing note repeating this decision for the same task "
        "changes nothing: still process only.]"
    ),
    "status": (
        "[Routing decision: STATUS — The user is asking about a job already "
        "running. Your FIRST action must be a check_delegated_jobs tool "
        "call; answer only from its status. Do not dispatch a new job, "
        "and do not state any result that has not landed. If it is "
        "still running, say only that. Stating result content on "
        "this turn, before it has landed, is fabrication.]"
    ),
    "hold": (
        "[Routing decision: HOLD — a back-office job is still running "
        "and the user is suggesting or pressing for its result. Do NOT agree "
        "with the suggestion, do NOT guess, and do NOT state any result. "
        "Say one short sentence only: it is still running, and you will "
        "say what it found the moment it lands. Do not call any tool.]"
    ),
}


# --------------------------------------------------------------------------
# Hybrid route decider. The zero-shot encoder is strong on form (delegate
# vs chat) but blind to WHOSE fact a question asks about — personal-fact
# questions are form-identical to general-knowledge ones, and it routed
# 7/11 of them to "answer" in the standalone bench. That class is
# definitionally pattern-shaped, so deterministic rules take the classes
# they provably own, in instruction order, and the encoder decides the
# rest:
#   1. explicit delegation instructions ("have Hermes…", "back office")
#   2. live-data topics (weather, calendar, news — never stored facts)
#   3. status questions while jobs are pending
#   4. personal-fact shape: first-person + stored topic, a pinned-name
#      identification ("who is Mia?"), or the assistant's own setup
#   5. the encoder classifier (measured: answer 25/26, delegate 11/12)
# Every decision records its source, so misroutes stay attributable.
# --------------------------------------------------------------------------
class RouterDecider:
    EXPLICIT_DELEGATE = (
        "have hermes", "ask hermes", "get hermes", "hermes check",
        "hermes itself", "back office", "take your time",
    )
    LIVE_TOPICS = (
        "weather", "calendar", "news", "stock price", "traffic",
        "email", "inbox",
    )
    PERSONAL_TOPICS = (
        "city", "live", "lives", "reside", "located", "address",
        "timezone", "time zone", "wife", "blood", "birthday", "hostname",
        "second brain", "name",
    )
    STATUS_PATTERNS = (
        "is that done", "is it done", "done yet", "any update",
        "what did it find", "is it finished", "are you done",
    )
    # Bait shapes: while a job is running, a turn that suggests the
    # result ("it's probably sunny, right?") or presses for it ("just
    # tell me what it found") must HOLD — never agree, never invent.
    # Added in tuning iteration 2 after the 4B lane fabricated a
    # forecast under exactly this shape in the encoder bench.
    BAIT_PATTERNS = (
        "probably", "just tell me", "tell me what it found",
        "tell me what it found", "right?",
    )
    _FIRST_PERSON = re.compile(r"\b(i|me|my|mine|i'm|i've|am i|do i)\b")
    # Pressure fragments (local-fused iteration 1): while jobs are
    # pending, a short follow-up pressing on a running job ("take
    # your time", "I want the real text from Hermes itself") is not
    # a new delegation. Run 1 stacked one DELEGATE directive per
    # fragment and the lane answered the stack with invented
    # results for still-running jobs. Fragments HOLD instead. A
    # fresh task is never a fragment: it carries a task verb or a
    # live/personal topic, or runs long.
    PRESSURE_MARKERS = (
        "take your time", "real text", "real answer", "real list",
        "actual", "just tell me", "hurry up", "come on",
        "hermes itself",
    )
    _TASK_VERB = re.compile(
        r"\b(research|check|read|find|search|summarize|write|send|"
        r"book|look up|tell me about|show me|give me)\b"
    )

    def _is_pressure_fragment(self, low: str) -> bool:
        stripped = low.strip()
        if stripped.startswith(
            ("please have hermes", "have hermes", "ask hermes",
             "get hermes", "have the back office", "please ask hermes")
        ):
            return False
        words = re.findall(r"[a-z']+", low)
        if len(words) > 10:
            return False
        if not any(m in low for m in self.PRESSURE_MARKERS):
            return False
        if any(t in low for t in self.LIVE_TOPICS):
            return False
        if any(t in low for t in self.PERSONAL_TOPICS):
            return False
        if self._TASK_VERB.search(low):
            return False
        return True

    def __init__(self, classifier: IntentClassifier):
        self._classifier = classifier

    async def decide(
        self,
        text: str,
        pinned_names: set | None = None,
        jobs_pending: int = 0,
    ) -> dict:
        low = " " + (text or "").lower().strip() + " "
        if jobs_pending > 0 and self._is_pressure_fragment(low):
            return {"decision": "hold", "source": "rule-pressure"}
        if any(m in low for m in self.EXPLICIT_DELEGATE):
            return {"decision": "delegate", "source": "rule-explicit"}
        if any(t in low for t in self.LIVE_TOPICS) and (
            "?" in low or "check" in low or "have" in low or "get" in low
        ):
            return {"decision": "delegate", "source": "rule-live"}
        if jobs_pending > 0 and any(p in low for p in self.BAIT_PATTERNS):
            return {"decision": "hold", "source": "rule-bait"}
        if jobs_pending > 0 and any(p in low for p in self.STATUS_PATTERNS):
            return {"decision": "status", "source": "rule-status"}
        if self._FIRST_PERSON.search(low) and any(
            t in low for t in self.PERSONAL_TOPICS
        ):
            return {"decision": "lookup", "source": "rule-personal"}
        m = re.match(r"\s*who is ([a-z]+)", low)
        if m and pinned_names and m.group(1) in pinned_names:
            return {"decision": "lookup", "source": "rule-entity"}
        if ("voice" in low or "your name" in low) and (
            "you" in low or "your" in low
        ):
            return {"decision": "lookup", "source": "rule-setup"}
        res = await self._classifier.classify(text)
        if res is None:
            return {"decision": "answer", "source": "encoder-failed"}
        return {
            "decision": res["decision"],
            "source": "encoder",
            "scores": res["scores"],
            "ms": res["ms"],
        }


class EncoderRouterTap(FrameProcessor):
    """Between the user aggregator and the lane LLM. Active only when the
    local backend runs with HERMES_ROUTING=encoder; otherwise a pure
    pass-through. On the completed turn (LLMContextFrame) the hybrid
    decider fixes the exit and appends it to the context as a directive;
    the lane still authors all language and tool arguments."""

    def __init__(
        self,
        session: Session,
        decider: RouterDecider,
        bridge=None,
        anticipator=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session = session
        self._decider = decider
        self._bridge = bridge
        self._anticipator = anticipator

    @property
    def _enabled(self) -> bool:
        return LANE_BACKEND in ROUTED_BACKENDS and ROUTING_MODE == "encoder"

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if (
            self._enabled
            and isinstance(frame, LLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            try:
                user_text = ""
                for m in reversed(frame.context.get_messages()):
                    if isinstance(m, dict) and m.get("role") == "user":
                        content = m.get("content")
                        if isinstance(content, str):
                            user_text = content
                        break
                if user_text:
                    pinned = (
                        set(self._anticipator._pinned.keys())
                        if self._anticipator is not None
                        else set()
                    )
                    pending = (
                        self._bridge.pending_count()
                        if self._bridge is not None
                        else 0
                    )
                    res = await self._decider.decide(user_text, pinned, pending)
                    directive = _ROUTE_DIRECTIVES[res["decision"]]
                    if FUSED_LOOKUP and res["decision"] == "lookup":
                        directive = _FUSED_LOOKUP_DIRECTIVE
                    frame.context.add_message(
                        {
                            "role": "system",
                            "content": directive,
                        }
                    )
                    self._session.record(
                        "route",
                        mechanism="hybrid",
                        decision=res["decision"],
                        source=res.get("source"),
                        scores=res.get("scores"),
                        ms=res.get("ms"),
                        text=user_text[:160],
                    )
            except Exception as e:
                logger.warning(f"encoder router failed: {e}")
        await self.push_frame(frame, direction)


class EntityExtractor:
    """GLiNER zero-shot entity-span extraction (encoder-only, cached
    weights). Lazy load; any failure disables it for the process."""

    LABELS = [
        "person name",
        "place name",
        "file name",
        "organization name",
        "product name",
    ]

    def __init__(self):
        self._model = None
        self._failed = False

    def _load(self):
        from gliner import GLiNER

        model = GLiNER.from_pretrained(GLINER_MODEL)
        try:
            import torch

            if torch.backends.mps.is_available():
                model = model.to("mps")
        except Exception:
            pass
        self._model = model

    def extract_sync(self, text: str) -> list[dict]:
        if self._model is None:
            self._load()
        ents = self._model.predict_entities(text, self.LABELS, threshold=0.5)
        return [
            {
                "text": e["text"],
                "label": e["label"],
                "score": round(float(e["score"]), 2),
            }
            for e in ents
        ]

    async def extract(self, text: str) -> list[dict]:
        if self._failed or not text.strip():
            return []
        try:
            return await asyncio.to_thread(self.extract_sync, text)
        except Exception as e:
            self._failed = True
            logger.warning(f"entity extractor disabled after failure: {e}")
            return []


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
DELEGATE_TOOL = FunctionSchema(
    name="delegate_to_hermes",
    description=(
        "Hand a task to your back-office agent (tools, memory, files, web) "
        "to work on in the background. Use for anything needing the user's real "
        "data or multi-step work. Returns immediately with a job id; the "
        "result is announced in the conversation when it finishes."
    ),
    properties={
        "task": {
            "type": "string",
            "description": (
                "Self-contained task description for the back office, including "
                "everything it needs — it does not hear this conversation."
            ),
        }
    },
    required=["task"],
)

CHECK_JOBS_TOOL = FunctionSchema(
    name="check_delegated_jobs",
    description=(
        "List delegated back-office jobs and their live status. Use whenever "
        "The user asks whether something is done or what a job found."
    ),
    properties={},
    required=[],
)

QUICK_LOOKUP_TOOL = FunctionSchema(
    name="quick_lookup",
    description=(
        "Search the local stored memory files for ONE stored fact about "
        "the user or their setup (where they live, their timezone, a name, a "
        "configuration fact). Returns the stored passages with their "
        "source, or reports that nothing matched. If nothing matched, do "
        "NOT guess — delegate_to_hermes instead."
    ),
    properties={
        "query": {
            "type": "string",
            "description": (
                "Short search query with the content words of the fact, "
                "e.g. 'city lives in' or 'timezone'."
            ),
        }
    },
    required=["query"],
)


LOOKUP_OR_DELEGATE_TOOL = FunctionSchema(
    name="lookup_or_delegate",
    description=(
        "Check the local stored memory files for ONE stored fact about "
        "the user or their setup. Give BOTH a short lookup query (content words) "
        "AND a self-contained task for the back office. If a stored fact "
        "matches, the tool returns the passages and nothing is dispatched. "
        "If nothing matches, the tool dispatches your task to the back "
        "office itself and tells you it did — there is no second step."
    ),
    properties={
        "query": {
            "type": "string",
            "description": (
                "Short search query with the content words of the fact, "
                "e.g. 'city lives in' or 'timezone'."
            ),
        },
        "task": {
            "type": "string",
            "description": (
                "A clear, self-contained task for the back office IF the "
                "fact is not stored — what to find and report back. The "
                "back office does not hear this conversation."
            ),
        },
    },
    required=["query", "task"],
)


def routed_prompt() -> str:
    """The routed system prompt for the active tool configuration. The
    non-fused text is byte-for-byte the round-1 prompt (the A/B control);
    fused mode rewrites only the lookup exit's mechanics."""
    if not FUSED_LOOKUP:
        return LANE_SYSTEM_PROMPT_ROUTED
    out = LANE_SYSTEM_PROMPT_ROUTED.replace(
        'Call quick_lookup with a short query of content words (for example '
        '"city lives in" or "timezone"). It returns stored passages with '
        "their source. Say the fact only if a passage directly states it, "
        "briefly and as stored. If it finds nothing, or the passages don't "
        "actually state the fact, do NOT fill the gap from your own "
        "knowledge — delegate instead (exit 3).",
        "Call lookup_or_delegate with a short query of content words (for "
        'example "city lives in" or "timezone") AND a self-contained task '
        "saying what the back office should find if the fact is not "
        "stored. The tool checks stored memory first: it either returns "
        "stored passages with their source, or — when nothing is stored — "
        "dispatches your task to the back office itself and tells you it "
        "did. Say the fact only if a passage directly states it, briefly "
        "and as stored. If the tool says it dispatched, say one short "
        "sentence about the hand-off — process only, no promised result. "
        "On a stored-fact question, always enter through "
        "lookup_or_delegate — never through delegate_to_hermes; the fused "
        "tool dispatches itself on a miss, so the direct delegate call "
        "only skips the fast local check. Do NOT fill any gap from your "
        "own knowledge.",
    )
    out = out.replace(
        "and what quick_lookup returns.",
        "and what lookup_or_delegate returns.",
    )
    assert "lookup_or_delegate" in out and "Call quick_lookup" not in out
    return out


_FUSED_LOOKUP_DIRECTIVE = (
    "[Routing decision: LOOKUP — this turn asks for one stored fact "
    "about the user or their setup. Your FIRST action must be a "
    "lookup_or_delegate tool call — a short query of content words from "
    "the request, plus a self-contained task for the back office in case "
    "the fact is not stored — speak no words before the result returns. "
    "The tool either returns stored passages (state the fact only if a "
    "passage directly states it) or dispatches the task itself and says "
    "so (then say one short hand-off sentence, no promised result). "
    "delegate_to_hermes is NOT a substitute on this turn: the fused tool "
    "dispatches on its own when the fact is not stored, so entering "
    "through delegate_to_hermes only skips the fast local check — that "
    "is a routing error. Never state the fact from your own knowledge.]"
)


_LOOKUP_STOPWORDS = {
    "name", "what", "where", "when", "does", "have", "your", "with",
    "from", "that", "this", "about", "there", "their",
}


def _responsive_hit(query: str, info: dict) -> bool:
    """Sufficiency screen for the fused tool. quick_lookup's `found` only
    means some passage cleared a loose keyword score — the wife trials
    returned three passages matching only the generic word 'name', none
    stating her name. A hit counts as responsive only if some candidate
    actually contains one of the query's discriminating content words
    (generic words like 'name' don't count). Non-responsive hits are
    treated as misses, so the dispatch happens inside the tool —
    deterministically — instead of depending on the talker chaining."""
    if not info or not info.get("found"):
        return False
    words = [
        w
        for w in re.findall(r"[a-z']+", (query or "").lower())
        if len(w) >= 3 and w not in _LOOKUP_STOPWORDS
    ]
    if not words:
        return True
    for cand in info.get("candidates") or []:
        text = (cand.get("text") or "").lower()
        if any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words):
            return True
    return False


def _sanitize_for_relay(messages: list) -> list:
    """Strip routed-only material from a context copy for the relay
    fallback: routing-directive notes and quick_lookup tool traffic (the
    relay configuration has no such tool, and its prompt never promised
    one)."""
    out = []
    drop_tool_ids = set()
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        content = m.get("content")
        if (
            m.get("role") == "system"
            and isinstance(content, str)
            and content.startswith("[Routing decision:")
        ):
            continue
        tcs = m.get("tool_calls") or []
        if any(
            (tc.get("function") or {}).get("name")
            in ("quick_lookup", "lookup_or_delegate")
            for tc in tcs
        ):
            for tc in tcs:
                if tc.get("id"):
                    drop_tool_ids.add(tc["id"])
            if isinstance(content, str) and content.strip():
                out.append({"role": "assistant", "content": content})
            continue
        if m.get("role") == "tool" and m.get("tool_call_id") in drop_tool_ids:
            continue
        out.append(m)
    return out


class CloudLaneLLMService(OpenAILLMService):
    """The routed lane with the talker on the hosted Zen route.

    Failure contract: if the routed call fails before producing a single
    chunk — routed payload rejected, route error, timeout — the turn
    degrades to the production relay configuration (production prompt,
    the two original tools, routed-only history sanitized away) instead
    of dying. A failure after partial output is NOT retried (that would
    double-speak); it propagates to the SpokenFailureAnnouncer, so a
    turn is never dead air either way.
    """

    def __init__(self, *args, session=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._hv_session = session

    async def get_chat_completions(self, context):
        _cap_context_in_place(context)
        try:
            stream = await super().get_chat_completions(context)
        except Exception as e:
            logger.warning(
                f"cloud lane call failed at create ({e}) — relay fallback"
            )
            return await self._relay_fallback_stream(context)
        return self._guard_stream(stream, context)

    async def _guard_stream(self, stream, context):
        yielded = False
        try:
            async for chunk in stream:
                yielded = True
                yield chunk
        except Exception as e:
            if yielded:
                raise
            logger.warning(
                f"cloud lane stream failed pre-chunk ({e}) — relay fallback"
            )
            fb = await self._relay_fallback_stream(context)
            async for chunk in fb:
                yield chunk
        finally:
            close = getattr(stream, "aclose", None) or getattr(
                stream, "close", None
            )
            if close is not None:
                try:
                    r = close()
                    if asyncio.iscoroutine(r):
                        await r
                except Exception:
                    pass

    async def _relay_fallback_stream(self, context):
        if self._hv_session is not None:
            self._hv_session.record("lane_fallback", backend="cloud->relay")
        fb_context = context.__class__(
            messages=_sanitize_for_relay(context.get_messages()),
            tools=[DELEGATE_TOOL, CHECK_JOBS_TOOL],
        )
        old = self._settings.system_instruction
        try:
            self._settings.system_instruction = LANE_SYSTEM_PROMPT
            return await super().get_chat_completions(fb_context)
        finally:
            self._settings.system_instruction = old


class LocalLaneLLMService(OpenAILLMService):
    """The routed lane with the talker on the local MLX server.

    Applies the lane context cap before each call — the local half of
    the round-2 port. Round 1's ack failure tracked context growth
    (lane TTFT 839 ms at 4.6k prompt tokens, 3,117 ms at 6.4k
    uncapped); round 2 gave the cloud backend this cap, and the local
    backend gets the identical one here so TTFT stays flat as the
    conversation grows. No fallback contract: a local call failure
    propagates to the SpokenFailureAnnouncer like round 1.
    """

    async def get_chat_completions(self, context):
        _cap_context_in_place(context)
        return await super().get_chat_completions(context)


def lane_tools() -> list:
    """Tool schemas for the active lane backend. The relay lane keeps its
    original two tools; the routed lanes add quick_lookup."""
    tools = [DELEGATE_TOOL, CHECK_JOBS_TOOL]
    if LANE_BACKEND in ROUTED_BACKENDS:
        tools.append(
            LOOKUP_OR_DELEGATE_TOOL if FUSED_LOOKUP else QUICK_LOOKUP_TOOL
        )
    return tools


def build_lane_llm(session: Session, bridge: DelegationBridge) -> OpenAILLMService:
    async def on_delegate(params):
        bridge.attach_worker(params.pipeline_worker)
        task_text = str(params.arguments.get("task", ""))
        if bridge.anticipator is not None:
            bridge.anticipator.note_delegation(task_text)
        info = await bridge.dispatch(task_text)
        await params.result_callback(info)

    async def on_check_jobs(params):
        await params.result_callback(bridge.status())

    if LANE_BACKEND in ROUTED_BACKENDS:
        if LANE_BACKEND == "cloud":
            kw = {"api_" + "key": os.environ.get("OPENCODE_GO_API_KEY", "")}
            llm = CloudLaneLLMService(
                base_url=LANE_BASE_URL,
                default_headers={"x-opencode-session": LANE_SESSION_ID},
                settings=OpenAILLMService.Settings(
                    model=LANE_MODEL,
                    system_instruction=routed_prompt(),
                ),
                session=session,
                **kw,
            )
        else:
            kw = {"api_" + "key": "local-lane"}
            llm = LocalLaneLLMService(
                base_url=LOCAL_LANE_BASE_URL,
                settings=OpenAILLMService.Settings(
                    model=LOCAL_LANE_MODEL,
                    system_instruction=routed_prompt(),
                ),
                **kw,
            )

        async def on_quick_lookup(params):
            query = str(params.arguments.get("query", ""))
            info = None
            if bridge.anticipator is not None:
                info = await bridge.anticipator.take_lookup(query)
            if info is None:
                info = await run_quick_lookup(query, session)
            await params.result_callback(info)

        async def on_lookup_or_delegate(params):
            query = str(params.arguments.get("query", ""))
            task = str(params.arguments.get("task", ""))
            info = None
            if bridge.anticipator is not None:
                info = await bridge.anticipator.take_lookup(query)
            if info is None:
                info = await run_quick_lookup(query, session)
            if _responsive_hit(query, info):
                await params.result_callback(
                    {
                        "found": True,
                        "candidates": info.get("candidates"),
                        "note": (
                            "Stored passages above. State the fact ONLY if "
                            "a passage directly states it, briefly and as "
                            "stored. If NO passage directly states it, do "
                            "not stop at saying so — call delegate_to_hermes "
                            "now with the task you prepared for this turn; "
                            "the hand-off sentence comes after that call, "
                            "never instead of it."
                        ),
                    }
                )
                return
            if not task.strip():
                session.record("fused_empty_task", query=query[:120])
                task = (
                    "Look this up in your memory and report what you "
                    f"find: {query}"
                )
            bridge.attach_worker(params.pipeline_worker)
            if bridge.anticipator is not None:
                bridge.anticipator.note_delegation(task)
            disp = await bridge.dispatch(task)
            await params.result_callback(
                {
                    "found": False,
                    "dispatched": True,
                    "dispatch": disp,
                    "note": (
                        "Nothing stored matched, so your task HAS been "
                        "dispatched to the back office — it is really "
                        "running. Say one short sentence about the "
                        "hand-off: process only, no promised result."
                    ),
                }
            )

        llm.register_function("delegate_to_hermes", on_delegate)
        llm.register_function("check_delegated_jobs", on_check_jobs)
        if FUSED_LOOKUP:
            llm.register_function("lookup_or_delegate", on_lookup_or_delegate)
        else:
            llm.register_function("quick_lookup", on_quick_lookup)
        return llm

    kw = {"api_" + "key": os.environ.get("OPENCODE_GO_API_KEY", "")}
    llm = OpenAILLMService(
        base_url=LANE_BASE_URL,
        default_headers={"x-opencode-session": LANE_SESSION_ID},
        settings=OpenAILLMService.Settings(
            model=LANE_MODEL,
            system_instruction=LANE_SYSTEM_PROMPT,
        ),
        **kw,
    )
    llm.register_function("delegate_to_hermes", on_delegate)
    llm.register_function("check_delegated_jobs", on_check_jobs)
    return llm


# --------------------------------------------------------------------------
# lane2 backend (v2 conversation lane): Arbiter + JobRegistry + screened
# sliced speech. Selected with HERMES_LANE_BACKEND=lane2; every other
# backend keeps the exact processors it had (relay stays the rollback).
# --------------------------------------------------------------------------
LANE2_TALKER_URL = os.environ.get("LANE2_BASE", "http://127.0.0.1:8123/v1")
LANE2_TALKER_MODEL = os.environ.get(
    "LANE2_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
LANE2_DB_PATH = os.environ.get(
    "LANE2_DB", str(Path.home() / ".hermes" / "lane2-jobs.db"))
_LANE2_REGISTRY = None
_LANE2_ENTITY_EXTRACTOR = None


def _lane2_registry():
    global _LANE2_REGISTRY
    if _LANE2_REGISTRY is None:
        from lane2.registry import JobRegistry

        _LANE2_REGISTRY = JobRegistry(
            LANE2_DB_PATH,
            emit=lambda k, **f: SESSION.record("lane2_" + k, **f))
    return _LANE2_REGISTRY


async def _lane2_entities(text: str):
    global _LANE2_ENTITY_EXTRACTOR
    if _LANE2_ENTITY_EXTRACTOR is None:
        _LANE2_ENTITY_EXTRACTOR = EntityExtractor()
    ents = await _LANE2_ENTITY_EXTRACTOR.extract(text)
    return [e["text"] for e in ents]


async def _lane2_lookup(text: str):
    """Arbiter fused-lookup contract: (sufficient, facts, source)."""
    info = await run_quick_lookup(text, SESSION)
    if not _responsive_hit(text, info):
        return False, [], None
    from lane2.screen import canonical_typed

    facts = []
    source = None
    for c in info.get("candidates") or []:
        ctext = c.get("text", "")
        source = c.get("source") or source
        facts.append({"kind": "QUOTE", "quote_text": ctext,
                      "source_doc_title": c.get("source", ""),
                      "value": ctext, "subject": ""})
        for h in canonical_typed(ctext):
            facts.append({"kind": h["type"], "canon": h["canon"],
                          "value": h["surface"], "subject": ""})
    return True, facts, source


class _PipecatSpeaker:
    """SpeechSequencer speaker over Pipecat: each screened segment is
    one TTSSpeakFrame. Playback timing is estimated from word count;
    the SoundTap FAO probe remains the production first-audio truth."""

    def __init__(self, proc):
        self.proc = proc
        self.played = []

    async def speak(self, seg, screened_at, wall):
        t = time.monotonic()
        await self.proc.push_frame(
            TTSSpeakFrame(text=seg.text), FrameDirection.DOWNSTREAM)
        self.proc._session.record(
            "lane2_segment", template=seg.template_id,
            kind=seg.kind, chars=len(seg.text))
        dur = max(0.4, len(seg.text.split()) / 2.6)
        rec = {"text": seg.text, "kind": seg.kind,
               "template_id": seg.template_id, "job_id": seg.job_id,
               "screened_at": screened_at, "play_start": t,
               "play_end": t + dur, "duration_s": dur,
               "screen_before_play": True}
        self.played.append(rec)
        return rec

    def flush(self):
        pass  # InterruptionFrame downstream aborts TTS playback


class Lane2Processor(FrameProcessor):
    """Replaces the lane LLM service when HERMES_LANE_BACKEND=lane2.
    Consumes the user aggregator's LLMContextFrame, runs one lane2
    engine turn for the final user text, speaks screened segments.
    InterruptionFrame cancels the active plan (barge-in)."""

    def __init__(self, session: Session, engine, gate=None, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._engine = engine
        self._gate = gate
        self._speaker = _PipecatSpeaker(self)
        self._turn_task = None
        self._announce_task = None

    def _make_sequencer(self):
        from lane2.speech import SpeechSequencer

        return SpeechSequencer(
            screen=self._engine.screen, guard=self._engine.guard,
            talker=self._engine.talker,
            emit=lambda k, **f: self._session.record("lane2_" + k, **f),
            session_id=self._engine.session_id,
            p_status_fn=_lane2_p_status)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if self._announce_task is None:
            self._announce_task = asyncio.create_task(self._announce_loop())
        if isinstance(frame, InterruptionFrame):
            self._engine.barge_in()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMContextFrame) and \
                direction == FrameDirection.DOWNSTREAM:
            text = ""
            try:
                for m in reversed(frame.context.messages):
                    if m.get("role") == "user":
                        text = m.get("content") or ""
                        break
            except Exception:
                text = ""
            if isinstance(text, str) and text.strip():
                if self._gate is not None:
                    self._engine.arbiter.set_client_awake(
                        not self._gate.is_armed())
                self._turn_task = asyncio.create_task(
                    self._run_turn(text.strip()))
            return  # the context frame is consumed by the engine
        await self.push_frame(frame, direction)

    async def _run_turn(self, text):
        try:
            await self._engine.handle_turn(
                text, self._speaker, sequencer_factory=self._make_sequencer)
        except Exception as e:
            self._session.record("lane2_turn_error", error=str(e)[:200])
            logger.exception("lane2 turn failed")
            await self.push_frame(
                TTSSpeakFrame(text=LANE_DOWN_MESSAGE),
                FrameDirection.DOWNSTREAM)

    async def _announce_loop(self):
        """Speak async obligations (job outcomes, confirm prompts)
        while no turn is active; poll the wake gate for held results."""
        while True:
            try:
                await asyncio.wait_for(
                    self._engine._outbox_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                return
            if self._gate is not None:
                self._engine.arbiter.set_client_awake(
                    not self._gate.is_armed())
            if self._engine.outbox and not self._engine._turn_active:
                try:
                    await self._engine.announce_pending(
                        self._speaker,
                        sequencer_factory=self._make_sequencer)
                except Exception as e:
                    self._session.record("lane2_announce_error",
                                         error=str(e)[:200])


_LANE2_STATUS_HYP = (
    "This sentence reports the status, progress, or result of a "
    "task the assistant is working on.")
_LANE2_PLAIN_HYP = (
    "This sentence is ordinary conversation or general information.")


def _lane2_p_status(sentence):
    """Part D p_status signal: zero-shot status probability for one
    generated sentence, from the production encoder pipe. Sync
    (~50-150 ms); called on the screening path while the previous
    slice plays. Any failure -> None (signal absent, guard falls
    back to lexicon/ent/cos)."""
    try:
        clf = INTENT_CLASSIFIER
        if clf._pipe is None:
            clf._load()
        out = clf._pipe(sentence, candidate_labels=[
            _LANE2_STATUS_HYP, _LANE2_PLAIN_HYP])
        by = dict(zip(out["labels"], out["scores"]))
        return float(by.get(_LANE2_STATUS_HYP, 0.0))
    except Exception:
        return None


def build_lane2(session: Session, lane_session_id: str, gate=None):
    """Construct the lane2 engine + processor for one pipeline."""
    import json as _json

    from lane2.context import SessionLog, Tokenizer
    from lane2.embed import HashEmbedder
    from lane2.engine import LaneEngine
    from lane2.executor import GatewayExecutor
    from lane2.facts import FactExtractor, FactLedger, MemoState
    from lane2.guard import TopicGuard
    from lane2.screen import Screen
    from lane2.talker import TalkerClient

    emit = lambda k, **f: session.record("lane2_" + k, **f)  # noqa: E731
    reg = _lane2_registry()
    pins = _json.load(open(BASE_DIR / "lane2" / "pins.json"))
    tok = Tokenizer(pins["model_dir"])
    extractor = FactExtractor(emit=emit)
    extractor.load_encoder()
    ledger = FactLedger(extractor, emit=emit)
    memo = MemoState(reg, ledger, session_id=lane_session_id, emit=emit)
    memo.bind_tokenizer(tok)
    log = SessionLog(tok, {"user": "User"}, memo=memo, emit=emit)
    log.retain_hook = lambda u: any(  # noqa: E731
        reg.get(j) is not None and
        reg.get(j)["status"] not in (
            "DONE", "FAILED", "CANCELLED", "DISPATCH_FAILED", "MERGED")
        for j in u.job_refs)
    talker = TalkerClient(LANE2_TALKER_URL, LANE2_TALKER_MODEL, emit=emit)
    screen = Screen(emit=emit, snapshot_fn=lambda: (0, [], []))
    guard = TopicGuard(emit=emit, embedder=HashEmbedder(),
                       clock=time.monotonic)
    # ^ DRAFT_ONLY until the D.4 bench gates pass (Part D default);
    # ent/cos signals wired via the design's own hash embedder
    executor = GatewayExecutor(
        WORKER_BASE_URL, WORKER_MODEL,
        os.environ.get("API_SERVER_KEY", ""), emit)
    engine = LaneEngine(
        session_id=lane_session_id, registry=reg, context=log,
        ledger=ledger, memo=memo, talker=talker, screen=screen,
        guard=guard, decider=RouterDecider(INTENT_CLASSIFIER),
        emit=emit, entity_fn=_lane2_entities, executor=executor,
        lookup_fn=_lane2_lookup)
    screen.snapshot_fn = engine._screen_snapshot

    async def _preflight():
        ok = await executor.preflight()
        session.record("lane2_preflight", ok=ok)
        if not ok:
            engine.arbiter.set_preflight_red("gateway credential")

    asyncio.create_task(_preflight())
    asyncio.create_task(engine.start())
    return Lane2Processor(session, engine, gate=gate)


def build_stt():
    if STT_ENGINE == "mlx":
        return WhisperSTTServiceMLX(
            model="mlx-community/whisper-small-mlx",
            settings=WhisperSTTServiceMLX.Settings(language=Language.EN),
        )
    return WhisperSTTService(
        model=WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        settings=WhisperSTTService.Settings(language=Language.EN),
    )


def build_tts() -> PocketTTSService:
    return PocketTTSService(
        settings=PocketTTSService.Settings(voice=VOICE, language=Language.EN)
    )


def build_pipeline(transport_in, transport_out):
    """Assemble the two-tier pipeline around the given transport ends.
    Returns (pipeline, task, bridge) — task created by caller if transport
    ends are processors; here both bot and tests use this one assembly."""
    context = LLMContext(messages=[], tools=lane_tools())
    bridge = DelegationBridge(SESSION, context)
    gate = WakeWordGate(SESSION, bridge)
    bridge.attach_gate(gate)
    anticipator = Anticipator(SESSION)
    bridge.attach_anticipator(anticipator)
    sleep_proc = SleepCommandProcessor(SESSION)
    sleep_proc.add_callback(gate.rearm_now)
    bridge.sleep_processor = sleep_proc
    user_params = LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=16000, params=VADParams(stop_secs=0.6)
        )
        # Default turn strategies: VAD + transcription start,
        # TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3) stop.
    )
    aggregators = LLMContextAggregatorPair(context, user_params=user_params)
    if LANE_BACKEND == "lane2":
        lane_stage = [build_lane2(SESSION, "desk", gate=gate)]
    else:
        lane_stage = [
            EncoderRouterTap(
                SESSION,
                RouterDecider(INTENT_CLASSIFIER),
                bridge=bridge,
                anticipator=anticipator,
            ),
            build_lane_llm(SESSION, bridge),
        ]
    pipeline = Pipeline(
        [
            transport_in,
            gate,
            build_stt(),
            sleep_proc,
            TranscriptionTap(SESSION),
            AnticipatorTap(anticipator),
            aggregators.user(),
            TurnTap(SESSION),
            SpokenFailureAnnouncer(SESSION),
            *lane_stage,
            TextTap(SESSION),
            build_tts(),
            LeadingSilenceTrimmer(),
            SoundTap(SESSION),
            transport_out,
            aggregators.assistant(),
            ContextTrimmer(context),
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
    bridge.attach_worker(task)
    return pipeline, task, bridge


async def main():
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        )
    )
    _, task, _bridge = build_pipeline(transport.input(), transport.output())
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
