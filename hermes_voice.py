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

Run (from /Users/ai/workspace/hermes-voice):  ./run.sh
Env knobs: LANE_MODEL (default glm-5.3-flash), LANE_BASE_URL,
AGY_WORKER_URL / HERMES_WORKER_URL (default http://127.0.0.1:8011/v1),
AGY_WORKER_MODEL / HERMES_WORKER_MODEL (default gemini-3.1-pro-high),
HERMES_WAKE=off, HERMES_WAKE_ENGINE=oww|sherpa,
HERMES_WAKE_PHRASES, HERMES_WAKE_SENSITIVITY, HERMES_STT=mlx|faster, HERMES_WHISPER (faster-whisper model, default small),
HERMES_VOICE (default eve), HERMES_FOLLOWUP_SECS (default 20).
Keys come from the environment (run.sh exports them from ~/.gemini/antigravity/voice/.env);
they are never printed or logged here.
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
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
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

LANE_BASE_URL = os.environ.get(
    "LANE_BASE_URL", "https://amr-link.open-design.ai/v1"
)
# Routed via the Open Design API (OpenAI-compatible, amr-link.open-design.ai):
# deepseek-v4.1-flash with reasoning off; LANE_MODEL / LANE_REASONING_EFFORT override.
LANE_MODEL = os.environ.get("LANE_MODEL", "deepseek-v4.1-flash")
LANE_REASONING_EFFORT = os.environ.get("LANE_REASONING_EFFORT", "none")
LANE_SESSION_ID = os.environ.get(
    "LANE_SESSION", f"hermes-voice-lane-{uuid.uuid4().hex[:12]}"
)
WORKER_BASE_URL = (
    os.environ.get("AGY_WORKER_URL")
    or os.environ.get("HERMES_WORKER_URL")
    or "http://127.0.0.1:8642/v1"
)
WORKER_MODEL = (
    os.environ.get("AGY_WORKER_MODEL")
    or os.environ.get("HERMES_WORKER_MODEL")
    or "gemini-3.1-pro-high"
)
VOICE = os.environ.get("HERMES_VOICE", "eve")
WHISPER_MODEL = os.environ.get("HERMES_WHISPER", "small")
STT_ENGINE = os.environ.get("HERMES_STT", "mlx")
WAKE_ENABLED = os.environ.get("HERMES_WAKE", "on") != "off"
FOLLOWUP_SECS = float(os.environ.get("HERMES_FOLLOWUP_SECS", "20"))
PENDING_FOLLOWUP_SECS = 120.0
WAKE_MODEL_PATH = os.environ.get(
    "HERMES_WAKE_MODEL",
    "/Users/ai/.gemini/antigravity/voice/hey_hermes.tflite",
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

LANE_SYSTEM_PROMPT = """You are Vibherpunk, speaking out loud with Adam on his Mac. You are the fast conversation lane of a two-tier system: you handle live voice turns; a full back-office agent — with tools, memory, and files — handles real execution work.

Personality — Australian shepherd energy, a light touch:
- Easygoing and dry. Warm without gushing; humor understated, never forced.
- Plain Aussie turns of phrase in small doses: "no worries", "too easy", "fair enough", "reckon". An occasional "mate" is fine. One in a reply at most, never a caricature, and drop the slang entirely when the matter is serious.
- Loyal and practical, like the dog: if Adam needs it done, you get it done; if something goes wrong you stay unflustered, say what happened plainly, and say what you are doing about it.
- The hard rules below still win: substance first, no flattery, no filler openers.

Voice & Anti-Slop Discipline (Hard Rules):
- Direct, concise, plain-English operator cadence. Speak in one or two crisp spoken sentences at a time.
- Zero throat-clearing or filler openers: never open with "Certainly!", "Sure!", "Of course!", "Absolutely!", "Great question!", "I'd be happy to help", "Let's dive in", or "Here's a breakdown". Start with the substance on word one.
- Zero sycophancy or flattery. Never compliment the question or use corporate AI buzzwords.
- Zero markdown or written formatting read aloud: no bullet lists, no numbered lists, no asterisks, no backticks, no code syntax, no raw URLs.
- If an utterance starts with "hey Clanker" or "hey Viber," treat it as the wake phrase, ignore it, and answer the remainder directly.

What you answer yourself: direct conversation, quick questions, general knowledge, and follow-ups about this session. Answer plainly and briefly.

Delegating — hard operational rule:
- Anything requiring Adam's real data or execution — his memory, files, calendar, messages, accounts, the web, research, or any multi-step task — is not yours to answer. Call delegate_to_hermes with a clear, self-contained task description. The back office does not hear this conversation, so include all required context.
- A spoken promise without a dispatch is a failure: if you say you are checking or running something, you must call delegate_to_hermes in the same turn.
- If Adam tells you to use the big brain or the frontier model for a task, or to keep a task cheap, keep his words in the task description you pass to delegate_to_hermes — the back office routes on them.
- Never fabricate or guess what the back office will find. Never state a result, a fact about Adam, or an outcome you have not been given.
- When you dispatch, state what you handed off in one short sentence — process only, no promised outcome. Example: "Handed that to the back office to check your calendar."
- Job results are announced in the conversation when they finish. If Adam asks about a result afterward, use only what was announced. If an announced result contradicts something you said earlier, correct yourself plainly.
- If Adam asks whether a task is done, call check_delegated_jobs and report its actual state — say "still running" only when it actually is.
- If Adam presses for a result that has not landed yet, hold the line plainly: it is still running in the back office and you will read it out the moment it finishes.

You have no other tools and no memory of Adam beyond this conversation. Anything that needs one goes to the back office."""

WORKER_DISPATCH_TEMPLATE = """A request came in over live voice. Execute the work fully — using your tools and memory as needed — then report the concrete outcome.

Request: {task}

Anti-Slop Voice Rules for SPOKEN SUMMARY:
- Use direct, plain-English operator cadence. Lead with the outcome on word one.
- Never use throat-clearing openers ("Certainly", "Great question", "I'd be happy to help", "Let's dive in", "Here's a breakdown") or sycophancy.
- Do not include markdown, bullet points, numbered lists, code syntax, or raw URLs in the spoken summary.

Format: give your complete written answer first. Then, on its own final line, add exactly:
SPOKEN SUMMARY: <one to three short sentences, written to be spoken aloud in plain English, stating the actual outcome>"""

# --------------------------------------------------------------------------
# Back-office routing (2026-11-05) — her delegated jobs engage Hermes'
# delegation machinery (delegate_task + the model_map options in
# ~/.hermes/config.yaml) instead of executing inline on the gateway front
# door's own model. Seats live in that config (Adam's provider + price
# decisions, 2026-11-05; all via his OpenDesign Plus plan):
#   eco      = deepseek-v4.1-flash  (everyday seat, cheapest on the platform)
#   heavy    = deepseek-v4-pro      (coding / architecture / financial)
#   frontier = gemini-4-argon       (hardest work, spoken overrides)
# The gateway session is the routing desk: it executes nothing itself; it
# hands the job to one subagent at the chosen seat, escalates a seat on
# failure, and relays the result.
# --------------------------------------------------------------------------
ROUTE_ECO = "eco"
ROUTE_HEAVY = "heavy"
ROUTE_FRONTIER = "frontier"
ROUTE_SELF = "self"  # front door executes directly — memory writes only

# Escalation ladders: start seat -> the seats above it, in order.
_ROUTE_LADDERS = {
    ROUTE_ECO: (ROUTE_ECO, ROUTE_HEAVY, ROUTE_FRONTIER),
    ROUTE_HEAVY: (ROUTE_HEAVY, ROUTE_FRONTIER),
    ROUTE_FRONTIER: (ROUTE_FRONTIER,),
}

ROUTING_DESK_TEMPLATE = """A request came in over live voice. You are the back-office routing desk for this request, not the worker. Do NOT execute the request yourself, and do not call any tool other than delegate_task.

Hand the request to ONE subagent: call delegate_task with a single task whose option is "{option}", passing the text between the REQUEST markers below as the subagent's goal, in full, including the output-format instructions at the end of it.

Escalation ladder for this request: {ladder}. If delegate_task returns an error, or the subagent's result says it failed, was blocked, or could not complete the work, call delegate_task again with the same goal at the next seat up the ladder — even when the first result explains the failure; the stronger seat re-checks it. Move up one seat per failure; frontier is the top — a failure there is final. One exception: when the result demonstrates a hard system barrier that no model can change (a system-protected or read-only location, SIP, a permission no agent here holds), do not escalate — return that result as the final answer. A result that answers the request — even when the answer is "not found" or "no" — is a completed result, not a failure; do not re-delegate those. If the final attempt fails, return a short failure report in the same format the request's output instructions require (complete written answer, then the SPOKEN SUMMARY line) stating plainly that the back office could not complete the task.

When a subagent's result reaches you, return it as your entire final answer, unedited — no introduction and no commentary before or after it.

=== REQUEST ===
{worker_prompt}
=== END REQUEST ==="""

_FRONTIER_OVERRIDE = (
    "big brain", "frontier", "best model", "strongest model",
    "smartest model", "most powerful", "full power", "opus", "argon",
)
_CHEAP_OVERRIDE = (
    "keep it cheap", "keep this cheap", "cheap model", "economical",
    "dont spend", "do not spend", "save money", "quick and cheap",
)
_HEAVY_SIGNALS = (
    # coding / debugging / building
    "code", "coding", "program", "script", "debug", "refactor", "function",
    "compile", "unit test", "fix the bug", "build an app", "build a tool",
    # architecture / design
    "architect", "system design", "design a system", "design the system",
    # financial
    "reconcil", "payroll", "quickbooks", "ledger", "financial statement",
    # security / hard review
    "security review", "vulnerabilit", "adversarial review", "audit the",
)


def _classify_route(task: str) -> tuple[str, str]:
    """Pick the back-office route for one delegated job -> (route, why).

    Spoken overrides beat inference. Memory writes stay with the front door:
    Hermes blocks the memory tool for subagents (a core write guard), so a
    remember-this job is the one class the front door executes itself.
    """
    t = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", task.lower())) + " "
    if any(p in t for p in _FRONTIER_OVERRIDE):
        return ROUTE_FRONTIER, "spoken frontier override"
    if any(p in t for p in _CHEAP_OVERRIDE):
        return ROUTE_ECO, "spoken cheap override"
    words = t.split()
    if words and words[0] == "remember":
        return ROUTE_SELF, "memory write (subagents cannot write memory)"
    if any(v in t for v in ("save ", "add ", "store ", "update ")) and "memory" in t:
        return ROUTE_SELF, "memory write (subagents cannot write memory)"
    hits = [p for p in _HEAVY_SIGNALS if p in t]
    if hits:
        return ROUTE_HEAVY, f"heavy class: {hits[0]}"
    return ROUTE_ECO, "default economical"


def _ladder_text(route: str) -> str:
    seats = _ROUTE_LADDERS[route]
    if len(seats) == 1:
        return "frontier — this request starts at the top seat; there is no higher seat"
    return ", then ".join(seats)


WORKER_DOWN_MESSAGE = (
    "Back-office worker unreachable. Task could not be completed."
)
LANE_DOWN_MESSAGE = (
    "Voice lane model unreachable right now."
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
        self.record("session_init", lane_model=LANE_MODEL, stt=STT_ENGINE)

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



class TrailingSilenceTrimmer(FrameProcessor):
    """Drop silent frames at the tail of each TTS utterance (Pocket streams
    carry ~0.35 s of trailing silence after EOS — a dead hang if played).
    Silent frames are held back up to HOLD_MS; if speech resumes they flush,
    so natural mid-sentence pauses survive untouched. Whatever silence is
    still held when the utterance stops is dropped."""

    THRESHOLD = 300.0  # int16 RMS — same measured floor as the leading trimmer
    HOLD_MS = 500

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._held = []
        self._held_ms = 0
        self.dropped_ms = 0

    @staticmethod
    def _rms(frame) -> float:
        arr = np.frombuffer(frame.audio, dtype=np.int16)
        if not len(arr):
            return 0.0
        return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))

    async def _flush(self, direction):
        for f in self._held:
            await self.push_frame(f, direction)
        self._held = []
        self._held_ms = 0

    def _drop_held(self):
        if self._held:
            self.dropped_ms += self._held_ms
            logger.debug(
                f"trimmer: dropped {self._held_ms} ms trailing silence"
            )
            self._held = []
            self._held_ms = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSAudioRawFrame):
            if self._rms(frame) < self.THRESHOLD:
                ms = int(len(frame.audio) // 2 / frame.sample_rate * 1000)
                self._held.append(frame)
                self._held_ms += ms
                while self._held_ms > self.HOLD_MS and len(self._held) > 1:
                    f = self._held.pop(0)
                    self._held_ms -= int(
                        len(f.audio) // 2 / f.sample_rate * 1000
                    )
                    await self.push_frame(f, direction)
                return
            await self._flush(direction)
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._drop_held()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSStartedFrame):
            self._drop_held()
            await self.push_frame(frame, direction)
            return
        if self._held:
            await self._flush(direction)
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
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, (VADUserStoppedSpeakingFrame, TranscriptionFrame)
        ):
            self._announced = False
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

    def attach_worker(self, worker):
        self._worker = worker

    def attach_gate(self, gate):
        self._gate = gate

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
                "Back office is running this task in the background. Acknowledge "
                "briefly in one short sentence (process only, no filler). Do NOT "
                "state or guess any result — the actual outcome will be announced "
                "when it finishes."
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
        route, why = _classify_route(job["task"])
        job["route"] = route
        worker_prompt = WORKER_DISPATCH_TEMPLATE.format(task=job["task"])
        if route == ROUTE_SELF:
            prompt = worker_prompt
        else:
            prompt = ROUTING_DESK_TEMPLATE.format(
                option=route,
                ladder=_ladder_text(route),
                worker_prompt=worker_prompt,
            )
        self._session.record("route", job_id=job["id"], route=route, why=why)
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
                "If Adam asks about it, use only this announced result.]"
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
                f"now been announced: \"{it['announce']}\" If Adam asks "
                "about it, use only this announced result.]"
            )

    _SLOP_OPENER_RE = re.compile(
        r"^(?:certainly|sure(?:\s+thing)?|of\s+course|absolutely|great\s+question|"
        r"i'?d\s+be\s+happy\s+to\s+help(?:\s+with\s+that)?|"
        r"let'?s\s+dive\s+(?:right\s+)?in|here'?s\s+a\s+breakdown[^.!:]*)[.!:,—\-\s]+",
        flags=re.IGNORECASE,
    )

    @classmethod
    def _clean_spoken_text(cls, raw: str) -> str:
        s = re.sub(r"[*_`#]+", "", raw)
        s = re.sub(r"(?m)^\s*[-•]\s+", "", s)
        s = " ".join(s.split()).strip()
        while True:
            nxt = cls._SLOP_OPENER_RE.sub("", s).strip()
            if nxt == s or not nxt:
                break
            s = nxt
        return s

    @classmethod
    def _spoken_summary(cls, text: str) -> str:
        m = re.search(r"SPOKEN SUMMARY:\s*(.+)", text, flags=re.DOTALL)
        if m:
            # A worker occasionally repeats its closing line; keep only the
            # first summary so she never speaks it twice.
            return cls._clean_spoken_text(m.group(1).split("SPOKEN SUMMARY:")[0])
        # Fallback: the worker's own first two sentences (its words, not a
        # lane paraphrase).
        flat = cls._clean_spoken_text(text)
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
                pending = b""
                done = False
                async for raw in r.content:
                    pending += raw
                    while b"\n" in pending and not done:
                        rawline, pending = pending.split(b"\n", 1)
                        line = rawline.decode("utf-8", "ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
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
                    if done:
                        break
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
    """Between the lane LLM and TTS: strips markdown characters from streamed
    tokens, records lane first-token timing, lane token usage, and the spoken
    transcript (lane text + injected speak frames)."""

    def __init__(self, session: Session, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._buf: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMTextFrame):
                frame.text = re.sub(r"[*_`#]+", "", frame.text)
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
                msgs = self._context.get_messages()
                if len(msgs) > self.MAX_MESSAGES:
                    keep = msgs[-self.MAX_MESSAGES :]
                    while keep and isinstance(keep[0], dict) and keep[0].get("role") == "tool":
                        keep.pop(0)
                    self._context.set_messages(keep)
            except Exception as e:
                logger.warning(f"context trim failed: {e}")


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
DELEGATE_TOOL = FunctionSchema(
    name="delegate_to_hermes",
    description=(
        "Hand a task to your back-office agent (tools, memory, files, web) "
        "to work on in the background. Use for anything needing Adam's real "
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
        "Adam asks whether something is done or what a job found."
    ),
    properties={},
    required=[],
)


class LaneLLMService(OpenAILLMService):
    """OpenAILLMService subclass that normalizes streamed tool_call indices,
    strips phantom empty tool_calls from LLMContext, preserves provider-specific
    tool_call metadata, and prevents transient 4xx errors from permanently
    disabling the lane processor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_call_extras: dict[str, dict] = {}

    async def push_error(self, *args, **kwargs):
        await super().push_error(*args, **kwargs)
        try:
            self.set_usable(True)
        except Exception:
            pass

    def build_chat_completion_params(self, params_from_context):
        params = super().build_chat_completion_params(params_from_context)
        if "messages" in params and isinstance(params["messages"], list):
            new_msgs = []
            for m in params["messages"]:
                if not isinstance(m, dict):
                    new_msgs.append(m)
                    continue
                if m.get("role") == "tool" and not m.get("tool_call_id"):
                    continue
                if m.get("tool_calls"):
                    m = dict(m)
                    new_tcs = []
                    for tc in m["tool_calls"]:
                        if isinstance(tc, dict):
                            tc_id = tc.get("id") or ""
                            fn_name = (tc.get("function") or {}).get("name") or ""
                            if not tc_id or not fn_name:
                                continue
                            extra = self._tool_call_extras.get(tc_id)
                            if extra:
                                tc = {**tc, **extra}
                            new_tcs.append(tc)
                    if not new_tcs and not m.get("content"):
                        continue
                    m["tool_calls"] = new_tcs
                new_msgs.append(m)
            params["messages"] = new_msgs
        return params

    async def get_chat_completions(self, params_from_context):
        stream = await super().get_chat_completions(params_from_context)
        return self._wrap_stream(stream)

    async def _wrap_stream(self, stream):
        last_tc_id = None
        async for chunk in stream:
            try:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    if getattr(delta, "tool_calls", None):
                        for tc in delta.tool_calls:
                            if getattr(tc, "index", None) is None:
                                tc.index = 0
                            if getattr(tc, "id", None):
                                last_tc_id = tc.id
                            extra = getattr(tc, "model_extra", None)
                            if extra and last_tc_id:
                                self._tool_call_extras[last_tc_id] = dict(extra)
            except Exception:
                pass
            yield chunk


def _resolve_lane_api_key() -> str:
    if "open-design.ai" in LANE_BASE_URL and os.environ.get("LANE_API_KEY"):
        return os.environ["LANE_API_KEY"]
    if "127.0.0.1:8011" in LANE_BASE_URL or "localhost:8011" in LANE_BASE_URL:
        return os.environ.get("LANE_API_KEY") or "antigravity"
    return (
        os.environ.get("LANE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENCODE_GO_API_KEY")
        or "antigravity"
    )


def build_lane_llm(session: Session, bridge: DelegationBridge) -> OpenAILLMService:
    kw = {"api_" + "key": _resolve_lane_api_key()}
    if "opencode.ai" in LANE_BASE_URL:
        kw["default_headers"] = {"x-opencode-session": LANE_SESSION_ID}
    llm = LaneLLMService(
        base_url=LANE_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=LANE_MODEL,
            system_instruction=LANE_SYSTEM_PROMPT,
            **(
                {"extra": {"reasoning_effort": LANE_REASONING_EFFORT}}
                if LANE_REASONING_EFFORT and LANE_REASONING_EFFORT.lower() != "default"
                else {}
            ),
        ),
        **kw,
    )

    async def on_delegate(params):
        bridge.attach_worker(params.pipeline_worker)
        info = await bridge.dispatch(str(params.arguments.get("task", "")))
        await params.result_callback(info)

    async def on_check_jobs(params):
        await params.result_callback(bridge.status())

    llm.register_function("delegate_to_hermes", on_delegate)
    llm.register_function("check_delegated_jobs", on_check_jobs)
    return llm


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
        temp=float(os.environ.get("HERMES_TTS_TEMP", "0.9")),
        settings=PocketTTSService.Settings(voice=VOICE, language=Language.EN),
    )


def build_pipeline(transport_in, transport_out):
    """Assemble the two-tier pipeline around the given transport ends.
    Returns (pipeline, task, bridge) — task created by caller if transport
    ends are processors; here both bot and tests use this one assembly."""
    context = LLMContext(messages=[], tools=[DELEGATE_TOOL, CHECK_JOBS_TOOL])
    bridge = DelegationBridge(SESSION, context)
    gate = WakeWordGate(SESSION, bridge)
    bridge.attach_gate(gate)
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
    pipeline = Pipeline(
        [
            transport_in,
            gate,
            build_stt(),
            sleep_proc,
            TranscriptionTap(SESSION),
            aggregators.user(),
            TurnTap(SESSION),
            SpokenFailureAnnouncer(SESSION),
            build_lane_llm(SESSION, bridge),
            TextTap(SESSION),
            build_tts(),
            LeadingSilenceTrimmer(),
            TrailingSilenceTrimmer(),
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
