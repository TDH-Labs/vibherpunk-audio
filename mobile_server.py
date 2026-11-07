#!/usr/bin/env python3
"""Vibherpunk Audio — MOBILE transport (SmallWebRTC), a second transport on
the Phase 1 pipeline. NOT a second voice stack: this module imports
hermes_voice.py and reuses its exact two-tier assembly (lane + DelegationBridge
worker, MLX Whisper STT, Pocket TTS, silence trimmer, failure announcer,
anti-slop operator prompt unchanged). The only differences from the desk bot:

  * Transport: SmallWebRTCTransport (phone browser) instead of
    LocalAudioTransport. This process NEVER opens the Mac's mic or speakers.
  * Wake gate: disabled (HERMES_WAKE=off) — on mobile the connect tap is the
    gate. The gate processor stays in the chain as a pass-through so the
    assembly is identical.
  * Session lifecycle: one phone session at a time. On disconnect the
    pipeline ends, but worker jobs keep running in this process; a job that
    finishes after its session ended is HELD (held_result_held +
    result_logged_disconnected) and announced first when the next session
    connects — "While you were asleep, Vibherpunk finished: ..." — the same
    held-result behavior as the desk bot's wake gate.
  * Spoken sleep: "go to sleep" / "stop listening" (shared SleepCommand
    Processor in hermes_voice.py) speaks its confirmation, then this
    session ends (reason=sleep_command). The hard off remains launchctl.

Serving: binds 127.0.0.1 only (MOBILE_PORT, default 8917). Adam reaches it
through `tailscale serve` (HTTPS on the tailnet FQDN = secure context).
Access control: tailnet membership PLUS a shared token (MOBILE_ACCESS_TOKEN,
loaded by run_mobile.sh from ~/.gemini/antigravity/voice/mobile-voice-token;
never printed). The token gates /api/offer (POST+PATCH) and /api/session.

Run (from /Users/ai/workspace/hermes-voice):  ./run_mobile.sh
"""
import os

# Env must be set BEFORE importing hermes_voice (it reads env at import time).
os.environ.setdefault("HERMES_WAKE", "off")
os.environ.setdefault(
    "HERMES_VOICE_LOGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs-mobile"),
)

import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.smallwebrtc.request_handler import (
    ConnectionMode,
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

import hermes_voice as hv
import viseme as viseme_mod

BASE_DIR = Path(__file__).resolve().parent
CLIENT_DIR = BASE_DIR / "client"
PORT = int(os.environ.get("MOBILE_PORT", "8917"))


# --------------------------------------------------------------------------
# Bridge: same DelegationBridge, plus disconnect-safe result handling and
# job-state pushes to the phone client (data channel app messages).
# --------------------------------------------------------------------------
class MobileBridge(hv.DelegationBridge):
    # Results that completed with no live session, held process-wide and
    # announced first on the next connect (the desk bot's held list, shared
    # across this process's phone sessions).
    SHARED_HELD: list = []

    def __init__(self, session, context, session_id: str):
        super().__init__(session, context)
        self.session_id = session_id
        self.alive = True
        self.notify = None  # callable(dict) — set by the session manager
        self.last_result = None

    def _push(self, msg: dict):
        if self.notify is not None:
            try:
                self.notify(msg)
            except Exception as e:
                logger.warning(f"mobile notify failed: {e}")

    async def dispatch(self, task_text: str) -> dict:
        info = await super().dispatch(task_text)
        self._push({"type": "job", "status": "running", "job_id": info.get("job_id")})
        return info

    async def _speak(self, text: str):
        if self.alive:
            self._push({"type": "job", "status": "result"})
            try:
                await super()._speak(text)
                return
            except Exception as e:
                # Pipeline died without a clean disconnect event (e.g. the
                # phone vanished mid-job). Treat as disconnected from here.
                self._session.record(
                    "speak_failed", session_id=self.session_id, error=str(e)[:200]
                )
                logger.warning(f"mobile speak failed: {e}")
                self.alive = False
        # Session is gone: the result goes to the log, never into a later
        # session's ears.
        self.last_result = text
        self._session.record(
            "result_logged_disconnected",
            session_id=self.session_id,
            text=text[:300],
        )
        logger.info(
            f"mobile session {self.session_id} gone — worker result "
            f"logged, not spoken: {text[:120]}"
        )


    async def _deliver(self, job, summary, is_error: bool = False) -> bool:
        """Result delivery, mobile edition. Live session: the base behavior
        (speak it now). Dead session (slept / disconnected): HOLD the result
        process-wide for announcement on the next connect, and keep the
        disconnect observability events."""
        if self.alive:
            return await super()._deliver(job, summary, is_error=is_error)
        announce = (
            hv.WORKER_DOWN_MESSAGE
            if is_error
            else f"While you were asleep, Vibherpunk finished: {summary}"
        )
        MobileBridge.SHARED_HELD.append(
            {
                "job_id": job["id"],
                "announce": announce,
                "summary": summary,
                "is_error": is_error,
                "completed_mono": time.monotonic(),
            }
        )
        self.last_result = summary
        self._session.record(
            "held_result_held",
            session_id=self.session_id,
            job_id=job["id"],
            summary=(summary or "")[:300],
        )
        self._session.record(
            "result_logged_disconnected",
            session_id=self.session_id,
            text=(summary or "")[:300],
        )
        logger.info(
            f"mobile session {self.session_id} gone — worker result held "
            f"for next session: {(summary or '')[:120]}"
        )
        return True


# --------------------------------------------------------------------------
# Pipeline assembly — mirrors hv.build_pipeline exactly, with MobileBridge
# and the WebRTC transport ends.
# --------------------------------------------------------------------------
def build_mobile_pipeline(transport: SmallWebRTCTransport, session_id: str):
    context = LLMContext(messages=[], tools=[hv.DELEGATE_TOOL, hv.CHECK_JOBS_TOOL])
    bridge = MobileBridge(hv.SESSION, context, session_id)
    gate = hv.WakeWordGate(hv.SESSION, bridge)  # disabled: pass-through
    bridge.attach_gate(gate)
    sleep_proc = hv.SleepCommandProcessor(hv.SESSION)
    sleep_proc.add_callback(gate.rearm_now)
    bridge.sleep_processor = sleep_proc
    user_params = LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=16000, params=VADParams(stop_secs=0.6)
        )
    )
    aggregators = LLMContextAggregatorPair(context, user_params=user_params)
    pipeline = Pipeline(
        [
            transport.input(),
            gate,
            hv.build_stt(),
            sleep_proc,
            hv.TranscriptionTap(hv.SESSION),
            aggregators.user(),
            hv.TurnTap(hv.SESSION),
            hv.SpokenFailureAnnouncer(hv.SESSION),
            hv.build_lane_llm(hv.SESSION, bridge),
            hv.TextTap(hv.SESSION),
            hv.build_tts(),
            hv.LeadingSilenceTrimmer(),
            hv.TrailingSilenceTrimmer(),
            viseme_mod.VisemeTap(notify=bridge._push),
            hv.SoundTap(hv.SESSION),
            transport.output(),
            aggregators.assistant(),
            hv.ContextTrimmer(context),
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
    return task, bridge


# --------------------------------------------------------------------------
# Session manager — one active phone session; a new offer replaces the old.
# --------------------------------------------------------------------------
class MobileSession:
    def __init__(self, session_id: str, connection):
        self.id = session_id
        self.connection = connection
        self.transport: SmallWebRTCTransport | None = None
        self.task: PipelineTask | None = None
        self.bridge: MobileBridge | None = None
        self.runner: PipelineRunner | None = None
        self.runner_task: asyncio.Task | None = None
        self.started_wall = time.time()
        self.ended: dict | None = None

    async def send_text_turn(self, text: str, use_llm: bool = False):
        """Inject a complete user phrasing + explicit reply request into the
        lane, bypassing STT (test hook). Goes through the real LLM+TTS path."""
        if self.bridge is None or self.bridge._context is None:
            raise RuntimeError("session has no context")
        ctx = self.bridge._context
        ctx.add_message({"role": "user", "content": text})
        if self.task is not None:
            if use_llm:
                await self.task.queue_frame(LLMContextFrame(ctx))
            else:
                await self.task.queue_frame(LLMFullResponseStartFrame())
                await self.task.queue_frame(TextFrame(text))
                await self.task.queue_frame(LLMFullResponseEndFrame())


async def _announce_held(sess: MobileSession):
    """On connect: if earlier sessions left held results, announce them
    first in this session (the mobile form of the desk bot's wake flush)."""
    await asyncio.sleep(1.5)
    bridge = sess.bridge
    if bridge is None or not bridge.alive:
        return
    if MobileBridge.SHARED_HELD:
        bridge.held.extend(MobileBridge.SHARED_HELD)
        MobileBridge.SHARED_HELD = []
    if bridge.held:
        await bridge.flush_held()


async def _end_session_after_sleep(sess: MobileSession):
    """Spoken sleep on mobile: let the confirmation finish playing, then
    end the session (the connect-tap session's form of gate re-arm)."""
    base = len(hv.SESSION.spoken)
    t0 = time.time()
    while time.time() - t0 < 6:
        if any(
            sp.get("text") == hv.SLEEP_CONFIRMATION
            for sp in hv.SESSION.spoken[base:]
        ):
            break
        await asyncio.sleep(0.3)
    await asyncio.sleep(1.5)
    await end_session(sess, "sleep_command")


MANAGER: dict = {"current": None, "last": None}


async def end_session(sess: MobileSession, reason: str):
    if sess.ended:
        return
    sess.ended = {"reason": reason, "wall": time.time()}
    if sess.bridge is not None:
        sess.bridge.alive = False
    hv.SESSION.record("mobile_session_end", session_id=sess.id, reason=reason)
    logger.info(f"mobile session {sess.id} ending ({reason})")
    if sess.task is not None:
        try:
            await sess.task.cancel(reason=reason)
        except Exception as e:
            logger.warning(f"task cancel failed: {e}")
    if sess.runner_task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(sess.runner_task), timeout=10)
        except Exception:
            sess.runner_task.cancel()
    if MANAGER["current"] is sess:
        MANAGER["current"] = None
    MANAGER["last"] = sess


async def start_session(connection) -> MobileSession:
    if MANAGER["current"] is not None:
        await end_session(MANAGER["current"], "replaced_by_new_connection")
    sid = f"mob-{uuid.uuid4().hex[:8]}"
    sess = MobileSession(sid, connection)
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )
    sess.transport = transport
    task, bridge = build_mobile_pipeline(transport, sid)
    sess.task = task
    sess.bridge = bridge
    bridge.notify = lambda msg: connection.send_app_message(msg)
    if bridge.sleep_processor is not None:
        bridge.sleep_processor.add_callback(
            lambda: asyncio.create_task(_end_session_after_sleep(sess))
        )

    @transport.event_handler("on_client_connected")
    async def _connected(transport_, conn):
        hv.SESSION.record("mobile_client_connected", session_id=sid)
        logger.info(f"mobile session {sid}: client connected")
        asyncio.create_task(_announce_held(sess))

    @transport.event_handler("on_client_disconnected")
    async def _disconnected(transport_, conn):
        hv.SESSION.record("mobile_client_disconnected", session_id=sid)
        logger.info(f"mobile session {sid}: client disconnected")
        if sess.bridge is not None:
            sess.bridge.alive = False  # flip NOW, before any await
        asyncio.create_task(end_session(sess, "client_disconnected"))

    sess.runner = PipelineRunner()
    sess.runner_task = asyncio.create_task(sess.runner.run(task))

    def _pipeline_done(_t, sess=sess):
        # If the pipeline ends while worker jobs are still out (dead peer,
        # transport failure), results must fall back to log-only.
        if (
            sess.bridge is not None
            and sess.bridge.alive
            and sess.bridge.pending_count() > 0
        ):
            sess.bridge.alive = False
            hv.SESSION.record(
                "mobile_pipeline_ended_with_pending_jobs", session_id=sess.id
            )

    sess.runner_task.add_done_callback(_pipeline_done)
    MANAGER["current"] = sess
    hv.SESSION.record("mobile_session_start", session_id=sid)
    logger.info(f"mobile session {sid} started")
    return sess


# --------------------------------------------------------------------------
# HTTP app
# --------------------------------------------------------------------------
app = FastAPI(title="Vibherpunk Audio Mobile")
webrtc_handler = SmallWebRTCRequestHandler(connection_mode=ConnectionMode.MULTIPLE)


def require_token(request: Request):
    expected = os.environ.get("MOBILE_ACCESS_TOKEN", "")
    supplied = request.query_params.get("token") or request.headers.get(
        "x-voice-token"
    )
    if not expected:
        raise HTTPException(status_code=503, detail="access token not configured")
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="bad or missing token")


# --------------------------------------------------------------------------
# TURN relay (coturn running on this Mac, tailnet-only). Why: ICE checks
# from a phone browser leave from per-interface sockets that cannot use
# the tailnet route, so host-candidate media never reaches this server
# (see delegation hermes-live-voice-mobile-fix1-20261027). A TURN
# allocation the phone opens over TCP across the tunnel is a media path
# that works. Credentials are minted per request (coturn REST scheme,
# 1 h TTL); the shared secret lives in ~/.gemini/antigravity/voice/turn-auth-secret
# (fallback ~/.hermes/turn-auth-secret) and is never printed.
# --------------------------------------------------------------------------
_AGY_TURN_SECRET = Path.home() / ".gemini" / "antigravity" / "voice" / "turn-auth-secret"
TURN_SECRET_FILE = (
    _AGY_TURN_SECRET
    if _AGY_TURN_SECRET.exists()
    else Path.home() / ".hermes" / "turn-auth-secret"
)
TURN_HOST = os.environ.get("MOBILE_TURN_HOST", "100.75.91.86")
TURN_PORT = int(os.environ.get("MOBILE_TURN_PORT", "3478"))


def mint_turn_ice_server():
    try:
        secret = TURN_SECRET_FILE.read_text().strip()
    except OSError:
        return None
    if not secret:
        return None
    username = str(int(time.time()) + 3600) + ":hermes"
    digest = hmac.new(secret.encode(), username.encode(), hashlib.sha1).digest()
    return {
        "urls": [
            "turn:" + TURN_HOST + ":" + str(TURN_PORT) + "?transport=tcp",
            "turn:" + TURN_HOST + ":" + str(TURN_PORT) + "?transport=udp",
        ],
        "username": username,
        "credential": base64.b64encode(digest).decode(),
    }


@app.get("/api/ice-config")
async def ice_config(request: Request):
    require_token(request)
    servers = []
    turn = mint_turn_ice_server()
    if turn:
        servers.append(turn)
    return {"iceServers": servers}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "hermes-voice-mobile"}


@app.get("/")
async def index():
    # no-cache: browsers must revalidate the shell on every load so client
    # fixes reach installed devices (see the standing rule in client/sw.js).
    return FileResponse(
        CLIENT_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/manifest.webmanifest")
async def manifest():
    return FileResponse(
        CLIENT_DIR / "manifest.webmanifest", media_type="application/manifest+json"
    )


@app.get("/sw.js")
async def service_worker():
    # no-cache: the service-worker update check must never be answered from
    # the HTTP cache, or a bumped cache version is never discovered.
    return FileResponse(
        CLIENT_DIR / "sw.js",
        media_type="text/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/icon-192.png")
async def icon192():
    return FileResponse(CLIENT_DIR / "icon-192.png", media_type="image/png")


@app.get("/icon-512.png")
async def icon512():
    return FileResponse(CLIENT_DIR / "icon-512.png", media_type="image/png")


@app.get("/avatar.png")
async def avatar():
    return FileResponse(CLIENT_DIR / "avatar.png", media_type="image/png")


@app.get("/avatar-loop.mp4")
async def avatar_loop():
    return FileResponse(
        CLIENT_DIR / "avatar-loop.mp4",
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/avatar-silent.mp4")
async def avatar_silent():
    return FileResponse(
        CLIENT_DIR / "avatar-silent.mp4",
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/avatar-speech.mp4")
async def avatar_speech():
    return FileResponse(
        CLIENT_DIR / "avatar-speech.mp4",
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/mouth-{letter}.jpg")
async def mouth_frame(letter: str):
    # Full-frame mouth shapes for the lip-sync layer (Rhubarb letters).
    if letter not in ("x", "a", "b", "c", "d", "e", "f", "g", "h"):
        return FileResponse(CLIENT_DIR / "avatar.png", media_type="image/png")
    return FileResponse(
        CLIENT_DIR / f"mouth-{letter}.jpg",
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/avatar-idle.mp4")
async def avatar_idle():
    return FileResponse(
        CLIENT_DIR / "avatar-idle.mp4",
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/avatar-talking.mp4")
async def avatar_talking():
    return FileResponse(
        CLIENT_DIR / "avatar-talking.mp4",
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/avatar-listening.mp4")
async def avatar_listening():
    return FileResponse(
        CLIENT_DIR / "avatar-listening.mp4",
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/apple-touch-icon.png")
async def apple_icon():
    return FileResponse(CLIENT_DIR / "icon-192.png", media_type="image/png")


@app.post("/api/offer")
async def offer(request: Request):
    require_token(request)
    body = await request.json()
    req = SmallWebRTCRequest.from_dict(body)

    async def _on_connection(connection):
        await start_session(connection)

    answer = await webrtc_handler.handle_web_request(req, _on_connection)
    return JSONResponse(answer)


@app.patch("/api/offer")
async def offer_patch(request: Request):
    require_token(request)
    body = await request.json()
    patch = SmallWebRTCPatchRequest(
        pc_id=body["pc_id"],
        candidates=[IceCandidate(**c) for c in body.get("candidates", [])],
    )
    await webrtc_handler.handle_patch_request(patch)
    return {"status": "ok"}


@app.get("/api/session")
async def session_state(request: Request):
    """Observability for tests + debugging: current/last session snapshot,
    event tail, spoken transcript, job states. Token-gated."""
    require_token(request)
    if request.query_params.get("flush"):
        hv.SESSION.flush_turn()
    sess = MANAGER["current"] or MANAGER["last"]
    out = {
        "active_session": MANAGER["current"].id if MANAGER["current"] else None,
        "last_session": MANAGER["last"].id if MANAGER["last"] else None,
        "events": hv.SESSION.events[-80:],
        "spoken": hv.SESSION.spoken,
        "turns": hv.SESSION.turns,
    }
    if sess is not None and sess.bridge is not None:
        out["session_id"] = sess.id
        out["session_ended"] = sess.ended
        out["jobs"] = sess.bridge.status()
        out["last_result"] = sess.bridge.last_result
    return JSONResponse(json.loads(json.dumps(out, default=str)))


@app.post("/api/_test_say")
async def test_say(request: Request):
    """Test hook: push a text turn straight into the live session's lane,
    skipping STT. Lets the headless suite prove the TTS -> WebRTC -> phone
    <audio> leg when the host has no usable fake mic. Token-gated, and
    loopback-only so it is unreachable from the tailnet."""
    if request.client and request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=404, detail="not found")
    require_token(request)
    sess = MANAGER["current"]
    if sess is None or sess.transport is None:
        raise HTTPException(status_code=409, detail="no active session")
    body = await request.json()
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    await sess.send_text_turn(text, use_llm=bool(body.get("llm")))
    return {"status": "queued", "session_id": sess.id, "text": text}


if __name__ == "__main__":
    import asyncio
    import uvicorn

    # asyncio, NOT uvloop: uvloop never retires an already-dead aiohttp
    # client connection for an inbound WebRTC offer (smallwebrtc fetches the
    # batch of ICE gathering). On a reconnect, the second POST /api/offer
    # then reuses that dead connection and 500s — the phone shows "connected"
    # but receives no audio. Proved with the mobile web test: uvloop =
    # "first remote audio" timeout, plain asyncio policy = audio flows.
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    logger.info(f"hermes-voice mobile server on 127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info",
                loop="asyncio")
