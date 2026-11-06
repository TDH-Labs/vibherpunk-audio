"""Voice turn hook: relay a completed user utterance from the voice pipeline
into the SAME Hermes session the desk UI has open.

Why this exists: the voice lane already dispatches worker jobs bound to a
session (lane2/executor.py sends X-Hermes-Session-Id), but the *conversation*
the person is having with the assistant never touched a Hermes session. This
module submits each final utterance into a real session with prompt.submit —
the same RPC Murmur's composer uses — so the turn, the assistant's reply, and
the tool activity show up in the desk transcript.

Binding modes, in order of preference:

  * explicit   — the desk told us which session it has open (bind() called by
                 the client; session id = the stored chat id Murmur shows).
  * attached   — we bound once and Hermes gave us a runtime id; turns go to it.
  * reaped     — the runtime id went away (Hermes reaps idle sessions ~20s
                 after the last viewer leaves). We re-resume the stored id.
  * unbound    — nothing yet: create a session of our own rather than silently
                 dropping the turn.

Env:
  VOICE_DESK_URL     dashboard base URL (default http://127.0.0.1:9119)
  VOICE_DESK_TOKEN   dashboard session token (default: read from
                     ~/.hermes/dashboard-session-token)
  VOICE_PROFILE      Hermes profile the desk chat belongs to (default: default)
  VOICE_BIND_FILE    optional JSON file {session_id, profile} written by the
                     desk; polled so a restart picks up the open chat
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DESK_URL = os.environ.get("VOICE_DESK_URL", "http://127.0.0.1:8011").rstrip("/")
DESK_BACKEND = os.environ.get("VOICE_DESK_BACKEND", "antigravity").strip().lower()
PROFILE = os.environ.get("VOICE_PROFILE", "default")
BIND_FILE = os.environ.get("VOICE_BIND_FILE", "").strip()
_AGY_TOKEN_PATH = Path.home() / ".gemini" / "antigravity" / "voice" / "mobile-voice-token"
TOKEN_PATH = (
    _AGY_TOKEN_PATH
    if _AGY_TOKEN_PATH.exists()
    else Path.home() / ".hermes" / "dashboard-session-token"
)
POLL_S = 2.0
RPC_TIMEOUT_S = 30


def _token() -> str:
    tok = os.environ.get("VOICE_DESK_TOKEN", "").strip()
    if tok:
        return tok
    try:
        return TOKEN_PATH.read_text().strip()
    except OSError:
        return ""


class DeskBinding:
    """Submits voice turns into the desk's Hermes session."""

    # The chat the desk last named — shared by every pipeline instance in
    # this process, so a reconnect (new Lane2Processor) keeps pointing at
    # the same chat rather than creating a fresh one.
    _last_target: tuple[str, str] | None = None

    def __init__(self, session=None):
        self._session = session            # hv.Session, for event records
        self.stored_id: str | None = None  # what the desk shows (chat id)
        self.runtime_id: str | None = None # what RPCs take
        self.profile: str = PROFILE
        self.source: str = "unbound"       # explicit | env | created | file
        self._ws = None
        self._http = None
        self._seq = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._poll_task: asyncio.Task | None = None
        self._last_file: tuple | None = None
        self._submitted = 0
        # The chat the desk last named. Process-wide, so a WebRTC reconnect
        # (which may carry no session hint) still lands in the open chat
        # instead of silently creating a new session.
        if self._last_target is not None:
            self.stored_id, self.profile, self.source = (
                self._last_target[0], self._last_target[1], "remembered")

    # ── observability ──────────────────────────────────────────────────────
    def _record(self, kind: str, **fields):
        if self._session is None:
            return
        try:
            self._session.record(kind, **fields)
        except Exception:
            pass

    def status(self) -> dict:
        return {
            "stored_session_id": self.stored_id,
            "runtime_session_id": self.runtime_id,
            "profile": self.profile,
            "bound_by": self.source,
            "voice_turns_submitted": self._submitted,
            "desk_url": DESK_URL,
        }

    # ── transport ──────────────────────────────────────────────────────────
    async def _connect(self):
        import aiohttp

        token = _token()
        if not token:
            raise RuntimeError("no dashboard session token")
        ws_url = DESK_URL.replace("https://", "wss://").replace("http://", "ws://")
        ws_url += f"/api/ws?token={token}"
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        self._ws = await self._http.ws_connect(ws_url)
        asyncio.create_task(self._reader())

    async def close(self):
        """Release the socket (tests and shutdown)."""
        self._poll_task and self._poll_task.cancel()
        self._poll_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._http is not None:
            try:
                await self._http.close()
            except Exception:
                pass
            self._http = None

    async def _reader(self):
        try:
            async for msg in self._ws:
                if msg.type != 1:  # TEXT
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                rid = data.get("id")
                fut = self._pending.pop(rid, None) if rid is not None else None
                if fut is not None and not fut.done():
                    fut.set_result(data)
        except Exception as e:
            logger.warning(f"desk socket closed: {e}")
            self._ws = None

    async def _rpc(self, method: str, params: dict) -> dict:
        if DESK_BACKEND == "antigravity" or DESK_URL.endswith(":8011"):
            import aiohttp
            import uuid

            self._seq += 1
            if method == "session.create":
                sid = f"agy-{uuid.uuid4().hex[:12]}"
                return {"session_id": sid, "stored_session_id": sid}
            if method == "session.resume":
                sid = str(params.get("session_id") or "")
                if not sid:
                    raise RuntimeError("session.resume: missing session_id")
                return {"session_id": sid, "resumed": sid}
            if method == "prompt.submit":
                sid = str(params.get("session_id") or "")
                text = str(params.get("text") or "").strip()
                base = DESK_URL if DESK_URL.endswith("/v1") else DESK_URL + "/v1"
                model = os.environ.get("AGY_WORKER_MODEL", "gemini-3.1-pro-high")
                if self._http is None or self._http.closed:
                    self._http = aiohttp.ClientSession()
                async with self._http.post(
                    f"{base}/chat/completions",
                    headers={
                        "X-Antigravity-Session-Id": sid,
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "stream": False,
                        "messages": [{"role": "user", "content": text}],
                    },
                    timeout=aiohttp.ClientTimeout(total=RPC_TIMEOUT_S),
                ) as resp:
                    if resp.status != 200:
                        body = (await resp.text())[:200]
                        raise RuntimeError(f"prompt.submit HTTP {resp.status}: {body}")
                    return {"status": "streaming", "user_row_id": self._seq}
            raise RuntimeError(f"unsupported method: {method}")

        if self._ws is None or self._ws.closed:
            await self._connect()
        self._seq += 1
        rid = self._seq
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._ws.send_json(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        )
        try:
            data = await asyncio.wait_for(fut, timeout=RPC_TIMEOUT_S)
        finally:
            self._pending.pop(rid, None)
        if data.get("error"):
            raise RuntimeError(f"{method}: {data['error']}")
        return data.get("result") or {}

    # ── binding ────────────────────────────────────────────────────────────
    async def bind(self, stored_session_id: str, profile: str | None = None,
                   source: str = "explicit"):
        """Point voice at a Hermes session the desk has open."""
        async with self._lock:
            self.stored_id = str(stored_session_id)
            self.profile = profile or self.profile
            self.source = source
            self.runtime_id = None
            DeskBinding._last_target = (self.stored_id, self.profile)
            await self._attach()
        self._record("desk_bound", stored_session_id=self.stored_id,
                     runtime_session_id=self.runtime_id, bound_by=source,
                     profile=self.profile)
        logger.info(
            f"voice turns -> desk session {self.stored_id} "
            f"(runtime {self.runtime_id}, by {source})"
        )
        return self.status()

    async def _attach(self):
        """Resume the stored session so we hold a live runtime id."""
        if not self.stored_id:
            await self._create()
            return
        try:
            res = await self._rpc(
                "session.resume",
                {
                    "session_id": self.stored_id,
                    "profile": self.profile,
                    "source": "desktop",
                    "omit_messages": True,
                    "lazy": True,
                },
            )
        except Exception as e:
            # Expired/unknown stored session: create rather than drop turns.
            self._record("desk_resume_failed", stored_session_id=self.stored_id,
                         error=str(e)[:200])
            logger.warning(f"resume of {self.stored_id} failed ({e}); creating one")
            await self._create()
            return
        self.runtime_id = res.get("session_id") or self.stored_id
        # Hermes may hand back the stored id it actually resumed.
        if res.get("resumed"):
            self.stored_id = str(res["resumed"])

    async def _create(self):
        res = await self._rpc(
            "session.create",
            {"profile": self.profile, "source": "desktop", "title": "Voice"},
        )
        self.stored_id = str(res.get("stored_session_id") or res.get("session_id"))
        self.runtime_id = str(res.get("session_id"))
        self.source = "created"
        DeskBinding._last_target = (self.stored_id, self.profile)
        self._record("desk_session_created", stored_session_id=self.stored_id,
                     runtime_session_id=self.runtime_id, profile=self.profile)

    # ── the turn ───────────────────────────────────────────────────────────
    async def submit(self, text: str) -> dict:
        """Submit one final user utterance as a turn in the bound session."""
        text = (text or "").strip()
        if not text:
            return {"submitted": False, "reason": "empty"}
        async with self._lock:
            if not self.runtime_id:
                await self._attach()
            try:
                res = await self._rpc(
                    "prompt.submit",
                    {
                        "session_id": self.runtime_id,
                        "profile": self.profile,
                        "text": text,
                        "voice_turn": True,
                        "surface": "voice",
                    },
                )
            except Exception as e:
                # A 4001-style "unknown runtime" means Hermes reaped it.
                self._record("desk_submit_retry", error=str(e)[:200],
                             runtime_session_id=self.runtime_id)
                logger.info(f"runtime {self.runtime_id} gone ({e}); re-attaching")
                self.runtime_id = None
                await self._attach()
                res = await self._rpc(
                    "prompt.submit",
                    {
                        "session_id": self.runtime_id,
                        "profile": self.profile,
                        "text": text,
                        "voice_turn": True,
                        "surface": "voice",
                    },
                )
        self._submitted += 1
        self._record("desk_turn_submitted", stored_session_id=self.stored_id,
                     runtime_session_id=self.runtime_id, text=text[:300])
        return {
            "submitted": True,
            "stored_session_id": self.stored_id,
            "runtime_session_id": self.runtime_id,
            "user_row_id": res.get("user_row_id"),
            "status": res.get("status"),
        }

    # ── bind-file polling (desk tells us which chat is open) ───────────────
    async def bind_file_or_create(self):
        """Adopt the desk's chat from the bind file, else create one.

        Called at session start so a turn arriving before the desk has ever
        spoken for itself still lands somewhere real instead of being
        dropped.
        """
        if BIND_FILE:
            try:
                path = Path(BIND_FILE).expanduser()
                if path.exists():
                    data = json.loads(path.read_text())
                    sid = (data.get("session_id") or "").strip()
                    if sid:
                        self._last_file = (sid, data.get("profile"))
                        return await self.bind(
                            sid, data.get("profile") or None, source="file")
            except Exception as e:
                logger.debug(f"bind file read failed at start: {e}")
        if self.runtime_id:
            return self.status()
        async with self._lock:
            await self._create()
        self._record("desk_bound", stored_session_id=self.stored_id,
                     runtime_session_id=self.runtime_id, bound_by="created",
                     profile=self.profile)
        return self.status()

    def start_polling(self):
        if not BIND_FILE or self._poll_task is not None:
            return
        self._poll_task = asyncio.create_task(self._poll())

    async def _poll(self):
        path = Path(BIND_FILE).expanduser()
        while True:
            try:
                if path.exists():
                    data = json.loads(path.read_text())
                    key = (data.get("session_id"), data.get("profile"))
                    if key[0] and key != self._last_file:
                        self._last_file = key
                        await self.bind(key[0], key[1] or None, source="file")
            except Exception as e:
                logger.debug(f"bind file read failed: {e}")
            await asyncio.sleep(POLL_S)
