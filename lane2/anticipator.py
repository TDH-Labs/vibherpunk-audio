"""LFM3-Audio anticipator client (lane3 Workstream B).

The anticipator is a pre-STT drafting service: while the user is
still speaking, partial audio is streamed to a local LFM3-Audio
service (MPS-resident, text-only mode) which drafts a candidate
reply + a route suggestion. Everything here is advisory:

- drafts never bypass the §2.7 screen (they enter the sequencer as a
  candidate generation and are screened sentence by sentence);
- the route suggestion is recorded and measured, never applied;
- any failure — service down, timeout, malformed reply — degrades
  to "no draft" and the lane runs exactly as without it.

Protocol (127.0.0.1 only):
  POST /partial  {session, audio_b64 (float32 LE 16kHz mono),
                  final: bool}
  GET  /draft?session=...  -> {ready, draft, route, gen_ms}
  GET  /health -> {ok, rss_mb, load_s, warm}
"""
import asyncio
import base64
import json
import re
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8126"
# Hard caps: a draft is a hint, not an essay (the model's verbosity
# is architectural — see the variant track report).
MAX_DRAFT_CHARS = 420
ROUTE_LINE = re.compile(
    r"^\s*rout(?:e)?\s*[:=]\s*(chat|lookup|delegate|status|unclear)"
    r"\s*$", re.I)


def parse_draft(raw):
    """Split a service reply into (route_suggestion, draft_text).
    The service contract is: first line ROUTE=<...>, remainder the
    draft. Anything else: route None, whole text is the draft."""
    if not raw:
        return None, ""
    lines = raw.strip().split("\n")
    route = None
    if lines:
        m = ROUTE_LINE.match(lines[0])
        if m:
            route = m.group(1).lower()
            # normalize to the decider's vocabulary for measurement
            route = {"chat": "answer"}.get(route, route)
            lines = lines[1:]
    draft = " ".join(l.strip() for l in lines if l.strip())
    return route, draft[:MAX_DRAFT_CHARS]


class AnticipatorClient:
    def __init__(self, base_url=DEFAULT_URL, session_id="lane",
                 emit=None):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.emit = emit or (lambda k, **f: None)
        self._partial_tasks = set()

    # ---------------------------------------------------------- tap side
    def note_partial(self, pcm_f32_bytes, final=False):
        """Fire-and-forget partial upload from the audio tap. Never
        blocks the audio path; failures are silent by design."""
        payload = json.dumps({
            "session": self.session_id, "final": final,
            "audio_b64": base64.b64encode(pcm_f32_bytes).decode(),
        }).encode()

        def _send():
            try:
                req = urllib.request.Request(
                    self.base_url + "/partial", data=payload,
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=2.0).read()
            except Exception:
                pass

        try:
            loop = asyncio.get_running_loop()
            t = loop.run_in_executor(None, _send)
            self._partial_tasks.add(t)
            t.add_done_callback(self._partial_tasks.discard)
        except RuntimeError:
            pass

    # ---------------------------------------------------------- turn side
    async def get_draft(self, budget_s=0.15):
        """Latest draft for this session, or None. Bounded by
        budget_s (default 150 ms per the design); any error -> None
        and the turn runs draftless."""
        def _get():
            try:
                url = (f"{self.base_url}/draft?session="
                       f"{self.session_id}")
                with urllib.request.urlopen(url,
                                            timeout=budget_s) as r:
                    return json.loads(r.read().decode())
            except Exception:
                return None

        try:
            data = await asyncio.get_running_loop().run_in_executor(
                None, _get)
        except Exception:
            return None
        if not data or not data.get("ready"):
            return None
        route, draft = parse_draft(data.get("draft") or "")
        if not draft:
            return None
        return {"route": route, "draft": draft,
                "gen_ms": data.get("gen_ms"),
                "audio_s": data.get("audio_s")}

    async def health(self):
        def _get():
            try:
                with urllib.request.urlopen(
                        self.base_url + "/health", timeout=2.0) as r:
                    return json.loads(r.read().decode())
            except Exception:
                return None

        return await asyncio.get_running_loop().run_in_executor(
            None, _get)
