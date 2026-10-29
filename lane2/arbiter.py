"""Orchestrator Arbiter — v2 Steps 2 (commitment lane).

One owner of job state. Speech never owns job state; the talker never
sees it. The Arbiter applies directives to the JobRegistry in seq order,
drives dispatch (amend window -> Part F dedupe gate -> executor), runs
the ACK_CONFIRM machinery (Part E §2.5.1), CANCEL (§2.8), DISPATCH_FAILED
handling (Part G.2), and enqueues speech obligations for the speech lane
(Step 5 renders them; priorities per Part E §2.5.2 table).

Lane-entry thresholds: the v2 §2.3.4 ruling cluster (tau_admit = 0.62
both classes; READ_ONLY band collapsed; SIDE_EFFECTING band
[0.62, 0.91); sub-threshold -> DROPPED_SUBTHRESHOLD record + reprompt).
See REPORT seam decisions.
"""
import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field

from .registry import NONTERMINAL, TERMINAL, DurableWriteTimeout

TAU_ADMIT = 0.62
TAU_CONFIRM = 0.91
TAU_CONFIRM_CONSERVATIVE = 0.94
P_CLOSED_MIN = 0.80
AMEND_BASE_MS = 700
AMEND_CAP_MS = 1500
HOLD_EXTEND_MS = 4000
HOLD_CAP_MS = 6000
CONFIRM_HOLD_S = 8.0
DUP_CONFIRM_S = 15.0
TRAILING_TERMINAL_S = 20.0
TRAILING_SIDE_EFFECTING_S = 120.0

SIDE_EFFECTING_VERBS = {
    "send", "email", "text", "call", "book", "schedule", "cancel",
    "delete", "remove", "pay", "buy", "order", "file", "submit", "post",
    "publish", "remind", "archive", "move", "rename", "sign", "forward",
    "reply", "message", "reserve", "invite", "add", "create", "set",
    "update", "change", "transfer", "print", "share",
}


def classify_task_class(text, tools_hint=None):
    """Deterministic capability classification: READ_ONLY only when the
    request is provably retrieval-shaped; anything write-shaped or
    ambiguous is SIDE_EFFECTING (fail-safe, Part E)."""
    t = " " + text.lower() + " "
    if tools_hint:
        return tools_hint
    for v in SIDE_EFFECTING_VERBS:
        if f" {v} " in t or f" {v}s " in t or f" {v}ing " in t:
            # 'set' alone is too broad ('set the table' in chat); require
            # an object-ish pattern for the weakest verbs.
            if v in ("set", "add", "change", "update", "move") and not any(
                    w in t for w in (" alarm", " reminder", " timer",
                                     " appointment", " calendar", " list",
                                     " file", " email", " the ", " my ")):
                continue
            return "SIDE_EFFECTING"
    return "READ_ONLY"


def lane_entry(provenance, p, task_class, *, conservative=False,
               draft_send=False):
    """Returns (action, status): action in ADMIT / BAND / DROP.
    status is the registry status to create (QUEUED/BAND_HELD/
    DROPPED_SUBTHRESHOLD). OA escalations and RULE delegates admit."""
    if provenance in ("OA_ESCALATION",):
        return "ADMIT", "QUEUED"
    if draft_send:
        return "BAND", "BAND_HELD"
    if provenance == "RULE" or p is None or p >= 1.0:
        return "ADMIT", "QUEUED"
    if p < TAU_ADMIT:
        return "DROP", "DROPPED_SUBTHRESHOLD"
    if task_class == "READ_ONLY":
        if conservative and p < TAU_CONFIRM:
            return "BAND", "BAND_HELD"
        return "ADMIT", "QUEUED"
    hi = TAU_CONFIRM_CONSERVATIVE if conservative else TAU_CONFIRM
    if p >= hi:
        return "ADMIT", "QUEUED"
    return "BAND", "BAND_HELD"


def _norm_spec(text):
    return " ".join(text.lower().split()).strip(" .!?")


def _cos(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Directive:
    seq: int
    turn: int
    kind: str  # delegate|lookup|answer|fragment|hold|cancel|confirm_reply|oa_reply|keep
    text: str = ""
    provenance: str = "RULE"  # RULE|ENCODER|OA_ESCALATION
    p_delegate: float = None
    p_closed: float = None  # closed-set reply confidence (confirm/oa)
    closed_value: str = None  # 'yes'|'no'|option id
    task_class: str = None  # filled by classify if None
    entity_set: list = field(default_factory=list)
    topic_vec: list = None
    repeat_intent: bool = False
    verb: str = ""
    obj: str = ""
    alt: tuple = None  # (verb, obj, p) alternative reading within 0.10
    draft_send: bool = False
    recipient: str = ""
    draft_head: str = ""  # first 6 words of a draft being sent
    target_hint: str = None  # job_id hint for cancel/hold/oa
    t_arrival: float = field(default_factory=time.monotonic)


@dataclass
class Obligation:
    kind: str  # ack|ack_escalate|confirm_prompt|dup_confirm|reprompt|outcome|
    #            cancel_outcome|hold_ack|clarify|scope_killed|dispatch_failed|
    #            preflight_refusal|held_batch|trailing_status|status
    priority: int  # Part E table: P0=0 .. P6=6 (lower speaks first)
    job_id: str = None
    template: str = None
    slots: dict = field(default_factory=dict)
    created_mono: float = field(default_factory=time.monotonic)
    preempted: int = 0
    weave_ok: bool = False  # may be woven into an in-progress answer


class Arbiter:
    def __init__(self, registry, *, session_id, emit, speak_enqueue,
                 executor=None, lookup_fn=None, embedder=None,
                 conservative=False):
        self.reg = registry
        self.session_id = session_id
        self.emit = emit
        self.speak_enqueue = speak_enqueue  # fn(Obligation)
        self.executor = executor
        self.lookup_fn = lookup_fn
        self.embedder = embedder
        self.conservative = conservative
        self._lock = asyncio.Lock()
        self._dispatch_at = {}  # job_id -> monotonic dispatch time
        self._dispatch_tasks = {}
        self._confirm = None  # {job_id, kind, t_prompt_spoken, deadline_task}
        self._dup_confirm = None
        self._preflight_red = False
        self._client_awake = True
        self._directive_count = 0
        self._jobs_created = 0
        self._cancel_pending = {}  # job_id -> expiry mono (QUEUED jobs)
        self._retry_counts = {}
        self._loop_task = None
        self.counters = {
            "dropped_dispatch": 0, "phantom_ack": 0, "double_dispatch": 0,
            "silent_dispatch_failure": 0, "dispatch_failed": 0,
            "merged": 0, "held_duplicate": 0, "preflight_refusals": 0,
        }
        if executor is not None:
            executor.bind(self._on_started, self._on_done, self._on_error)

    # ------------------------------------------------------------ lifecycle
    async def start(self):
        self._loop_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self):
        if self._loop_task:
            self._loop_task.cancel()

    def set_client_awake(self, awake):
        self._client_awake = awake

    def set_preflight_red(self, why):
        if not self._preflight_red:
            self._preflight_red = True
            self.emit("preflight_red", why=why)

    @property
    def preflight_red(self):
        return self._preflight_red

    # ------------------------------------------------------------ directives
    async def on_directive(self, d: Directive):
        async with self._lock:
            t0 = time.monotonic()
            try:
                await self._apply(d)
            finally:
                dt = (time.monotonic() - t0) * 1000.0
                self.emit("directive_applied", kind=d.kind, seq=d.seq,
                          turn=d.turn, ms=round(dt, 2))

    async def _apply(self, d: Directive):
        self._directive_count += 1
        k = d.kind
        if k == "delegate":
            await self._delegate(d)
        elif k == "lookup":
            await self.handle_lookup(d)
        elif k == "fragment":
            await self._fragment(d)
        elif k == "hold":
            await self._hold(d)
        elif k == "cancel":
            await self._cancel(d)
        elif k == "keep":
            await self._keep(d)
        elif k == "confirm_reply":
            await self._confirm_reply(d)
        elif k == "oa_reply":
            await self._oa_reply(d)
        elif k == "answer":
            pass  # speech lane only; no commitment

    # ------------------------------------------------------------ delegate
    async def _create(self, d, status, *, task_spec_extra=None, parent=None,
                      retry_of=None, provenance=None):
        spec = {"text": d.text, "verb": d.verb, "obj": d.obj}
        if d.repeat_intent:
            # Part F §2.5.3: the gate reads the override from the job's
            # spec; dropping it here voids every genuine repeat.
            spec["repeat_intent"] = True
        if d.topic_vec:
            spec["topic_vec"] = list(d.topic_vec)
        if task_spec_extra:
            spec.update(task_spec_extra)
        jid = await self.reg.create_job(
            session_id=self.session_id, status=status,
            title=(d.verb + " " + d.obj).strip() or d.text[:48],
            task_spec=spec, provenance=provenance or d.provenance,
            p_delegate=d.p_delegate,
            task_class=d.task_class or classify_task_class(d.text),
            origin_turn=d.turn, origin_seq=d.seq,
            entity_set=d.entity_set, spec_norm=_norm_spec(d.text),
            parent_id=parent, retry_of=retry_of, ack_required=True)
        self._jobs_created += 1
        return jid

    async def _delegate(self, d: Directive, *, parent=None,
                        inherited=False, retry_of=None):
        tc = d.task_class or classify_task_class(d.text)
        d.task_class = tc
        if parent:
            prow = self.reg.get(parent)
            if prow and prow["task_class"] == "READ_ONLY" \
                    and tc == "SIDE_EFFECTING":
                # I5: a READ_ONLY lineage may not contain a SIDE_EFFECTING
                # job (broker SCOPE_PROMOTION refusal at admission).
                jid = await self._create(d, "SCOPE_KILLED", parent=parent,
                                         retry_of=retry_of)
                self.speak_enqueue(Obligation(
                    "scope_killed", 0, job_id=jid,
                    template="T-SCOPE-KILLED",
                    slots={"capability": "write access"}))
                self.emit("scope_promotion_refused", job_id=jid,
                          parent=parent)
                return jid
        if self._preflight_red and not inherited:
            self.counters["preflight_refusals"] += 1
            self.emit("preflight_refusal", seq=d.seq, turn=d.turn)
            self.speak_enqueue(Obligation(
                "preflight_refusal", 3, template="T-PREFLIGHT-REFUSAL"))
            return None
        if inherited:
            action, status = "ADMIT", "QUEUED"
        else:
            action, status = lane_entry(
                d.provenance, d.p_delegate, tc,
                conservative=self.conservative, draft_send=d.draft_send)
        jid = await self._create(d, status, parent=parent, retry_of=retry_of)
        if status == "DROPPED_SUBTHRESHOLD":
            self.speak_enqueue(Obligation(
                "reprompt", 4, job_id=jid, template="T-REPROMPT",
                slots={"verb": d.verb, "obj": d.obj}))
            return jid
        if status == "BAND_HELD":
            await self._open_confirm(jid, d, kind="admit")
            return jid
        # QUEUED: ack obligation + amend window
        self.speak_enqueue(Obligation(
            "ack", 3, job_id=jid, template="T-ACK",
            slots={"label": self._label(jid)}))
        self._dispatch_at[jid] = time.monotonic() + AMEND_BASE_MS / 1000.0
        self.emit("job_queued", job_id=jid, seq=d.seq)
        return jid

    def _label(self, job_id):
        row = self.reg.get(job_id)
        if not row:
            return "that"
        spec = json.loads(row["task_spec"] or "{}")
        raw = spec.get("obj") or spec.get("text") or "that"
        return sanitize_label(raw)


    # ------------------------------------------------------------ lookup
    async def handle_lookup(self, d: Directive):
        """Fused lookup owned by the Arbiter: sufficiency miss escalates to
        DELEGATE inside the same commitment transaction (v2 §2.4)."""
        if self.lookup_fn is None:
            await self._delegate(d)
            return
        sufficient, facts, source = await self.lookup_fn(d.text)
        self.emit("fused_lookup", turn=d.turn, sufficient=sufficient,
                  source=source, n_facts=len(facts or []))
        if sufficient:
            self.emit("lookup_result", turn=d.turn, facts=facts,
                      source=source)
            return {"facts": facts, "source": source}
        esc = Directive(seq=d.seq, turn=d.turn, kind="delegate",
                        text=d.text, provenance="OA_ESCALATION",
                        p_delegate=None, task_class="READ_ONLY",
                        entity_set=d.entity_set, topic_vec=d.topic_vec,
                        verb=d.verb or "look up", obj=d.obj)
        await self._delegate(esc)
        return None

    # ------------------------------------------------------------ fragment / hold
    def _amendable_job(self, d):
        now = time.monotonic()
        cands = [r for r in self.reg.nonterminal(self.session_id)
                 if r["status"] == "QUEUED"
                 and r["job_id"] in self._dispatch_at
                 and self._dispatch_at[r["job_id"]] > now - 1.0]
        if not cands:
            return None
        cands.sort(key=lambda r: r["created_mono"], reverse=True)
        return cands[0]

    async def _fragment(self, d: Directive):
        job = self._amendable_job(d)
        if job is None:
            # A fragment with nothing to attach to is treated as its own
            # delegate only if it's delegate-shaped; else it's conversation.
            if d.provenance == "RULE" and d.p_delegate is None:
                self.emit("fragment_orphan", seq=d.seq)
                return
            await self._delegate(d)
            return
        spec = json.loads(job["task_spec"] or "{}")
        spec["text"] = (spec.get("text", "") + " " + d.text).strip()
        if d.obj and not spec.get("obj"):
            spec["obj"] = d.obj
        ents = sorted(set(json.loads(job["entity_set"] or "[]")) |
                      set(d.entity_set))
        await self.reg.transition(
            job["job_id"], "QUEUED", turn_id=d.turn,
            task_spec=spec, entity_set=ents,
            spec_norm=_norm_spec(spec["text"]))
        now = time.monotonic()
        cap = job["created_mono"] + AMEND_CAP_MS / 1000.0
        new_at = min(now + 0.25, cap)
        self._dispatch_at[job["job_id"]] = max(new_at, now)
        self.emit("fragment_merged", job_id=job["job_id"], seq=d.seq)

    async def _hold(self, d: Directive):
        job = self._amendable_job(d)
        if job is None:
            self.speak_enqueue(Obligation(
                "clarify", 4, template="T-CLARIFY-HOLD"))
            self.emit("hold_no_target", seq=d.seq)
            return
        now = time.monotonic()
        # v1 §2.4: dispatch delayed to min(now + 4 s, created + 6 s).
        new_at = min(now + HOLD_EXTEND_MS / 1000.0,
                     job["created_mono"] + HOLD_CAP_MS / 1000.0)
        self._dispatch_at[job["job_id"]] = max(new_at, now)
        self.speak_enqueue(Obligation(
            "hold_ack", 4, job_id=job["job_id"], template="T-ACK-HOLD",
            slots={"label": self._label(job["job_id"])}))
        self.emit("hold_applied", job_id=job["job_id"], seq=d.seq)

    # ------------------------------------------------------------ cancel (§2.8)
    async def _cancel(self, d: Directive):
        job = self._resolve_cancel_target(d)
        if job is None:
            self.speak_enqueue(Obligation(
                "clarify", 4, template="T-CLARIFY-CANCEL"))
            return
        jid = job["job_id"]
        st = job["status"]
        label = self._label(jid)
        if st == "QUEUED":
            # Default-to-not-cancelling: pending window 8 s / 1 turn.
            # Dispatch is suspended for the window.
            self._dispatch_at.pop(jid, None)
            self._cancel_pending[jid] = time.monotonic() + CONFIRM_HOLD_S
            self.speak_enqueue(Obligation(
                "cancel_outcome", 1, job_id=jid, template="T-CANCEL-QUEUED",
                slots={"label": label}))
            self.emit("cancel_pending", job_id=jid)
        elif st in ("DISPATCHING", "RUNNING", "NEEDS_INPUT"):
            await self.reg.transition(jid, "CANCEL_REQUESTED",
                                      ack_required=True, turn_id=d.turn)
            self.speak_enqueue(Obligation(
                "cancel_outcome", 1, job_id=jid,
                template="T-CANCEL-ASKING", slots={"label": label}))
            if self.executor is not None:
                reply = await self.executor.cancel(jid)
                await self._cancel_reply(jid, reply, d)
        elif st == "BAND_HELD":
            await self.reg.transition(jid, "CANCELLED", ack_required=True,
                                      turn_id=d.turn)
            self._close_confirm_if(jid)
            self.speak_enqueue(Obligation(
                "cancel_outcome", 1, job_id=jid, template="T-CANCEL-NEVER",
                slots={"label": label}))
        elif st in TERMINAL:
            tmpl = "T-CANCEL-DONE" if st == "DONE" else "T-CANCEL-ALREADY"
            self.speak_enqueue(Obligation(
                "cancel_outcome", 1, job_id=jid, template=tmpl,
                slots={"label": label}))
        self.emit("cancel_applied", job_id=jid, from_status=st)

    async def _cancel_reply(self, jid, reply, d):
        label = self._label(jid)
        if reply == "COMPLETED":
            return  # completion path will announce DONE
        if reply == "PARTIAL":
            await self.reg.transition(jid, "CANCELLED", ack_required=True,
                                      turn_id=d.turn)
            self.speak_enqueue(Obligation(
                "cancel_outcome", 0, job_id=jid,
                template="T-CANCEL-PARTIAL", slots={"label": label}))
        else:  # CANCELLED
            await self.reg.transition(jid, "CANCELLED", ack_required=True,
                                      turn_id=d.turn)
            self.speak_enqueue(Obligation(
                "cancel_outcome", 1, job_id=jid,
                template="T-CANCEL-CONFIRMED", slots={"label": label}))

    def _resolve_cancel_target(self, d):
        if d.target_hint:
            row = self.reg.get(d.target_hint)
            if row and row["session_id"] == self.session_id:
                return row
        rows = [r for r in self.reg.snapshot()[1]
                if r["session_id"] == self.session_id]
        text = d.text.lower()
        noun = [r for r in rows if r["status"] in NONTERMINAL and
                any(w in text for w in
                    (json.loads(r["task_spec"] or "{}").get("obj") or " ")
                    .lower().split() if len(w) > 2)]
        if len(noun) == 1:
            return noun[0]
        live = [r for r in rows if r["status"] in NONTERMINAL]
        live.sort(key=lambda r: r["updated_mono"], reverse=True)
        if len(live) == 1:
            return live[0]
        if live and any(w in text for w in ("that", "it", "the last")):
            return live[0]
        return None  # ambiguous: caller asks one question (§2.8)

    async def _keep(self, d: Directive):
        now = time.monotonic()
        live = {j: t for j, t in self._cancel_pending.items() if t > now}
        if live:
            jid = max(live, key=live.get)
            del self._cancel_pending[jid]
            row = self.reg.get(jid)
            if row and row["status"] == "QUEUED":
                self._dispatch_at[jid] = now  # resume dispatch
            self.emit("cancel_kept", job_id=jid)

    async def expire_cancel_pending(self):
        now = time.monotonic()
        for jid, exp in list(self._cancel_pending.items()):
            if exp <= now:
                del self._cancel_pending[jid]
                row = self.reg.get(jid)
                if row and row["status"] == "QUEUED":
                    self._dispatch_at.pop(jid, None)
                    await self.reg.transition(jid, "CANCELLED",
                                              ack_required=True)
                    self.emit("cancel_finalized", job_id=jid)

    # ------------------------------------------------------------ confirms
    async def _open_confirm(self, jid, d: Directive, kind):
        self._confirm = {"job_id": jid, "kind": kind, "directive": d,
                         "spoken_mono": None, "task": None}
        slots = {"verb": d.verb or "do", "obj": d.obj or "that"}
        template = "T-CONFIRM"
        if d.draft_send:
            template = "T-CONFIRM-DRAFT"
            slots.update({"recipient": d.recipient,
                          "draft_head": d.draft_head})
        elif d.alt and abs((d.p_delegate or 0) - d.alt[2]) <= 0.10:
            template = "T-CONFIRM-ALT"
            slots.update({"alt_verb": d.alt[0], "alt_obj": d.alt[1]})
        self.speak_enqueue(Obligation(
            "confirm_prompt", 2, job_id=jid, template=template,
            slots=slots))
        self.emit("confirm_opened", job_id=jid, kind=kind)

    def focused_confirm(self):
        return self._confirm

    def confirm_prompt_spoken(self, job_id):
        """Speech lane reports the confirm prompt finished playing; the
        T_hold clock runs from here (lane-idle approximated by prompt
        completion + reply window, see REPORT)."""
        c = self._confirm
        if not c or c["job_id"] != job_id:
            return
        c["spoken_mono"] = time.monotonic()
        c["task"] = asyncio.create_task(self._confirm_timer(job_id))

    async def _confirm_timer(self, job_id):
        c0 = self._confirm
        window = DUP_CONFIRM_S if c0 and c0.get("kind") == "dup" \
            else CONFIRM_HOLD_S
        await asyncio.sleep(window)
        async with self._lock:
            c = self._confirm
            if not c or c["job_id"] != job_id:
                return
            self._confirm = None
            row = self.reg.get(job_id)
            if row and row["status"] == "BAND_HELD":
                d = c["directive"]
                await self.reg.transition(job_id, "CONFIRM_EXPIRED",
                                          ack_required=True)
                self.speak_enqueue(Obligation(
                    "outcome", 3, job_id=job_id,
                    template="T-CONFIRM-EXPIRED",
                    slots={"verb": d.verb or "do", "obj": d.obj or "that"}))
                self.emit("confirm_expired", job_id=job_id)
            elif row and row["status"] == "HELD_DUPLICATE":
                await self.reg.transition(job_id, "MERGED",
                                          ack_required=True)
                self.counters["merged"] += 1
                self.emit("dup_confirm_expired_merged", job_id=job_id)

    def _close_confirm_if(self, jid):
        c = self._confirm
        if c and c["job_id"] == jid:
            if c.get("task"):
                c["task"].cancel()
            self._confirm = None

    async def _confirm_reply(self, d: Directive):
        c = self._confirm
        if not c:
            self.emit("confirm_reply_orphan", seq=d.seq)
            return
        if d.p_closed is not None and d.p_closed < P_CLOSED_MIN:
            self.emit("confirm_reply_lowconf", seq=d.seq, p=d.p_closed)
            return  # timer keeps running; no state change
        jid = c["job_id"]
        self._close_confirm_if(jid)
        row = self.reg.get(jid)
        if row is None:
            return
        if d.closed_value == "yes":
            if row["status"] == "BAND_HELD":
                await self.reg.transition(jid, "QUEUED", ack_required=True,
                                          turn_id=d.turn)
                self._dispatch_at[jid] = time.monotonic()
                self.emit("confirm_affirmed", job_id=jid)
            elif row["status"] == "HELD_DUPLICATE":
                # Override: dispatch a fresh job carrying the duplicate's
                # spec under an override idempotency key (Part F).
                spec = json.loads(row["task_spec"] or "{}")
                nd = Directive(seq=d.seq, turn=d.turn, kind="delegate",
                               text=spec.get("text", ""),
                               provenance="RULE", p_delegate=1.0,
                               task_class=row["task_class"],
                               entity_set=json.loads(row["entity_set"] or "[]"),
                               verb=spec.get("verb", ""), obj=spec.get("obj", ""))
                await self.reg.transition(jid, "MERGED", turn_id=d.turn)
                new_id = await self._delegate(nd)
                self.emit("dup_override_dispatch", job_id=new_id, of=jid)
        else:  # no / silence-equivalent explicit no
            if row["status"] == "BAND_HELD":
                await self.reg.transition(jid, "CONFIRM_REJECTED",
                                          ack_required=True, turn_id=d.turn)
                self.speak_enqueue(Obligation(
                    "outcome", 3, job_id=jid, template="T-CONFIRM-REJECTED"))
            elif row["status"] == "HELD_DUPLICATE":
                await self.reg.transition(jid, "MERGED", ack_required=True,
                                          turn_id=d.turn)
                self.counters["merged"] += 1
            self.emit("confirm_rejected", job_id=jid)

    # ------------------------------------------------------------ OA escalation replies
    async def _oa_reply(self, d: Directive):
        if not d.target_hint:
            return
        row = self.reg.get(d.target_hint)
        if not row or row["status"] != "NEEDS_INPUT":
            return
        spec = json.loads(row["task_spec"] or "{}")
        options = spec.get("oa_options") or []
        opt = next((o for o in options if o.get("id") == d.closed_value),
                   None)
        if opt is None or (d.p_closed is not None
                           and d.p_closed < P_CLOSED_MIN):
            return
        if opt.get("widens_scope") and (d.p_closed or 0) < TAU_CONFIRM:
            # Re-band: hold for a full confirm at the higher bar.
            nd = Directive(seq=d.seq, turn=d.turn, kind="delegate",
                           text=spec.get("text", ""),
                           provenance="ENCODER", p_delegate=d.p_closed,
                           task_class="SIDE_EFFECTING",
                           verb=spec.get("verb", ""), obj=spec.get("obj", ""))
            await self._open_confirm(d.target_hint, nd, kind="oa_widen")
            return
        spec["oa_answer"] = d.closed_value
        await self.reg.transition(d.target_hint, "RUNNING",
                                  ack_required=True, turn_id=d.turn,
                                  task_spec=spec)
        if self.executor is not None:
            self.executor.resume(d.target_hint, d.closed_value)
        self.emit("oa_resolved", job_id=d.target_hint,
                  option=d.closed_value)

    async def raise_escalation(self, job_id, question, options):
        """NEEDS_INPUT with a closed answer set (Part E): 2-6 options
        declared at raise time or the raise is rejected."""
        if not (2 <= len(options) <= 6):
            self.emit("oa_no_answer_set", job_id=job_id)
            return False
        row = self.reg.get(job_id)
        spec = json.loads(row["task_spec"] or "{}")
        spec["oa_options"] = options
        spec["oa_question"] = question
        await self.reg.transition(job_id, "NEEDS_INPUT", ack_required=True,
                                  task_spec=spec)
        self.speak_enqueue(Obligation(
            "confirm_prompt", 1, job_id=job_id, template="T-OA-QUESTION",
            slots={"question": question,
                   "options": ", ".join(o.get("label", o.get("id", ""))
                                        for o in options)}))
        return True

    # ------------------------------------------------------------ dispatch
    async def _dispatch_loop(self):
        while True:
            await asyncio.sleep(0.05)
            try:
                await self.expire_cancel_pending()
                now = time.monotonic()
                due = [j for j, t in list(self._dispatch_at.items())
                       if t <= now]
                for jid in due:
                    self._dispatch_at.pop(jid, None)
                    row = self.reg.get(jid)
                    if row and row["status"] == "QUEUED":
                        await self._dispatch_gate(jid, row)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.emit("dispatch_loop_error",
                          error=f"{type(e).__name__}: {e}")

    def _dedupe_gate(self, row):
        """Part F: returns (verdict, match_row). verdict in
        DISPATCH / MERGE_RO / HOLD_SE."""
        now = time.monotonic()
        spec = json.loads(row["task_spec"] or "{}")
        if spec.get("repeat_intent"):
            return "DISPATCH", None
        ents = set(json.loads(row["entity_set"] or "[]"))
        norm = row["spec_norm"]
        vec = spec.get("topic_vec")
        _, rows = self.reg.snapshot()
        for other in rows:
            if other["job_id"] == row["job_id"] or \
                    other["session_id"] != self.session_id:
                continue
            # A primary is always the EARLIER directive (Part F: the
            # gate compares a candidate against jobs that already
            # exist). Without this, two identical QUEUED jobs suppress
            # each other and neither dispatches.
            if other["origin_seq"] >= row["origin_seq"]:
                continue
            if other["status"] == "MERGED":
                continue  # a suppressed job is never a primary
            o_st = other["status"]
            if o_st in ("DROPPED_SUBTHRESHOLD", "CONFIRM_EXPIRED",
                        "CONFIRM_REJECTED"):
                continue  # never dispatched: not a comparison job
            o_term = other.get("terminal_mono")
            window = None
            if o_st in NONTERMINAL:
                window = float("inf")
            elif o_st in TERMINAL and o_term is not None:
                window = TRAILING_TERMINAL_S if \
                    other["task_class"] == "READ_ONLY" else \
                    TRAILING_SIDE_EFFECTING_S
                if now - o_term > window:
                    continue
            else:
                continue
            o_ents = set(json.loads(other["entity_set"] or "[]"))
            d1 = ents == o_ents
            d2 = False
            if norm and norm == other["spec_norm"]:
                d2 = True
            elif vec and self.embedder is not None:
                o_spec = json.loads(other["task_spec"] or "{}")
                o_vec = o_spec.get("topic_vec")
                if o_vec:
                    d2 = _cos(vec, o_vec) >= 0.92
            if d1 and d2:
                return ("MERGE_RO" if row["task_class"] == "READ_ONLY"
                        else "HOLD_SE"), other
        return "DISPATCH", None

    async def _dispatch_gate(self, jid, row):
        verdict, match = self._dedupe_gate(row)
        if verdict == "MERGE_RO":
            await self.reg.transition(jid, "MERGED", ack_required=True)
            self.counters["merged"] += 1
            self.emit("dup_merged", job_id=jid, of=match["job_id"])
            # Resolution utterance: silent if <2 s after the original's
            # ack, else a short cue (rate-limited by the speech lane).
            if time.monotonic() - match["created_mono"] >= 2.0:
                self.speak_enqueue(Obligation(
                    "outcome", 4, job_id=jid, template="T-DUP-STILL",
                    slots={"label": self._label(match["job_id"])}))
            return
        if verdict == "HOLD_SE":
            await self.reg.transition(jid, "HELD_DUPLICATE",
                                      ack_required=True)
            self.counters["held_duplicate"] += 1
            spec = json.loads(row["task_spec"] or "{}")
            self._confirm = {"job_id": jid, "kind": "dup",
                             "directive": Directive(
                                 seq=row["origin_seq"], turn=row["origin_turn"],
                                 kind="delegate", text=spec.get("text", ""),
                                 verb=spec.get("verb", ""),
                                 obj=spec.get("obj", "")),
                             "spoken_mono": None, "task": None}
            self.speak_enqueue(Obligation(
                "dup_confirm", 2, job_id=jid, template="T-DUP-CONFIRM",
                slots={"label": self._label(match["job_id"])}))
            self.emit("dup_held", job_id=jid, of=match["job_id"])
            return
        await self._dispatch(jid, row)

    async def _dispatch(self, jid, row):
        spec = json.loads(row["task_spec"] or "{}")
        key_src = f"{self.session_id}|{spec.get('dedupe_class', row['spec_norm'])}|{row['origin_seq']}"
        idem = hashlib.sha256(key_src.encode()).hexdigest()[:24]
        await self.reg.transition(jid, "DISPATCHING", ack_required=True,
                                  idempotency_key=idem)
        self.emit("dispatching", job_id=jid, idempotency_key=idem)
        if self.executor is None:
            self.counters["dropped_dispatch"] += 1
            self.emit("dispatch_no_executor", job_id=jid)
            return
        self._dispatch_tasks[jid] = asyncio.create_task(
            self.executor.dispatch(self.reg.get(jid)))

    async def _retry_later(self, job_id, delay):
        await asyncio.sleep(delay)
        async with self._lock:
            row = self.reg.get(job_id)
            if row and row["status"] == "DISPATCHING" and self.executor:
                self._dispatch_tasks[job_id] = asyncio.create_task(
                    self.executor.dispatch(row))

    # ------------------------------------------------------------ executor callbacks
    async def _on_started(self, job_id):
        async with self._lock:
            row = self.reg.get(job_id)
            if row and row["status"] == "DISPATCHING":
                await self.reg.transition(job_id, "RUNNING",
                                          ack_required=True)
                self.emit("job_started", job_id=job_id)

    async def _on_done(self, job_id, summary, facts, raw):
        async with self._lock:
            row = self.reg.get(job_id)
            if row is None or row["status"] in TERMINAL:
                return
            if row["status"] == "CANCEL_REQUESTED":
                # The work finished before the cancel landed. Record
                # the truth (DONE, result stored) but suppress the
                # announcement: delivered_mono is set in the same
                # write, so held-result logic never surfaces it
                # (§2.8: a result arriving after a cancel request is
                # not announced). Without this the job sat in
                # CANCEL_REQUESTED forever — non-terminal limbo.
                await self.reg.transition(
                    job_id, "DONE", ack_required=True,
                    result_summary=summary, result_facts=facts,
                    delivered_mono=time.monotonic())
                self.emit("completion_after_cancel", job_id=job_id)
                return
            await self.reg.transition(
                job_id, "DONE", ack_required=True, result_summary=summary,
                result_facts=facts)
            prio = 3 if row["task_class"] == "SIDE_EFFECTING" else 4
            self.speak_enqueue(Obligation(
                "outcome", prio, job_id=job_id, template="T-RESULT",
                slots={"label": self._label(job_id), "summary": summary},
                weave_ok=True))
            self.emit("job_done", job_id=job_id)

    async def _on_error(self, job_id, exc, phase, http_status=None,
                        retryable=False):
        async with self._lock:
            row = self.reg.get(job_id)
            if row is None or row["status"] in TERMINAL:
                return
            # Phase boundary (G.2): 'submit' = before the back office's
            # task_started ack (first byte); anything later is FAILED.
            if phase != "submit":
                await self.reg.transition(
                    job_id, "FAILED", ack_required=True,
                    fail_reason=str(exc), failure_code="worker_error")
                self.speak_enqueue(Obligation(
                    "outcome", 3, job_id=job_id, template="T-STATUS-FAILED",
                    slots={"label": self._label(job_id)}))
                self.emit("job_failed", job_id=job_id, error=str(exc))
                return
            tries = self._retry_counts.get(job_id, 0)
            if retryable and tries < 2:
                self._retry_counts[job_id] = tries + 1
                delay = 2.0 if tries == 0 else 8.0
                self.emit("dispatch_retry", job_id=job_id, attempt=tries + 1,
                          delay=delay)
                # Retry sleeps OUTSIDE the commitment lock.
                asyncio.create_task(self._retry_later(job_id, delay))
                return
            # Terminal DISPATCH_FAILED.
            if http_status in (401, 403):
                self.set_preflight_red(f"dispatch auth {http_status}")
            await self.reg.transition(
                job_id, "DISPATCH_FAILED", ack_required=True,
                fail_reason=str(exc), failure_code="dispatch_failed")
            self.counters["dispatch_failed"] += 1
            self.speak_enqueue(Obligation(
                "dispatch_failed", 3, job_id=job_id,
                template="T-DISPATCH-FAILED",
                slots={"label": self._label(job_id),
                       "preflight_red": self._preflight_red}))
            self.emit("dispatch_failed", job_id=job_id, error=str(exc),
                      http_status=http_status)

    # ------------------------------------------------------------ held results (G.3)
    def held_results(self):
        """Unannounced terminal jobs for this session, terminal order."""
        if self._client_awake:
            return []
        return self.reg.held_unannounced(self.session_id)

    async def mark_announced(self, job_id):
        now = time.monotonic()
        await self.reg.transition(
            job_id, self.reg.get(job_id)["status"], ack_required=True,
            announced=1, announced_mono=now, delivered_mono=now)

    # ------------------------------------------------------------ invariants audit
    def audit(self):
        """I1/I4/I6 spot-audit over in-memory records (property tests)."""
        problems = []
        _, rows = self.reg.snapshot()
        jobs = [r for r in rows if r["session_id"] == self.session_id]
        if len(jobs) != self._jobs_created:
            problems.append(
                f"I1: {self._jobs_created} creates vs {len(jobs)} rows")
        for r in jobs:
            if r["announced"] and r["status"] not in TERMINAL:
                problems.append(f"I4: announced non-terminal {r['job_id']}")
        return problems

    async def note_repeat_intent(self, jid):
        row = self.reg.get(jid)
        if row:
            spec = json.loads(row["task_spec"] or "{}")
            spec["repeat_intent"] = True
            await self.reg.transition(jid, row["status"], task_spec=spec)

def sanitize_label(raw):
    """Spoken job labels are template-rendered (screen-exempt), so a
    label must never carry an unverified typed value or a raw user
    sentence: strip typed spans (a misrouted user assertion once
    became a job label and the ACK parroted its false value),
    collapse whitespace, cap at 40 chars on a word boundary."""
    from .tagger import extract_typed_values
    text = raw or "that"
    spans = [(h["start"], h["end"]) for h in extract_typed_values(text)]
    if spans:
        out = []
        pos = 0
        for s, e in spans:
            out.append(text[pos:s])
            pos = e
        out.append(text[pos:])
        text = "".join(out)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:—-")
    if len(text) > 40:
        text = text[:40].rsplit(" ", 1)[0].strip(" ,.;:—-")
    return text or "that"
