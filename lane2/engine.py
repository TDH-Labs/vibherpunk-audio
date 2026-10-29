"""Lane engine — one user turn end to end (v2 §2 + SCOPE Step 6/7).

Pipeline per turn: router decision -> Directive(s) -> Arbiter (all
commitments land in the registry before anything is spoken) ->
SpeechPlan (arbiter obligations + at most one talker generation) ->
SpeechSequencer (sliced speech; every generated slice screened) ->
context commit + fact extraction.

The engine is transport-agnostic: Pipecat (production) and the bench
both drive it through a speaker object. Async obligations (job
outcomes, confirm prompts) arrive via the arbiter's speak_enqueue
callback into an outbox; the transport drains the outbox when idle,
and the sequencer weaves result obligations into an active answer.
"""
import asyncio
import hashlib
import re
import time

from .arbiter import Arbiter, Directive, Obligation
from .facts import outcome_line
from .screen import canonical_typed
from . import templates

_CANCEL_RE = re.compile(
    r"\b(cancel|never ?mind|stop that|forget it|don'?t bother|"
    r"call it off|scrap that)\b", re.I)
_KEEP_RE = re.compile(r"\b(keep it|keep going|go ahead|carry on|"
                      r"leave it running|don'?t cancel)\b", re.I)
_YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go ahead",
        "do it", "please do", "yes please", "correct", "right"}
_NO = {"no", "nope", "don't", "do not", "never mind", "cancel",
       "not now", "no thanks", "stop"}
_REPEAT_RE = re.compile(
    r"\b(again|another (?:one|time)|repeat|same again|once more|"
    r"do it again|run it again)\b", re.I)
_TASK_VERB = re.compile(
    r"\b(research|check|read|find|search|summarize|write|send|email|"
    r"text|call|book|look up|tell me about|show me|give me|pay|file|"
    r"submit|forward|schedule|remind|sort|review|arrange|order)\b",
    re.I)

# Screen state map: registry -> screen/guard coarse states.
_STATE_MAP = {
    "QUEUED": "QUEUED", "BAND_HELD": "QUEUED",
    "HELD_DUPLICATE": "QUEUED", "DISPATCHING": "RUNNING",
    "RUNNING": "RUNNING", "NEEDS_INPUT": "RUNNING",
    "CANCEL_REQUESTED": "RUNNING", "DONE": "DONE",
    "FAILED": "FAILED", "DISPATCH_FAILED": "FAILED",
    "CANCELLED": "CANCELLED",
}


def closed_value(text):
    t = " ".join(re.findall(r"[a-z']+", (text or "").lower())).strip()
    if t in _YES:
        return "yes", 0.95
    if t in _NO:
        return "no", 0.95
    if t.startswith("yes") and len(t.split()) <= 3:
        return "yes", 0.8
    if t.startswith("no") and len(t.split()) <= 3:
        return "no", 0.8
    return None, 0.0


def _verb_obj(text):
    m = _TASK_VERB.search(text.lower())
    if not m:
        return "", text.strip()[:60]
    verb = m.group(1)
    obj = text[m.end():].strip(" .,")
    obj = re.sub(r"^(for|to|about|on|into)\s+", "", obj)
    return verb, (obj or text.strip()[:60])


class LaneEngine:
    def __init__(self, *, session_id, registry, context, ledger, memo,
                 talker, screen, guard, decider, emit,
                 entity_fn=None, embedder=None, executor=None,
                 lookup_fn=None):
        self.session_id = session_id
        self.reg = registry
        self.log = context
        self.ledger = ledger
        self.memo = memo
        self.talker = talker
        self.screen = screen
        self.guard = guard
        self.decider = decider
        self.emit = emit
        self.entity_fn = entity_fn
        self.seq = 0
        self.turn = 0
        self.outbox = []            # Obligations awaiting speech
        self._outbox_event = asyncio.Event()
        self._turn_active = False
        self._seq_sequencer = None
        self._last_turn_end = 0.0
        self._started = False
        self.user_history = []      # recent user utterances (screen U)
        self.arbiter = Arbiter(
            registry, session_id=session_id, emit=emit,
            speak_enqueue=self._enqueue_obligation,
            executor=executor, lookup_fn=lookup_fn, embedder=embedder)
        self._lookup_facts = []
        self._lookup_source = None
        self._warm_task = None

    # ------------------------------------------------------------ lifecycle
    async def start(self):
        await self.arbiter.start()
        self._started = True

    async def stop(self):
        await self.arbiter.stop()

    def barge_in(self):
        """InterruptionFrame: cancel the active speech plan. Unplayed
        slices are discarded by the sequencer, never resumed."""
        if self._seq_sequencer is not None:
            self._seq_sequencer.cancel()

    def _enqueue_obligation(self, ob):
        self.outbox.append(ob)
        self._outbox_event.set()
        # mid-answer weave: result obligations may join the active plan
        if self._seq_sequencer is not None and ob.weave_ok:
            self._seq_sequencer.note_result_obligation(
                self._ob_to_dict(ob))

    # ------------------------------------------------------------ snapshots
    def _task_rows(self):
        _, rows = self.reg.snapshot()
        out = []
        for r in rows:
            if r["session_id"] != self.session_id:
                continue
            st = _STATE_MAP.get(r["status"])
            if st is None:
                continue
            import json as _json
            spec = _json.loads(r["task_spec"] or "{}")
            out.append({
                "task_id": r["job_id"], "job_id": r["job_id"],
                "task_label": (spec.get("obj") or spec.get("text")
                               or r.get("title") or "that")[:40],
                "intent_class": spec.get("verb", ""),
                "state": st, "registry_state": r["status"],
                "eta_hint": None, "failure_code": r.get("failure_code"),
                "session_id": self.session_id,
                "completed_mono": r.get("terminal_mono"),
                "delivered_mono": r.get("delivered_mono"),
                "outcome_line": outcome_line(r),
                "title": r.get("title"),
            })
        return out

    def _screen_snapshot(self):
        version = self.reg.mirror_version()
        facts = list(self._lookup_facts)
        _, rows = self.reg.snapshot()
        for r in rows:
            if r["session_id"] != self.session_id or \
                    r["status"] != "DONE":
                continue
            import json as _json
            try:
                rf = _json.loads(r["result_facts"] or "[]")
            except Exception:
                rf = []
            for h in rf:
                facts.append({
                    "kind": h.get("type", "TEXT"), "task_id": r["job_id"],
                    "subject": self.arbiter._label(r["job_id"]),
                    "value": h.get("surface", ""),
                    "canon": None})
            summary = r.get("result_summary") or ""
            from .screen import NULL_RESULT
            for p in NULL_RESULT:
                m = re.search(p, " " + summary.lower() + " ")
                if m:
                    facts.append({
                        "kind": "NULL_RESULT", "task_id": r["job_id"],
                        "subject": self.arbiter._label(r["job_id"]),
                        "value": m.group(0).strip(), "canon": None})
                    break
        return version, self._task_rows(), facts

    # ------------------------------------------------------------ obligations
    _SPEECH_KIND = {
        "ack": "ack", "ack_escalate": "ack", "hold_ack": "ack",
        "outcome": "result", "cancel_outcome": "cancel_confirm",
        "confirm_prompt": "status", "dup_confirm": "status",
        "trailing_status": "status", "status": "status",
        "scope_killed": "status", "dispatch_failed": "status",
        "preflight_refusal": "status", "clarify": "clarify",
        "reprompt": "clarify", "held_batch": "held",
    }

    def _ob_to_dict(self, ob):
        d = {"kind": self._SPEECH_KIND.get(ob.kind, "chatter"),
             "arb_kind": ob.kind,
             "template_id": ob.template,
             "slots": dict(ob.slots or {}), "job_id": ob.job_id,
             "priority": ob.priority}
        return d

    def _drain_outbox(self):
        obs = self.outbox
        self.outbox = []
        self._outbox_event.clear()
        return [self._ob_to_dict(o) for o in obs]

    # ------------------------------------------------------------ directives
    async def _directives_for(self, text, route):
        self.seq += 1
        seq = self.seq
        decision = route["decision"]
        source = route.get("source", "")
        provenance = "RULE" if source.startswith("rule") else "ENCODER"
        scores = route.get("scores") or {}
        p_del = scores.get("delegate")
        if provenance == "RULE" and decision == "delegate":
            p_del = 1.0
        verb, obj = _verb_obj(text)
        ents = []
        if self.entity_fn is not None:
            try:
                ents = await self.entity_fn(text)
            except Exception:
                ents = []
        base = dict(seq=seq, turn=self.turn, text=text, verb=verb,
                    obj=obj, entity_set=ents)
        if decision == "delegate":
            rep = bool(_REPEAT_RE.search(text.lower()))
            return [Directive(kind="delegate", provenance=provenance,
                              p_delegate=p_del,
                              repeat_intent=rep, **base)]
        if decision == "hold":
            return [Directive(kind="hold", provenance=provenance, **base)]
        if decision == "lookup":
            return [Directive(kind="lookup", provenance=provenance,
                              **base)]
        return [Directive(kind="answer", provenance=provenance, **base)]

    # ------------------------------------------------------------ the turn
    async def handle_turn(self, text, speaker, *, sequencer_factory):
        """Run one full user turn. speaker: object with async speak().
        Returns the turn result dict (segments played + counters)."""
        turn_t0 = time.monotonic()   # first-audio latency is measured
        # from turn entry (Part B: end of user speech -> first audio)
        self.turn += 1
        self._turn_active = True
        self._lookup_facts = []
        self._lookup_source = None
        turn_id = self.turn
        result = {"turn": turn_id, "route": None, "played": [],
                  "counters": {}}
        try:
            # held results first after a sleep/disconnect gap
            pre_obs = []
            if not self._started:
                await self.start()
            held = self.arbiter.held_results()
            if held:
                self.arbiter.set_client_awake(True)
                # the wake announcement below speaks these from the
                # registry; drop their queued outcome obligations so
                # nothing is announced twice
                held_ids = {h["job_id"] for h in held}
                self.outbox = [o for o in self.outbox
                               if o.job_id not in held_ids]
            # 1) confirm replies pre-empt routing (§2.7 band / Part F)
            focused = self.arbiter.focused_confirm()
            cv, cp = closed_value(text)
            route = None
            if focused and cv:
                self.seq += 1
                await self.arbiter.on_directive(Directive(
                    seq=self.seq, turn=turn_id, kind="confirm_reply",
                    text=text, closed_value=cv, p_closed=cp))
                route = {"decision": "confirm", "source": "engine-closed"}
            elif _CANCEL_RE.search(text) and \
                    self.arbiter.reg.nonterminal(self.session_id):
                self.seq += 1
                verb, obj = _verb_obj(text)
                await self.arbiter.on_directive(Directive(
                    seq=self.seq, turn=turn_id, kind="cancel", text=text,
                    verb=verb, obj=obj))
                route = {"decision": "cancel", "source": "engine-rule"}
            elif _KEEP_RE.search(text) and \
                    self.arbiter._cancel_pending:
                self.seq += 1
                await self.arbiter.on_directive(Directive(
                    seq=self.seq, turn=turn_id, kind="keep", text=text))
                route = {"decision": "keep", "source": "engine-rule"}
            if route is None:
                pending = len(self.arbiter.reg.nonterminal(
                    self.session_id))
                route = await self.decider.decide(
                    text, jobs_pending=pending)
                if route["decision"] == "status":
                    # status turns are answered from the registry via
                    # the guard's templates, not the talker: mark the
                    # utterance for the guard recovery path and let a
                    # STATUS obligation speak.
                    jobs = self._task_rows()
                    live = [j for j in jobs if j["state"] in
                            ("QUEUED", "RUNNING")]
                    done = [j for j in jobs if j["state"] == "DONE"]
                    tgt = live[0] if live else (done[0] if done else None)
                    if tgt is not None:
                        from .guard import _status_template_for
                        tid, slots = _status_template_for(tgt)
                        pre_obs.append({"kind": "status",
                                        "template_id": tid, "slots": slots,
                                        "job_id": tgt["job_id"],
                                        "priority": 2})
                    route = {"decision": "answer", "source": "status"}
                    result["route"] = route
                    obs = pre_obs + self._drain_outbox()
                    return await self._speak_plan(
                        turn_id, obs, None, speaker, sequencer_factory,
                        text, result, turn_t0=turn_t0)
                dirs = await self._directives_for(text, route)
                status_flag = None
                if isinstance(dirs, tuple):
                    dirs, status_flag = dirs
                for d in dirs:
                    if d.kind == "lookup":
                        res = await self.arbiter.handle_lookup(d)
                        if res:
                            self._lookup_facts = res["facts"]
                            self._lookup_source = res["source"]
                    else:
                        await self.arbiter.on_directive(d)
            result["route"] = route
            # 2) register dispatch canaries for new jobs
            self._register_canaries()
            # 3) plan: obligations + generation when conversational
            obs = pre_obs + self._drain_outbox()
            for h in held:
                obs.insert(0, {
                    "kind": "held", "arb_kind": "held_batch",
                    "template_id": "T-RESULT",
                    "slots": {"label": self.arbiter._label(h["job_id"]),
                              "summary": h.get("result_summary")
                              or outcome_line(h)},
                    "job_id": h["job_id"], "priority": 5})
            generation = None
            decision = (route or {}).get("decision")
            if decision in ("answer", "lookup", "status", None) and \
                    not any(o["kind"] in ("clarify",)
                            and o.get("template_id") == "T-CLARIFY-HOLD"
                            for o in obs):
                # §3.3: never send the real turn while a post-eviction
                # warm request is in flight (2.0 s timeout, then go).
                if self._warm_task is not None and \
                        not self._warm_task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(self._warm_task), 2.0)
                    except (asyncio.TimeoutError, Exception):
                        self.emit("warm_timeout")
                profile = "LOOKUP" if self._lookup_facts else "ANSWER"
                facts_txt = []
                for f in self._lookup_facts:
                    facts_txt.append(f.get("quote_text")
                                     or f"{f.get('subject', '')}: "
                                     f"{f.get('value', '')}")
                ledger_lines = [
                    f"{j['task_label']}: {j['state'].lower()}"
                    for j in self._task_rows()
                    if j["state"] in ("QUEUED", "RUNNING")]
                prompt_b, meta = self.log.build_prompt(
                    mode_directive=f"MODE: {profile}", user_text=text,
                    lookup_facts=facts_txt or None,
                    ledger_lines=ledger_lines or None)
                self.emit("prompt_built", total_tokens=meta["total_tokens"],
                          expected_cacheable=meta["expected_cacheable"])
                generation = {"messages": prompt_b.decode("utf-8"),
                              "profile": profile}
            return await self._speak_plan(
                turn_id, obs, generation, speaker, sequencer_factory,
                text, result, turn_t0=turn_t0)
        finally:
            self._turn_active = False
            self._last_turn_end = time.monotonic()

    async def _speak_plan(self, turn_id, obligations, generation, speaker,
                          sequencer_factory, user_text, result,
                          turn_t0=None):
        seq = sequencer_factory()
        self._seq_sequencer = seq
        utterances = (self.user_history[-7:] + [user_text])
        self.screen.begin_turn(
            profile=(generation or {}).get("profile", "ANSWER"),
            user_utterances=utterances,
            lookup_facts=self._lookup_facts)
        p_status = None
        res = await seq.run_turn(
            turn_id=turn_id, obligations=obligations,
            generation=generation, speaker=speaker,
            jobs_snapshot_fn=lambda: (self._task_rows(),
                                      self._lookup_facts),
            p_status=p_status)
        self._seq_sequencer = None
        result["played"] = res["played"]
        result["counters"] = res["counters"]
        if turn_t0 is not None and res["played"]:
            result["first_slice_latency_s"] = (
                min(r["play_start"] for r in res["played"]) - turn_t0)
        else:
            result["first_slice_latency_s"] = res["first_slice_latency_s"]
        result["screen_before_play_all"] = res["screen_before_play_all"]
        # confirm windows start when the prompt is actually PLAYED
        played_keys = {(r.get("template_id"), r.get("job_id"))
                       for r in res["played"]}
        for o in obligations:
            if o.get("arb_kind") in ("confirm_prompt", "dup_confirm") \
                    and (o["template_id"], o.get("job_id")) in played_keys:
                self.arbiter.confirm_prompt_spoken(o["job_id"])
        spoken_text = " ".join(r["text"] for r in res["played"])
        # context commit + extraction
        job_refs = tuple({r_job for r_job in
                          [o.get("job_id") for o in obligations]
                          if r_job})
        self.log.commit_exchange(
            turn_id, user_text, spoken_text, job_refs=job_refs,
            stub_text=None, low_content=len(user_text.split()) <= 6)
        self.user_history.append(user_text)
        await self.ledger.extract_turn_async(turn_id, user_text)
        if self.log.history_tokens() > 2110:
            self.log.evict_if_needed()
            self._schedule_warm()
        # mark held announced for held obligations actually played
        for o in obligations:
            if o["kind"] == "held" and o.get("job_id"):
                await self.arbiter.mark_announced(o["job_id"])
        return result

    def _schedule_warm(self):
        """§3.3 post-eviction warm: one 1-token completion over the
        new prefix, in the background; the next real turn waits for
        it (bounded) in handle_turn."""
        async def _warm():
            try:
                prompt_b, _ = self.log.build_prompt(
                    mode_directive="MODE: ANSWER", user_text="(warm)")
                await self.talker.complete(prompt_b.decode("utf-8"),
                                           max_tokens=1, temperature=0.0)
                self.emit("warm_done")
            except Exception as e:
                self.emit("warm_failed", error=str(e)[:120])
        self._warm_task = asyncio.create_task(_warm())

    def _register_canaries(self):
        _, rows = self.reg.snapshot()
        for r in rows:
            if r["session_id"] != self.session_id:
                continue
            if r["status"] in ("QUEUED", "DISPATCHING", "RUNNING") and \
                    r["job_id"] not in getattr(self, "_canaried", set()):
                self._canaried = getattr(self, "_canaried", set())
                self._canaried.add(r["job_id"])
                h = hashlib.sha256(r["job_id"].encode()).hexdigest()
                ident = f"QX-{int(h[:4], 16) % 10000:04d}"
                money = f"${9000 + int(h[4:8], 16) % 900}.{42}"
                self.screen.register_canary(r["job_id"], [ident, money])

    # ------------------------------------------------------------ async speech
    async def announce_pending(self, speaker, *, sequencer_factory,
                               timeout_s=0.0):
        """Speak outbox obligations while idle (job outcomes, confirm
        prompts). Confirm windows start when the prompt is PLAYED:
        the caller wires speaker playback to confirm_prompt_spoken."""
        if not self.outbox or not self.arbiter._client_awake:
            return None  # asleep: outcomes stay held for the wake turn
        obs = self._drain_outbox()
        seq = sequencer_factory()
        utterances = self.user_history[-7:]
        self.screen.begin_turn(profile="ANSWER",
                               user_utterances=utterances,
                               lookup_facts=[])
        res = await seq.run_turn(
            turn_id=self.turn, obligations=obs, generation=None,
            speaker=speaker,
            jobs_snapshot_fn=lambda: (self._task_rows(), []),
            p_status=None)
        for o in obs:
            if o["kind"] == "held" and o.get("job_id"):
                await self.arbiter.mark_announced(o["job_id"])
        return res
