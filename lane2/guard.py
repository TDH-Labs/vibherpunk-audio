"""Topic guard (v2 Part D) — blocks status-shaped talker sentences in
ANSWER generations that reference a tracked job.

Final safety net behind prompt discipline and the §2.7 screen. Three
actions: PASS (untouched), DRAFT (replaced by a STATUS.v2 template,
event logged), BLOCK (same replacement, stronger event). Ships in
DRAFT_ONLY mode: BLOCK decisions are computed, logged, and executed as
DRAFT until the D.4 bench gates pass.
"""
import json
import re
import time

from . import templates

DEFAULT_THRESHOLDS = {
    "p_status_block": 0.85,
    "cos_block": 0.82,
    "p_status_draft_lo": 0.60,
    "cos_draft_lo": 0.74,
    "cos_draft_hi": 0.82,
    "t_recent_s": 180.0,
    "t_delivered_s": 60.0,
    "suppression_s": 300.0,
    "recovery_escalation": 2,
    "mode": "DRAFT_ONLY",  # OFF | DRAFT_ONLY | BLOCK
}

STATUS_LEX = {
    "still", "yet", "so far", "by now",
}
STATUS_NOUN = {"update", "status", "progress", "report", "result",
               "results", "news", "word"}
_DONE_WORDS = {"done", "finished", "ready", "back", "complete",
               "completed"}
_DEICTIC = {"that", "it", "this", "the one", "that one"}

_PATTERNS = [
    # (a) (any|an) (update|status|word|news) on NP
    re.compile(r"\bany (?:update|status|word|news) on\b", re.I),
    re.compile(r"\b(?:an? )?(?:update|status|word|news|progress report) on\b", re.I),
    # (b) is NP still/already V-ing | done/finished/ready/back
    re.compile(r"\bis [^.?!]{0,40}?\b(?:still|already)\s+\w+ing\b", re.I),
    re.compile(r"\bis [^.?!]{0,40}?\b(?:done|finished|ready|back)\b", re.I),
    # (c) has NP V-en / come back / reported
    re.compile(r"\bhas [^.?!]{0,40}?\b(?:come back|reported|finished|"
               r"completed|been (?:done|sent|delivered))\b", re.I),
    re.compile(r"\bhas [^.?!]{0,40}?\s\w+(?:ed|en)\b", re.I),
    # (d) NP is still/already V-ing
    re.compile(r"\bis (?:still|already) \w+ing\b", re.I),
    # (e) heard back / anything back / come through yet
    re.compile(r"\b(?:heard|got) (?:anything|any word|back)\b", re.I),
    re.compile(r"\bcome through (?:yet|already)\b", re.I),
    # status question shapes
    re.compile(r"\bhow(?:'s| is| goes) [^.?!]{1,40}\b(?:going|coming)\b",
               re.I),
    re.compile(r"\bwhere (?:are we|do we stand) (?:on|with)\b", re.I),
    # declarative status assertions in answer clothing
    re.compile(r"\bstill (?:working|running|pending|in progress|"
               r"processing|underway)\b", re.I),
    re.compile(r"\bno (?:update|news|word|result)s? (?:yet|so far)\b",
               re.I),
    re.compile(r"\bnothing (?:back|yet|so far)\b", re.I),
    re.compile(r"\bnot (?:done|back|finished|ready) yet\b", re.I),
    # past-tense completion assertions ("just finished", "wrapped up")
    re.compile(r"\b(?:just\s+)?(?:finished|completed|wrapped up)\b",
               re.I),
]


def status_shape(sentence):
    """status_lex signal: grammar match over the sentence."""
    low = sentence.lower()
    return any(p.search(low) for p in _PATTERNS)


def _tokens(text):
    return [w for w in re.findall(r"[a-z']+", text.lower())]


def _label_tokens(label):
    stop = {"the", "a", "an", "to", "for", "of", "on", "that"}
    return {w for w in _tokens(label) if w not in stop}


def _cos(a, b):
    import math
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if not da or not db:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


class TopicGuard:
    def __init__(self, *, emit, thresholds=None, embedder=None,
                 clock=None):
        self.emit = emit
        self.th = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.th.update(thresholds)
        self.embedder = embedder
        self.clock = clock or time.time
        self.mode = self.th["mode"]
        self.suppressed_until = {}   # job_id -> epoch (recovery)
        self.recovery_count = 0
        self.escalated = False
        self._gen_replaced = set()   # (turn_id, job_id)

    @classmethod
    def from_file(cls, path, **kw):
        try:
            with open(path) as f:
                th = json.load(f)
        except Exception:
            th = None
        return cls(thresholds=th, **kw)

    # -------------------------------------------------- session state
    def begin_generation(self, turn_id):
        self._gen_replaced = {k for k in self._gen_replaced
                              if k[0] != turn_id}

    def note_completed_announced(self, job_id):
        """RecoverySuppress bookkeeping is engine-driven via
        note_status_like_utterance; per-generation suppression resets
        in begin_generation."""

    def note_status_like_utterance(self, *, user_text, p_status,
                                   jobs, now=None):
        """D.2 recovery: a status-like USER utterance about job j means
        the conversation legitimately moved to status — bypass the
        guard for j and suppress re-firing for suppression_s."""
        now = now or self.clock()
        shaped = status_shape(user_text)
        if not (shaped or (p_status is not None
                           and p_status >= self.th["p_status_draft_lo"])):
            return None
        ref = self._reference(user_text, None, p_status, jobs, now)
        job = ref[0] if ref else None
        self.recovery_count += 1
        if job is not None:
            self.suppressed_until[job["job_id"]] = \
                now + self.th["suppression_s"]
        if self.recovery_count >= self.th["recovery_escalation"] \
                and not self.escalated:
            self.escalated = True
            if self.mode == "BLOCK":
                self.mode = "DRAFT_ONLY"
            self.emit("guard_recovery_bypass", job_id=(
                job or {}).get("job_id"), escalated=True)
        else:
            self.emit("guard_recovery_bypass",
                      job_id=(job or {}).get("job_id"), escalated=False)
        return (job or {}).get("job_id")

    # -------------------------------------------------- eligibility
    def _eligible_jobs(self, jobs, session_id, now):
        out = []
        for j in jobs:
            if j.get("session_id") not in (None, session_id):
                continue
            if j["job_id"] in self.suppressed_until and \
                    now < self.suppressed_until[j["job_id"]]:
                continue
            st = j["state"]
            if st in ("QUEUED", "RUNNING"):
                out.append(j)
            elif st in ("DONE", "FAILED", "CANCELLED"):
                recent = j.get("completed_mono") is not None and \
                    now - j["completed_mono"] <= self.th["t_recent_s"]
                delivered_recent = j.get("delivered_mono") is not None \
                    and now - j["delivered_mono"] <= \
                    self.th["t_delivered_s"]
                if recent or delivered_recent:
                    out.append(j)
        return out

    def _reference(self, sentence, sent_lemmas, p_status, jobs, now):
        """Returns (job, evidence) for the uniquely referenced job."""
        eligible = jobs
        if not eligible:
            return None
        sent_tokens = set(_tokens(sentence))
        best = None
        for j in eligible:
            lab = _label_tokens(j.get("task_label") or "")
            ent = bool(lab & sent_tokens)
            cos = None
            if self.embedder is not None:
                try:
                    cos = _cos(self.embedder(sentence),
                               self.embedder(j.get("task_label") or ""))
                except Exception:
                    cos = None
            strong = (ent and cos is not None
                      and cos >= self.th["cos_block"]) or \
                (p_status is not None
                 and p_status >= self.th["p_status_block"])
            # Deictic reference counts only in subject position:
            # among the first tokens, or immediately before a status
            # predicate. A trailing object pronoun ("...loved it")
            # refers to the sentence's own topic, not to a job.
            toks = _tokens(sentence)
            status_words = _DONE_WORDS | {"still", "already"}
            deictic = bool(set(toks[:4]) & _DEICTIC) or any(
                toks[i] in _DEICTIC and toks[i + 1] in status_words
                for i in range(len(toks) - 1))
            if strong or deictic:
                score = (cos or 0.0) + (0.1 if ent else 0.0)
                if best is None or score > best[1]:
                    best = (j, score, {"ent": ent, "cos": cos,
                                       "p_status": p_status})
        if best is None:
            return None
        job, _, ev = best
        # deictic-only reference requires a unique open job
        if not ((ev["ent"] and ev["cos"] is not None
                 and ev["cos"] >= self.th["cos_block"])
                or (ev["p_status"] is not None
                    and ev["p_status"] >= self.th["p_status_block"])):
            open_jobs = [j for j in eligible
                         if j["state"] in ("QUEUED", "RUNNING")]
            if len(open_jobs) != 1 or open_jobs[0] is not job:
                return None
        return (job, ev)

    # -------------------------------------------------- evaluate
    def evaluate(self, *, sentence, turn_id, turn_type, jobs,
                 session_id, p_status=None, now=None):
        """Returns dict: action PASS|DRAFT|BLOCK, job, template_id,
        slots, evidence. Only ANSWER generations are candidates."""
        now = now or self.clock()
        base = {"action": "PASS", "job": None, "template_id": None,
                "slots": None, "evidence": {}}
        if self.mode == "OFF" or turn_type != "ANSWER":
            return base
        if not status_shape(sentence):
            return base
        eligible = self._eligible_jobs(jobs, session_id, now)
        if not eligible:
            return base
        ref = self._reference(sentence, None, p_status, eligible, now)
        shaped = True
        sent_tokens = set(_tokens(sentence))
        any_ent = any(_label_tokens(j.get("task_label") or "")
                      & sent_tokens for j in eligible)
        any_cos = None
        if self.embedder is not None:
            try:
                any_cos = max(_cos(self.embedder(sentence),
                                   self.embedder(
                                       j.get("task_label") or ""))
                              for j in eligible)
            except Exception:
                any_cos = None
        if ref is not None:
            job, ev = ref
            action = "BLOCK"
        elif shaped and (
                (p_status is not None
                 and self.th["p_status_draft_lo"] <= p_status
                 < self.th["p_status_block"])
                or any_ent
                or (any_cos is not None
                    and self.th["cos_draft_lo"] <= any_cos
                    < self.th["cos_draft_hi"])):
            # DRAFT needs a probable UNIQUE referent (D.4 NEAR gate:
            # an ambiguous reference must not be replaced with a
            # guessed job's status). Probable = some entity overlap
            # with the best job, or exactly one eligible open job.
            job = None
            best_score = -1
            for j in eligible:
                lab = _label_tokens(j.get("task_label") or "")
                s = len(lab & sent_tokens)
                if s > best_score:
                    best_score, job = s, j
            open_eligible = [j for j in eligible
                             if j["state"] in ("QUEUED", "RUNNING")]
            if job is None or (best_score < 1
                               and len(open_eligible) != 1):
                return base
            if best_score < 1:
                job = open_eligible[0]
            ev = {"ent": any_ent, "cos": any_cos, "p_status": p_status}
            action = "DRAFT"
        else:
            return base
        if (turn_id, job["job_id"]) in self._gen_replaced:
            return {**base, "action": "DROP",
                    "evidence": {"reason": "already_replaced"}}
        self._gen_replaced.add((turn_id, job["job_id"]))
        tid, slots = _status_template_for(job)
        executed = action
        if action == "BLOCK" and self.mode == "DRAFT_ONLY":
            executed = "DRAFT"
        self.emit("guard_block" if executed == "BLOCK" else "guard_draft",
                  job_id=job["job_id"], decision=action,
                  evidence={k: (round(v, 3) if isinstance(v, float)
                                else v) for k, v in ev.items()},
                  sentence=sentence[:80])
        return {"action": executed, "decision": action, "job": job,
                "template_id": tid, "slots": slots, "evidence": ev}

    # -------------------------------------------------- late re-check
    def recheck(self, *, job, jobs_now, now=None):
        """At plan commit: if the referenced job was announced/
        delivered meanwhile, the replacement is dropped (returns
        'DROP'); if its state changed, re-render ('RERENDER', tid,
        slots); version-identical eligibility returns 'KEEP'."""
        now = now or self.clock()
        cur = next((j for j in jobs_now
                    if j["job_id"] == job["job_id"]), None)
        if cur is None:
            return ("DROP", None, None)
        if cur.get("delivered_mono") is not None and \
                job.get("delivered_mono") is None:
            return ("DROP", None, None)
        if cur["state"] != job["state"]:
            tid, slots = _status_template_for(cur)
            return ("RERENDER", tid, slots)
        return ("KEEP", None, None)


def _status_template_for(job):
    from .facts import outcome_line
    label = job.get("task_label") or "that"
    st = job["state"]
    if st == "QUEUED":
        return ("T-GUARD-QUEUED", {"label": label})
    if st == "RUNNING":
        if job.get("eta_hint"):
            return ("T-GUARD-RUNNING-ETA",
                    {"label": label, "eta": str(job["eta_hint"])})
        return ("T-GUARD-RUNNING", {"label": label})
    if st == "DONE":
        line = job.get("outcome_line") or outcome_line(job)
        return ("T-GUARD-DONE", {"label": label, "summary": line})
    if st == "FAILED":
        return ("T-GUARD-FAILED", {"label": label})
    return ("T-GUARD-CANCELLED", {"label": label})
