"""§2.7 output screen (v2 Part C) — mandatory, fail-closed.

The screen is the only path from talker text to TTS. Every generated
sentence passes C6/C0/C5/C1/C3a/C3b/C4/C2 (precedence in that order)
before release. Actions are DROP / REPLACE (template) / FALLBACK
(template, ends generation) / KILL (C6 only). Generated text is never
edited in place. Regeneration is DISABLED (SCOPE seam decision 3,
overriding §2.7.4's single retry): a sentence that fails is dropped
or replaced, never regenerated.

Deterministic only: lexicons, regexes, set comparisons, the shared
typed tagger. No model calls.
"""
import re
import time
from datetime import datetime, timedelta

from .tagger import extract_typed_values, parse_number_words
from . import templates

SCREEN_BUDGET_MS = 40.0

# ------------------------------------------------------------ lexicons
AGENT_NOUNS = [
    "back office", "the back office", "our team", "the team", "team",
    "the office", "our office", "specialists", "the specialists",
    "someone", "the system", "our records", "the records", "records",
    "the file", "our file", "the account", "our people", "staff",
    "they", "the agent", "our agent", "support", "the desk",
    "head office", "the department", "billing", "the crew",
]
RESULT_PREDICATES = [
    "find", "found", "confirm", "confirmed", "verify", "verified",
    "check", "checked", "review", "reviewed", "determine", "determined",
    "show", "shows", "showed", "say", "says", "said", "come back",
    "came back", "locate", "located", "flag", "flagged", "clear",
    "cleared", "approve", "approved", "reject", "rejected", "match",
    "matched", "reconcile", "reconciled", "look into", "looked into",
    "turn up", "turned up", "see", "sees", "saw", "report back",
    "reported", "pull up", "pulled up", "go through", "went through",
]
NULL_RESULT = [
    r"no discrepanc(?:y|ies)", r"nothing (?:unusual|out of the "
    r"ordinary|wrong|to report)", r"everything (?:checks out|matches|"
    r"lines up|looks (?:good|fine|correct))", r"all (?:clear|good|set)",
    r"no issues", r"no problems(?: found)?", r"nothing came up",
    r"clean bill", r"all in order", r"everything is in order",
    r"no errors(?: found)?", r"no flags", r"nothing flagged",
]
DOC_NOUNS = [
    "lease", "contract", "agreement", "policy", "statement", "invoice",
    "receipt", "bill", "terms", "section", "clause", "paragraph",
    "page", "document", "letter", "email", "notice", "report", "form",
    "application", "permit", "license", "quote", "proposal", "memo",
    "file", "record", "note", "schedule", "statement", "handbook",
    "manual", "guide", "waiver", "deed", "title", "certificate",
    "transcript", "summary",
]
OUTCOME_PREDICATES = [
    "approved", "processed", "cancelled", "canceled", "refunded",
    "confirmed", "booked", "resolved", "fixed", "sorted", "cleared",
    "completed", "sent", "received", "paid", "filed", "submitted",
    "delivered", "scheduled", "renewed", "transferred", "deposited",
    "charged", "credited", "done", "finished", "complete",
]
PRESUPPOSE = [
    r"now that", r"since (?:your|the) [\w ]{0,30} (?:has|have) been",
    r"glad (?:that's|it's|we got)", r"great[,\u2014-] (?:your|the|that)",
    r"good news[,\u2014-]", r"once (?:that's|it's)",
    r"with that (?:done|sorted|taken care of)",
    r"now (?:that|since) (?:your|the|it)",
]
ETA_VAGUE = ["soon", "shortly", "any minute", "later today",
             "by tonight", "by tomorrow", "in a bit", "right away"]
ASSENT = [
    "yes", "yeah", "yep", "right", "that's right", "correct",
    "exactly", "you're right", "as you said", "like you said",
    "indeed", "sure is", "it is", "that is right", "absolutely",
]
ATTRIBUTION = ["says", "states", "mentions", "requires", "allows",
               "specifies", "according to", "per the"]
STOPWORDS = {
    "the", "a", "an", "is", "it", "that", "this", "to", "of", "on",
    "in", "for", "and", "or", "you", "your", "my", "i", "was", "were",
    "has", "have", "had", "be", "been", "with", "at", "by", "from",
    "what", "did", "do", "does", "are", "if", "so", "but", "not",
}

_C1B = re.compile(
    r"\bI(?:'ve| have|'d| had)?\s+(?:checked|looked (?:into|up|at)|"
    r"found|verified|confirmed|reviewed|pulled up|went through)\b|"
    r"\bI (?:can )?see (?:that|here)\b", re.I)
_C3A_FIRST = re.compile(
    r"\bI(?:'ll| will| am going to|'m going to)\s+(?:have|get|ask|send|"
    r"text|email|call|remind|book|schedule|cancel|file|submit|pass|"
    r"forward|flag|escalate|follow up|check|look into|arrange|set up)\b",
    re.I)
_C3A_THIRD = re.compile(
    r"\b(?:they|the back office|someone|the team)(?:'ll| will)\s+"
    r"(?:call|text|email|get back|reach out|send|follow up|contact|"
    r"check|handle)\b", re.I)
_C3A_DONE = re.compile(
    r"\bI(?:'ve| have)\s+(?:passed|sent|forwarded|booked|scheduled|"
    r"cancelled|canceled|filed|submitted|flagged|escalated|put in|"
    r"asked|arranged)\b|\bit's (?:done|in|submitted|on its way)\b|"
    r"\bthat's been (?:sent|passed along|submitted|taken care of)\b",
    re.I)
_COMMIT_VERB_CLASS = {
    "send": "send", "email": "send", "text": "send", "call": "call",
    "book": "book", "schedule": "book", "cancel": "cancel",
    "file": "file", "submit": "file", "forward": "send",
    "remind": "remind",
}


def _lemmas(text):
    return [w for w in re.findall(r"[a-z']+", text.lower())
            if w not in STOPWORDS]


def _lemma_set(text):
    out = set()
    for w in _lemmas(text):
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


# ------------------------------------------------------------ typed values
_UNIT_RE = {
    "DURATION": re.compile(
        r"(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s*$", re.I),
    "QUANTITY": re.compile(
        r"(items?|files?|emails?|messages?|pages?|people|tickets?|"
        r"photos?|songs?|calls?|meetings?|tasks?|jobs?|entries|"
        r"records?|results?)\s*$", re.I),
}
_WORD_MONEY = re.compile(
    r"\b((?:(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|and)[ -]+){1,7}?"
    r"(?:dollars|bucks))\b", re.I)
_TIME_PHRASE = re.compile(
    r"\b(quarter|half) (?:past|to) (\w+)\b", re.I)


def canonical_typed(text):
    """Typed spans with screen canonical forms:
    (type, canonical, surface, start, end, hedged).
    MONEY -> minor units int; DATE -> ISO or relative-resolved key;
    TIME -> 'HH:MM'; DURATION -> seconds; PERCENT -> basis points;
    QUANTITY -> (value, unit); IDENT -> stripped upper."""
    out = []
    spans = []
    for h in extract_typed_values(text):
        s, e = h["start"], h["end"]
        spans.append((s, e))
        typ, surf = h["type"], h["surface"]
        v = h["value"]
        if typ == "MONEY":
            try:
                canon = int(round(float(v) * 100))
            except Exception:
                canon = v
        elif typ == "PERCENT":
            try:
                canon = int(round(float(v) * 100))
            except Exception:
                canon = v
        elif typ == "DURATION":
            m = _UNIT_RE["DURATION"].search(surf)
            unit = (m.group(1).lower().rstrip("s") if m else "second")
            mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400,
                    "week": 604800, "month": 2592000, "year": 31536000}
            try:
                canon = int(float(v) * mult.get(unit, 1))
            except Exception:
                canon = v
        elif typ == "QUANTITY":
            m = _UNIT_RE["QUANTITY"].search(surf)
            unit = (m.group(1).lower() if m else "")
            canon = (v, unit)
        elif typ == "TIME":
            canon = _canon_time(surf)
        elif typ == "DATE":
            canon = re.sub(r"[\s,]+", " ", surf.lower()).strip()
        else:
            canon = re.sub(r"[^A-Z0-9]", "", surf.upper())
        out.append({"type": typ, "canon": canon, "surface": surf,
                    "start": s, "end": e, "hedged": h["hedged"]})
    # long-form word money the base tagger misses
    for m in _WORD_MONEY.finditer(text):
        s, e = m.span()
        if any(not (e <= a or s >= b) for a, b in spans):
            continue
        words = m.group(1).lower()
        dollars = parse_number_words(
            " ".join(w for w in re.findall(r"[a-z]+", words)))
        if dollars is None:
            continue
        spans.append((s, e))
        out.append({"type": "MONEY", "canon": int(dollars * 100),
                    "surface": m.group(0), "start": s, "end": e,
                    "hedged": False})
    for m in _TIME_PHRASE.finditer(text):
        s, e = m.span()
        if any(not (e <= a or s >= b) for a, b in spans):
            continue
        hour = parse_number_words(m.group(2))
        if hour is None:
            continue
        if m.group(1) == "quarter" and " to " in m.group(0):
            hh, mm = hour - 1, 45
        elif m.group(1) == "quarter":
            hh, mm = hour, 15
        else:
            hh, mm = hour, 30
        out.append({"type": "TIME", "canon": f"{hh % 24:02d}:{mm:02d}",
                    "surface": m.group(0), "start": s, "end": e,
                    "hedged": False})
    out.sort(key=lambda h: h["start"])
    return out


def _canon_time(surf):
    s = surf.lower().replace(" ", "").replace(".", "")
    if s == "noon":
        return "12:00"
    if s == "midnight":
        return "00:00"
    m = re.match(r"(\d{1,2})(?::(\d{2}))?(?::\d{2})?(am|pm)?$", s)
    if not m:
        return s
    hh, mm, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and hh < 12:
        hh += 12
    if ap == "am" and hh == 12:
        hh = 0
    return f"{hh:02d}:{mm:02d}"


# ------------------------------------------------------------ segments
# The minting key lives at module level in THIS module only (lane3 H2:
# in v2 it was a public class attribute and the speech module minted
# segments with it — the "type is the enforcement" claim was false).
# Nothing outside screen.py can construct a ScreenedSegment without
# importing a private name, and both speakers isinstance-check.
_SEGMENT_KEY = object()


class ScreenedSegment:
    """Only the screen constructs these (build-time checked path to
    TTS, §2.7.1 item 1)."""
    __slots__ = ("text", "kind", "template_id", "job_id", "screened_ms",
                 "snapshot_version", "hits")

    def __init__(self, key, text, kind, template_id=None, job_id=None,
                 screened_ms=0.0, snapshot_version=0, hits=()):
        if key is not _SEGMENT_KEY:
            raise TypeError("ScreenedSegment is screen-constructed only")
        self.text = text
        self.kind = kind  # generated | template
        self.template_id = template_id
        self.job_id = job_id
        self.screened_ms = screened_ms
        self.snapshot_version = snapshot_version
        self.hits = hits


class ScreenFault(Exception):
    pass


# ------------------------------------------------------------ the screen
class Screen:
    def __init__(self, *, emit, snapshot_fn, tz_name="America/Los_Angeles",
                 assistant_name="Vibherpunk", user_name="Adam",
                 global_canaries=()):
        self.emit = emit
        self.snapshot_fn = snapshot_fn  # () -> (version, [task rows], facts)
        self.assistant_name = assistant_name
        self.user_name = user_name
        try:
            from zoneinfo import ZoneInfo
            self.tz = ZoneInfo(tz_name)
        except Exception:
            self.tz = None
        self.global_canaries = list(global_canaries)
        self.task_canaries = {}  # task_id -> [values]
        self.leak_suspect = False
        self._reset_turn()

    # -------------------------------------------------- per-turn state
    def _reset_turn(self):
        self.profile = "ANSWER"
        self.S0_version = -1
        self.tasks = []
        self.facts = []
        self.allow = {}       # (type, canon) -> set(provenance)
        self.quotes = []      # (normalized quote, doc noun)
        self.Q = []           # user-assertion claims
        self.used_templates = set()
        self.sentences_seen = 0
        self.hits_log = []

    def register_canary(self, task_id, values):
        self.task_canaries.setdefault(task_id, []).extend(values)

    def begin_turn(self, *, profile, user_utterances, lookup_facts=(),
                   now=None):
        self._reset_turn()
        self.profile = profile
        version, tasks, facts = self.snapshot_fn()
        self.S0_version = version
        self.tasks = tasks
        self.facts = list(facts) + list(lookup_facts)
        now = now or datetime.now(self.tz)
        self._build_allowlist(user_utterances, now)
        self._build_Q(user_utterances)
        self.quotes = []
        for f in self.facts:
            qt = f.get("quote_text")
            if qt:
                self.quotes.append((_norm_quote(qt),
                                    (f.get("source_doc_title") or "").lower()))
        self.user_utterances = list(user_utterances)

    # -------------------------------------------------- allowlist / Q
    def _allow_add(self, type_, canon, prov):
        self.allow.setdefault((type_, canon), set()).add(prov)

    def _build_allowlist(self, user_utterances, now):
        # HEADER: today / now in user tz
        self._allow_add("DATE", now.strftime("%Y-%m-%d"), "HEADER")
        self._allow_add("TIME", now.strftime("%H:%M"), "HEADER")
        # CONST: names
        # REGISTRY: labels are text, eta hints are typed
        for t in self.tasks:
            eta = t.get("eta_hint")
            if eta:
                for h in canonical_typed(str(eta)):
                    self._allow_add(h["type"], h["canon"], "REGISTRY")
        # LOOKUP facts
        for f in self.facts:
            if f.get("canon") is not None:
                self._allow_add(f["kind"], f["canon"], "LOOKUP")
            elif f.get("value") is not None:
                for h in canonical_typed(str(f["value"])):
                    self._allow_add(h["type"], h["canon"], "LOOKUP")
        # USER typed values
        for u in user_utterances:
            for h in canonical_typed(u):
                self._allow_add(h["type"], h["canon"], "USER")

    def _build_Q(self, user_utterances):
        self.Q = []
        pats = [
            re.compile(r"\bmy ([\w ]{2,30}?) (?:is|was|ends|expires|"
                       r"costs|was set to) ([^.,;]+)", re.I),
            re.compile(r"\b(?:the back office|they|you) (?:said|told me|"
                       r"already|confirmed|approved) ([^.,;]+)", re.I),
            re.compile(r"\bit's supposed to be ([^.,;]+)", re.I),
        ]
        for idx, u in enumerate(user_utterances):
            claims = []
            for h in canonical_typed(u):
                claims.append({"subject": _lemma_set(u), "value":
                               (h["type"], h["canon"]), "turn": idx,
                               "tag": False})
            for p in pats:
                for m in p.finditer(u):
                    claims.append({"subject": _lemma_set(m.group(0)),
                                   "value": None, "turn": idx,
                                   "tag": False})
            if re.search(r",\s*(?:right|correct)\?\s*$", u.strip(), re.I):
                for c in claims:
                    c["tag"] = True
            for c in claims:
                c["corroborated"] = self._corroborated(c)
                self.Q.append(c)

    def _corroborated(self, claim):
        if claim.get("value") is None:
            return False
        provs = self.allow.get(claim["value"], set())
        return bool(provs - {"USER"})

    # -------------------------------------------------- task binding
    def _bind_task(self, sentence):
        lem = _lemma_set(sentence)
        for t in self.tasks:
            lab = _lemma_set(t.get("task_label") or "")
            if lab and lab <= lem:
                return t
        live = [t for t in self.tasks
                if t["state"] in ("DRAFT", "DISPATCHED", "RUNNING")]
        done_recent = [t for t in self.tasks if t["state"] == "DONE"]
        if len(live) == 1:
            return live[0]
        if not live and len(done_recent) == 1:
            return done_recent[0]
        return None

    def _facts_for(self, task, kind=None):
        if task is None:
            return []
        out = [f for f in self.facts
               if f.get("task_id") in (None, task["task_id"])]
        if kind:
            out = [f for f in out if f["kind"] == kind]
        return out

    # -------------------------------------------------- checks
    def _c6(self, sentence):
        for v in self.global_canaries:
            if v and v in sentence:
                return True
        for vals in self.task_canaries.values():
            for v in vals:
                if v and v in sentence:
                    return True
        # (c) unreleased-payload values: typed values in non-DONE tasks'
        # payload digest that are absent from the allowlist are checked
        # by C2; C6 covers the explicit canary channels (a)/(b) plus any
        # payload digest the engine registered as canaries.
        return False

    def _c5(self, sentence):
        low = sentence.lower()
        quote_spans = re.findall(
            r"[\"“”«»]([^\"“”«»]{3,})[\"“”«»]", sentence)
        colon = re.search(
            r"\b(?:states|says|reads|specifies|provides|stipulates|"
            r"notes)\s*:\s*(.+)$", sentence, re.I)
        doc_hit = None
        dm = None
        for d in DOC_NOUNS:
            if re.search(rf"\b{d}\b", low):
                dm = d
                break
        attrib = any(re.search(rf"\b{a}\b", low) for a in ATTRIBUTION) \
            or "according to" in low
        if quote_spans:
            for q in quote_spans:
                nq = _norm_quote(q)
                ok = any(nq and nq in qt and (not dm or dm in title)
                         for qt, title in self.quotes)
                if not ok:
                    return {"doc": dm}
            return None
        if colon and dm:
            return {"doc": dm}
        if dm and attrib:
            bound = self._bind_task(sentence)
            facts = [f for f in self._facts_for(bound)
                     if f["kind"] in ("QUOTE", "TEXT")]
            if not facts:
                return {"doc": dm}
        return None

    def _c1(self, sentence):
        low = " " + sentence.lower() + " "
        # (b) first-person investigative completion
        if _C1B.search(sentence):
            return True
        # (c) null-result phrases (unconditional)
        for p in NULL_RESULT:
            if re.search(p, low):
                return True
        # (a) agent noun within 8 tokens of a result predicate
        toks = re.findall(r"[a-z']+", low)
        agent_pos = []
        for i, t in enumerate(toks):
            for noun in AGENT_NOUNS:
                parts = noun.split()
                if toks[i:i + len(parts)] == parts:
                    agent_pos.append(i)
        pred_pos = [i for i, t in enumerate(toks)
                    if t in RESULT_PREDICATES]
        for a in agent_pos:
            for p in pred_pos:
                if abs(a - p) <= 8:
                    return True
        return False

    def _c3a(self, sentence):
        hit = None
        m = _C3A_FIRST.search(sentence) or _C3A_THIRD.search(sentence) \
            or _C3A_DONE.search(sentence)
        if not m:
            return None
        verb = None
        vm = re.search(
            r"\b(send|email|text|call|book|schedule|cancel|file|submit|"
            r"forward|remind|passed|sent|forwarded|booked|scheduled|"
            r"cancelled|filed|submitted)\b", sentence.lower())
        if vm:
            verb = _COMMIT_VERB_CLASS.get(vm.group(1))
        bound = self._bind_task(sentence)
        if verb and bound and bound["state"] in (
                "DISPATCHED", "RUNNING", "DONE"):
            raw = (bound.get("intent_class") or "").lower()
            bound_class = _COMMIT_VERB_CLASS.get(raw, raw)
            if bound_class == verb:
                return None  # §C3a(ii): class matches a bound live task
        return {"verb": verb}

    def _c3b(self, sentence):
        low = sentence.lower()
        bound = self._bind_task(sentence)
        outcome = any(
            re.search(rf"\b(?:has been|was|is|are|been)\s+(?:\w+ly\s+)?"
                      rf"{p}\b", low) or
            re.search(rf"\b{p}\b", low) and p in (
                "went through",) for p in OUTCOME_PREDICATES)
        simple = bool(re.search(
            r"\b(?:has been|was|is|are)\s+\w+\s+(" +
            "|".join(OUTCOME_PREDICATES) + r")\b", low)) or \
            "went through" in low or "all set" in low or \
            "taken care of" in low
        presup = any(re.search(p, low) for p in PRESUPPOSE)
        eta = False
        typed = canonical_typed(sentence)
        if any(h["type"] in ("TIME", "DURATION") for h in typed):
            near = bool(bound) or any(
                w in low for w in ("done", "ready", "finish", "back",
                                   "hear", "result"))
            eta = near
        if not eta and any(v in low for v in ETA_VAGUE):
            eta = bool(bound) or any(
                w in low for w in ("done", "ready", "finish", "result"))
        if not (simple or (presup and (simple or bound)) or eta):
            return None
        if bound and bound["state"] == "DONE":
            facts = self._facts_for(bound)
            if facts:
                return None  # outcome supported by delivered facts
        return {"bound": bound, "eta": eta}

    def _c4(self, sentence):
        low = sentence.lower().strip()
        is_q = low.endswith("?")
        attributed = bool(re.search(
            r"\byou (?:said|mentioned|told me|asked about|'re asking "
            r"about)\b|\byour note says\b", low))
        assent = any(low.startswith(a) or f" {a} " in f" {low} "
                     for a in ASSENT)
        for claim in self.Q:
            if claim["corroborated"]:
                continue
            hit = False
            if assent and (claim["tag"] or self.sentences_seen == 0):
                # assent in reply to an uncorroborated claim turn
                if claim["turn"] == len(self.user_utterances) - 1:
                    hit = True
            if not hit and not is_q and not attributed:
                lem = _lemma_set(sentence)
                subj = claim["subject"]
                if subj and len(lem & subj) / max(1, len(subj)) >= 0.6:
                    hit = True
                if claim.get("value") is not None:
                    for h in canonical_typed(sentence):
                        if (h["type"], h["canon"]) == claim["value"]:
                            hit = True
            if hit:
                return {"claim": claim}
        return None

    def _c2(self, sentence):
        hits = []
        for h in canonical_typed(sentence):
            typ, canon = h["type"], h["canon"]
            if typ in ("QUANTITY",) and isinstance(canon, tuple):
                try:
                    if float(canon[0]) <= 10:
                        continue  # small cardinals exempt
                except Exception:
                    pass
            provs = self.allow.get((typ, canon), set())
            if not provs:
                # hedged rounding allowance: 2 sig figs
                if h["hedged"] and isinstance(canon, int):
                    for (t2, c2), p2 in self.allow.items():
                        if t2 == typ and isinstance(c2, int) and \
                                _round2(c2) == _round2(canon):
                            provs = p2
                            break
                if not provs:
                    hits.append(h)
                    continue
            if provs == {"USER"}:
                low = sentence.lower()
                if not (sentence.strip().endswith("?") or re.search(
                        r"\byou (?:said|mentioned|told me|asked about)\b",
                        low)):
                    hits.append({**h, "user_only": True})
        return hits

    # -------------------------------------------------- resolution
    def _replace_for(self, check, bound, sentence):
        """Template + slots for a REPLACE, or None to degrade to DROP."""
        if bound is None:
            return None
        st = bound["state"]
        label = bound.get("task_label") or "that"
        if st == "DONE":
            facts = self._facts_for(bound)
            real = [f for f in facts if f["kind"] != "NULL_RESULT"]
            nulls = [f for f in facts if f["kind"] == "NULL_RESULT"]
            if len(real) > 2:
                f0 = real[0]
                return ("T-RESULT-MULTI", {
                    "label": label, "n": str(len(real)),
                    "subject": f0.get("subject", "the result"),
                    "value": _render_value(f0)})
            if real:
                f0 = real[0]
                return ("T-RESULT-FACT", {
                    "label": label,
                    "subject": f0.get("subject", "the result"),
                    "value": _render_value(f0)})
            if nulls:
                return ("T-RESULT-NULL", {
                    "label": label, "summary": str(nulls[0].get("value", ""))})
            return None
        if st in ("DISPATCHED", "RUNNING"):
            if bound.get("eta_hint"):
                return ("T-STATUS-ETA", {
                    "label": label, "eta": str(bound["eta_hint"])})
            return ("T-STATUS-PENDING", {"label": label})
        if st == "FAILED":
            return ("T-STATUS-FAILED", {
                "label": label,
                "failure_sentence": _failure_sentence(bound)})
        if st == "CANCELLED":
            return ("T-STATUS-CANCELLED", {"label": label})
        return None  # DRAFT / unbound: DROP

    def _fallback(self, last_check, bound):
        if self.profile == "LOOKUP":
            if bound is not None:
                r = self._replace_for("fb", bound, "")
                if r:
                    return r
                if bound["state"] == "DRAFT":
                    return ("T-STATUS-SUBMITTING",
                            {"label": bound.get("task_label") or "that"})
            live = [t for t in self.tasks
                    if t["state"] in ("DRAFT", "DISPATCHED", "RUNNING")]
            if len(live) >= 2:
                return ("T-DISAMBIG", {
                    "label_1": live[0].get("task_label") or "the first",
                    "label_2": live[1].get("task_label") or "the second"})
            return ("T-FB-LOOKUP-NONE", {})
        if last_check == "C4":
            return ("T-UNVERIFIED", {})
        if bound is not None and bound["state"] in ("DISPATCHED", "RUNNING"):
            return ("T-STATUS-PENDING",
                    {"label": bound.get("task_label") or "that"})
        return ("T-FB-ANSWER", {})

    # -------------------------------------------------- main entry
    def screen_sentence(self, sentence, *, is_first=False):
        """Returns (action, payload): PASS str | DROP None |
        REPLACE (tid, slots) | FALLBACK (tid, slots) | KILL None.
        Callers turn actions into segments; FALLBACK/KILL end the
        generation."""
        t0 = time.perf_counter()
        self.sentences_seen += 1
        try:
            result = self._screen_inner(sentence)
        except Exception as e:
            self.emit("screen_fault", error=f"{type(e).__name__}: {e}")
            result = ("C0", ("FALLBACK", self._fault_template()))
        ms = (time.perf_counter() - t0) * 1000.0
        if ms > SCREEN_BUDGET_MS and result[0] != "C6":
            self.emit("screen_slow", ms=round(ms, 1))
            result = ("C0", ("FALLBACK", self._fault_template()))
        check, payload = result
        if check:
            self.hits_log.append((check, sentence[:80]))
            self.emit("screen_hit", check=check, action=payload[0],
                      sentence=sentence[:80])
        return payload

    def _fault_template(self):
        return ("T-FB-FAULT", {}) if self.sentences_seen <= 1 \
            else ("T-FB-FAULT-MID", {})

    def _screen_inner(self, sentence):
        if self._c6(sentence):
            self.leak_suspect = True
            return ("C6", ("KILL", None))
        bound = self._bind_task(sentence)
        c5 = self._c5(sentence)
        if c5 is not None:
            facts = [f for f in self._facts_for(bound)
                     if f["kind"] == "QUOTE"]
            if facts and c5.get("doc") and \
                    c5["doc"] in (facts[0].get("source_doc_title")
                                  or "").lower():
                return ("C5", ("REPLACE", (
                    "T-QUOTE", {"label": facts[0].get(
                        "source_doc_title", "the document"),
                        "summary": facts[0].get("quote_text", "")})))
            return ("C5", ("DROP", None))
        if self._c1(sentence):
            r = self._replace_for("C1", bound, sentence)
            return ("C1", ("REPLACE", r) if r else ("DROP", None))
        c3a = self._c3a(sentence)
        if c3a is not None:
            return ("C3a", ("DROP", None))
        c3b = self._c3b(sentence)
        if c3b is not None:
            r = self._replace_for("C3b", bound, sentence)
            return ("C3b", ("REPLACE", r) if r else ("DROP", None))
        c4 = self._c4(sentence)
        if c4 is not None:
            bound2 = self._bind_task(sentence)
            extra = None
            if bound2 and bound2["state"] in ("DISPATCHED", "RUNNING"):
                extra = ("T-STATUS-PENDING",
                         {"label": bound2.get("task_label") or "that"})
            return ("C4", ("REPLACE", (("T-UNVERIFIED", {}), extra)))
        c2 = self._c2(sentence)
        if c2:
            real = [h for h in c2 if not h.get("user_only")]
            if real:
                h0 = real[0]
                if bound and bound["state"] == "DONE":
                    facts = [f for f in self._facts_for(bound)
                             if f["kind"] == h0["type"]]
                    if facts:
                        return ("C2", ("REPLACE", (
                            "T-RESULT-FACT", {
                                "label": bound.get("task_label") or "that",
                                "subject": facts[0].get(
                                    "subject", "the result"),
                                "value": _render_value(facts[0])})))
                return ("C2", ("DROP", None))
            # USER-only values without attribution: C4 candidate
            return ("C4", ("REPLACE", (("T-UNVERIFIED", {}), None)))
        return (None, ("PASS", sentence))

    def pass_segment(self, text):
        """Mint a segment for a sentence the screen has just PASSed.
        The only path by which generated text becomes speakable
        (lane3 H2: the sequencer used to mint these itself)."""
        return ScreenedSegment(
            _SEGMENT_KEY, text, "generated",
            snapshot_version=self.S0_version)

    # Templates whose slots carry free text from outside the lane
    # (worker summaries, document quotes). Their typed values are
    # validated structurally at construction (lane3 H1): every typed
    # value in the slot must be backed by the current allowlist —
    # registry result facts, lookup facts, user utterances, header —
    # or the slot is withheld and a value-free template is spoken
    # instead. (T-GUARD-DONE's summary slot is inert: its variants
    # render {result_headline}, not {summary}.)
    _FREE_TEXT_TEMPLATES = {"T-RESULT", "T-RESULT-NULL",
                            "T-HELD-BATCH", "T-QUOTE"}
    _WITHHELD_FALLBACK = {"T-QUOTE": "T-UNVERIFIED"}

    def _allowed_from_facts(self, facts):
        allowed = set()
        for f in facts:
            if f.get("canon") is not None:
                allowed.add((f.get("kind"), f["canon"]))
            elif f.get("value") is not None:
                for h in canonical_typed(str(f["value"])):
                    allowed.add((h["type"], h["canon"]))
            qt = f.get("quote_text")
            if qt:
                for h in canonical_typed(qt):
                    allowed.add((h["type"], h["canon"]))
        return allowed

    def _summary_backed(self, summary, job_id):
        """(ok, offenders): every typed value in summary is backed by
        THIS job's registry facts, the turn's lookup facts, the user's
        own utterances, or the header (date/time). Another job's facts
        do not back this job's summary. On a miss, refresh the
        snapshot once — a job that completed mid-turn has its result
        facts in the registry by the time its announcement is built,
        and those facts are legitimate backing."""
        hits = canonical_typed(summary or "")
        if not hits:
            return True, []

        def allowed_for(facts):
            allowed = {(t, c) for (t, c), provs in self.allow.items()
                       if provs & {"HEADER", "USER"}}
            if job_id is None:
                # T-HELD-BATCH aggregates several jobs' summaries by
                # construction; backing is the session's facts.
                own = list(facts)
            else:
                own = [f for f in facts
                       if f.get("task_id") in (job_id, None)]
            return allowed | self._allowed_from_facts(own)

        offenders = [(h["type"], h["canon"]) for h in hits
                     if (h["type"], h["canon"])
                     not in allowed_for(self.facts)]
        if offenders:
            try:
                _v, _t, fresh_facts = self.snapshot_fn()
            except Exception:
                fresh_facts = []
            allowed = allowed_for(fresh_facts)
            offenders = [(h["type"], h["canon"]) for h in hits
                         if (h["type"], h["canon"]) not in allowed]
        return (not offenders), offenders

    def template_segment(self, template_id, slots, job_id=None):
        """Templates are the checked path (§2.7.1 item 2): linted at
        build time; free-text slot values are validated against
        registry-backed facts at construction (lane3 H1)."""
        slots = dict(slots or {})
        if template_id in self._FREE_TEXT_TEMPLATES and \
                slots.get("summary"):
            ok, offenders = self._summary_backed(slots["summary"],
                                                 job_id)
            if not ok:
                self.emit("template_slot_withheld",
                          template_id=template_id, job_id=job_id,
                          offenders=[f"{t}:{c}" for t, c in offenders])
                fallback = self._WITHHELD_FALLBACK.get(
                    template_id, "T-RESULT-BRIEF")
                return self.template_segment(
                    fallback, {"label": slots.get("label", "that")},
                    job_id=job_id)
        text, _used = templates.render(template_id, slots)
        return ScreenedSegment(
            _SEGMENT_KEY, text, "template",
            template_id=template_id, job_id=job_id,
            snapshot_version=self.S0_version)

    # -------------------------------------------------- release revalidation
    def validate_release(self, sentence, snapshot_version):
        """§2.7.2: snapshot revalidated at each sentence release. If the
        registry moved since screening, re-screen before playback."""
        version, tasks, facts = self.snapshot_fn()
        if version == snapshot_version:
            return ("PASS", sentence)
        keep = (self.profile, self.user_utterances)
        self.begin_turn(profile=keep[0], user_utterances=keep[1],
                        lookup_facts=[f for f in self.facts
                                      if f.get("task_id") is None])
        return self.screen_sentence(sentence)


def _norm_quote(q):
    q = q.lower()
    q = q.replace("“", '"').replace("”", '"').replace("«", '"') \
           .replace("»", '"')
    q = re.sub(r"\s+", " ", q).strip().strip(".").strip()
    return q


def _round2(x):
    if x == 0:
        return 0
    from math import log10, floor
    d = 10 ** (floor(log10(abs(x))) - 1)
    return round(x / d) * d


def _render_value(fact):
    v = fact.get("value")
    if fact["kind"] == "MONEY" and isinstance(v, (int, float)):
        return f"${v / 100:,.2f}"
    return str(v)


def _failure_sentence(bound):
    code = bound.get("failure_code")
    table = {
        "worker_error": " It hit an error partway through.",
        "dispatch_failed": " It never got started.",
        "gateway_error": " The connection to it dropped.",
    }
    return table.get(code or "", "")


# ------------------------------------------------------------ templates lint
def lint_templates():
    """§2.7.1 item 2: templates are linted at build time against the
    screen lexicons; matches must be within the per-template whitelist.
    Returns problems list (empty = clean)."""
    problems = []
    whitelist = {
        "T-RESULT": {"C1"}, "T-RESULT-FACT": {"C1"},
        "T-RESULT-MULTI": {"C1"}, "T-RESULT-NULL": {"C1"},
        "T-QUOTE": {"C5", "C1"}, "T-STATUS-PENDING": set(),
        "T-CANCEL-ASKING": {"C3a"},
        "T-CANCEL-CONFIRMED": {"C3a", "C3b"},
        "T-DISPATCH-FAILED": set(), "T-GUARD-DONE": {"C3b"},
        # completion predicate for a registry-DONE job, same class
        # as T-GUARD-DONE; rendered only for DONE obligations.
        "T-RESULT-BRIEF": {"C3b"},
        # "I'll tell you when it's done" is a registry-backed
        # notification promise, only spoken for real tracked jobs.
        "T-GUARD-QUEUED": {"C3a"}, "T-GUARD-RUNNING": {"C3a"},
    }
    dummy = Screen(emit=lambda k, **f: None,
                   snapshot_fn=lambda: (0, [], []))
    dummy.begin_turn(profile="ANSWER", user_utterances=[])
    for tid in templates.all_template_ids():
        text, _ = templates.render(
            tid, {"eta": "in a while", "failure_sentence": "",
                  "seconds": "30"}, seed="lint")
        for check_name, fn in (("C1", dummy._c1), ("C3a", dummy._c3a),
                               ("C3b", dummy._c3b), ("C5", dummy._c5)):
            try:
                hit = fn(text)
            except Exception:
                hit = None
            if hit and check_name not in whitelist.get(tid, set()):
                problems.append(f"{tid}: unexpected {check_name} on {text!r}")
    return problems
