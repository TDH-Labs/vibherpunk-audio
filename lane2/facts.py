"""FactLedger + structured memo (v2 Part I.1 / §3.3 v2 — Step 4).

User-fact extraction runs at turn-write time, never in the talker:
the union of deterministic detectors (typed tagger + commitment cues +
number-word shorthand) and a GLiNER-class encoder pass over user turns
only. Records are purely extractive: {turn_id, type, span, anchor} with
the anchor a verbatim clause of the user's transcript (<=13 tokens).

Memo M renders in three tiers against a 240-token cap:
  P1 pinned job lines (registry-derived title + outcome, never text),
  P2 fact lines for evicted source turns,
  P3 topic stubs (fill-only, suppressed for units with P1/P2 content
     and for low-content units).
M is re-rendered only at eviction events (SessionLog calls absorb()).
"""
import asyncio
import re
import time

from .tagger import extract_typed_values, parse_number_words

MEMO_CAP = 240
P1_CAP = 96
P1_LINE_CAP = 24
P2_LINE_CAP = 22
P3_LINE_CAP = 12
ANCHOR_CAP = 13

_ENT_STOP = {"i", "me", "we", "you", "it", "they", "he", "she", "the",
             "a", "an", "my", "your", "his", "her", "our", "their"}
_NUM_LABELS = {"quantity", "money", "date", "time"}
_ENT_LABELS = {"person", "organization", "location", "product"}
GLINER_LABELS = sorted(_NUM_LABELS | _ENT_LABELS | {"user commitment"})

_COMMIT_CUES = re.compile(
    r"\b(I'll|I will|I'm going to|I am going to|I need to|I have to|"
    r"I promised|I volunteered|remind me to|don't let me forget to|"
    r"do not let me forget to)\b|"
    r"\bI (?:told|said to)\b[^,.]{0,50}?\bI'?d\b", re.I)
# Plan statements: first-person present-progressive + a date-ish
# token in the same turn ("I'm planting the garlic in October",
# "I'm hosting on the 19th") — stated plans, Exp-6 commitment class.
_PLAN = re.compile(r"\bI'?m\s+(?!going\b)\w+ing\b", re.I)
_DATEISH = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december|tomorrow|tonight|the\s+\d{1,2}"
    r"(?:st|nd|rd|th))\b", re.I)
# Labeled numbers ("budget code 4417"), comma-grouped numbers
# ("82,410"), and dimensions ("30 by 12 feet") — deterministic
# numeric facts the base tagger's unit patterns miss.
_LABELED_NUM = re.compile(
    r"\b(code|room|unit|account|order|invoice|suite|ext|building|lot|"
    r"po|id)\s*(?:#|no\.?|number)?\s*(\d{3,6})\b", re.I)
_COMMA_NUM = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_DIMENSION = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:by|x|×)\s*\d+(?:\.\d+)?\s*"
    r"(?:feet|ft|inches|in|meters?|cm|mm|yards?)?\b", re.I)

_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|and)")
_SHORTHAND = re.compile(
    r"\b((?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?"
    r" (?:ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))\b",
    re.I)

_UNIT_WORDS = {
    "dollar", "dollars", "usd", "buck", "bucks", "percent", "%", "am",
    "pm", "a.m", "p.m", "month", "months", "year", "years", "week",
    "weeks", "day", "days", "hour", "hours", "minute", "minutes",
    "second", "seconds", "items", "files", "emails", "messages",
    "pages", "people", "tickets", "photos", "calls", "meetings",
    "tasks", "jobs", "entries", "records", "results",
}


def _clauses(text):
    """Split into clause spans (start, end) on sentence/clause marks."""
    out, start = [], 0
    for m in re.finditer(r"[.!?;]\s+|\n", text):
        out.append((start, m.start()))
        start = m.end()
    out.append((start, len(text)))
    return [(s, e) for s, e in out if text[s:e].strip()]


def _anchor(text, s, e, cap=ANCHOR_CAP):
    """Verbatim clause containing [s,e), trimmed to <=cap tokens
    centered on the span, at word boundaries."""
    for cs, ce in _clauses(text):
        if cs <= s and e <= ce:
            clause = text[cs:ce]
            off = s - cs
            break
    else:
        clause, off = text, s
    words = clause.split()
    if len(words) <= cap:
        return " ".join(words).strip()
    # word index of the span start
    pos = len(clause[:off].split())
    lo = max(0, pos - cap // 2)
    hi = min(len(words), lo + cap)
    lo = max(0, hi - cap)
    return " ".join(words[lo:hi]).strip()


def _head_noun(text, end):
    """First noun-ish token after a span (skipping unit words): the
    supersession key for num facts ('the deposit is ...')."""
    toks = re.findall(r"[a-z']+", text[end:end + 60].lower())
    for t in toks:
        if t in _UNIT_WORDS or t in ("the", "a", "an", "of", "per"):
            continue
        return t
    return ""


def _truncate_tokens(text, cap, count):
    """Truncate to <=cap tokens at a word boundary (tokenizer count)."""
    words = text.split()
    while words and count(" ".join(words)) > cap:
        words.pop()
    return " ".join(words)


class FactRecord:
    __slots__ = ("turn_id", "type", "span", "anchor", "head", "unit",
                 "last_ref_turn", "evicted", "dropped")

    def __init__(self, turn_id, type_, span, anchor, head="", unit=""):
        self.turn_id = turn_id
        self.type = type_
        self.span = span
        self.anchor = anchor
        self.head = head
        self.unit = unit
        self.last_ref_turn = turn_id
        self.evicted = False
        self.dropped = False

    def key(self):
        return (self.type, self.span.lower(), self.head)


# ------------------------------------------------------------ extractor
class FactExtractor:
    """Deterministic detectors + GLiNER encoder pass (union)."""

    def __init__(self, gliner_model=None, threshold=0.5, emit=None):
        self._model = gliner_model
        self.threshold = threshold
        self.emit = emit or (lambda k, **f: None)
        self.last_latency_ms = None

    def load_encoder(self):
        if self._model is None:
            from gliner import GLiNER
            self._model = GLiNER.from_pretrained(
                "urchade/gliner_medium-v2.1")
        return self._model

    def extract(self, text):
        t0 = time.perf_counter()
        # Precision rule (Step-4 iteration, measured): purely
        # interrogative turns state no user facts — trivia questions
        # were flooding P2 with junk entities/numbers and evicting
        # real facts under the drop order. Questions carrying a
        # commitment cue or a possessive statement still extract.
        stripped = text.strip()
        if stripped.endswith("?") and not _COMMIT_CUES.search(text) \
                and not re.search(
                    r"\bmy\b[^?]{0,60}\b(?:is|was|costs|ends|expires)\b",
                    text, re.I):
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            return []
        recs = self._deterministic(text)
        if self._model is not None:
            recs = self._merge(recs, self._encoder(text))
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return recs

    # -- deterministic floor
    def _deterministic(self, text):
        out = []
        for h in extract_typed_values(text):
            out.append({
                "type": "num", "span": h["surface"],
                "anchor": _anchor(text, h["start"], h["end"]),
                "head": _head_noun(text, h["end"]), "unit": h["type"],
                "start": h["start"], "end": h["end"]})
        for m in _SHORTHAND.finditer(text):
            surface = m.group(1)
            words = surface.lower().replace("-", " ").split()
            if len(words) == 2:
                hi = parse_number_words(words[0])
                lo = parse_number_words(words[1])
                if hi and lo and 10 <= hi <= 99:
                    out.append({
                        "type": "num", "span": surface,
                        "anchor": _anchor(text, m.start(), m.end()),
                        "head": _head_noun(text, m.end()),
                        "unit": "SHORTHAND",
                        "start": m.start(), "end": m.end(),
                        "value": hi * 100 + lo})
        covered = [(h["start"], h["end"]) for h in
                   extract_typed_values(text)]

        def _free(s, e):
            return all(e <= a or s >= b for a, b in covered)

        for pat, head in ((_LABELED_NUM, None), (_COMMA_NUM, None),
                          (_DIMENSION, None)):
            for m in pat.finditer(text):
                s, e = (m.span(2) if pat is _LABELED_NUM else m.span())
                if pat is _DIMENSION:
                    # a dimension SUBSUMES a contained unit hit
                    # ("12 feet" inside "30 by 12 feet")
                    if not all(e2 <= s or s2 >= e or
                               (s <= s2 and e2 <= e)
                               for s2, e2 in covered):
                        continue
                elif not _free(s, e):
                    continue
                covered.append((s, e))
                out.append({
                    "type": "num", "span": text[s:e],
                    "anchor": _anchor(text, s, e),
                    "head": (m.group(1).lower()
                             if pat is _LABELED_NUM else ""),
                    "unit": "NUMBER", "start": s, "end": e})
        pm = _PLAN.search(text)
        if pm and _DATEISH.search(text):
            clause = re.split(r"[.!?;]", text[pm.start():])[0]
            out.append({
                "type": "commit", "span": clause.strip(),
                "anchor": _anchor(text, pm.start(),
                                  pm.start() + len(clause)),
                "head": "", "unit": "PLAN",
                "start": pm.start(), "end": pm.start() + len(clause)})
        for m in _COMMIT_CUES.finditer(text):
            end = m.end()
            # span runs to the clause end (<=12 words) so the
            # commitment's object/day is inside the span, not cut off
            clause = re.split(r"[.!?;,]", text[end:end + 120])[0]
            tail = clause.split()
            span = (m.group(0) + " " + " ".join(tail[:12])).strip()
            out.append({
                "type": "commit", "span": span,
                "anchor": _anchor(text, m.start(),
                                  min(len(text), end + len(span))),
                "head": "", "unit": "CUE",
                "start": m.start(), "end": end + len(span)})
        return out

    # -- encoder pass
    def _encoder(self, text):
        try:
            ents = self._model.predict_entities(
                text, GLINER_LABELS, threshold=self.threshold)
        except Exception as e:
            self.emit("fact_encoder_error", error=str(e))
            return []
        out = []
        for e in ents:
            lab = e["label"]
            if lab in _NUM_LABELS:
                typ = "num"
            elif lab in _ENT_LABELS:
                typ = "ent"
            elif lab == "user commitment":
                typ = "commit"
            else:
                continue
            # Shape validation (Step-4 iteration, measured): the
            # encoder emits junk spans at useful-looking scores —
            # "I" as a person (0.81), "flights" as a product (0.75),
            # bare words as quantities. A num span must itself parse
            # as a typed value; an ent span must be capitalized
            # somewhere and not be a pronoun/stopword.
            span = e["text"].strip()
            if typ == "num" and not extract_typed_values(span) \
                    and not parse_number_words(span.lower()) \
                    and not re.fullmatch(r"\$?\d[\d,]*(?:\.\d+)?", span):
                continue
            if typ == "ent":
                if span.lower() in _ENT_STOP or not any(
                        w[:1].isupper() for w in span.split()):
                    continue
            out.append({
                "type": typ, "span": e["text"],
                "anchor": _anchor(text, e["start"], e["end"]),
                "head": _head_noun(text, e["end"]) if typ == "num" else "",
                "unit": lab.upper(), "start": e["start"], "end": e["end"],
                "score": e.get("score")})
        return out

    @staticmethod
    def _merge(det, enc):
        out = list(det)
        for r in enc:
            dup = False
            for d in out:
                if d["type"] != r["type"]:
                    continue
                if d["span"].lower() == r["span"].lower():
                    dup = True
                    break
                # overlapping spans of the same class: keep both only
                # if neither contains the other
                if not (r["end"] <= d["start"] or r["start"] >= d["end"]):
                    if d["start"] <= r["start"] and r["end"] <= d["end"]:
                        dup = True
                        break
            if not dup:
                out.append(r)
        out.sort(key=lambda r: r["start"])
        return out


# ------------------------------------------------------------ ledger
class FactLedger:
    """Session-scoped, in-memory. Appended by the extraction pass over
    each committed user turn; assistant turns are never extracted."""

    def __init__(self, extractor, emit=None):
        self.extractor = extractor
        self.emit = emit or (lambda k, **f: None)
        self.records = []
        self._pending = {}  # turn_id -> asyncio.Future
        self.counters = {"fact_suspect_miss": {}, "fact_extract_timeout": 0}

    # -- sync path (probe / tests)
    def extract_turn(self, turn_id, text):
        raw = self.extractor.extract(text)
        return self._absorb(turn_id, text, raw)

    # -- async path (engine): encoder pass off the turn path
    async def extract_turn_async(self, turn_id, text, loop=None):
        loop = loop or asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[turn_id] = fut
        try:
            raw = await loop.run_in_executor(
                None, self.extractor.extract, text)
            recs = self._absorb(turn_id, text, raw)
            if not fut.done():
                fut.set_result(recs)
            return recs
        finally:
            self._pending.pop(turn_id, None)

    def _absorb(self, turn_id, text, raw):
        made = []
        for r in raw:
            rec = FactRecord(turn_id, r["type"], r["span"], r["anchor"],
                             head=r.get("head", ""), unit=r.get("unit", ""))
            self._supersede_or_refresh(rec)
            self.records.append(rec)
            made.append(rec)
        self._suspect(text, made)
        return made

    def _supersede_or_refresh(self, new):
        for old in list(self.records):
            if old.dropped or old.type != new.type:
                continue
            if old.turn_id == new.turn_id:
                continue
            if new.type == "num":
                if old.head and old.head == new.head and \
                        old.unit == new.unit:
                    old.dropped = True  # superseded by the newer record
                    self.emit("fact_superseded", turn=old.turn_id)
                    return
            elif new.type == "ent":
                if old.span.lower() == new.span.lower():
                    old.dropped = True
                    self.emit("fact_superseded", turn=old.turn_id)
                    return
            # re-mention refreshes the P2 eviction rank
            if old.span.lower() == new.span.lower():
                old.last_ref_turn = new.turn_id

    def _suspect(self, text, made):
        """fact_suspect_miss: an evicted-bound turn with candidate
        signals but no record of that type (surfaced, never shown to
        the talker)."""
        have = {r.type for r in made}
        signals = set()
        if re.search(r"\d", text) or parse_number_words(
                " ".join(w for w in re.findall(r"[a-z]+", text.lower())
                         if w in ("one two three four five six seven eight "
                                  "nine ten eleven twelve twenty thirty "
                                  "forty fifty sixty seventy eighty ninety "
                                  "hundred thousand").split())) is not None:
            signals.add("num")
        if re.search(r"(?<!^)(?<!\.\s)\b[A-Z][a-z]{2,}\b", text):
            signals.add("ent")
        if re.search(r"\bI(?:'ll| will| am going to|'m going to)\b", text):
            signals.add("commit")
        for s in signals - have:
            c = self.counters["fact_suspect_miss"]
            c[s] = c.get(s, 0) + 1

    async def wait_turn(self, turn_id, timeout=0.2):
        """Eviction barrier: True if the turn's extraction is settled."""
        fut = self._pending.get(turn_id)
        if fut is None:
            return True
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout)
            return True
        except asyncio.TimeoutError:
            self.counters["fact_extract_timeout"] += 1
            return False

    def records_for_turn(self, turn_id):
        return [r for r in self.records
                if r.turn_id == turn_id and not r.dropped]

    def mark_evicted(self, turn_id):
        out = []
        for r in self.records:
            if r.turn_id == turn_id and not r.dropped:
                r.evicted = True
                out.append(r)
        return out


# ------------------------------------------------------------ memo
def outcome_line(row):
    """Deterministic outcome text from registry fields only (I.1.2):
    result_summary, else the typed primary value; <=16 tokens."""
    st = row["status"]
    if st == "DONE":
        s = (row.get("result_summary") or "").strip()
        if not s:
            facts = row.get("result_facts")
            if facts:
                try:
                    import json
                    fl = json.loads(facts) if isinstance(facts, str) \
                        else facts
                    if fl:
                        s = str(fl[0].get("surface", ""))
                except Exception:
                    s = ""
        words = s.split()
        return " ".join(words[:16])
    if st == "FAILED":
        return ("failed: " + (row.get("failure_code") or "error"))[:60]
    if st == "CANCELLED":
        return "cancelled"
    if st == "DISPATCH_FAILED":
        return "failed: dispatch_failed"
    return st.lower()


class MemoState:
    """Three-tier memo. SessionLog calls absorb(unit, tok) per evicted
    unit and render() when the memo span is rebuilt."""

    def __init__(self, registry, ledger, session_id, emit=None):
        self.reg = registry
        self.ledger = ledger
        self.session_id = session_id
        self.emit = emit or (lambda k, **f: None)
        self._p1 = {}       # job_id -> (terminal_mono, line)
        self._p1_dropped = set()
        self._p3 = []       # (turn_id, stub)
        self.counters = {"memo_job_spill": 0, "memo_fact_drop": {}}
        self.stub_only_turns = set()  # extraction timed out at eviction

    # ------------------------------------------------------------- absorb
    def absorb(self, unit, tok):
        # P1 lines are rendered from the registry at render time
        # (I.1.2: one line per job in the session, terminal or not) —
        # absorb only records that this unit is a job's origin turn,
        # which suppresses its P3 stub (its content is represented
        # by the job's P1 line).
        _, rows = self.reg.snapshot()
        for r in rows:
            if r["session_id"] == self.session_id and \
                    r["origin_turn"] == unit.turn_id:
                unit.produced_p1 = True
                break
        # P2: facts from this unit's user turn (unless barrier timed out)
        if unit.turn_id not in self.stub_only_turns:
            recs = self.ledger.mark_evicted(unit.turn_id)
            if recs:
                unit.produced_p2 = True
        # P3: stub, fill-only and suppressed
        if not unit.produced_p1 and not unit.produced_p2 \
                and not unit.low_content:
            stub = unit.stub_text or _derive_stub(unit, tok)
            if stub:
                self._p3.append((unit.turn_id, stub))

    # ------------------------------------------------------------- render
    def bind_tokenizer(self, tok):
        self._tok = tok

    def render(self):
        return self.render_with(self._tok)

    def render_with(self, tok):
        # P1: registry-derived at render (I.1.2) — one line per job
        # in the session, creation order, first-fit under the tier
        # cap; jobs that do not fit count one spill each.
        _, rows = self.reg.snapshot()
        jobs = [r for r in rows if r["session_id"] == self.session_id]
        jobs.sort(key=lambda r: r["origin_seq"])
        p1_lines, p1_used = [], 0
        for r in jobs:
            line = _p1_line(r, tok)
            c = tok.count(line)
            if p1_used + c > P1_CAP:
                if r["job_id"] not in self._p1_dropped:
                    self._p1_dropped.add(r["job_id"])
                    self.counters["memo_job_spill"] += 1
                    self.emit("memo_job_spill", job_id=r["job_id"])
                continue
            p1_lines.append(line)
            p1_used += c
        budget = MEMO_CAP - p1_used
        # P2: evicted, not dropped; drop order = oldest non-commit by
        # last-reference turn first, commit lines last
        p2_recs = [r for r in self.ledger.records
                   if r.evicted and not r.dropped]
        p2_lines, p2_used = [], 0
        # choose survivors from the newest end of the drop order
        order = sorted(
            p2_recs,
            key=lambda r: (r.type == "commit", r.last_ref_turn))
        keep = []
        for r in reversed(order):
            line = _p2_line(r, tok)
            c = tok.count(line)
            if p2_used + c > budget:
                r.dropped = True
                d = self.counters["memo_fact_drop"]
                d[r.type] = d.get(r.type, 0) + 1
                self.emit("memo_fact_drop", type=r.type, turn=r.turn_id)
                continue
            keep.append((line, r))
            p2_used += c
        p2_lines = [ln for ln, _ in reversed(keep)]
        p2_recs_kept = [r for _, r in reversed(keep)]
        budget -= p2_used
        # P3: fill-only, oldest-first drop
        p3_lines, p3_used = [], 0
        stubs = list(self._p3)
        kept = []
        for turn_id, stub in reversed(stubs):
            line = f"[t{turn_id}] {stub}"
            line = _truncate_tokens(line, P3_LINE_CAP, tok.count)
            c = tok.count(line)
            if p3_used + c > budget:
                continue
            kept.append(line)
            p3_used += c
        p3_lines = list(reversed(kept))
        # Hard cap: per-line counts are not additive under a real
        # tokenizer, so enforce MEMO_CAP on the joined text — drop
        # P3 lines first, then the oldest non-commit P2 lines.
        lines = p1_lines + p2_lines + p3_lines
        text = "\n".join(lines)
        while lines and tok.count(text) > MEMO_CAP:
            if p3_lines:
                lines.remove(p3_lines.pop())
            elif p2_lines:
                victim = None
                for i, r in enumerate(p2_recs_kept):
                    if r.type != "commit":
                        victim = i
                        break
                if victim is None:
                    victim = 0
                lines.remove(p2_lines.pop(victim))
                p2_recs_kept.pop(victim)
            else:
                lines.pop()
            text = "\n".join(lines)
        self._last_total = tok.count(text) if text else 0
        return text

    @property
    def last_total(self):
        return getattr(self, "_last_total", 0)


def _p1_line(row, tok):
    """[J{id} {STATE}] {title} — {outcome}; <= P1_LINE_CAP tokens.
    Truncation order: outcome words AFTER its last typed value first,
    then title words — the typed value (the recall-bearing token)
    and as much title as fits both survive."""
    oc = outcome_line(row)
    # Session-local job number (origin_seq): real ULID suffixes
    # tokenize 2+ tokens worse than the reference renderer's
    # placeholder, which alone pushed exactly-at-cap lines into
    # title truncation (measured, dense corpus). The memo is
    # session-scoped, so the sequence number is unambiguous here;
    # the registry holds the full id.
    head = f"[J{row.get('origin_seq', 0)} {row['status']}] "
    title = " ".join((row.get("title") or "job").split())
    if tok.count(f"{head}{title} — {oc}") <= P1_LINE_CAP:
        return f"{head}{title} — {oc}"
    oc_words = oc.split()
    prot = len(oc_words)
    surfaces = [h["surface"] for h in extract_typed_values(oc)]
    for s in surfaces:
        sw = s.split()
        for i in range(len(oc_words) - len(sw) + 1):
            if oc_words[i:i + len(sw)] == sw:
                prot = min(prot, i + len(sw))
    title_words = title.split()
    while tok.count(f"{head}{' '.join(title_words)} — "
                    f"{' '.join(oc_words)}") > P1_LINE_CAP:
        if len(oc_words) > prot:
            oc_words.pop()
        elif title_words:
            title_words.pop()
        elif oc_words:
            oc_words.pop()
        else:
            break
    return f"{head}{' '.join(title_words)} — {' '.join(oc_words)}"


def _p2_line(r, tok):
    """[{type} t{turn}] {anchor} => {span}; the value is ALWAYS the
    explicit tail (never left inside the anchor, where truncation
    could delete it): the anchor is rendered with the span's own
    occurrence removed, then absorbs truncation alone."""
    head = f"[{r.type} t{r.turn_id}] "
    anchor = " ".join((r.anchor or "").split())
    span = (r.span or "").strip()
    if span:
        i = anchor.lower().find(span.lower())
        if i >= 0:
            anchor = " ".join(
                (anchor[:i] + " " + anchor[i + len(span):]).split())
    tail = f" => {span}" if span else ""
    line = head + anchor + tail
    if tok.count(line) <= P2_LINE_CAP:
        return line
    words = anchor.split()
    while len(words) > 4 and \
            tok.count(head + " ".join(words) + tail) > P2_LINE_CAP:
        words.pop(len(words) // 2)
    return head + " ".join(words) + tail


def _derive_stub(unit, tok):
    """Topic stub from the unit's user text with typed values removed
    (values never enter stubs), <=12 tokens."""
    try:
        data = unit.span.data.decode("utf-8", "replace")
    except Exception:
        return None
    m = re.search(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>)", data, re.S)
    if not m:
        return None
    text = m.group(1)
    for h in extract_typed_values(text):
        text = text.replace(h["surface"], " ")
    words = [w for w in re.findall(r"[A-Za-z']+", text)
             if len(w) > 2][:P3_LINE_CAP - 2]
    if not words:
        return None
    return " ".join(words)
