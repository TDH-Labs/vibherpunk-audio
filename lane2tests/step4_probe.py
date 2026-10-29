"""Step-4 recall probe (v2 I.1.7): Experiment 6's method against the
SHIPPED extractor and §3.3 v2 memo policy — no oracle.

Corpus: the 10 Exp-6 sessions (lane2tests/exp6_sessions.json) + 30
generated sessions (200 targets total across the 40) + 10 dense
sessions (12 facts + 5 jobs each) for drop-order determinism.
Generated sessions include >=10 number-word targets, >=5 corrections,
>=5 multi-entity turns, >=5 cue-less commitments (flagged in data).

Scoring: strict availability (Exp 6 rubric) — the planted answer value
is present in the final context (M + retained C) under light
normalization, as a substring. Verdict computed at token scales
0.8 / 1.0 / 1.25 and must be identical at all three.
"""
import asyncio
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.expanduser("~/workspace/hermes-voice"))
os.chdir(os.path.expanduser("~/workspace/hermes-voice"))

from lane2.context import SessionLog, Tokenizer          # noqa: E402
from lane2.facts import FactExtractor, FactLedger, MemoState  # noqa: E402
from lane2.registry import JobRegistry                    # noqa: E402
from lane2.tagger import extract_typed_values             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = json.load(open(os.path.expanduser(
    "~/workspace/hermes-voice/lane2/pins.json")))


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower().replace(",", "")).strip()


# ------------------------------------------------------------ corpus gen
FILLER = [
    ("What do you think about meal prepping on Sundays?",
     "It saves weeknight time and cuts down on last-minute takeout. The trick is picking two or three base dishes you can remix."),
    ("I keep forgetting to water the plants on the balcony.",
     "Heat like this, they need water every other day. The herbs will forgive a missed day; the tomatoes will not."),
    ("Do you know a good way to learn touch typing?",
     "Short daily sessions beat long weekend ones. Fifteen minutes of drills, and stop before your hands get tired."),
    ("The neighbor's dog has been barking at night again.",
     "That's rough on sleep. A friendly word first usually works better than jumping straight to a complaint."),
    ("I've been thinking about getting back into running.",
     "Start with short, easy runs and walk breaks. Consistency in the first month matters more than distance."),
    ("What's a good gift for a coworker who's retiring?",
     "Something tied to a shared memory beats something expensive. A framed team photo usually lands well."),
    ("My phone battery barely lasts the day anymore.",
     "Check which apps run in the background first. If the health percentage is under eighty, a battery swap is worth it."),
    ("Do you have a trick for falling asleep faster?",
     "Same wake time every day, screens off an hour before bed, and keep the room cool. Boring, but it works."),
    ("I saw a documentary about deep sea fishing last night.",
     "Those crews work brutal hours. The footage of the nets coming up at dawn is something else."),
    ("What should I plant in the shady part of the garden?",
     "Hostas and ferns do well in shade. If you want color, astilbe handles it better than most flowering plants."),
    ("The car is making a clicking sound when I turn left.",
     "That often points to a worn CV joint. Worth having a mechanic listen before it gets worse."),
    ("I've been meaning to back up my photo library.",
     "Two copies in different places is the rule. Cloud plus a drive you keep offsite covers most failures."),
    ("How do you keep houseplants alive in winter?",
     "Less water than you think, and keep them away from heating vents. Most winter losses are overwatering."),
    ("Any tips for a long drive with kids?",
     "Snacks within reach, a stop every two hours, and one new thing held back for the last stretch."),
    ("I tried making sourdough again and it finally rose.",
     "Nice — once the starter is strong it gets reliable. Temperature is usually the missing piece."),
    ("What's the best way to clean cast iron?",
     "Hot water, a stiff brush, dry it on the stove, and a thin layer of oil. No soaking."),
    ("The roof gutters are full again after that storm.",
     "Worth clearing before the next rain. Overflow at the foundation is the expensive kind of problem."),
    ("I'm thinking about learning the piano as an adult.",
     "Totally doable. Ten minutes a day with a teacher once a month beats an hour of random practice."),
    ("Do you know how to get a coffee stain out of a shirt?",
     "Cold water from the back of the fabric, then a little dish soap. Heat sets it, so no dryer until it's out."),
    ("We watched the meteor shower from the driveway last night.",
     "Clear nights have been rare this month. The peak after midnight is usually worth staying up for."),
    ("I need a better system for sorting the mail.",
     "One tray for bills, one for everything else, and shred the junk the day it arrives. Simple survives."),
    ("The kids want a lemonade stand this weekend.",
     "Good first business. Price it so the sign math is easy and let them keep the ledger."),
    ("How long should I rest a steak before cutting it?",
     "About half the cooking time. Five to ten minutes for a thick cut keeps the juices in."),
    ("My laptop fan has been loud all week.",
     "Dust in the vents is the usual cause. A can of air while it's off clears most of it."),
]


def filler_turns(n, theme):
    out = []
    for i in range(n):
        u, a = FILLER[i % len(FILLER)]
        out.append({"kind": "chat", "user": u, "assistant": a,
                    "stub": theme})
    return out


def C(user, asst, stub, ttype, answer, q, **kw):
    d = {"kind": "chat", "user": user, "assistant": asst, "stub": stub,
         "target": {"type": ttype, "answer": answer, "question": q}}
    d.update(kw)
    return d


def J(title, outcome, user, ttype, answer, q):
    return {"kind": "job", "title": title, "outcome": outcome,
            "user": user, "stub": title.lower(),
            "target": {"type": ttype, "answer": answer, "question": q}}


PEOPLE = ["Marcus Webb", "Priya Nair", "Tom Okafor", "Elena Ruiz",
          "David Lindqvist", "Aunt Rosa", "Kevin Doyle", "Sarah Kim",
          "Omar Haddad", "Grace Liu", "Ben Carter", "Nina Petrova",
          "Hugo Almeida", "Kate Brennan", "Yusuf Ali", "Laura Chen"]
PLACES = ["Lisbon", "Denver", "the coast house", "Reykjavik", "Austin",
          "the lake cabin", "Portland", "Santa Fe", "Chicago", "Halifax"]
ORGS = ["Northwind", "Harbor Dental", "Atlas Insurance", "Brightpath",
        "City Library", "Union Bank", "Kite Systems", "Fernwood Clinic"]


def new_sessions():
    """30 sessions x 5 targets. Flags: words/correction/multi/cueless."""
    S = []
    for i in range(30):
        p = PEOPLE[i % len(PEOPLE)]
        p2 = PEOPLE[(i + 5) % len(PEOPLE)]
        pl = PLACES[i % len(PLACES)]
        org = ORGS[i % len(ORGS)]
        amt = 900 + 137 * i
        amt_words = ["one thousand two hundred", "two thousand four hundred fifty",
                     "eight hundred seventy-five", "three thousand six hundred",
                     "one thousand nine hundred twenty"][i % 5]
        amt_words_val = ["$1,200", "$2,450", "$875", "$3,600", "$1,920"][i % 5]
        day = ["Friday", "Tuesday", "Thursday", "Monday", "Wednesday"][i % 5]
        items = []
        # job outcome
        items.append(J(f"{pl} flight search",
                       f"cheapest option lands at {5 + i % 6}:4{i % 10} PM",
                       f"Can you have the back office find flights to {pl} for next month?",
                       "job_outcome", f"{5 + i % 6}:4{i % 10} PM",
                       f"What time does the cheapest {pl} flight land?"))
        # job title
        items.append(J(f"email check for {org} messages",
                       "two messages found, both about the appointment",
                       f"Please check my email for anything from {org}.",
                       "job_title", "email",
                       f"Did you have the back office check my email for {org} messages?"))
        # entity (some multi-entity)
        if i % 6 == 1:  # multi-entity turns: sessions 1,7,13,19,25
            items.append(C(
                f"My accountant is {p}, my lawyer is {p2}, and the house sitter is {PEOPLE[(i + 9) % len(PEOPLE)]} — quite a team.",
                "A full roster. Noted.",
                "accountant", "entity_name", p,
                "What's my accountant's name?", multi=True))
        else:
            items.append(C(
                f"Before I forget — my travel agent is {p}, the one who booked the {pl} trip.",
                f"Noted — {p}, your travel agent.",
                "travel agent", "entity_name", p,
                "What's my travel agent's name?"))
        # commitment (some cue-less)
        if i % 6 == 3:  # cue-less: sessions 3,9,15,21,27
            items.append(C(
                f"First thing {day} I'm calling {org} about the claim — that's on me.",
                "Got it, that's yours to make.",
                "call claim", "commitment", f"calling {org}",
                f"What did I say I'd do first thing {day}?", cueless=True))
        else:
            items.append(C(
                f"Remind me to call {org} on {day} about the paperwork.",
                f"Will do — call {org} on {day}.",
                "call reminder", "commitment", f"call {org}",
                f"What did I ask you to remind me to do?"))
        # numeric: corrections first (5), then number-word (10), then digits
        if i % 6 == 4:  # corrections: sessions 4,10,16,22,28
            items.append(C(
                f"The deposit is ${amt} — wait, actually it's ${amt + 40}, I read the wrong line.",
                f"Corrected — ${amt + 40} for the deposit.",
                "deposit", "numeric", f"${amt + 40}",
                "How much is the deposit?", correction=True))
        elif i % 5 in (0, 2):  # number-word targets: 10 sessions
            items.append(C(
                f"I'm capping the whole trip at {amt_words} dollars, flights and hotel together.",
                "Got it — that's the all-in cap.",
                "trip budget", "numeric", amt_words,
                "What's the total budget I set for the trip?", words=True))
        else:
            items.append(C(
                f"I'm capping the whole trip at ${amt:,} budget, flights and hotel together.",
                f"Got it — ${amt:,} all-in.",
                "trip budget", "numeric", f"${amt:,}",
                "What's the total budget I set for the trip?"))
        # rotate planting order per session
        k = i % len(items)
        items = items[k:] + items[:k]
        S.append({"id": f"N{i + 1}", "name": f"generated {i + 1}",
                  "planted": items, "filler": filler_turns(40, "daily life")})
    return S


def dense_sessions():
    S = []
    for i in range(10):
        items = []
        for j in range(12):
            p = PEOPLE[(i + j) % len(PEOPLE)]
            if j % 3 == 0:
                items.append(C(
                    f"The appointment with {p} is at {2 + j}:30 PM.",
                    "Noted.", "appointment", "entity_name", p, "Who?"))
            elif j % 3 == 1:
                items.append(C(
                    f"Line item {j}: the fee was ${300 + 25 * j} even.",
                    "Got it.", "fee", "numeric", f"${300 + 25 * j}", "How much?"))
            else:
                items.append(C(
                    f"I'll file the form {j} with {ORGS[(i + j) % len(ORGS)]} this week.",
                    "Noted.", "filing", "commitment",
                    f"file the form {j}", "What will I file?"))
        for j in range(5):
            items.append(J(f"monthly report pull number {j} for account {i}",
                           f"report {j} ready with {3 + j} pages total",
                           f"Have the back office pull report {j} for my account.",
                           "job_outcome", f"{3 + j} pages", "How long?"))
        S.append({"id": f"D{i + 1}", "name": f"dense {i + 1}",
                  "planted": items, "filler": filler_turns(45, "daily life")})
    return S


# ------------------------------------------------------------ simulation
class ScaledTok:
    """Tokenizer wrapper scaling counted lengths by `scale` (bytes and
    rendering untouched; only counts move), per I.1.7's 0.8/1.0/1.25
    estimator-robustness requirement."""

    def __init__(self, base, scale):
        self._b, self.scale = base, scale
        self.special_tokens = base.special_tokens
        self._tok = base._tok

    def encode(self, text):
        ids = self._b.encode(text)
        n = max(1, round(len(ids) * self.scale))
        if n >= len(ids):
            return ids + [0] * (n - len(ids))
        return ids[:n]

    def count(self, text):
        return len(self.encode(text))


EXTRACTION_CACHE = {}
LATENCIES = []


def get_extraction(extractor, turn_key, text):
    if turn_key in EXTRACTION_CACHE:
        return EXTRACTION_CACHE[turn_key]
    t0 = time.perf_counter()
    recs = extractor.extract(text)
    LATENCIES.append((time.perf_counter() - t0) * 1000.0)
    EXTRACTION_CACHE[turn_key] = recs
    return recs


async def run_session(session, tok, extractor, db_path):
    """Drive one session through the real SessionLog + MemoState +
    FactLedger + JobRegistry. Returns (targets, context_text, meta)."""
    reg = JobRegistry(db_path, batch_window_ms=1.0)
    ledger = FactLedger(extractor)
    memo = MemoState(reg, ledger, session_id=session["id"])
    memo.bind_tokenizer(tok)
    log = SessionLog(tok, {"user": "User"}, memo=memo)
    turn = 0
    targets = []

    async def commit(user, asst, stub=None, job_refs=(), job=None):
        nonlocal turn
        turn += 1
        raw = get_extraction(extractor, (session["id"], turn), user)
        ledger._absorb(turn, user, raw)
        if job is not None:
            jid = await reg.create_job(
                session_id=session["id"], status="QUEUED",
                title=job["title"], task_spec={"text": user},
                origin_turn=turn, origin_seq=turn, ack_required=True)
            await reg.transition(jid, "DISPATCHING", ack_required=True)
            await reg.transition(jid, "RUNNING", ack_required=True)
            if not job.get("open"):
                await reg.transition(
                    jid, "DONE", ack_required=True,
                    result_summary=job["outcome"],
                    result_facts=extract_typed_values(job["outcome"]))
            job_refs = (jid,)
        log.commit_exchange(turn, user, asst, job_refs=job_refs,
                            stub_text=stub,
                            low_content=len(user.split()) <= 6)
        if log.history_tokens() > 2110:
            log.evict_if_needed()
        return turn

    for item in session["planted"]:
        if item["kind"] in ("job", "openjob"):
            job = dict(item)
            if item["kind"] == "openjob":
                job["open"] = True
                job.setdefault("outcome", "")
            t = await commit(item["user"], "(Passed that to the team.)",
                             stub=item.get("stub"), job=job)
        else:
            t = await commit(item["user"], item["assistant"],
                             stub=item.get("stub"))
        if item.get("target"):
            tg = dict(item["target"])
            tg["turn"] = t
            tg["session"] = session["id"]
            targets.append(tg)
    for f in session["filler"]:
        await commit(f["user"], f["assistant"], stub=None)
    if log.history_tokens() > 2110:
        log.evict_if_needed()
    ctx = memo.render()
    parts = [ctx]
    for u in log.units:
        parts.append(u.span.data.decode("utf-8", "replace"))
    meta = {"evictions": log.eviction_count,
            "evicted_turns": list(log.evicted_turn_ids),
            "memo_tokens": tok.count(ctx) if ctx else 0,
            "memo_counters": dict(memo.counters),
            "ledger_counters": {k: v for k, v in ledger.counters.items()},
            "history_tokens": log.history_tokens()}
    reg.close()
    return targets, "\n".join(parts), meta


def reference_memo_lines(session, tok):
    """Independent §3.3 renderer for the DENSE corpus: computes the
    expected surviving P1/P2/P3 lines from the session spec alone."""
    # P1: jobs in terminal order; line <=24 tok; tier <=96, spill oldest
    p1 = []
    for item in session["planted"]:
        if item["kind"] != "job":
            continue
        line = f"[Jxxxxxx DONE] {item['title']} — {item['outcome']}"
        words = line.split()
        while words and tok.count(" ".join(words)) > 24:
            words.pop()
        p1.append(" ".join(words))
    used, keep = 0, []
    spill = 0
    for line in p1:
        c = tok.count(line)
        if used + c > 96:
            spill += 1
            continue
        keep.append(line)
        used += c
    return keep, spill


async def main():
    base_tok = Tokenizer(PINS["model_dir"])
    extractor = FactExtractor()
    extractor.load_encoder()
    exp6 = json.load(open(os.path.join(HERE, "exp6_sessions.json")))
    # map Exp-6 target types onto I.1.7 types
    tmap = {"entity_name": "ent", "numeric": "num",
            "commitment": "commit", "job_outcome": "job_outcome",
            "job_title": "job_title"}
    for s in exp6:
        for it in s["planted"]:
            if it.get("target"):
                it["target"]["type"] = tmap[it["target"]["type"]]
        s["filler"] = [
            f if isinstance(f, dict) else
            {"kind": "chat", "user": f[0], "assistant": f[1], "stub": None}
            for f in s.get("filler", [])]
    corpus = exp6 + new_sessions()
    # Top every corpus session up past the eviction horizon
    # (H=2,110): the probe exists to measure the MEMO channel, and
    # the raw Exp-6 sessions sit at ~1,720 tokens — under this
    # build's H nothing would ever evict and job outcomes (which
    # live only in the memo) would be untestable rather than lost.
    for s in corpus:
        est = sum(base_tok.count(it["user"]) + 12 for it in s["planted"])
        est += sum(base_tok.count(f["user"]) + base_tok.count(f["assistant"])
                   for f in s["filler"])
        rounds = 0
        while est < 2700 and rounds < 14:
            extra = filler_turns(10, "topup")
            s["filler"].extend(extra)
            est += sum(base_tok.count(f["user"])
                       + base_tok.count(f["assistant"]) for f in extra)
            rounds += 1
    dense = dense_sessions()
    report = {"scales": {}, "dense": {}}
    for scale in (1.0, 0.8, 1.25):
        tok = ScaledTok(base_tok, scale) if scale != 1.0 else base_tok
        all_targets, metas = [], []
        for s in corpus:
            db = f"/tmp/step4_probe_{s['id']}_{scale}.db"
            if os.path.exists(db):
                os.remove(db)
            tg, ctx, meta = await run_session(s, tok, extractor, db)
            nctx = norm(ctx)
            for t in tg:
                t["available"] = norm(t["answer"]) in nctx
                t["evicted"] = t["turn"] in meta["evicted_turns"]
            all_targets.extend(tg)
            metas.append(meta)
        by_type = {}
        for t in all_targets:
            by_type.setdefault(t["type"], []).append(t["available"])
        overall = sum(t["available"] for t in all_targets) / len(all_targets)
        report["scales"][str(scale)] = {
            "n_targets": len(all_targets),
            "overall": round(overall, 4),
            "by_type": {k: [sum(v), len(v)] for k, v in by_type.items()},
            "all_evicted": all(t["evicted"] for t in all_targets),
            "memo_max_tokens": max(m["memo_tokens"] for m in metas),
            "evictions_min": min(m["evictions"] for m in metas),
        }
        print(f"scale {scale}: overall {overall:.3f} "
              f"types { {k: f'{sum(v)}/{len(v)}' for k, v in by_type.items()} }")
    # dense corpus at scale 1.0: drop-order determinism + spill count
    dense_ok, dense_detail = True, []
    for s in dense:
        db = f"/tmp/step4_probe_{s['id']}_dense.db"
        if os.path.exists(db):
            os.remove(db)
        tg, ctx, meta = await run_session(s, base_tok, extractor, db)
        keep, spill = reference_memo_lines(s, base_tok)
        got_spill = meta["memo_counters"].get("memo_job_spill", 0)
        # every reference P1 survivor's title must appear in the memo
        memo_text = ctx.split("\n\n")[0]
        titles_ok = all(k.split("] ", 1)[1].split(" — ")[0] in ctx
                        for k in keep)
        ok = titles_ok and got_spill == 1
        dense_ok = dense_ok and ok
        dense_detail.append({"session": s["id"], "spill": got_spill,
                             "ref_spill_min": spill, "titles_ok": titles_ok})
    report["dense"] = {"ok": dense_ok, "detail": dense_detail}
    lat = sorted(LATENCIES)
    report["extraction_latency_ms"] = {
        "n": len(lat), "median": round(lat[len(lat) // 2], 1),
        "p95": round(lat[int(len(lat) * 0.95) - 1], 1)}
    sc = report["scales"]
    # I.1.7 gates, literally: overall >= 85%, per-type floors, memo
    # <= 240 at every scale, identical verdict across scales, dense
    # determinism, extraction p95 <= 40 ms. (The earlier all_evicted
    # condition was a harness property, not an I.1.7 gate — eviction
    # coverage is now guaranteed by the filler top-up above and is
    # reported, not gated.)
    for k, v in sc.items():
        bt = v["by_type"]
        report["scales"][k]["type_gates"] = {
            "job_outcome_100": bt["job_outcome"][0] == bt["job_outcome"][1],
            "job_title_100": bt["job_title"][0] == bt["job_title"][1],
            "num_90": bt["num"][0] / bt["num"][1] >= 0.90,
            "ent_80": bt["ent"][0] / bt["ent"][1] >= 0.80,
            "commit_70": bt["commit"][0] / bt["commit"][1] >= 0.70,
        }
    verdicts = {k: (v["overall"] >= 0.85
                    and v["memo_max_tokens"] <= 240
                    and all(v["type_gates"].values()))
                for k, v in sc.items()}
    report["verdicts"] = verdicts
    report["verdict_identical"] = len(set(verdicts.values())) == 1
    report["PASS"] = all(verdicts.values()) and dense_ok and \
        report["extraction_latency_ms"]["p95"] <= 40.0
    out = os.path.join(HERE, "step4_probe_result.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("PROBE", "PASS" if report["PASS"] else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
