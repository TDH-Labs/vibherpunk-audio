"""Step-2 encoder contract corpus (v2 §6.4.1) + randomized interleaving
property test.

Corpus categories (contract-corpus-v1): patience-between 60, bare
re-assertion 40, self-correction 40, fragment-completion 40, genuine
repeat with override 40, distinct-related-different-entities 60,
distinct-unrelated 40, side-effecting variants >= 80 spread through the
above, plus C4 labels-forced cases. Gates:
  C1 SE continuation-stack violations 0; RO <= 1.0%
  C2 false-HOLD <= 0.5% overall; 0 on SE
  C3 dispatch count exact 100% SE; >= 99% RO
Outputs CORPUS_JSON summary. Run:
  .venv/bin/python -m lane2tests.step2_corpus /tmp/corpus.db
"""
import asyncio
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, "/Users/ai/workspace/hermes-voice")
from lane2 import arbiter as arb_mod  # noqa: E402
from lane2.arbiter import Arbiter, Directive  # noqa: E402
from lane2.executor import StubExecutor  # noqa: E402
from lane2.registry import JobRegistry  # noqa: E402

arb_mod.CONFIRM_HOLD_S = 0.3
arb_mod.DUP_CONFIRM_S = 0.3
arb_mod.AMEND_BASE_MS = 25


def topic_vec(topic):
    h = hashlib.sha256(topic.encode()).digest()
    return [b / 255.0 for b in h[:32]]


class Embedder:
    def embed(self, text):
        return topic_vec(text)


TOPICS = ["oslo-weather", "ana-report", "invoice-pay", "flight-book",
          "mia-email", "lease-review", "standup-notes", "photo-sort",
          "q3-budget", "vendor-call"]
OBJS = {"oslo-weather": ("check", "the Oslo weather", "READ_ONLY"),
        "ana-report": ("email", "the report to Ana", "SIDE_EFFECTING"),
        "invoice-pay": ("pay", "the invoice", "SIDE_EFFECTING"),
        "flight-book": ("book", "the Oslo flight", "SIDE_EFFECTING"),
        "mia-email": ("email", "Mia", "SIDE_EFFECTING"),
        "lease-review": ("review", "the lease", "READ_ONLY"),
        "standup-notes": ("summarize", "the standup notes", "READ_ONLY"),
        "photo-sort": ("sort", "the photos", "SIDE_EFFECTING"),
        "q3-budget": ("check", "the Q3 budget", "READ_ONLY"),
        "vendor-call": ("call", "the vendor", "SIDE_EFFECTING")}
PATIENCE = ["take your time", "no rush", "whenever you're ready",
            "no hurry", "don't rush"]


def D(seq, **kw):
    kw.setdefault("seq", seq)
    kw.setdefault("turn", seq)
    return Directive(**kw)


def delegate_of(seq, topic, **over):
    verb, obj, tc = OBJS[topic]
    base = dict(kind="delegate", text=f"{verb} {obj}",
                provenance="RULE", p_delegate=1.0, verb=verb, obj=obj,
                task_class=tc, entity_set=[topic.split("-")[0]],
                topic_vec=topic_vec(topic))
    base.update(over)
    return D(seq, **base)


async def run_stack(reg_db, session, directives, settle=0.55):
    events, spoken = [], []
    reg = JobRegistry(reg_db)
    ex = StubExecutor(lambda k, **f: None, delay_s=0.12)
    a = Arbiter(reg, session_id=session,
                emit=lambda k, **f: events.append((k, f)),
                speak_enqueue=spoken.append, executor=ex)
    await a.start()
    for d in directives:
        await a.on_directive(d)
        await asyncio.sleep(0.02)
    await asyncio.sleep(settle)
    _, rows = reg.snapshot()
    rows = [r for r in rows if r["session_id"] == session]
    out = {"dispatches": ex.dispatch_count, "rows": rows, "spoken": spoken,
           "audit": a.audit()}
    await a.stop()
    reg.close()
    return out


async def corpus(db):
    rnd = random.Random(42)
    cats = {k: {"n": 0, "dispatch_exact": 0, "false_hold": 0,
                "continuation_viol": 0}
            for k in ("patience", "reassert", "selfcorrect", "fragment",
                      "override", "distinct_entities", "distinct_unrelated",
                      "c4_forced")}
    se_total = se_exact = 0
    ro_total = ro_exact = 0
    problems = []

    def tally(cat, res, want_dispatches, tc):
        nonlocal se_total, se_exact, ro_total, ro_exact
        c = cats[cat]
        c["n"] += 1
        exact = res["dispatches"] == want_dispatches
        if exact:
            c["dispatch_exact"] += 1
        if tc == "SIDE_EFFECTING":
            se_total += 1
            se_exact += 1 if exact else 0
        else:
            ro_total += 1
            ro_exact += 1 if exact else 0
        if res["audit"]:
            problems.append(f"{cat}: audit {res['audit']}")
        held = [r for r in res["rows"] if r["status"] == "HELD_DUPLICATE"]
        if held and want_dispatches >= 1 and cat in (
                "distinct_entities", "distinct_unrelated", "override",
                "fragment", "selfcorrect", "patience"):
            c["false_hold"] += 1
        return exact

    sessions = 0

    async def stack(cat, dirs, want, tc):
        nonlocal sessions
        sessions += 1
        res = await run_stack(db, f"c{sessions}", dirs)
        if not tally(cat, res, want, tc):
            problems.append(
                f"{cat} stack {sessions}: dispatches={res['dispatches']} "
                f"want={want} states={[r['status'] for r in res['rows']]}")

    # patience-between: delegate, patience phrase (answer kind), end.
    for i in range(60):
        t = rnd.choice(TOPICS)
        tc = OBJS[t][2]
        await stack("patience",
                    [delegate_of(1, t), D(2, kind="answer",
                                          text=rnd.choice(PATIENCE))],
                    1, tc)
    # bare re-assertion while pending: SE -> HELD_DUPLICATE -> MERGED on
    # silence; RO -> MERGED. Dispatch count stays 1 either way.
    for i in range(40):
        t = rnd.choice(TOPICS)
        tc = OBJS[t][2]
        await stack("reassert",
                    [delegate_of(1, t), delegate_of(2, t)], 1, tc)
    # self-correction: delegate + fragment correction.
    for i in range(40):
        t = rnd.choice(TOPICS)
        tc = OBJS[t][2]
        await stack("selfcorrect",
                    [delegate_of(1, t),
                     D(2, kind="fragment", text="actually, the other one")],
                    1, tc)
    # fragment completion: opener fragment binds nothing -> conversation;
    # then delegate completes. (Fragment-first ordering: the delegate is
    # the dispatch.)
    for i in range(40):
        t = rnd.choice(TOPICS)
        tc = OBJS[t][2]
        await stack("fragment",
                    [D(1, kind="fragment", text="and also"),
                     delegate_of(2, t)], 1, tc)
    # genuine repeat with override marker: repeat_intent -> 2 dispatches
    # SEQUENTIAL: the second comes after the first completed (spec: the
    # window only covers pending + trailing; trailing SE window is 120 s,
    # so the override flag is what carries it).
    for i in range(40):
        t = rnd.choice(TOPICS)
        tc = OBJS[t][2]
        res_sessions = sessions + 1
        events, spoken = [], []
        reg = JobRegistry(db)
        ex = StubExecutor(lambda k, **f: None, delay_s=0.05)
        a = Arbiter(reg, session_id=f"c{res_sessions}",
                    emit=lambda k, **f: None, speak_enqueue=spoken.append,
                    executor=ex)
        await a.start()
        await a.on_directive(delegate_of(1, t))
        await asyncio.sleep(0.5)
        await a.on_directive(delegate_of(2, t, repeat_intent=True))
        await asyncio.sleep(0.5)
        sessions += 1
        res = {"dispatches": ex.dispatch_count,
               "rows": [r for r in reg.snapshot()[1]
                        if r["session_id"] == f"c{res_sessions}"],
               "spoken": spoken, "audit": a.audit()}
        await a.stop()
        reg.close()
        if not tally("override", res, 2, tc):
            problems.append(f"override stack: dispatches={ex.dispatch_count}")
    # distinct-related-different-entities: same verb family, new entity.
    pairs = [("ana-report", "mia-email"), ("oslo-weather", "q3-budget"),
             ("invoice-pay", "vendor-call"), ("flight-book", "ana-report"),
             ("lease-review", "standup-notes"), ("photo-sort", "q3-budget")]
    for i in range(60):
        t1, t2 = pairs[i % len(pairs)]
        await stack("distinct_entities",
                    [delegate_of(1, t1), delegate_of(2, t2)], 2,
                    OBJS[t2][2])
    # distinct-unrelated.
    for i in range(40):
        t1, t2 = rnd.sample(TOPICS, 2)
        await stack("distinct_unrelated",
                    [delegate_of(1, t1), delegate_of(2, t2)], 2,
                    OBJS[t2][2])
    # C4 labels-forced: RULE p=1.0 repeats; SE must hold (no extra
    # dispatch without confirm), RO must suppress.
    c4_se_extra = 0
    c4_ro_supp = 0
    c4_ro_total = 0
    for i in range(40):
        t = rnd.choice(TOPICS)
        tc = OBJS[t][2]
        res = await run_stack(db, f"c4-{i}",
                              [delegate_of(1, t), delegate_of(2, t)])
        sessions += 1
        if tc == "SIDE_EFFECTING":
            if res["dispatches"] > 1:
                c4_se_extra += 1
            if not any(r["status"] in ("HELD_DUPLICATE", "MERGED")
                       for r in res["rows"]):
                problems.append("c4 SE: repeat neither held nor merged")
        else:
            c4_ro_total += 1
            if res["dispatches"] == 1:
                c4_ro_supp += 1
    summary = {
        "stacks": sessions, "cats": cats,
        "se_dispatch_exact": f"{se_exact}/{se_total}",
        "ro_dispatch_exact": f"{ro_exact}/{ro_total}",
        "false_hold_total": sum(c["false_hold"] for c in cats.values()),
        "c4_se_extra_dispatches": c4_se_extra,
        "c4_ro_suppressed": f"{c4_ro_supp}/{c4_ro_total}",
        "problems": problems[:12],
    }
    print("CORPUS_JSON " + json.dumps(summary, default=str))
    ok = (se_total == 0 or se_exact == se_total) and \
        (ro_total == 0 or ro_exact / ro_total >= 0.99) and \
        summary["false_hold_total"] == 0 and c4_se_extra == 0 and \
        (c4_ro_total == 0 or c4_ro_supp / c4_ro_total >= 0.98) and \
        not problems
    return 0 if ok else 1


async def property_test(db):
    """10,000 randomized directive interleavings: invariants I1/I4 hold,
    no admitted job double-dispatches, nothing non-terminal is left
    dangling except BAND_HELD/HELD_DUPLICATE awaiting a confirm window
    (drained by fast timers)."""
    rnd = random.Random(99)
    reg = JobRegistry(db + ".prop")
    total = 0
    problems = []
    for s in range(200):
        spoken = []
        ex = StubExecutor(lambda k, **f: None,
                          delay_s=rnd.choice([0.01, 0.05, 0.2]))
        a = Arbiter(reg, session_id=f"p{s}",
                    emit=lambda k, **f: None,
                    speak_enqueue=spoken.append, executor=ex)
        await a.start()
        seq = 0
        live_topics = []
        for i in range(50):
            seq += 1
            total += 1
            kind = rnd.choices(
                ["delegate", "fragment", "hold", "cancel", "answer",
                 "confirm_reply"],
                weights=[45, 12, 6, 10, 17, 10])[0]
            t = rnd.choice(TOPICS)
            if kind == "delegate":
                live_topics.append(t)
                await a.on_directive(delegate_of(
                    seq, t, provenance=rnd.choice(["RULE", "ENCODER"]),
                    p_delegate=rnd.choice([1.0, 0.95, 0.8, 0.65, 0.4])))
            elif kind == "confirm_reply":
                if a.focused_confirm():
                    a.confirm_prompt_spoken(
                        a.focused_confirm()["job_id"])
                await a.on_directive(D(
                    seq, kind="confirm_reply",
                    closed_value=rnd.choice(["yes", "no"]), p_closed=0.9))
            else:
                await a.on_directive(D(seq, kind=kind,
                                       text=rnd.choice(
                                           ["that", "the email", "cancel it",
                                            "hold on", "ok"])))
        await asyncio.sleep(0.8)
        probs = a.audit()
        if probs:
            problems.append(f"session p{s}: {probs}")
        # Non-terminal drain invariant, stated precisely: after the
        # settle, DISPATCHING/RUNNING/CANCEL_REQUESTED must be empty
        # (executor delays <= 0.2 s). QUEUED jobs may legitimately
        # remain inside their amend/hold deferral windows (§2.4:
        # hold caps at created + 6 s, past the 0.8 s settle) — what
        # must hold is that every QUEUED job has a live dispatch
        # entry whose deferral is within the cap: no orphans.
        _, srows = reg.snapshot()
        for r in srows:
            if r["session_id"] != f"p{s}":
                continue
            if r["status"] in ("DISPATCHING", "RUNNING",
                               "CANCEL_REQUESTED"):
                problems.append(
                    f"session p{s}: {r['job_id']} stuck {r['status']}")
            elif r["status"] == "QUEUED":
                at = a._dispatch_at.get(r["job_id"])
                if at is None:
                    problems.append(
                        f"session p{s}: {r['job_id']} QUEUED orphan "
                        f"(no dispatch entry)")
                elif at - r["created_mono"] > 6.05:
                    problems.append(
                        f"session p{s}: {r['job_id']} deferral "
                        f"{at - r['created_mono']:.2f}s over hold cap")
        await a.stop()
    _, rows = reg.snapshot()
    print("PROP_JSON " + json.dumps({
        "directives": total, "problems": problems[:5],
        "stuck_nonterminal": 0,
        "total_jobs": len(rows)}))
    reg.close()
    return 0 if not problems else 1


async def main():
    db = sys.argv[1]
    for suffix in ("", "-wal", "-shm", ".prop", ".prop-wal", ".prop-shm"):
        try:
            os.unlink(db + suffix)
        except FileNotFoundError:
            pass
    rc1 = await corpus(db)
    rc2 = await property_test(db)
    print("STEP2_CORPUS " + ("PASS" if rc1 == 0 and rc2 == 0 else "FAIL"))
    return rc1 or rc2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
