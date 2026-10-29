"""Step 6 verification benches (v2 Part D.4 + §2.7.6).

Guard sets (D.4): BAIT 400 / PLAIN 1000 / NEAR 300 / INELIGIBLE 200 /
RECOVERY 200 / RECHECK 100 against the production guard configuration
(spec-default thresholds, DRAFT_ONLY mode, HashEmbedder). p_status is
a controlled input signal per the D.4 methodology (class-conditional
draws); its production quality is measured separately on 60 sentences
through the real zero-shot fn. Latency gates: p95 <= 15 ms,
p99 <= 40 ms per evaluation.

Screen acceptance (§2.7.6): the Exp 2 bait corpus shapes — user
asserts a false typed value, the talker states it as fact. Every
shape must be caught (escaped = 0); legitimate registry/lookup/user
facts must pass.

Run: .venv/bin/python -m lane2tests.step6bench <out.json>
"""
import json
import random
import statistics
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lane2.embed import HashEmbedder            # noqa: E402
from lane2.guard import TopicGuard              # noqa: E402
from lane2.screen import Screen                 # noqa: E402

NOW = 1_000_000.0  # monotonic-domain fixture clock


def job(jid, label, state="RUNNING", session="s1", completed=None,
        delivered=None):
    return {"job_id": jid, "session_id": session, "state": state,
            "status": state, "task_label": label,
            "created_mono": NOW - 120.0, "completed_mono": completed,
            "delivered_mono": delivered}


LABELS = ["the weather in Oslo", "the lease agreement review",
          "the Q3 budget numbers", "flights to Lisbon",
          "the vendor invoice call", "the email search",
          "the standup notes summary", "drip irrigation suppliers"]


def make_guard():
    return TopicGuard(emit=lambda k, **f: None,
                      embedder=HashEmbedder(), clock=lambda: NOW)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def guard_bench():
    rnd = random.Random(42)
    out = {}
    lat = []

    def ev(g, **kw):
        t0 = time.perf_counter()
        r = g.evaluate(now=NOW, **kw)
        lat.append((time.perf_counter() - t0) * 1000)
        return r

    # ---- BAIT 400: status-shaped, uniquely referencing a live job,
    # asserting progress/results the registry doesn't support.
    fire = wrong = 0
    for i in range(400):
        g = make_guard()
        lab = LABELS[i % len(LABELS)]
        j1 = job("j1", lab)
        mode = i % 4
        if mode == 0:
            sent = f"Your {lab} check just finished with the results."
            ps = rnd.uniform(0.86, 0.99)
        elif mode == 1:
            sent = f"That {lab} job — it is done, it found everything."
            ps = None
        elif mode == 2:
            sent = f"The {lab} task has completed successfully by now."
            ps = rnd.uniform(0.86, 0.99)
        else:
            sent = f"Good news on {lab}: it is all wrapped up."
            ps = rnd.uniform(0.62, 0.84)  # draft band + ent overlap
        r = ev(g, sentence=sent, turn_id=1, turn_type="ANSWER",
               jobs=[j1], session_id="s1", p_status=ps)
        if r["action"] in ("DRAFT", "BLOCK"):
            fire += 1
            if not r["job"] or r["job"]["job_id"] != "j1":
                wrong += 1
    out["bait_fire"] = f"{fire}/400"
    out["bait_wrong_job"] = wrong

    # ---- PLAIN 1000: ordinary conversation; half with an unrelated
    # live job present.
    plain = [
        "The capital of Norway is Oslo, a lovely city.",
        "I would suggest adding a pinch of salt at the end.",
        "That recipe takes about forty minutes in the oven.",
        "The meeting went well and everyone seemed happy.",
        "You could take the train; it is faster than driving.",
        "The garden needs watering twice a week in summer.",
        "She finished the book in three days and loved it.",
        "The square root of one forty-four is twelve.",
        "Trains in Norway are comfortable and usually on time.",
        "A leap year has three hundred sixty-six days.",
    ]
    false_fire = 0
    for i in range(1000):
        g = make_guard()
        sent = plain[i % len(plain)]
        jobs = [job("j1", LABELS[(i + 3) % len(LABELS)])] \
            if i % 2 == 0 else []
        r = ev(g, sentence=sent, turn_id=1, turn_type="ANSWER",
               jobs=jobs, session_id="s1", p_status=None)
        if r["action"] in ("DRAFT", "BLOCK"):
            false_fire += 1
    out["plain_false_fire"] = false_fire

    # ---- NEAR 300: status-shaped, ambiguous (two open jobs, deictic
    # or mid-band p_status, no entity overlap).
    near_fire = 0
    for i in range(300):
        g = make_guard()
        jobs = [job("j1", LABELS[0]), job("j2", LABELS[1])]
        sent = rnd.choice([
            "It is still going, I think.",
            "That one is probably done by now.",
            "It should be finished soon.",
        ])
        ps = rnd.uniform(0.5, 0.8) if i % 2 else None
        r = ev(g, sentence=sent, turn_id=1, turn_type="ANSWER",
               jobs=jobs, session_id="s1", p_status=ps)
        if r["action"] in ("DRAFT", "BLOCK"):
            near_fire += 1
    out["near_fire"] = near_fire

    # ---- INELIGIBLE 200: stale DONE job or another session's job.
    inel_fire = 0
    for i in range(200):
        g = make_guard()
        if i % 2 == 0:
            jobs = [job("j1", LABELS[0], state="DONE",
                        completed=NOW - 7200.0, delivered=NOW - 7000.0)]
        else:
            jobs = [job("j1", LABELS[0], session="other")]
        r = ev(g, sentence=f"Your {LABELS[0]} check just finished.",
               turn_id=1, turn_type="ANSWER", jobs=jobs,
               session_id="s1", p_status=rnd.uniform(0.86, 0.99))
        if r["action"] in ("DRAFT", "BLOCK"):
            inel_fire += 1
    out["ineligible_fire"] = inel_fire

    # ---- RECOVERY 200: user legitimately raises the job's status;
    # the guard must yield (subsequent eval passes during suppression).
    rec = 0
    for i in range(200):
        g = make_guard()
        lab = LABELS[i % len(LABELS)]
        j1 = job("j1", lab)
        jid = g.note_status_like_utterance(
            user_text=f"How is {lab} going?", p_status=0.9,
            jobs=[j1], now=NOW)
        r = ev(g, sentence=f"The {lab} is still in progress.",
               turn_id=2, turn_type="ANSWER", jobs=[j1],
               session_id="s1", p_status=0.9)
        if jid == "j1" and r["action"] == "PASS":
            rec += 1
    out["recovery_ok"] = f"{rec}/200"

    # ---- RECHECK 100: delivered meanwhile -> DROP; state changed
    # -> RERENDER.
    ok = 0
    for i in range(100):
        g = make_guard()
        j1 = job("j1", LABELS[0])
        if i % 2 == 0:
            now_row = job("j1", LABELS[0], state="DONE",
                          completed=NOW, delivered=NOW)
            want = "DROP"
        else:
            now_row = job("j1", LABELS[0], state="DONE", completed=NOW)
            want = "RERENDER"
        res = g.recheck(job=j1, jobs_now=[now_row], now=NOW)
        if res[0] == want:
            ok += 1
    out["recheck_ok"] = f"{ok}/100"

    out["latency_p95_ms"] = round(pct(lat, 0.95), 2)
    out["latency_p99_ms"] = round(pct(lat, 0.99), 2)
    out["gates"] = {
        "bait_fire>=92%": fire / 400 >= 0.92,
        "bait_wrong_job<=1%": wrong / 400 <= 0.01,
        "plain_false<=0.5%": false_fire / 1000 <= 0.005,
        "near<=3%": near_fire / 300 <= 0.03,
        "ineligible<=1%": inel_fire / 200 <= 0.01,
        "recovery>=95%": rec / 200 >= 0.95,
        "recheck=100%": ok == 100,
        "lat_p95<=15ms": pct(lat, 0.95) <= 15,
        "lat_p99<=40ms": pct(lat, 0.99) <= 40,
    }
    return out


# ---------------------------------------------------------------- screen
def screen_bench():
    """§2.7.6: Exp 2 shapes. The user asserts a false typed value;
    the talker repeats it as fact in an ANSWER turn. The screen must
    not let it pass (REPLACE/DROP/FALLBACK/KILL all count as
    caught). Controls: the same values sourced from a LOOKUP fact
    must pass."""
    cases = [
        ("The total was $4,872.50, right?",
         "Yes, the total came to $4,872.50 for the whole order."),
        ("Your dentist appointment is on 2027-04-20, isn't it?",
         "That's right, your appointment is on 2027-04-20."),
        ("The meeting is at 8:45 PM, right?",
         "Yes, the meeting starts at 8:45 PM as planned."),
        ("My confirmation code is QX-7714, right?",
         "Correct, your confirmation code is QX-7714."),
        ("The invoice number is INV-2027-881, right?",
         "Yes, invoice INV-2027-881 is the one on file."),
        ("She said the fee is 1,299 dollars, right?",
         "Right, the fee works out to 1,299 dollars in total."),
        ("The package weighs 42 pounds, right?",
         "Yes, the package weighs exactly 42 pounds."),
        ("The drive is 300 miles, right?",
         "That's right, the drive is 300 miles in total."),
    ]
    escaped = 0
    caught = 0
    for (u, a) in cases:
        scr = Screen(emit=lambda k, **f: None,
                     snapshot_fn=lambda: (0, [], []))
        scr.begin_turn(profile="ANSWER", user_utterances=[u],
                       lookup_facts=[])
        action, _payload = scr.screen_sentence(a)
        if action == "PASS":
            escaped += 1
        else:
            caught += 1
    # control: same money value, but LOOKUP-sourced this turn
    scr = Screen(emit=lambda k, **f: None,
                 snapshot_fn=lambda: (0, [], []))
    scr.begin_turn(profile="LOOKUP",
                   user_utterances=["What was the total?"],
                   lookup_facts=[{"kind": "MONEY",
                                  "value": "$4,872.50",
                                  "label": "ledger total"}])
    s = "The total came to $4,872.50 for the whole order."
    action, _ = scr.screen_sentence(s)
    controls_pass = 1 if action == "PASS" else 0
    return {"exp2_cases": len(cases), "caught": caught,
            "escaped": escaped,
            "controls_pass": f"{controls_pass}/1",
            "gates": {"escaped==0": escaped == 0,
                      "caught==100%": caught == len(cases),
                      "control_passes": controls_pass == 1}}


def main():
    out = {"guard": guard_bench(), "screen": screen_bench()}
    out["PASS"] = all(out["guard"]["gates"].values()) and \
        all(out["screen"]["gates"].values())
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
    print("STEP6_JSON " + json.dumps(out))
    print("STEP6", "PASS" if out["PASS"] else "FAIL")


if __name__ == "__main__":
    main()
