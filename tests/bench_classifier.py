"""Standalone encoder intent-classifier bench: the three-exit routing
decision measured directly on labeled utterances (the pipeline suite's
scripted texts + authored paraphrases). This is mechanism (b)'s misroute
rate — no pipeline, no lane model, just the classifier's argmax vs the
expected exit. Prints CLF_JSON per item + CLF_SUMMARY.

Env: HERMES_ENCODER_MODEL selects the zero-shot encoder checkpoint.
"""
import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/path/to/hermes-voice")

import hermes_voice as hv

ITEMS = [
    # chat / answer (scripted bench texts)
    ("What is the capital of France?", "answer"),
    ("How many legs does a spider have?", "answer"),
    ("What color do you get when you mix blue and yellow paint?", "answer"),
    ("Tell me one interesting fact about the moon.", "answer"),
    ("How many days are there in a leap year?", "answer"),
    ("What is the capital of Japan?", "answer"),
    ("How many continents are there on Earth?", "answer"),
    ("What is the largest planet in our solar system?", "answer"),
    ("How many minutes are there in a full day?", "answer"),
    ("What is the boiling point of water in degrees Celsius?", "answer"),
    ("Who wrote the play Romeo and Juliet?", "answer"),
    ("What is the fastest land animal?", "answer"),
    ("How many sides does a hexagon have?", "answer"),
    ("What gas do plants absorb from the air?", "answer"),
    ("What is the square root of one hundred forty four?", "answer"),
    ("Which ocean is the largest on Earth?", "answer"),
    ("What is the main ingredient in guacamole?", "answer"),
    ("How many keys does a standard piano have?", "answer"),
    ("What is the currency of the United Kingdom?", "answer"),
    ("In what year did humans first land on the moon?", "answer"),
    ("Quick question while that is running: what is two plus two?", "answer"),
    ("Tell me a joke.", "answer"),
    ("What do you think about electric cars?", "answer"),
    ("How many moons does Jupiter have?", "answer"),
    # lookup (stored-fact questions; misses escalate inside the flow)
    ("What city do I live in?", "lookup"),
    ("What time zone am I in?", "lookup"),
    ("Who is Alex?", "lookup"),
    ("Where does my second brain live?", "lookup"),
    ("What is my wife's name?", "lookup"),
    ("What is my blood type?", "lookup"),
    ("Remind me where I live again.", "lookup"),
    ("What's the name of the assistant agent you mentioned?", "lookup"),
    ("What is my Mac's hostname?", "lookup"),
    ("When is my birthday?", "lookup"),
    ("What voice do you speak with?", "lookup"),
    # delegate (work / live data / multi-step)
    ("Please have Hermes check its memory and tell me what city the user lives in, and what timezone he is in. I want the real answer from Hermes itself, so take your time.", "delegate"),
    ("Please have Hermes read the first line of its SOUL file and tell me the exact words it finds there. Take your time.", "delegate"),
    ("Please have Hermes list the names of the profiles it has on this machine. Take your time.", "delegate"),
    ("Please have Hermes check what operating system version this Mac is running, using its tools. Take your time.", "delegate"),
    ("Please have Hermes research the weather in Portland this weekend and summarize it for me. Take your time.", "delegate"),
    ("Please have Hermes check my email for anything urgent from this morning and summarize it. Take your time.", "delegate"),
    ("Could you get Hermes to look into train tickets to Portland for next week?", "delegate"),
    ("Get the back office to pull my calendar for tomorrow morning.", "delegate"),
    ("Do I have anything on my calendar tomorrow?", "delegate"),
    ("How's the weather outside right now?", "delegate"),
    ("Summarize the RUNBOOK file for me.", "delegate"),
    ("Send an email to Jordan about the invoices.", "delegate"),
    # flow turns (no new work; lane answers / holds / reports status)
    ("Is that done yet?", "answer"),
    ("It's probably going to be sunny in Portland, right? Just tell me what it found.", "answer"),
]


LABELSETS = {
    "desc": None,  # hv.IntentClassifier.LABELS as shipped
    "short": [
        ("answer", "general conversation"),
        ("lookup", "personal fact"),
        ("delegate", "task request"),
    ],
    "nli": [
        ("answer", "This is casual conversation or a general knowledge question."),
        ("lookup", "This is a question about the user's own personal information or stored details."),
        ("delegate", "This is a request for the assistant to do something, check something, or find something out."),
    ],
    "short2": [
        ("answer", "chatting or asking a general knowledge question"),
        ("lookup", "asking about a personal detail stored in memory"),
        ("delegate", "asking for a task to be done or information to be fetched"),
    ],
}


async def main():
    labelset = os.environ.get("CLF_LABELSET", "desc")
    if LABELSETS.get(labelset):
        hv.IntentClassifier.LABELS = LABELSETS[labelset]
    clf = hv.IntentClassifier()
    rows = []
    for text, expect in ITEMS:
        t0 = time.monotonic()
        res = await clf.classify(text)
        wall_ms = round((time.monotonic() - t0) * 1000)
        row = {
            "text": text[:80],
            "expect": expect,
            "decision": res["decision"] if res else None,
            "scores": res["scores"] if res else None,
            "ms": res["ms"] if res else wall_ms,
            "ok": bool(res) and res["decision"] == expect,
        }
        rows.append(row)
        print("CLF_JSON " + json.dumps(row), flush=True)
    with open("/tmp/clf_rows.json", "w") as f:
        json.dump(rows, f, indent=1)
    ok = sum(1 for r in rows if r["ok"])
    by_class = {}
    for r in rows:
        c = by_class.setdefault(r["expect"], {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += 1 if r["ok"] else 0
    lat = [r["ms"] for r in rows[1:]]  # drop the load/first-call outlier
    print(
        "CLF_SUMMARY "
        + json.dumps(
            {
                "model": hv.ENCODER_MODEL,
                "accuracy": f"{ok}/{len(rows)}",
                "misroute_rate": round(1 - ok / len(rows), 3),
                "by_class": by_class,
                "latency_median_ms": statistics.median(lat) if lat else None,
                "latency_p90_ms": sorted(lat)[int(0.9 * len(lat))] if lat else None,
                "first_call_ms": rows[0]["ms"] if rows else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
