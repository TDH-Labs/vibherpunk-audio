"""Standalone HYBRID router bench: RouterDecider (rules + encoder) over
the same 49 labeled utterances as bench_classifier.py, with GLiNER
extraction run per item first (as the anticipator does during speech, so
pinned names are available to the decider). jobs_pending=0 throughout —
the status rule is context-dependent and is verified in-pipeline instead
(the 'Is that done yet?' item is reported separately).

Prints ROUTER_JSON per item + ROUTER_SUMMARY with strict accuracy and
the answer-boundary (dangerous) error count.
"""
import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/Users/ai/workspace/hermes-voice")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hermes_voice as hv
from bench_classifier import ITEMS


async def main():
    decider = hv.RouterDecider(hv.IntentClassifier())
    extractor = hv.EntityExtractor()
    rows = []
    for text, expect in ITEMS:
        ents = await extractor.extract(text)
        pinned = {
            e["text"].lower()
            for e in ents
            if e["label"] == "person name" and e["score"] >= 0.55
        }
        t0 = time.monotonic()
        res = await decider.decide(text, pinned, jobs_pending=0)
        ms = round((time.monotonic() - t0) * 1000)
        row = {
            "text": text[:80],
            "expect": expect,
            "decision": res["decision"],
            "source": res.get("source"),
            "scores": res.get("scores"),
            "ms": res.get("ms", ms),
            "ok": res["decision"] == expect,
        }
        rows.append(row)
        print("ROUTER_JSON " + json.dumps(row), flush=True)
    with open("/tmp/router_rows.json", "w") as f:
        json.dump(rows, f, indent=1)
    ok = sum(1 for r in rows if r["ok"])
    dangerous = [
        r
        for r in rows
        if r["expect"] in ("lookup", "delegate") and r["decision"] == "answer"
    ]
    by_class = {}
    for r in rows:
        c = by_class.setdefault(r["expect"], {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += 1 if r["ok"] else 0
    lat = [r["ms"] for r in rows if isinstance(r.get("ms"), (int, float))]
    print(
        "ROUTER_SUMMARY "
        + json.dumps(
            {
                "accuracy": f"{ok}/{len(rows)}",
                "misroute_rate": round(1 - ok / len(rows), 3),
                "by_class": by_class,
                "dangerous_answer_boundary_errors": len(dangerous),
                "latency_median_ms": statistics.median(lat) if lat else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
