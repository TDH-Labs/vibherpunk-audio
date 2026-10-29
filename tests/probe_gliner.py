"""GLiNER span-extraction probe: load time, per-call latency on sample
transcripts, and what spans it pins. Validates the anticipator's encoder
detection layer (cached urchade/gliner_medium-v2.1 weights)."""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/path/to/hermes-voice")

import hermes_voice as hv

SAMPLES = [
    "Please have Hermes read the first line of its SOUL file and tell me the exact words it finds there.",
    "Please have Hermes research the weather in Portland this weekend and summarize it.",
    "What city do I live in?",
    "Send an email to Jordan about the Beaverton invoices.",
    "Have Alex check the calendar for the office meeting tomorrow.",
    "Ask the back office to open the RUNBOOK file on this Mac.",
]


async def main():
    ex = hv.EntityExtractor()
    t0 = time.monotonic()
    for s in SAMPLES:
        t1 = time.monotonic()
        ents = await ex.extract(s)
        print(
            "GLINER_JSON "
            + json.dumps(
                {
                    "text": s[:70],
                    "ms": round((time.monotonic() - t1) * 1000),
                    "ents": ents,
                }
            ),
            flush=True,
        )
    print(f"GLINER_TOTAL_MS {round((time.monotonic() - t0) * 1000)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
