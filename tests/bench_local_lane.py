"""Local lane candidate probe: TTFT on the real lane payload + tool-call
reliability. Pattern follows tests/bench_lane2.py (real system prompt, real
tool schemas, streaming), extended to score tool-call deltas.

Env: LANE_BASE_URL (e.g. http://127.0.0.1:8123/v1), LANE_MODEL (repo id).
Prints LOCALBENCH_JSON lines + a LOCALBENCH_SUMMARY line.
"""
import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/path/to/hermes-voice")
from openai import AsyncOpenAI

import hermes_voice as hv


def tool_schema(fn):
    return {
        "type": "function",
        "function": {
            "name": fn.name,
            "description": fn.description,
            "parameters": {
                "type": "object",
                "properties": fn.properties,
                "required": fn.required,
            },
        },
    }


TOOLS = [tool_schema(hv.DELEGATE_TOOL), tool_schema(hv.CHECK_JOBS_TOOL)]
if hasattr(hv, "QUICK_LOOKUP_TOOL"):
    TOOLS.append(tool_schema(hv.QUICK_LOOKUP_TOOL))

HISTORY = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "How many legs does a spider have?"},
    {"role": "assistant", "content": "Eight."},
]

CHAT_Q = "What color do you get mixing blue and yellow paint?"

MUST_DELEGATE = [
    "Please have Hermes check its memory and tell me what city the user lives in, and what timezone he is in.",
    "Have the back office research the weather in Portland this weekend and give me a summary.",
    "Ask Hermes to read the first line of its SOUL file and tell me the exact words it finds there.",
]
MUST_ANSWER = [
    "What is the capital of France?",
    "How many legs does a spider have?",
    "Tell me a quick fact about the moon.",
]


async def one_call(client, model, messages, tools=None):
    t0 = time.monotonic()
    first_any = None
    first_content = None
    tool_names = []
    tool_args = ""
    content = ""
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        now = time.monotonic()
        if first_any is None and (
            delta.content or getattr(delta, "tool_calls", None)
        ):
            first_any = now
        if delta.content:
            if first_content is None:
                first_content = now
            content += delta.content
        for tc in getattr(delta, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            if fn is not None:
                if fn.name:
                    tool_names.append(fn.name)
                if fn.arguments:
                    tool_args += fn.arguments
    return {
        "first_any_ms": round((first_any - t0) * 1000) if first_any else None,
        "first_content_ms": round((first_content - t0) * 1000)
        if first_content
        else None,
        "total_ms": round((time.monotonic() - t0) * 1000),
        "tool_names": tool_names,
        "tool_args": tool_args[:200],
        "content": content[:120],
    }


SYSTEM_PROMPT = (
    hv.LANE_SYSTEM_PROMPT_ROUTED
    if os.environ.get("LANE_PROMPT") == "routed"
    else hv.LANE_SYSTEM_PROMPT
)

MUST_LOOKUP = [
    "What city do I live in?",
    "What time zone am I in?",
    "Who is Alex?",
]


async def main():
    client = AsyncOpenAI(
        base_url=os.environ["LANE_BASE_URL"],
        api_key="test-key-not-real",
    )
    model = os.environ["LANE_MODEL"]
    out = {"model": model, "ttft": [], "delegate": [], "answer": [], "lookup": []}

    # Phase 1: TTFT on the real payload (9 calls, drop the first as warm-up).
    for i in range(9):
        msgs = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + HISTORY
            + [{"role": "user", "content": CHAT_Q}]
        )
        r = await one_call(client, model, msgs, tools=TOOLS)
        out["ttft"].append(r)
        print("LOCALBENCH_JSON " + json.dumps({"phase": "ttft", **r}), flush=True)

    # Phase 2: tool reliability, 3 repeats each.
    for q in MUST_DELEGATE:
        for rep in range(3):
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            r = await one_call(client, model, msgs, tools=TOOLS)
            r["expect"] = "delegate_to_hermes"
            r["ok"] = "delegate_to_hermes" in r["tool_names"]
            out["delegate"].append(r)
            print(
                "LOCALBENCH_JSON " + json.dumps({"phase": "delegate", **r}),
                flush=True,
            )
    for q in MUST_ANSWER:
        for rep in range(3):
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            r = await one_call(client, model, msgs, tools=TOOLS)
            r["expect"] = "direct"
            r["ok"] = not r["tool_names"] and bool(r["content"])
            out["answer"].append(r)
            print(
                "LOCALBENCH_JSON " + json.dumps({"phase": "answer", **r}),
                flush=True,
            )
    if os.environ.get("LANE_PROMPT") == "routed":
        for q in MUST_LOOKUP:
            for rep in range(3):
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ]
                r = await one_call(client, model, msgs, tools=TOOLS)
                r["expect"] = "quick_lookup"
                r["ok"] = "quick_lookup" in r["tool_names"]
                out["lookup"].append(r)
                print(
                    "LOCALBENCH_JSON " + json.dumps({"phase": "lookup", **r}),
                    flush=True,
                )

    ttfts = [r["first_content_ms"] for r in out["ttft"][1:] if r["first_content_ms"]]
    summary = {
        "model": model,
        "prompt": os.environ.get("LANE_PROMPT", "relay"),
        "ttft_median_ms": statistics.median(ttfts) if ttfts else None,
        "ttft_all_ms": [r["first_content_ms"] for r in out["ttft"]],
        "delegate_ok": f"{sum(1 for r in out['delegate'] if r['ok'])}/{len(out['delegate'])}",
        "answer_ok": f"{sum(1 for r in out['answer'] if r['ok'])}/{len(out['answer'])}",
        "lookup_ok": f"{sum(1 for r in out['lookup'] if r['ok'])}/{len(out['lookup'])}",
        "delegate_first_any_ms": [r["first_any_ms"] for r in out["delegate"]],
        "delegate_replies": [r["content"][:60] for r in out["delegate"]],
        "lookup_replies": [r["content"][:60] for r in out["lookup"]],
    }
    print("LOCALBENCH_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
