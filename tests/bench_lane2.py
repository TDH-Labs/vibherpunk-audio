"""Lane TTFT with the EXACT in-pipeline payload: the bot's real system
prompt, both tool schemas, and a few history turns — isolates whether the
in-pipeline lane segment (0.85–1.6 s) is payload/relay-side or pipecat-side.
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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": hv.DELEGATE_TOOL.name,
            "description": hv.DELEGATE_TOOL.description,
            "parameters": {
                "type": "object",
                "properties": hv.DELEGATE_TOOL.properties,
                "required": hv.DELEGATE_TOOL.required,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": hv.CHECK_JOBS_TOOL.name,
            "description": hv.CHECK_JOBS_TOOL.description,
            "parameters": {
                "type": "object",
                "properties": hv.CHECK_JOBS_TOOL.properties,
                "required": hv.CHECK_JOBS_TOOL.required,
            },
        },
    },
]

HISTORY = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "How many legs does a spider have?"},
    {"role": "assistant", "content": "Eight."},
]


async def main():
    kw = {"api_" + "key": os.environ.get("OPENCODE_GO_API_KEY", "")}
    client = AsyncOpenAI(
        base_url=hv.LANE_BASE_URL,
        default_headers={"x-opencode-session": "hermes-voice-lane-bench2"},
        **kw,
    )
    rows = []
    for i in range(6):
        msgs = (
            [{"role": "system", "content": hv.LANE_SYSTEM_PROMPT}]
            + HISTORY
            + [{"role": "user", "content": "What color do you get mixing blue and yellow paint?"}]
        )
        t0 = time.monotonic()
        first = None
        usage = None
        stream = await client.chat.completions.create(
            model=hv.LANE_MODEL,
            messages=msgs,
            tools=TOOLS,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.monotonic()
        rows.append(
            {
                "turn": i,
                "ttft_ms": round((first - t0) * 1000) if first else None,
                "prompt_tokens": usage.prompt_tokens if usage else None,
            }
        )
        print("LANE2_JSON " + json.dumps(rows[-1]), flush=True)
    ttfts = [r["ttft_ms"] for r in rows if r["ttft_ms"]]
    print(
        "LANE2_SUMMARY "
        + json.dumps(
            {"median_ms": statistics.median(ttfts), "all_ms": ttfts,
             "prompt_tokens": [r["prompt_tokens"] for r in rows]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
