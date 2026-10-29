"""Direct lane-route benchmark: streaming TTFT + usage for the conversation
lane's model route (opencode-go / deepseek-v4.1-flash), small voice prompt.
Key comes from env OPENCODE_GO_API_KEY (exported by the runner from
~/.hermes/.env — never printed here).
"""
import asyncio
import json
import os
import statistics
import time

from openai import AsyncOpenAI

SYSTEM = (
    "You are Hermes, a voice assistant on this Mac. You are the fast "
    "conversation lane: you speak briefly and naturally, one or two short "
    "sentences at a time. No lists, no markdown, no code. You can chat and "
    "answer simple questions directly. For anything that needs real work - "
    "tools, memory, files, the web, or more than a moment's thought - call "
    "delegate_to_hermes instead of guessing."
)

PROMPTS = [
    "Hi there, how's it going?",
    "What's the capital of France?",
    "How many legs does a spider have?",
    "Tell me a quick fact about the moon.",
    "What color do you get mixing blue and yellow paint?",
]


async def main():
    _kw = {"api_key": os.environ.get("OPENCODE_GO_API_KEY", "")}
    client = AsyncOpenAI(
        base_url=os.environ.get("LANE_BASE_URL", "https://opencode.ai/zen/go/v1"),
        default_headers={"x-opencode-session": os.environ.get("LANE_SESSION", "hermes-voice-lane-bench")},
        **_kw,
    )
    model = os.environ.get("LANE_MODEL", "deepseek-v4.1-flash")
    rows = []
    for i, p in enumerate(PROMPTS):
        t0 = time.monotonic()
        first = None
        usage = None
        text = ""
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": p},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.monotonic()
                text += chunk.choices[0].delta.content
        rows.append(
            {
                "turn": i,
                "ttft_ms": round((first - t0) * 1000) if first else None,
                "total_ms": round((time.monotonic() - t0) * 1000),
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "reply": text[:80],
            }
        )
        print("LANE_JSON " + json.dumps(rows[-1]), flush=True)
    ttfts = [r["ttft_ms"] for r in rows[1:] if r["ttft_ms"]]
    print(
        "LANE_SUMMARY "
        + json.dumps(
            {
                "ttft_median_ms": statistics.median(ttfts) if ttfts else None,
                "ttft_all_ms": [r["ttft_ms"] for r in rows],
                "prompt_tokens": [r["prompt_tokens"] for r in rows],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
