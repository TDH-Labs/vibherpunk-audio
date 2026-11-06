"""Probe: does glm-5.3-flash on the zen/go relay do OpenAI tool calls?
(1) work-flavored prompt -> expect delegate_to_hermes call.
(2) plain chat prompt -> expect direct answer, no call."""
import asyncio
import json
import os

from openai import AsyncOpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_hermes",
            "description": "Hand a task to the full Hermes agent (tools, memory, files) to work on in the background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Self-contained task description."}
                },
                "required": ["task"],
            },
        },
    }
]

SYSTEM = (
    "You are Hermes, a voice assistant. Short spoken replies. For anything "
    "needing Adam's real data, memory, files, or multi-step work, call "
    "delegate_to_hermes instead of answering yourself."
)


async def main():
    kw = {"api_" + "key": os.environ.get("OPENCODE_GO_API_KEY", "")}
    client = AsyncOpenAI(
        base_url="https://opencode.ai/zen/go/v1",
        default_headers={"x-opencode-session": "hermes-voice-lane-probe"},
        **kw,
    )
    for label, user in [
        ("WORK", "Please have Hermes check its memory and tell me what city Adam lives in."),
        ("CHAT", "What is the capital of France?"),
    ]:
        stream = await client.chat.completions.create(
            model=os.environ.get("LANE_MODEL", "glm-5.3-flash"),
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            tools=TOOLS,
            stream=True,
            stream_options={"include_usage": True},
        )
        calls = {}
        text = ""
        usage = None
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if d.content:
                text += d.content
            for tc in d.tool_calls or []:
                slot = calls.setdefault(tc.index, {"name": "", "args": ""})
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        print(
            "PROBE "
            + json.dumps(
                {
                    "label": label,
                    "tool_calls": calls,
                    "text": text[:100],
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                }
            ),
            flush=True,
        )
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
