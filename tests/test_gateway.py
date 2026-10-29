"""Direct gateway timing: POST streaming chat completions, no pipecat.
Records: time to first response line, first CONTENT delta, prompt_tokens."""
import json
import os
import time
import urllib.request

KEY = os.environ["API_SERVER_KEY"]
URL = os.environ.get("HERMES_GATEWAY_URL", "http://127.0.0.1:8642/v1") + "/chat/completions"
PROMPTS = [
    "In one short sentence, what is the capital of France?",
    "In one short sentence, how many legs does a spider have?",
    "Tell me one short interesting fact about the moon.",
]
for p in PROMPTS:
    body = json.dumps({
        "model": "hermes-agent",
        "stream": True,
        "messages": [
            {"role": "system", "content": "Reply in short spoken sentences."},
            {"role": "user", "content": p},
        ],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}",
    })
    t0 = time.time()
    first_line = first_content = None
    usage = None
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            if first_line is None:
                first_line = time.time()
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content") and first_content is None:
                first_content = time.time()
    print(json.dumps({
        "prompt": p[:40],
        "first_line_ms": round((first_line - t0) * 1000) if first_line else None,
        "first_content_ms": round((first_content - t0) * 1000) if first_content else None,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
    }), flush=True)
print("GATEWAY_TEST_DONE", flush=True)
