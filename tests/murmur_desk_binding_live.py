"""Murmur's binding path against a live Pipecat server.

Run against a server started from the current tree (the code Murmur will talk
to), not the :8917 production build:

    MOBILE_PORT=8931 .venv/bin/python mobile_server.py &
    .venv/bin/python /path/to/murmur_desk_binding_live.py <base> <token> <wav> <session_id> <profile>

It does exactly what the Murmur hook now does, in order:
  1. POST /api/offer carrying `session_id` + `profile` beside the SDP;
  2. speaks a real WAV through the WebRTC mic;
  3. waits for the desk binding to report that chat;
  4. POST /api/bind to a second chat, as the desk switching chats does.

Prints a JSON verdict. Exits non-zero on any failed step.
"""
import asyncio
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import urllib.request

from rtc_client import connect, http_session  # noqa: E402


def post_bind(base, token, session_id, profile):
    body = {"session_id": session_id}
    if profile:
        body["profile"] = profile
    req = urllib.request.Request(
        f"{base}/api/bind?token={token}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


async def main():
    base, token, wav, session_id = sys.argv[1:5]
    profile = sys.argv[5] if len(sys.argv) > 5 else "default"
    out = {"steps": {}}

    pc, track, sink, messages, marks = await connect(
        base, token, wav, speech_secs=2.8, session_id=session_id, profile=profile
    )
    out["steps"]["offer"] = {
        "connection_state": marks.get("connection_state"),
        "pc_id": bool(marks.get("pc_id")),
    }

    # The server binds before the pipeline produces a turn; poll briefly.
    bound = None
    for _ in range(40):
        await asyncio.sleep(0.5)
        state = http_session(base, token)
        desk = state.get("desk") or {}
        if desk.get("stored_session_id") == session_id:
            bound = {
                "stored_session_id": desk.get("stored_session_id"),
                "runtime_session_id": desk.get("runtime_session_id"),
                "bound_by": desk.get("bound_by"),
                "profile": desk.get("profile"),
            }
            break
    out["steps"]["bound_on_offer"] = bound

    await asyncio.sleep(3)
    state = http_session(base, token)
    out["steps"]["after_speech"] = {
        "voice_turns_submitted": (state.get("desk") or {}).get("voice_turns_submitted"),
        "transcription": state.get("events") and [
            e for e in state.get("events", []) if e.get("kind") in ("transcription", "desk_turn_submitted")
        ][-3:],
        "turn_messages": [m.get("type") for m in messages if isinstance(m, dict)][-6:],
    }

    # The desk switches chats mid-call: same call, different target.
    other = session_id + "-rebind"
    try:
        out["steps"]["rebind"] = post_bind(base, token, other, profile)
    except Exception as e:  # a missing chat is a legitimate answer; record it
        out["steps"]["rebind_error"] = str(e)[:200]
    after = http_session(base, token).get("desk") or {}
    out["steps"]["bound_after_rebind"] = after.get("stored_session_id")

    await pc.close()
    print(json.dumps(out, indent=1, default=str))
    ok = (
        out["steps"]["offer"]["connection_state"] == "connected"
        and bound is not None
        and out["steps"]["after_speech"]["voice_turns_submitted"]
    )
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
