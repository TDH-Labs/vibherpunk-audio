"""Real-audio proof that a spoken turn lands in the desk's Hermes session.

Uses the production aiortc harness (tests/rtc_client.py) to connect to the
mobile voice server with a real 48 kHz WAV as the microphone, so the turn
travels the true path: WebRTC -> VAD/STT -> lane2 -> LLMContextFrame ->
Lane2Processor._run_turn -> desk mirror. Then it reads the bound chat's
transcript out of state.db.

Usage: .venv/bin/python tests/live_voice_to_desk.py <base> <token> <session> <wav>
"""
import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
from rtc_client import connect  # noqa: E402

BASE = sys.argv[1]
TOKEN = sys.argv[2]
SESSION = sys.argv[3]
WAV = sys.argv[4]
DB = Path.home() / ".hermes" / "state.db"


def rows(session_id):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return con.execute(
            "SELECT rowid, role, substr(content, 1, 120) FROM messages "
            "WHERE session_id = ? ORDER BY rowid",
            (session_id,)).fetchall()
    finally:
        con.close()


def session_view(base, token):
    import urllib.request
    with urllib.request.urlopen(f"{base}/api/session?token={token}", timeout=15) as r:
        return json.loads(r.read().decode())


async def main():
    before = rows(SESSION)
    print(f"transcript before: {len(before)} rows")
    print(f"connecting to {BASE} with {WAV}")
    pc, track, sink, messages, marks = await connect(
        BASE, TOKEN, WAV, speech_secs=2.8, session_id=SESSION, profile="default")
    print("connection:", marks.get("connection_state"))

    t0 = time.monotonic()
    while track.speech_end_wall is None and time.monotonic() - t0 < 30:
        await asyncio.sleep(0.2)
    print("speech sent at:", track.speech_end_wall)

    # Wait for the pipeline to transcribe and start the turn.
    end = None
    for _ in range(60):
        await asyncio.sleep(1)
        data = session_view(BASE, TOKEN)
        kinds = [e["kind"] for e in data.get("events", [])]
        if "transcription" in kinds:
            end = data
            break
    print("server events:", [e for e in (end or {}).get("events", []) if e["kind"] in
                             ("transcription", "desk_turn_submitted", "desk_turn_mirrored",
                              "desk_turn_failed", "desk_bound")])

    # Give the mirrored submit time to be written and answered.
    for _ in range(40):
        await asyncio.sleep(1)
        got = rows(SESSION)
        if len(got) > len(before) and any(r[1] == "assistant" for r in got[len(before):]):
            break

    await asyncio.sleep(3)
    try:
        await pc.close()
    except Exception:
        pass

    after = rows(SESSION)
    print(f"\ntranscript of {SESSION} after ({len(after)} rows):")
    for rowid, role, content in after:
        mark = "  NEW" if (rowid, role, content) not in before else "     "
        print(f"{mark} {rowid:>6} {role:<12} {content}")

    new = [r for r in after if (r[0], r[1], r[2]) not in before]
    print("\nnew rows:", len(new))
    if new:
        print("PROOF OK: the spoken turn reached the desk session", SESSION)
    else:
        print("PROOF FAILED: nothing new in the desk session")
    return new


if __name__ == "__main__":
    asyncio.run(main())
