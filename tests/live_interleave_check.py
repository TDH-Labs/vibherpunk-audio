"""Acceptance check: text and voice turns interleave in ONE conversation,
and a reaped runtime is re-attached rather than dropped.

Uses the live gateway over the same RPCs the desk and the voice binding use.
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import desk_binding as db  # noqa: E402

DB = Path.home() / ".hermes" / "state.db"


def rows(sid):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return con.execute(
            "SELECT rowid, role, substr(content,1,80) FROM messages "
            "WHERE session_id=? ORDER BY rowid", (sid,)).fetchall()
    finally:
        con.close()


async def wait_for(sid, n, tries=60):
    for _ in range(tries):
        await asyncio.sleep(1)
        got = rows(sid)
        if len(got) >= n:
            return got
    return rows(sid)


async def main():
    desk = db.DeskBinding(session=None)
    created = await desk._rpc("session.create",
                              {"profile": "default", "source": "desktop",
                               "title": "Interleave check"})
    sid = str(created.get("stored_session_id"))
    print("chat:", sid)
    st = await desk.bind(sid, "default")
    print("bound:", st["runtime_session_id"], "by", st["bound_by"])

    # 1. A TEXT turn, through the desk's own path (prompt.submit, no voice mark).
    t = await desk._rpc("prompt.submit", {
        "session_id": st["runtime_session_id"], "profile": "default",
        "text": "Reply with exactly: text-turn-1"})
    print("text turn:", t)
    await wait_for(sid, 2)

    # 2. A VOICE turn through the binding (voice_turn + surface=voice).
    v = await desk.submit("Reply with exactly: voice-turn-2")
    print("voice turn:", v)
    got = await wait_for(sid, 4)

    # 3. A second TEXT turn, to prove interleaving (text, voice, text).
    t2 = await desk._rpc("prompt.submit", {
        "session_id": st["runtime_session_id"], "profile": "default",
        "text": "Reply with exactly: text-turn-3"})
    got = await wait_for(sid, 6)

    print(f"\ntranscript of {sid}:")
    for r in got:
        print(f"  {r[0]:>6} {r[1]:<10} {r[2]}")

    users = [r[2] for r in got if r[1] == "user"]
    order = [u.split(":")[-1].strip() for u in users]
    print("\nuser turns in order:", order)
    assert order == ["text-turn-1", "voice-turn-2", "text-turn-3"], order

    # 4. Reaped runtime: force a stale id and confirm the binding recovers.
    desk.runtime_id = "deadbeef"
    r = await desk.submit("Reply with exactly: after-reap-4")
    print("after forced reap:", r["submitted"], "->", r["runtime_session_id"])
    got = await wait_for(sid, len(got) + 2)
    users = [x[2] for x in got if x[1] == "user"]
    assert any("after-reap-4" in u for u in users), "turn lost after reap"
    print("\nPROOF OK: text and voice turns interleaved; reaped runtime re-attached")
    print("chat:", sid)


if __name__ == "__main__":
    asyncio.run(main())
