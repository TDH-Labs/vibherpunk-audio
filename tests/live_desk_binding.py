"""Live proof: DeskBinding submits a turn into a real Hermes session.

Creates a session through the same RPC the desk uses, binds to it, submits
a voice turn through DeskBinding, then reads the transcript back from
state.db to show the turn and the reply in the one stored session.

Usage: .venv/bin/python tests/live_desk_binding.py
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import desk_binding as db  # noqa: E402

DB = Path.home() / ".hermes" / "state.db"


def rows(session_id):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return con.execute(
            "SELECT rowid, role, substr(content, 1, 90) FROM messages "
            "WHERE session_id = ? ORDER BY rowid",
            (session_id,),
        ).fetchall()
    finally:
        con.close()


async def main():
    desk = db.DeskBinding(session=None)
    # Create a session the way the desk does, then bind voice to it.
    res = await desk._rpc("session.create", {"profile": "default", "source": "desktop", "title": "Voice binding proof"})
    stored = str(res.get("stored_session_id") or res.get("session_id"))
    print("created stored session:", stored)

    st = await desk.bind(stored, "default", source="explicit")
    print("bound:", st)

    out = await desk.submit("Reply with exactly this and nothing else: voice-binding-ok")
    print("submit:", out)
    assert out["stored_session_id"] == stored, "voice turn landed in a different session!"
    assert out["runtime_session_id"] == st["runtime_session_id"]

    # Wait for the turn to be recorded, then read it back out of state.db.
    for _ in range(40):
        await asyncio.sleep(0.75)
        got = rows(stored)
        roles = [r[1] for r in got]
        if "assistant" in roles and "user" in roles:
            if any("voice-binding-ok" in (r[2] or "") for r in got if r[1] == "assistant"):
                break

    print(f"\ntranscript of {stored}:")
    for rowid, role, content in rows(stored):
        print(f"  {rowid:>6} {role:<12} {content}")

    got = rows(stored)
    user_rows = [r for r in got if r[1] == "user"]
    bot_rows = [r for r in got if r[1] == "assistant"]
    assert any("Reply with exactly this" in (r[2] or "") for r in user_rows), "user turn missing"
    assert any("voice-binding-ok" in (r[2] or "") for r in bot_rows), "assistant reply missing"
    print("\nPROOF OK: voice turn and reply are in the same stored session", stored)
    return stored


if __name__ == "__main__":
    asyncio.run(main())
