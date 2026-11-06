"""Tests for desk_binding: voice turns land in the desk's Hermes session.

No network: a fake transport records the RPCs, so the binding's decisions
(resume vs create, retry after a reaped runtime) are asserted directly.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import desk_binding as db  # noqa: E402


class FakeDesk(db.DeskBinding):
    def __init__(self, script):
        super().__init__(session=None)
        self.calls = []
        self.script = list(script)

    async def _rpc(self, method, params):
        self.calls.append((method, dict(params)))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def run(coro):
    return asyncio.run(coro)


def test_bind_resumes_the_desk_chat():
    desk = FakeDesk([{"session_id": "aaaa1111", "resumed": "20261029_101010_abc123"}])
    st = run(desk.bind("20261029_101010_abc123", "default"))
    assert [c[0] for c in desk.calls] == ["session.resume"]
    assert desk.calls[0][1]["session_id"] == "20261029_101010_abc123"
    assert st["runtime_session_id"] == "aaaa1111"
    assert st["stored_session_id"] == "20261029_101010_abc123"
    assert st["bound_by"] == "explicit"


def test_submit_uses_prompt_submit_on_the_runtime_id():
    desk = FakeDesk([
        {"session_id": "aaaa1111", "resumed": "chat-A"},
        {"status": "streaming", "user_row_id": 42},
    ])
    run(desk.bind("chat-A", "default"))
    res = run(desk.submit("  what's on my calendar  "))
    method, params = desk.calls[1]
    assert method == "prompt.submit"
    assert params["session_id"] == "aaaa1111"
    assert params["profile"] == "default"
    assert params["text"] == "what's on my calendar"
    # The desk marks voice turns so the transcript can tell them apart.
    assert params["voice_turn"] is True and params["surface"] == "voice"
    assert res["submitted"] and res["user_row_id"] == 42
    assert res["stored_session_id"] == "chat-A"


def test_expired_runtime_is_reattached_not_dropped():
    desk = FakeDesk([
        {"session_id": "aaaa1111", "resumed": "chat-A"},   # bind
        RuntimeError("prompt.submit: unknown session"),     # runtime reaped
        {"session_id": "bbbb2222", "resumed": "chat-A"},   # re-attach
        {"status": "streaming", "user_row_id": 7},          # retry succeeds
    ])
    run(desk.bind("chat-A", "default"))
    res = run(desk.submit("still there?"))
    methods = [c[0] for c in desk.calls]
    assert methods == ["session.resume", "prompt.submit", "session.resume", "prompt.submit"]
    assert desk.calls[3][1]["session_id"] == "bbbb2222"
    assert res["submitted"] is True


def test_missing_session_creates_one_instead_of_failing():
    desk = FakeDesk([
        RuntimeError("session.resume: not found"),
        {"session_id": "cccc3333", "stored_session_id": "20261029_110000_new01"},
    ])
    st = run(desk.bind("does-not-exist", "default"))
    assert [c[0] for c in desk.calls] == ["session.resume", "session.create"]
    assert st["stored_session_id"] == "20261029_110000_new01"
    assert st["runtime_session_id"] == "cccc3333"
    assert st["bound_by"] == "created"


def test_empty_utterance_is_not_submitted():
    desk = FakeDesk([])
    res = run(desk.submit("   "))
    assert res["submitted"] is False and desk.calls == []


def test_status_reports_the_session_the_desk_shows():
    desk = FakeDesk([{"session_id": "aaaa1111", "resumed": "chat-A"}])
    run(desk.bind("chat-A", "work"))
    st = desk.status()
    assert st["stored_session_id"] == "chat-A"
    assert st["profile"] == "work"
    assert st["bound_by"] == "explicit"


if __name__ == "__main__":
    import traceback

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print("failed:", failed)
    raise SystemExit(1 if failed else 0)
