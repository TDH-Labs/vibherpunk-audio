"""Mobile disconnect test via the aiortc test peer.

Connect -> work request dispatches a real Hermes job -> peer process drops
(pc.close + exit, no goodbye) -> job finishes server-side -> assert the
result was LOGGED (result_logged_disconnected), never spoken (no
spoken_inject of the summary), server healthy -> fresh peer connects and
asserts no stale result is injected into the new session.
Prints RTC_DISCONNECT_JSON.
"""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
from rtc_client import connect, http_session, read_token  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://ais-mac-studio.tail3b031e.ts.net:8443"
WORK_WAV = "/Users/ai/workspace/hermes-voice/tests/rtc_work.wav"
CHAT_WAV = "/Users/ai/workspace/hermes-voice/tests/rtc_chat.wav"
WORK_SPEECH_SECS = 8.56


async def wait_event(token, kind, timeout, pred=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = http_session(BASE, token)
        for e in data.get("events", []):
            if e["kind"] == kind and (pred is None or pred(e)):
                return data, e
        await asyncio.sleep(1.0)
    raise TimeoutError(f"event {kind} not seen in {timeout}s")


async def main():
    token = read_token()
    results = {"asserts": {}}
    pc, track, sink, messages, marks = await connect(
        BASE, token, WORK_WAV, speech_secs=WORK_SPEECH_SECS
    )
    results["connected"] = marks.get("connection_state")
    data, disp = await wait_event(token, "dispatch", 120)
    results["dispatch"] = disp
    session_id = disp.get("session_id") if "session_id" in disp else None
    # Hard drop.
    await pc.close()
    results["peer_dropped"] = True
    data, done = await wait_event(token, "worker_done", 300)
    results["worker_done"] = {k: done.get(k) for k in ("job_id", "latency_s", "summary")}
    summary = (done.get("summary") or "").strip()
    data, logged = await wait_event(token, "result_logged_disconnected", 60)
    results["result_logged"] = {k: logged.get(k) for k in ("session_id", "text")}
    evs = data.get("events", [])
    results["saw_disconnect_event"] = any(
        e["kind"] in ("mobile_client_disconnected", "mobile_session_end") for e in evs
    )
    stale_inj = [
        e
        for e in evs
        if e["kind"] == "spoken_inject" and summary and summary[:40] in (e.get("text") or "")
    ]
    results["stale_injections_total"] = len(stale_inj)
    import urllib.request

    with urllib.request.urlopen(f"{BASE}/healthz", timeout=10) as r:
        results["healthz"] = json.loads(r.read())

    # Fresh peer: no stale blast.
    n_events = len(http_session(BASE, token).get("events", []))
    pc2, track2, sink2, messages2, marks2 = await connect(
        BASE, token, CHAT_WAV, speech_secs=2.16
    )
    results["reconnected"] = marks2.get("connection_state")
    await asyncio.sleep(18)
    data2 = http_session(BASE, token)
    evs2 = data2.get("events", [])[n_events:]
    stale2 = [
        e
        for e in evs2
        if e["kind"] == "spoken_inject" and summary and summary[:40] in (e.get("text") or "")
    ]
    results["stale_blast_on_reconnect"] = len(stale2)
    results["new_session_spoken"] = [
        s["text"] for s in data2.get("spoken", [])[-3:]
    ]
    await pc2.close()

    a = results["asserts"]
    a["dispatched"] = bool(disp)
    a["worker_finished_after_drop"] = bool(done)
    a["result_logged_not_spoken"] = bool(logged) and results["stale_injections_total"] == 0
    a["server_healthy"] = results.get("healthz", {}).get("status") == "ok"
    a["no_stale_blast_on_reconnect"] = (
        results.get("reconnected") == "connected" and results["stale_blast_on_reconnect"] == 0
    )
    print("RTC_DISCONNECT_JSON " + json.dumps(results, default=str), flush=True)
    ok = all(a.values())
    print("RTC_DISCONNECT_STRUCTURAL_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
