"""Mobile disconnect acceptance test.

Flow: connect (fake mic plays the work request once, then 150 s silence)
-> server dispatches a worker job -> KILL the browser (hard drop, no clean
close) -> worker finishes on its own -> assert the result was LOGGED, not
spoken (result_logged_disconnected, no spoken summary injection) -> the
server is still healthy -> reconnect with a fresh browser and assert the
stale result is NOT blasted into the new session.
Prints DISCONNECT_JSON. Run from the project root with the venv python.
"""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/path/to/hermes-voice/tests")
from cdp import Chrome, read_token, session_events, wait_event  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://your-mac.your-tailnet.ts.net:8443"
WORK_WAV = "/path/to/hermes-voice/tests/mob_work_once.wav"
CHAT_WAV = "/path/to/hermes-voice/tests/mob_chat_once.wav"


async def connect_and_dispatch(token):
    chrome = Chrome(WORK_WAV, port=9334, profile="/tmp/hv-chrome-disc")
    await chrome.open_page(f"{BASE}/?token={token}")
    await chrome.evaluate("window.hermesVoice.connect()")
    await chrome.poll("window.hermesVoice.state === 'connected'", 60, "pc connected")
    data, idx, ev = await wait_event(BASE, token, "dispatch", 90)
    return chrome, data, ev


async def main():
    token = read_token()
    results = {"asserts": {}}
    chrome, data0, disp = await connect_and_dispatch(token)
    results["dispatch"] = disp
    since = len(session_events(BASE, token).get("events", []))
    # Hard drop: kill the whole browser mid-job.
    chrome.kill()
    results["browser_killed"] = True
    # Worker runs to completion on its own (~10-60 s for this task).
    data, idx, done = await wait_event(BASE, token, "worker_done", 300, since=0)
    results["worker_done"] = done
    # The bridge must log the result instead of speaking it.
    data, idx, logged = await wait_event(
        BASE, token, "result_logged_disconnected", 60, since=0
    )
    results["result_logged"] = logged
    evs = data.get("events", [])
    results["disconnected_event"] = any(
        e["kind"] in ("mobile_client_disconnected", "mobile_session_end")
        for e in evs
    )
    summary = (done.get("summary") or "").strip()
    results["summary_head"] = summary[:80]
    # No spoken injection of that summary anywhere after the kill.
    inj_after = [
        e
        for e in evs
        if e["kind"] == "spoken_inject" and summary and summary[:40] in (e.get("text") or "")
    ]
    results["stale_injections"] = len(inj_after)
    # Server still healthy?
    import urllib.request

    with urllib.request.urlopen(f"{BASE}/healthz", timeout=10) as r:
        results["healthz"] = json.loads(r.read())

    # Reconnect fresh: the stale result must not be spoken into the new session.
    n_events_before = len(session_events(BASE, token).get("events", []))
    chrome2 = Chrome(CHAT_WAV, port=9335, profile="/tmp/hv-chrome-disc2")
    try:
        await chrome2.open_page(f"{BASE}/?token={token}")
        await chrome2.evaluate("window.hermesVoice.connect()")
        await chrome2.poll("window.hermesVoice.state === 'connected'", 60, "reconnect")
        results["reconnected"] = True
        await asyncio.sleep(20)  # window in which a stale blast would arrive
        data2 = session_events(BASE, token)
        evs2 = data2.get("events", [])[n_events_before:]
        stale = [
            e
            for e in evs2
            if e["kind"] == "spoken_inject"
            and summary
            and summary[:40] in (e.get("text") or "")
        ]
        results["stale_blast_on_reconnect"] = len(stale)
        results["new_session_events"] = [e["kind"] for e in evs2][:20]
    finally:
        chrome2.kill()

    a = results["asserts"]
    a["dispatched"] = bool(disp)
    a["worker_finished_after_drop"] = bool(done)
    a["result_logged_not_spoken"] = bool(logged) and results["stale_injections"] == 0
    a["server_healthy"] = results.get("healthz", {}).get("status") == "ok"
    a["no_stale_blast_on_reconnect"] = (
        results.get("reconnected") and results.get("stale_blast_on_reconnect") == 0
    )
    print("DISCONNECT_JSON " + json.dumps(results, default=str), flush=True)
    ok = all(a.values())
    print("DISCONNECT_STRUCTURAL_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
