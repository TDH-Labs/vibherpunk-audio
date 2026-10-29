"""Mobile acceptance — held results across sessions (aiortc peer).

Session 1: work request dispatches -> "go to sleep" before completion ->
session ends -> job finishes with no live session: result HELD
(held_result_held), never spoken. Session 2 (reconnect): the held result
is announced FIRST ("While you were asleep, Vibherpunk finished: ..."),
before the new session's own chat answer.
Prints MOBHELD_JSON / MOBHELD_OK.
"""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/path/to/hermes-voice/tests")
from rtc_client import connect, http_session, read_token  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://your-mac.your-tailnet.ts.net:8443"
WORK_SLEEP_WAV = "/path/to/hermes-voice/tests/rtc_work_sleep.wav"
CHAT_WAV = "/path/to/hermes-voice/tests/rtc_chat.wav"
OUT_WAV = "/path/to/hermes-voice/tests/rtc_held_reply.wav"
PREFIX = "While you were asleep, Vibherpunk finished:"


async def wait_event(token, kind, timeout, since=0, pred=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = http_session(BASE, token)
        for e in data.get("events", [])[since:]:
            if e["kind"] == kind and (pred is None or pred(e)):
                return data, e
        await asyncio.sleep(1.0)
    raise TimeoutError(f"event {kind} not seen in {timeout}s")


async def main():
    token = read_token()
    results = {"asserts": {}}
    n_ev0 = len(http_session(BASE, token).get("events", []))
    # ---- Session 1: dispatch, then sleep mid-job. ----
    pc, track, sink, messages, marks = await connect(BASE, token, WORK_SLEEP_WAV)
    results["connected1"] = marks.get("connection_state")
    data, disp = await wait_event(token, "dispatch", 90, since=n_ev0)
    results["dispatch"] = disp.get("job_id")
    # The track plays "go to sleep" at ~11.6 s; wait for the session to end.
    data, end_ev = await wait_event(
        token, "mobile_session_end", 90, since=n_ev0,
        pred=lambda e: e.get("reason") == "sleep_command",
    )
    results["session1_end_reason"] = end_ev.get("reason")
    try:
        await pc.close()
    except Exception:
        pass
    # ---- Job completes into the held store. ----
    data, done = await wait_event(token, "worker_done", 300, since=n_ev0)
    results["worker_summary"] = done.get("summary")
    data, held = await wait_event(token, "held_result_held", 30, since=n_ev0)
    results["held_event"] = {k: held.get(k) for k in ("job_id", "session_id")}
    await asyncio.sleep(3)
    data = http_session(BASE, token)
    evs = data.get("events", [])[n_ev0:]
    a = results["asserts"]
    a["dispatched"] = bool(disp)
    a["session1_slept"] = end_ev.get("reason") == "sleep_command"
    a["result_held"] = bool(held)
    a["no_announcement_while_away"] = not any(
        e["kind"] == "held_result_announced" for e in evs
    ) and not any(
        PREFIX in (s.get("text") or "") for s in data.get("spoken", [])
    )
    # ---- Session 2: held result announced first. ----
    n_spoken_before = len(data.get("spoken", []))
    pc2, track2, sink2, messages2, marks2 = await connect(
        BASE, token, CHAT_WAV, speech_secs=2.16
    )
    results["connected2"] = marks2.get("connection_state")
    data2, ann = await wait_event(token, "held_result_announced", 60, since=n_ev0)
    results["announced"] = {k: ann.get(k) for k in ("job_id", "summary")}
    # Wait for this session's own chat answer too.
    t0 = time.time()
    lane_after = None
    while time.time() - t0 < 60:
        d = http_session(BASE, token)
        sp = d.get("spoken", [])[n_spoken_before:]
        lane = [s for s in sp if s["source"] == "lane"]
        if lane:
            lane_after = lane[0]["text"]
            break
        await asyncio.sleep(1.0)
    await asyncio.sleep(3)
    sink2.dump_wav(OUT_WAV)
    await pc2.close()
    d = http_session(BASE, token)
    sp = d.get("spoken", [])[n_spoken_before:]
    texts = [s["text"] for s in sp]
    results["session2_spoken_order"] = texts
    idx_ann = next((i for i, t in enumerate(texts) if t.startswith(PREFIX)), None)
    idx_lane = next(
        (i for i, s in enumerate(sp) if s["source"] == "lane"), None
    )
    a["announced_on_reconnect"] = idx_ann is not None
    a["announcement_before_chat_answer"] = (
        idx_ann is not None and (idx_lane is None or idx_ann < idx_lane)
    )
    a["chat_still_answered"] = lane_after is not None
    a["announcement_audio_at_peer"] = sink2.total_samples > 48000 * 3
    print("MOBHELD_JSON " + json.dumps(results, default=str), flush=True)
    ok = all(a.values())
    print("MOBHELD_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
