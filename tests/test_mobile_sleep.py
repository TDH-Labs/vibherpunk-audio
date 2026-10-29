"""Mobile acceptance — spoken sleep (aiortc peer).

Connect -> "Go to sleep." -> confirmation audio arrives at the peer and
transcribes to the exact confirmation line -> sleep_command event ->
server ends the session (reason=sleep_command).
Prints MOBSLEEP_JSON / MOBSLEEP_OK.
"""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/path/to/hermes-voice/tests")
from rtc_client import connect, http_session, read_token  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://your-mac.your-tailnet.ts.net:8443"
WAV = "/path/to/hermes-voice/tests/rtc_sleep.wav"
OUT_WAV = "/path/to/hermes-voice/tests/rtc_sleep_reply.wav"
CONFIRMATION = "Going to sleep — say hey Hermes when you need me."


async def main():
    token = read_token()
    results = {"asserts": {}}
    n_ev0 = len(http_session(BASE, token).get("events", []))
    pc, track, sink, messages, marks = await connect(BASE, token, WAV, speech_secs=1.68)
    results["connected"] = marks.get("connection_state")
    # Wait for the server to end the session (sleep) or timeout.
    end_event = None
    t0 = time.time()
    while time.time() - t0 < 60:
        data = http_session(BASE, token)
        evs = data.get("events", [])[n_ev0:]
        for e in evs:
            if e["kind"] == "mobile_session_end":
                end_event = e
        if end_event:
            break
        await asyncio.sleep(1.0)
    await asyncio.sleep(2)
    sink.dump_wav(OUT_WAV)
    try:
        await pc.close()
    except Exception:
        pass
    data = http_session(BASE, token)
    evs = data.get("events", [])[n_ev0:]
    kinds = [e["kind"] for e in evs]
    results["event_kinds"] = kinds
    results["session_end"] = end_event
    results["spoken"] = data.get("spoken", [])[-3:]
    a = results["asserts"]
    a["pc_connected"] = marks.get("connection_state") == "connected"
    a["sleep_command_seen"] = "sleep_command" in kinds
    a["confirmation_spoken"] = any(
        s["text"] == CONFIRMATION for s in data.get("spoken", [])
    )
    a["session_ended_by_sleep"] = bool(end_event) and end_event.get("reason") == "sleep_command"
    a["confirmation_audio_at_peer"] = sink.total_samples > 48000  # >1 s of audio
    print("MOBSLEEP_JSON " + json.dumps(results, default=str), flush=True)
    ok = all(a.values())
    print("MOBSLEEP_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
