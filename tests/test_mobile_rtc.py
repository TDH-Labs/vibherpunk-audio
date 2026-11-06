"""Mobile full-loop test via the aiortc test peer — TWO chat turns in one
session (turn 1 = cold pipeline, turn 2 = warm), over the real URL/token.

Verifies per turn: server transcription, lane answer, reply audio arriving
at the peer; and computes:
  client_e2e_ms   speech-end (peer clock) -> first audible bot frame
  server_first_audio_ms  pipeline metric (Phase 1 definition) via ?flush=1
  transport_overhead_ms  client_e2e - 600 (VAD stop) - server_first_audio
Also saves the reply audio + an energy-trimmed version for transcription.
Prints RTC_JSON.
"""
import asyncio
import json
import sys
import time
import urllib.request

import numpy as np
import scipy.io.wavfile

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
from rtc_client import connect, http_session, read_token  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://ais-mac-studio.tail3b031e.ts.net:8443"
WAV = "/Users/ai/workspace/hermes-voice/tests/rtc_chat2turn.wav"
SPEECH_ENDS = [2.16, 12.16]
REPLY_WAV = "/Users/ai/workspace/hermes-voice/tests/rtc_reply.wav"
REPLY_TRIM = "/Users/ai/workspace/hermes-voice/tests/rtc_reply_trim.wav"


def session_flush(base, token):
    req = urllib.request.Request(
        f"{base}/api/session?flush=1", headers={"x-voice-token": token}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


async def wait_audible(sink, after_wall, timeout=120):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        v = sink.first_audible_after(after_wall)
        if v:
            return v
        await asyncio.sleep(0.25)
    return None


async def main():
    token = read_token()
    results = {"base": BASE, "asserts": {}}
    before = http_session(BASE, token)
    n_turns0 = len(before.get("turns", []))
    pc, track, sink, messages, marks = await connect(
        BASE, token, WAV, speech_ends=SPEECH_ENDS
    )
    results["marks"] = marks
    results["asserts"]["pc_connected"] = marks.get("connection_state") == "connected"
    e2es = []
    # Wait for both speech ends, then each reply.
    t0 = time.monotonic()
    while len(track.speech_end_walls) < 2 and time.monotonic() - t0 < 60:
        await asyncio.sleep(0.2)
    results["speech_ends_sent"] = len(track.speech_end_walls)
    for i, end_wall in enumerate(list(track.speech_end_walls)):
        first = await wait_audible(sink, end_wall, timeout=120)
        e2es.append(round((first - end_wall) * 1000) if first else None)
    results["client_e2e_ms_per_turn"] = e2es
    await asyncio.sleep(4)
    if sink.dump_wav(REPLY_WAV):
        sr, data = scipy.io.wavfile.read(REPLY_WAV)
        x = data.astype(np.float64)
        idx = np.where(np.abs(x) > 300)[0]
        if len(idx):
            lo = max(0, idx[0] - int(0.3 * sr))
            hi = min(len(x), idx[-1] + int(0.5 * sr))
            scipy.io.wavfile.write(REPLY_TRIM, sr, data[lo:hi])
            results["reply_trim_secs"] = round((hi - lo) / sr, 2)
    results["dc_job_messages"] = [m for m in messages if m.get("type") == "job"]
    results["dc_rtvi_types"] = sorted({m.get("type") for m in messages if m.get("label") == "rtvi-ai"})
    await pc.close()

    data = session_flush(BASE, token)
    turns = data.get("turns", [])[n_turns0:]
    results["turns"] = turns
    results["transcriptions"] = [
        e.get("text") for e in data.get("events", []) if e["kind"] == "transcription"
    ][-4:]
    results["spoken_lane"] = [s["text"] for s in data.get("spoken", []) if s["source"] == "lane"]
    overheads = []
    for i, t in enumerate(turns[:2]):
        if i < len(e2es) and e2es[i] is not None and t.get("first_audio_ms"):
            overheads.append(round(e2es[i] - 600 - t["first_audio_ms"]))
    results["transport_overhead_ms_per_turn"] = overheads
    a = results["asserts"]
    a["two_turns_transcribed"] = (
        any("capital of france" in (t or "").lower() for t in results["transcriptions"])
        and any("spider" in (t or "").lower() for t in results["transcriptions"])
    )
    a["lane_answered_both"] = len(results["spoken_lane"]) >= 2
    a["both_replies_audible_at_peer"] = all(e is not None for e in e2es) and len(e2es) == 2
    a["turn_metrics_flushed"] = len(turns) >= 2
    print("RTC_JSON " + json.dumps(results, default=str), flush=True)
    ok = all(a.values())
    print("RTC_STRUCTURAL_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
