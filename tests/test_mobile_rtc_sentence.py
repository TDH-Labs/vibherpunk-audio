"""One-turn sentence-reply variant of the RTC loop test (intelligibility)."""
import asyncio
import json
import sys
import time

sys.path.insert(0, "/path/to/hermes-voice/tests")
from rtc_client import connect, http_session, read_token  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://your-mac.your-tailnet.ts.net:8443"
WAV = "/path/to/hermes-voice/tests/rtc_sentence.wav"
OUT_WAV = "/path/to/hermes-voice/tests/rtc_sentence_reply.wav"


async def main():
    token = read_token()
    pc, track, sink, messages, marks = await connect(BASE, token, WAV, speech_secs=2.8)
    t0 = time.monotonic()
    while track.speech_end_wall is None and time.monotonic() - t0 < 30:
        await asyncio.sleep(0.2)
    end = track.speech_end_wall
    first = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < 90:
        if end:
            first = sink.first_audible_after(end)
            if first:
                break
        await asyncio.sleep(0.25)
    await asyncio.sleep(6)
    sink.dump_wav(OUT_WAV)
    await pc.close()
    data = http_session(BASE, token)
    print("SENTENCE_JSON " + json.dumps({
        "connected": marks.get("connection_state"),
        "e2e_ms": round((first - end) * 1000) if (first and end) else None,
        "transcriptions": [e.get("text") for e in data.get("events", []) if e["kind"] == "transcription"][-2:],
        "spoken_lane": [s["text"] for s in data.get("spoken", []) if s["source"] == "lane"][-2:],
        "reply_wav": OUT_WAV,
    }, default=str), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
