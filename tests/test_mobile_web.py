"""Mobile full-loop acceptance test (headless Chrome, fake mic, real server).

Drives the shipped web client over the given base URL (the tailnet FQDN in
the real run), connects with the token, lets the looping fake-mic WAV ask
"What is the capital of France?", and captures:
  * client-side: connection marks + first remote audio time (page clock),
    speech-end latch, and a MediaRecorder capture of the spoken reply;
  * server-side (/api/session): transcription, spoken lane text, and the
    turn's first_audio_ms under the SAME definition Phase 1 used
    (VAD end-of-turn -> first audible sound).
Transport overhead = client end-to-end - VAD stop window (0.6 s)
                     - server first_audio_ms.
Prints MOBILE_JSON. Run from the project root with the venv python.
"""
import asyncio
import base64
import json
import subprocess
import sys
import time

sys.path.insert(0, "/path/to/hermes-voice/tests")
from cdp import Chrome, read_token, session_events  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://your-mac.your-tailnet.ts.net:8443"
WAV = sys.argv[2] if len(sys.argv) > 2 else "/path/to/hermes-voice/tests/mob_chat_loop.wav"
OUT_WEBM = "/path/to/hermes-voice/tests/mob_reply.webm"

RECORD_JS = """
(() => {
  const stream = window.hermesVoice && window.hermesVoice.remoteStream;
  if (!stream) return "no-stream";
  const rec = new MediaRecorder(stream);
  window.__recChunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) window.__recChunks.push(e.data); };
  rec.start(250);
  window.__rec = rec;
  return "recording";
})()
"""

FETCH_RECORDING_JS = """
(async () => {
  const rec = window.__rec;
  if (!rec) return null;
  const done = new Promise((res) => { rec.onstop = res; });
  rec.stop();
  await done;
  const blob = new Blob(window.__recChunks, { type: rec.mimeType || "audio/webm" });
  const buf = await blob.arrayBuffer();
  let bin = "";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i += 8192) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
  }
  return { mime: blob.type, size: blob.size, b64: btoa(bin) };
})()
"""


async def main():
    token = read_token()
    results = {"base": BASE, "asserts": {}}
    ev0 = len(session_events(BASE, token).get("events", []))
    results["events_before"] = ev0
    chrome = Chrome(WAV, port=9333, profile="/tmp/hv-chrome-web")
    try:
        await chrome.open_page(f"{BASE}/?token={token}")
        title = await chrome.evaluate("document.title")
        results["page_title"] = title
        await chrome.evaluate("window.hermesVoice.connect()")
        state = await chrome.poll(
            "window.hermesVoice && window.hermesVoice.state", 30, "pc state"
        )
        await chrome.poll(
            "window.hermesVoice.state === 'connected'", 60, "pc connected"
        )
        results["connected"] = True
        rec = await chrome.evaluate(RECORD_JS)
        results["recorder"] = rec
        # Wait for the first spoken reply on the client side.
        await chrome.poll(
            "window.hermesVoice.debug.firstRemoteAudioAt", 120, "first remote audio"
        )
        marks = await chrome.evaluate("window.hermesVoice.debug.marks")
        dbg = await chrome.evaluate(
            "({first: window.hermesVoice.debug.firstRemoteAudioAt, "
            "speechEnd: window.hermesVoice.debug.speechEndBeforeReply})"
        )
        results["client_marks"] = marks
        if dbg and dbg.get("first") and dbg.get("speechEnd"):
            results["client_e2e_ms"] = round(dbg["first"] - dbg["speechEnd"])
        await asyncio.sleep(4)  # let the reply finish + maybe a second turn
        payload = await chrome.evaluate(FETCH_RECORDING_JS, await_promise=True)
        if payload and payload.get("b64"):
            with open(OUT_WEBM, "wb") as f:
                f.write(base64.b64decode(payload["b64"]))
            results["recording_bytes"] = payload["size"]
            results["recording_mime"] = payload["mime"]
    finally:
        chrome.kill()

    # Server-side view.
    data = session_events(BASE, token)
    evs = data.get("events", [])[ev0:]
    results["transcriptions"] = [e.get("text") for e in evs if e["kind"] == "transcription"]
    results["spoken_lane"] = [s["text"] for s in data.get("spoken", []) if s["source"] == "lane"]
    turns = [t for t in data.get("turns", []) if t.get("transcript")]
    results["turns"] = turns[-3:]
    if turns:
        results["server_first_audio_ms"] = turns[-1].get("first_audio_ms")
    if results.get("client_e2e_ms") is not None and results.get("server_first_audio_ms"):
        results["transport_overhead_ms"] = round(
            results["client_e2e_ms"] - 600 - results["server_first_audio_ms"]
        )
    a = results["asserts"]
    a["connected"] = bool(results.get("connected"))
    a["server_heard_question"] = any(
        "capital of france" in (t or "").lower() for t in results["transcriptions"]
    )
    a["lane_answered_paris"] = any(
        "paris" in (t or "").lower() for t in results["spoken_lane"]
    )
    a["client_got_audio"] = bool(results.get("recording_bytes", 0) > 2000)
    a["client_e2e_measured"] = results.get("client_e2e_ms") is not None
    print("MOBILE_JSON " + json.dumps(results, default=str), flush=True)
    ok = all(a.values())
    print("MOBILE_STRUCTURAL_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
