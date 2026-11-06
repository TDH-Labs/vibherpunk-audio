"""Probe 2: is the fake mic actually delivering speech, and for how long?
Samples the page-side mic level over time and records the local stream."""
import asyncio
import base64
import json
import sys

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
from cdp import Chrome, read_token  # noqa: E402

BASE = "https://ais-mac-studio.tail3b031e.ts.net:8443"
WAV = sys.argv[1] if len(sys.argv) > 1 else "/Users/ai/workspace/hermes-voice/tests/mob_chat_loop.wav"
OUT = "/Users/ai/workspace/hermes-voice/tests/mob_local_capture.webm"

LEVEL_JS = """
(() => {
  const hv = window.hermesVoice;
  if (!hv || !hv.localStream) return null;
  if (!window.__lvlCtx) {
    window.__lvlCtx = new AudioContext();
    window.__lvlAn = window.__lvlCtx.createAnalyser();
    window.__lvlCtx.createMediaStreamSource(hv.localStream).connect(window.__lvlAn);
    window.__lvlBuf = new Uint8Array(2048);
  }
  const an = window.__lvlAn, buf = window.__lvlBuf;
  an.getByteTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
  return Math.sqrt(sum / buf.length);
})()
"""

REC_JS = """
(() => {
  const s = window.hermesVoice.localStream;
  const rec = new MediaRecorder(s);
  window.__chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) window.__chunks.push(e.data); };
  rec.start(250);
  window.__rec = rec;
  return "rec";
})()
"""

FETCH_JS = """
(async () => {
  const rec = window.__rec;
  const done = new Promise((res) => { rec.onstop = res; });
  rec.stop();
  await done;
  const blob = new Blob(window.__chunks, { type: rec.mimeType || "audio/webm" });
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
  return { size: blob.size, b64: btoa(bin) };
})()
"""


async def main():
    token = read_token()
    mute = not (len(sys.argv) > 2 and sys.argv[2] == "nomute")
    chrome = Chrome(WAV, port=9337, profile="/tmp/hv-chrome-probe2", mute=mute)
    try:
        await chrome.open_page(f"{BASE}/?token={token}")
        await chrome.evaluate("window.hermesVoice.connect()")
        await chrome.poll("window.hermesVoice.state === 'connected'", 60, "connected")
        print("REC " + str(await chrome.evaluate(REC_JS)), flush=True)
        levels = []
        for i in range(16):
            lv = await chrome.evaluate(LEVEL_JS)
            levels.append(round(lv, 4) if lv is not None else None)
            await asyncio.sleep(1.0)
        print("LEVELS " + json.dumps(levels), flush=True)
        payload = await chrome.evaluate(FETCH_JS, await_promise=True)
        if payload and payload.get("b64"):
            with open(OUT, "wb") as f:
                f.write(base64.b64decode(payload["b64"]))
            print("CAPTURE_BYTES " + str(payload["size"]), flush=True)
    finally:
        chrome.kill()


if __name__ == "__main__":
    asyncio.run(main())
