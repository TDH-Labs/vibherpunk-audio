"""Record what the phone hears and transcribe it locally, to prove the
spoken reply actually arrived intact over WebRTC."""
import asyncio
import base64
import json
import sys
import urllib.request

sys.path.insert(0, "/path/to/hermes-voice/tests")
from cdp import Chrome, read_token, session_events  # noqa: E402

BASE = "https://your-mac.your-tailnet.ts.net:8443"
OUT = "/path/to/hermes-voice/tests/phone_heard.webm"

RECORD_JS = """
(() => {
  const s = window.hermesVoice.remoteStream;
  if (!s) return "no-stream";
  const rec = new MediaRecorder(s);
  window.__chunks = [];
  rec.ondataavailable = e => { if (e.data && e.data.size) window.__chunks.push(e.data); };
  rec.start(200);
  window.__rec = rec;
  return "recording";
})()
"""

FETCH_JS = """
(async () => {
  const rec = window.__rec;
  const done = new Promise(res => { rec.onstop = res; });
  rec.stop();
  await done;
  const blob = new Blob(window.__chunks, {type: rec.mimeType || "audio/webm"});
  const buf = await blob.arrayBuffer();
  let bin = "";
  const b = new Uint8Array(buf);
  for (let i = 0; i < b.length; i += 8192)
    bin += String.fromCharCode.apply(null, b.subarray(i, i + 8192));
  return { size: blob.size, mime: blob.type, b64: btoa(bin) };
})()
"""


def say(text):
    token = read_token()
    req = urllib.request.Request(
        "http://127.0.0.1:8917/api/_test_say",
        data=json.dumps({"text": text}).encode(),
        headers={"content-type": "application/json", "x-voice-token": token},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


async def main():
    token = read_token()
    chrome = Chrome("/path/to/hermes-voice/tests/mob_chat_loop.wav",
                    port=9344, profile="/tmp/hv-chrome-rec", mute=False)
    try:
        await chrome.open_page(f"{BASE}/?token={token}")
        await chrome.evaluate("window.hermesVoice.connect()")
        await chrome.poll("window.hermesVoice.state === 'connected'", 60, "pc connected")
        await asyncio.sleep(1.0)
        print("REC", await chrome.evaluate(RECORD_JS), flush=True)
        print("SAY", say("Please say exactly this sentence back to me: the phone version is "
                         "running and the audio path works."), flush=True)
        await asyncio.sleep(14)
        payload = await chrome.evaluate(FETCH_JS, await_promise=True)
        if payload and payload.get("b64"):
            with open(OUT, "wb") as f:
                f.write(base64.b64decode(payload["b64"]))
            print("RECORDED", payload["size"], payload["mime"], OUT, flush=True)
        else:
            print("NO_RECORDING", flush=True)
    finally:
        chrome.kill()
    data = session_events(BASE, token)
    print("SPOKEN", json.dumps(data.get("spoken", []), default=str)[:600], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
