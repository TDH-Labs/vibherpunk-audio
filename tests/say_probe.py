"""Send a text turn straight into a live mobile session's lane, bypassing
STT, and confirm the phone-side <audio> element receives and plays audio.

Proves everything except speech recognition (which needs a real mic).
Server-side helper endpoint is added by mobile_server.py (/api/_test_say,
token-gated, loopback-only).
"""
import asyncio
import json
import sys
import urllib.request

sys.path.insert(0, "/path/to/hermes-voice/tests")
from cdp import Chrome, read_token, session_events  # noqa: E402

BASE = "https://your-mac.your-tailnet.ts.net:8443"

AUDIO_JS = """
(() => {
  const el = document.getElementById("remoteAudio");
  const out = { paused: el.paused, t: el.currentTime, ready: el.readyState,
                hasSrc: !!el.srcObject, muted: el.muted, vol: el.volume };
  try { out.ctx = el.webkitAudioContext ? 1 : 0; } catch (e) {}
  return out;
})()
"""


def say(text):
    token = read_token()
    req = urllib.request.Request(
        f"http://127.0.0.1:8917/api/_test_say",
        data=json.dumps({"text": text}).encode(),
        headers={"content-type": "application/json", "x-voice-token": token},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


async def main():
    token = read_token()
    chrome = Chrome("/path/to/hermes-voice/tests/mob_chat_loop.wav",
                    port=9343, profile="/tmp/hv-chrome-say", mute=False)
    try:
        await chrome.open_page(f"{BASE}/?token={token}")
        await chrome.evaluate("window.hermesVoice.connect()")
        await chrome.poll("window.hermesVoice.state === 'connected'", 60, "pc connected")
        await asyncio.sleep(1.5)
        print("SAY", say("Yes, this is the phone version of Hermes. I can hear you."),
              flush=True)
        await asyncio.sleep(2)
        a = await chrome.evaluate(AUDIO_JS)
        await asyncio.sleep(4)
        b = await chrome.evaluate(AUDIO_JS)
        print("AUDIO_T2", json.dumps(a, default=str), flush=True)
        print("AUDIO_T6", json.dumps(b, default=str), flush=True)
        log = await chrome.evaluate("window.hermesVoice.debug.events.map(e => e.text)")
        print("PAGE_LOG", log, flush=True)
    finally:
        chrome.kill()
    data = session_events(BASE, token)
    print("SPOKEN", [s.get("text") for s in data.get("spoken", [])], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
