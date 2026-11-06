"""Probe: does the mobile voice lane actually answer speech end-to-end?

Connects headless Chrome (fake mic fed from mob_chat_loop.wav — asks
"what is the capital of France"), records the bot's inbound audio, then
reports transcriptions / lane speech / recorded bytes from the server view.
"""
import asyncio
import sys

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
from cdp import Chrome, read_token, session_events  # noqa: E402

BASE = "https://ais-mac-studio.tail3b031e.ts.net:8443"
WAV = "/Users/ai/workspace/hermes-voice/tests/mob_chat_loop.wav"

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


async def main():
    token = read_token()
    ev0 = len(session_events(BASE, token).get("events", []))
    chrome = Chrome(WAV, port=9337, profile="/tmp/hv-chrome-e2e", mute=False)
    try:
        await chrome.open_page(f"{BASE}/?token={token}")
        await chrome.evaluate("window.hermesVoice.connect()")
        await chrome.poll("window.hermesVoice.state === 'connected'", 60, "pc connected")
        print("PC_CONNECTED", flush=True)
        print("REC", await chrome.evaluate(RECORD_JS), flush=True)
        try:
            await chrome.poll("window.hermesVoice.debug.firstRemoteAudioAt", 120,
                              "first remote audio")
            print("AUDIO_ARRIVED", flush=True)
        except Exception as e:
            print("NO_AUDIO", e, flush=True)
        await asyncio.sleep(5)
        rec_bytes = await chrome.evaluate(
            "window.__recChunks.reduce((a, c) => a + c.size, 0)")
        print("RECORDED_BYTES", rec_bytes, flush=True)
        log = await chrome.evaluate("window.hermesVoice.debug.events.map(e => e.text)")
        print("PAGE_LOG", log, flush=True)
    finally:
        chrome.kill()
    data = session_events(BASE, token)
    evs = data.get("events", [])[ev0:]
    print("EVENT_KINDS", [e["kind"] for e in evs], flush=True)
    print("TRANSCRIPTIONS", [e.get("text") for e in evs if e["kind"] == "transcription"],
          flush=True)
    print("SPOKEN", [s.get("text") for s in data.get("spoken", [])], flush=True)
    print("TURNS", data.get("turns"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
