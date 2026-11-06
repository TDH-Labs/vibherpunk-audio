"""Probe: is the phone->server audio path carrying packets, and is the mic
content non-silent? Answers "why no transcription" without guessing.
"""
import asyncio
import json
import sys

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
from cdp import Chrome, read_token, session_events  # noqa: E402

BASE = "https://ais-mac-studio.tail3b031e.ts.net:8443"
WAV = "/Users/ai/workspace/hermes-voice/tests/mob_chat_loop.wav"

AUDIO_JS = """
(async () => {
  const hv = window.hermesVoice;
  const out = { conn: hv.state };
  const pc = hv.pc;
  if (pc) {
    const stats = await pc.getStats();
    out.rtp = [];
    stats.forEach(r => {
      if (r.type === "outbound-rtp" && r.kind === "audio")
        out.rtp.push({ dir: "out", pkts: r.packetsSent, bytes: r.bytesSent });
      if (r.type === "inbound-rtp" && r.kind === "audio")
        out.rtp.push({ dir: "in", pkts: r.packetsReceived, bytes: r.bytesReceived });
      if (r.type === "media-source" && r.kind === "audio")
        out.src = { level: r.audioLevel, total: r.totalAudioEnergy };
    });
  }
  const ls = hv.localStream;
  out.tracks = ls ? ls.getTracks().map(t => ({ label: t.label, state: t.readyState, muted: t.muted, enabled: t.enabled })) : null;
  return out;
})()
"""


async def main():
    token = read_token()
    chrome = Chrome(WAV, port=9338, profile="/tmp/hv-chrome-audio", mute=False)
    try:
        await chrome.open_page(f"{BASE}/?token={token}")
        await chrome.evaluate("window.hermesVoice.connect()")
        await chrome.poll("window.hermesVoice.state === 'connected'", 60, "pc connected")
        a = await chrome.evaluate(AUDIO_JS, await_promise=True)
        await asyncio.sleep(10)
        b = await chrome.evaluate(AUDIO_JS, await_promise=True)
        print("T0", json.dumps(a, default=str), flush=True)
        print("T10", json.dumps(b, default=str), flush=True)
    finally:
        chrome.kill()
    data = session_events(BASE, token)
    print("EVENTS", [e["kind"] for e in data.get("events", [])][-10:], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
