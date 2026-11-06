"""Probe: connect the client, dump WebRTC stats + track state."""
import asyncio
import json
import sys

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
from cdp import Chrome, read_token, session_events  # noqa: E402

BASE = "https://ais-mac-studio.tail3b031e.ts.net:8443"
WAV = "/Users/ai/workspace/hermes-voice/tests/mob_chat_loop.wav"
MUTE = len(sys.argv) < 2 or sys.argv[1] != "nomute"

STATS_JS = """
(async () => {
  const pc = window.hermesVoice.pc;
  if (!pc) return { err: "no pc" };
  const out = { conn: pc.connectionState, ice: pc.iceConnectionState, sig: pc.signalingState, stats: [] };
  const ls = window.hermesVoice.localStream;
  out.localTracks = ls ? ls.getTracks().map(t => ({ kind: t.kind, state: t.readyState, muted: t.muted, enabled: t.enabled, label: t.label })) : null;
  const stats = await pc.getStats();
  stats.forEach(r => {
    if (r.type === "outbound-rtp" || r.type === "inbound-rtp" || r.type === "candidate-pair") {
      out.stats.push({ type: r.type, kind: r.kind || r.mediaType, bytesSent: r.bytesSent, bytesReceived: r.bytesReceived, packetsSent: r.packetsSent, packetsReceived: r.packetsReceived, state: r.state, nominated: r.nominated });
    }
  });
  out.pageLog = window.hermesVoice.debug.events.map(e => e.text);
  return out;
})()
"""


async def main():
    token = read_token()
    chrome = Chrome(WAV, port=9336, profile="/tmp/hv-chrome-probe", mute=MUTE)
    try:
        await chrome.open_page(f"{BASE}/?token={token}")
        await chrome.evaluate("window.hermesVoice.connect()")
        await chrome.poll("window.hermesVoice.state === 'connected'", 60, "pc connected")
        await asyncio.sleep(14)
        info = await chrome.evaluate(STATS_JS, await_promise=True)
        print("PROBE_JSON " + json.dumps(info, default=str), flush=True)
    finally:
        chrome.kill()
    data = session_events(BASE, token)
    kinds = [e["kind"] for e in data.get("events", [])][-20:]
    print("SERVER_EVENT_KINDS " + json.dumps(kinds), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
