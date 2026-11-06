"""Browser-half proof: the SHIPPED client page, over the real tailnet URL.

Asserts: page loads, hermesVoice.connect() reaches pc=connected, RTP bytes
flow BOTH directions (getStats), the server registered the session, and the
page's own disconnect tears it down cleanly. (Audio CONTENT proof lives in
the aiortc full-loop test: headless Chrome's fake mic is digital silence in
this environment — see REPORT.)
Prints BROWSER_JSON.
"""
import asyncio
import json
import sys

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
from cdp import Chrome, read_token, session_events  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://ais-mac-studio.tail3b031e.ts.net:8443"
WAV = "/Users/ai/workspace/hermes-voice/tests/mob_chat_loop.wav"

STATS_JS = """
(async () => {
  const pc = window.hermesVoice.pc;
  if (!pc) return { err: "no pc" };
  let sent = 0, recv = 0, pair = null;
  const stats = await pc.getStats();
  stats.forEach(r => {
    if (r.type === "outbound-rtp" && (r.kind === "audio" || r.mediaType === "audio")) sent += r.bytesSent || 0;
    if (r.type === "inbound-rtp" && (r.kind === "audio" || r.mediaType === "audio")) recv += r.bytesReceived || 0;
    if (r.type === "candidate-pair" && r.state === "succeeded") pair = { sent: r.bytesSent, recv: r.bytesReceived };
  });
  return { conn: pc.connectionState, ice: pc.iceConnectionState, sent, recv, pair,
           status: document.getElementById("status").textContent };
})()
"""


async def main():
    token = read_token()
    results = {"asserts": {}}
    before = session_events(BASE, token)
    n0 = len(before.get("events", []))
    chrome = Chrome(WAV, port=9338, profile="/tmp/hv-chrome-browser")
    try:
        await chrome.open_page(f"{BASE}/?token={token}")
        results["title"] = await chrome.evaluate("document.title")
        results["manifest_linked"] = await chrome.evaluate(
            "!!document.querySelector('link[rel=\"manifest\"]')"
        )
        await chrome.evaluate("window.hermesVoice.connect()")
        await chrome.poll("window.hermesVoice.state === 'connected'", 60, "pc connected")
        await asyncio.sleep(10)
        results["stats"] = await chrome.evaluate(STATS_JS, await_promise=True)
        await chrome.evaluate("window.hermesVoice.disconnect()")
        await asyncio.sleep(3)
        results["state_after_disconnect"] = await chrome.evaluate(
            "window.hermesVoice.state"
        )
    finally:
        chrome.kill()
    data = session_events(BASE, token)
    evs = data.get("events", [])[n0:]
    kinds = [e["kind"] for e in evs]
    results["server_event_kinds"] = kinds
    a = results["asserts"]
    st = results.get("stats") or {}
    a["page_loaded"] = results.get("title") == "Hermes Voice"
    a["pc_connected"] = st.get("conn") == "connected"
    a["rtp_both_directions"] = (st.get("sent") or 0) > 1000 and (st.get("recv") or 0) > 1000
    a["server_saw_session"] = "mobile_session_start" in kinds and "mobile_client_connected" in kinds
    a["clean_disconnect"] = "mobile_session_end" in kinds
    print("BROWSER_JSON " + json.dumps(results, default=str), flush=True)
    ok = all(a.values())
    print("BROWSER_STRUCTURAL_OK " + str(ok), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
