"""Wake-lock proof: the SHIPPED client page, over the real tailnet URL.

Asserts: fresh-profile page load serves the wake-lock client, connect()
reaches pc=connected, the wake-lock request path executes (a "Wake lock:"
diagnostics line appears in the page log; the sentinel state is read back),
and disconnect() releases the lock (sentinel null, wanted flag false).
Prints WAKELOCK_JSON.
"""
import asyncio
import json
import sys

sys.path.insert(0, "/path/to/hermes-voice/tests")
from cdp import Chrome, read_token  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://your-mac.your-tailnet.ts.net:8443"
WAV = "/path/to/hermes-voice/tests/mob_chat_loop.wav"

LOG_JS = "document.getElementById('log').textContent"
STATE_JS = """(() => ({
  state: window.hermesVoice.state,
  status: document.getElementById('status').textContent,
  wakeLockHeld: !!window.hermesVoice.wakeLock,
  wakeLockWanted: window.hermesVoice.wakeLockWanted,
  wakeLockInNavigator: 'wakeLock' in navigator,
}))()"""


async def main():
    token = read_token()
    results = {"asserts": {}}
    chrome = Chrome(WAV, port=9341, profile="/tmp/hv-chrome-wakelock")
    try:
        await chrome.open_page(f"{BASE}/?token={token}")
        results["title"] = await chrome.evaluate("document.title")
        results["client_has_wakelock_code"] = await chrome.evaluate(
            "document.documentElement.outerHTML.includes('wakeLock') || "
            "fetch('/sw.js').then(r => r.text()).then(t => t.includes('hermes-voice-v3'))"
        )
        await chrome.evaluate("window.hermesVoice.connect()")
        await chrome.poll("window.hermesVoice.state === 'connected'", 60, "pc connected")
        # Give the async wake-lock request a moment to settle, then read state.
        await asyncio.sleep(3)
        results["while_connected"] = await chrome.evaluate(STATE_JS)
        log_text = await chrome.evaluate(LOG_JS)
        wakelock_lines = [ln for ln in log_text.split("\n") if "Wake lock" in ln]
        results["wakelock_log_lines"] = wakelock_lines
        results["asserts"]["wakelock_path_executed"] = len(wakelock_lines) > 0
        results["asserts"]["listening"] = (
            results["while_connected"]["status"] == "Listening"
        )
        await chrome.evaluate("window.hermesVoice.disconnect()")
        await asyncio.sleep(3)
        results["after_disconnect"] = await chrome.evaluate(STATE_JS)
        log_text = await chrome.evaluate(LOG_JS)
        results["wakelock_log_lines_final"] = [
            ln for ln in log_text.split("\n") if "Wake lock" in ln
        ]
        results["asserts"]["lock_released_on_disconnect"] = (
            results["after_disconnect"]["wakeLockHeld"] is False
            and results["after_disconnect"]["wakeLockWanted"] is False
        )
        results["asserts"]["state_after_disconnect"] = (
            results["after_disconnect"]["state"] == "none"
        )
    finally:
        chrome.kill()
    results["ok"] = all(results["asserts"].values())
    print("WAKELOCK_JSON " + json.dumps(results, indent=1))
    return 0 if results["ok"] else 1


sys.exit(asyncio.run(main()))
