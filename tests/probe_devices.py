"""Probe 5: scan each fake audio input device for the file-fed one."""
import asyncio
import json
import sys

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
import cdp as cdpmod  # noqa: E402

PAGE = "https://ais-mac-studio.tail3b031e.ts.net:8443/"
import sys as _s; WAV = _s.argv[1] if len(_s.argv) > 1 else "/Users/ai/workspace/hermes-voice/tests/mob_chat_loop.wav"

SCAN_JS = """
(async () => {
  const unlock = await navigator.mediaDevices.getUserMedia({ audio: true });
  unlock.getTracks().forEach(t => t.stop());
  const devs = (await navigator.mediaDevices.enumerateDevices())
    .filter(d => d.kind === "audioinput");
  const out = [];
  for (const d of devs) {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: d.deviceId } } });
    } catch (e) { out.push({ label: d.label, error: String(e) }); continue; }
    const ctx = new AudioContext();
    const an = ctx.createAnalyser();
    ctx.createMediaStreamSource(stream).connect(an);
    const buf = new Uint8Array(2048);
    let peak = 0;
    for (let i = 0; i < 14; i++) {
      an.getByteTimeDomainData(buf);
      let sum = 0;
      for (let j = 0; j < buf.length; j++) { const v = (buf[j] - 128) / 128; sum += v * v; }
      peak = Math.max(peak, Math.sqrt(sum / buf.length));
      await new Promise(r => setTimeout(r, 450));
    }
    out.push({ label: d.label, peak: Math.round(peak * 1000) / 1000 });
    stream.getTracks().forEach(t => t.stop());
    await ctx.close();
  }
  return out;
})()
"""


async def main():
    chrome = cdpmod.Chrome(WAV, port=9342, profile="/tmp/hv-chrome-p5")
    try:
        await chrome.open_page(PAGE)
        res = await chrome.evaluate(SCAN_JS, await_promise=True)
        print("SCAN " + json.dumps(res), flush=True)
    finally:
        chrome.kill()


if __name__ == "__main__":
    asyncio.run(main())
