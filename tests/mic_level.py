import asyncio
import json
import sys

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
from cdp import Chrome  # noqa: E402

WAV = sys.argv[1] if len(sys.argv) > 1 else "/Users/ai/workspace/hermes-voice/tests/mob_chat_loop_48k_stereo.wav"

JS = """
(async () => {
  const s = await navigator.mediaDevices.getUserMedia({audio:true});
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(s);
  const an = ctx.createAnalyser(); an.fftSize = 2048; src.connect(an);
  let peak = 0, sum = 0, n = 0; const buf = new Float32Array(2048);
  for (let i = 0; i < 40; i++) {
    await new Promise(r => setTimeout(r, 100));
    an.getFloatTimeDomainData(buf);
    for (const v of buf) { const a = Math.abs(v); if (a > peak) peak = a; sum += a; n++; }
  }
  return { peak, mean: sum / n, label: s.getAudioTracks()[0].label };
})()
"""


async def main():
    c = Chrome(WAV, port=9342, profile="/tmp/hv-chrome-lvl4", mute=False)
    try:
        await c.open_page("https://ais-mac-studio.tail3b031e.ts.net:8443/healthz")
        print("LEVEL", json.dumps(await c.evaluate(JS, await_promise=True), default=str),
              flush=True)
    finally:
        c.kill()


asyncio.run(main())
