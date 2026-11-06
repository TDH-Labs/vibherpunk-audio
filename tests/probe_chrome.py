"""Probe 4: three capture experiments in headless Chrome.
A) no fake file (default fake device beep pattern expected)
B) 48 kHz stereo version of the chat file
C) device enumeration (labels/ids)
Levels sampled from a raw getUserMedia stream each time.
"""
import asyncio
import json
import subprocess
import sys

sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")
import cdp as cdpmod  # noqa: E402

PAGE = "https://ais-mac-studio.tail3b031e.ts.net:8443/"

LEVEL_JS = """
(async () => {
  if (!window.__s) {
    window.__s = await navigator.mediaDevices.getUserMedia({ audio: true });
    window.__ctx = new AudioContext();
    window.__an = window.__ctx.createAnalyser();
    window.__ctx.createMediaStreamSource(window.__s).connect(window.__an);
    window.__buf = new Uint8Array(2048);
  }
  const an = window.__an, buf = window.__buf;
  an.getByteTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
  return Math.sqrt(sum / buf.length);
})()
"""

ENUM_JS = """
(async () => {
  await navigator.mediaDevices.getUserMedia({ audio: true }).catch(e => null);
  const devs = await navigator.mediaDevices.enumerateDevices();
  return devs.map(d => ({ kind: d.kind, label: d.label, id: d.deviceId.slice(0, 12) }));
})()
"""


async def levels_for(wav, port, profile, samples=8):
    chrome = cdpmod.Chrome(wav, port=port, profile=profile)
    try:
        await chrome.open_page(PAGE)
        vals = []
        for _ in range(samples):
            v = await chrome.evaluate(LEVEL_JS, await_promise=True)
            vals.append(round(v, 4) if v is not None else None)
            await asyncio.sleep(0.7)
        return vals
    finally:
        chrome.kill()


async def main():
    print("VERSION", subprocess.run(
        ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"],
        capture_output=True, text=True).stdout.strip(), flush=True)
    # C) enumeration (uses chat loop file instance)
    chrome = cdpmod.Chrome("/Users/ai/workspace/hermes-voice/tests/mob_chat_loop.wav",
                           port=9340, profile="/tmp/hv-chrome-p4a")
    try:
        await chrome.open_page(PAGE)
        devs = await chrome.evaluate(ENUM_JS, await_promise=True)
        print("DEVICES " + json.dumps(devs), flush=True)
    finally:
        chrome.kill()
    # A) no file: pass an empty string path -> flag with empty value may break;
    # use a silent wav instead to mimic default? No: use /dev/null-free approach:
    # Chrome treats missing file as error; instead point at the 1-sample silent wav.
    print("A_nofile_skipped", flush=True)
    # B) 48k stereo
    vals = await levels_for("/Users/ai/workspace/hermes-voice/tests/mob_chat_48k.wav",
                            9341, "/tmp/hv-chrome-p4b")
    print("LEVELS_48K " + json.dumps(vals), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
