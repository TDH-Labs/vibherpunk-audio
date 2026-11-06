"""Minimal Chrome DevTools Protocol driver (aiohttp websockets) for the
mobile headless tests. Launches headless Chrome with a fake mic fed from a
WAV file, opens the voice client page, and exposes evaluate/poll helpers.
"""
import asyncio
import json
import subprocess
import sys
import time
import urllib.request

import aiohttp

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


class Chrome:
    def __init__(self, wav_path: str, port: int = 9333, profile: str = "/tmp/hv-chrome", mute: bool = True):
        self.port = port
        self.proc = subprocess.Popen(
            [
                CHROME,
                "--headless=new",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}",
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={wav_path}",
                "--autoplay-policy=no-user-gesture-required",
                *(["--mute-audio"] if mute else []),
                "--no-first-run",
                "--no-default-browser-check",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.ws = None
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task = None

    async def _wait_devtools(self, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/json/version", timeout=2
                ) as r:
                    return json.loads(r.read())
            except Exception:
                await asyncio.sleep(0.5)
        raise TimeoutError("chrome devtools did not come up")

    async def open_page(self, url: str):
        await self._wait_devtools()
        async with aiohttp.ClientSession() as s:
            async with s.put(
                f"http://127.0.0.1:{self.port}/json/new?{urllib.parse.quote(url, safe='')}"
            ) as r:
                target = await r.json()
        ws_url = target["webSocketDebuggerUrl"]
        session = aiohttp.ClientSession()
        self.ws = await session.ws_connect(ws_url)
        self._reader_task = asyncio.create_task(self._reader())
        await self.cmd("Page.enable")
        await self.cmd("Runtime.enable")
        # /json/new already navigates; give the page a moment to load.
        await asyncio.sleep(2.0)
        return target

    async def _reader(self):
        async for msg in self.ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            fut = self._pending.pop(data.get("id", -1), None)
            if fut and not fut.done():
                fut.set_result(data)

    async def cmd(self, method: str, **params):
        self._id += 1
        fut = asyncio.get_event_loop().create_future()
        self._pending[self._id] = fut
        await self.ws.send_json({"id": self._id, "method": method, "params": params})
        return await asyncio.wait_for(fut, timeout=30)

    async def evaluate(self, expr: str, await_promise: bool = False):
        res = await self.cmd(
            "Runtime.evaluate",
            expression=expr,
            returnByValue=True,
            awaitPromise=await_promise,
        )
        result = res.get("result", {}).get("result", {})
        if "exceptionDetails" in res.get("result", {}):
            raise RuntimeError(str(res["result"]["exceptionDetails"])[:400])
        return result.get("value")

    async def poll(self, expr: str, timeout: float, desc: str, interval=0.5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = await self.evaluate(expr)
            if v:
                return v
            await asyncio.sleep(interval)
        raise TimeoutError(f"poll timed out: {desc}")

    def kill(self):
        if self._reader_task:
            self._reader_task.cancel()
        try:
            self.proc.kill()
        except Exception:
            pass


import urllib.parse  # noqa: E402  (used in open_page)


def read_token() -> str:
    with open(os.path.expanduser("~/.hermes/mobile-voice-token")) as f:
        return f.read().strip()


import os  # noqa: E402


def session_events(base: str, token: str) -> dict:
    req = urllib.request.Request(
        f"{base}/api/session", headers={"x-voice-token": token}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


async def wait_event(base: str, token: str, kind: str, timeout: float, since: int = 0):
    """Poll /api/session until an event of `kind` appears at index >= since."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = session_events(base, token)
        evs = data.get("events", [])
        for i, e in enumerate(evs):
            if i >= since and e.get("kind") == kind:
                return data, i, e
        await asyncio.sleep(1.0)
    raise TimeoutError(f"event {kind} not seen within {timeout}s")


if __name__ == "__main__":
    print("cdp helper module — import it", file=sys.stderr)
