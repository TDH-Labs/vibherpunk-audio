"""Live handshake with the voice server exactly as Murmur sends it.

Murmur cannot reach the Pipecat server cross-origin (no CORS, OPTIONS -> 405),
so this probe reproduces the two things that matter on the wire:
  * the browser-ish Origin/Referer headers Murmur's same-origin relay forwards;
  * the offer body built by src/features/voice/desk-binding.ts
    ({sdp, type, session_id, profile}).

Usage:
    .venv/bin/python tests/offer_wire_probe.py <base> <session_id> <profile>
"""
import json
import sys
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import asyncio  # noqa: E402

from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import AudioStreamTrack  # noqa: E402

from rtc_client import read_token  # noqa: E402


async def main():
    base = sys.argv[1]
    session_id = sys.argv[2] if len(sys.argv) > 2 else ""
    profile = sys.argv[3] if len(sys.argv) > 3 else "default"
    token = read_token()

    pc = RTCPeerConnection()
    pc.addTrack(AudioStreamTrack())
    pc.createDataChannel("pipecat")
    await pc.setLocalDescription(await pc.createOffer())
    for _ in range(50):
        if pc.iceGatheringState == "complete":
            break
        await asyncio.sleep(0.1)

    # Same shape desk-binding.ts emits: no session -> SDP only.
    body = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    if session_id:
        body["session_id"] = session_id
        if profile:
            body["profile"] = profile

    req = urllib.request.Request(
        f"{base}/api/offer?token={token}",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            # What the browser sends and the Murmur relay forwards.
            "Origin": "http://localhost:3000",
            "Referer": "http://localhost:3000/",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            answer = json.loads(r.read())
        print(json.dumps({"status": 200, "pc_id": bool(answer.get("pc_id")), "type": answer.get("type")}))
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
        for _ in range(120):
            if pc.connectionState in ("connected", "failed", "closed"):
                break
            await asyncio.sleep(0.25)
        print(json.dumps({"connection_state": pc.connectionState}))
    finally:
        await pc.close()


if __name__ == "__main__":
    asyncio.run(main())
