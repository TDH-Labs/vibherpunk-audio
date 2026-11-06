"""Pure-STT proof through the real (phone, not mic) pipeline.

Force-loads the mobile pipeline's PipelineTask but pipes the test WAV into
its transport.input() instead of a WebRTC track, then reports whether the
session recorded a transcription + lane reply. Settles "does the mobile
brain hear speech" despite headless Chrome's silent fake mic.
"""
import asyncio
import sys
import time
import wave

sys.path.insert(0, "/Users/ai/workspace/hermes-voice")
sys.path.insert(0, "/Users/ai/workspace/hermes-voice/tests")

import os

os.environ.setdefault("HERMES_WAKE", "off")
os.environ.setdefault("HERMES_VOICE_LOGDIR",
                      "/Users/ai/workspace/hermes-voice/logs-mobile")

import numpy as np  # noqa: E402
from pipecat.frames.frames import InputAudioRawFrame  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport  # noqa: E402

import mobile_server as ms  # noqa: E402
import hermes_voice as hv  # noqa: E402

WAV = "/Users/ai/workspace/hermes-voice/tests/mob_chat_loop.wav"


class _FakeConn:
    """Minimal stand-in for a SmallWebRTCConnection so the transport can be
    built without a real peer connection (it only needs the event API)."""

    def event_handler(self, *_a, **_k):
        return lambda fn: fn

    def add_event_handler(self, *_a, **_k):
        return None

    def __getattr__(self, _name):
        def _noop(*_a, **_k):
            return None

        return _noop


def pcm_16k_mono():
    w = wave.open(WAV)
    raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if w.getframerate() != 16000:
        n = int(len(a) * 16000 / w.getframerate())
        a = np.interp(np.arange(n) * (w.getframerate() / 16000), np.arange(len(a)), a)
    return (a * 32767).astype(np.int16).tobytes()


async def main():
    session = hv.SESSION
    # Begin a mobile session without a real peer connection: the pipeline's
    # input frame source is replaced by our WAV feed below.
    transport = SmallWebRTCTransport(
        webrtc_connection=_FakeConn(),
        params=ms.TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )
    sess = ms.MobileSession("synthetic-stt", None)
    sess.transport = transport
    task, bridge = ms.build_mobile_pipeline(transport, sess.id)
    sess.task, sess.bridge = task, bridge
    ms.MANAGER["current"] = sess
    session.record("mobile_session_start", session_id=sess.id)

    runner = PipelineRunner(handle_sigint=False)
    run_task = asyncio.create_task(runner.run(task))
    await asyncio.sleep(6)  # let MLX whisper + TTS warm up

    pcm = pcm_16k_mono()
    chunk = 16000 * 2 * 20 // 1000  # 20 ms of 16-bit mono
    for i in range(0, len(pcm), chunk):
        frame = InputAudioRawFrame(audio=pcm[i:i + chunk], sample_rate=16000, num_channels=1)
        await task.queue_frame(frame)
        await asyncio.sleep(0.02)
    print("AUDIO_PUSHED", len(pcm), flush=True)

    deadline = time.time() + 90
    heard, spoken = None, None
    while time.time() < deadline:
        await asyncio.sleep(2)
        evs = [e for e in session.events if e["kind"] == "transcription"]
        if evs:
            heard = evs[-1].get("text")
        if session.spoken:
            spoken = [s.get("text") for s in session.spoken]
        if spoken:
            break
    print("TRANSCRIPTION", repr(heard), flush=True)
    print("LANE_SPOKEN", repr(spoken), flush=True)
    print("TURNS", session.turns[-2:] if session.turns else None, flush=True)

    await task.cancel()
    try:
        await asyncio.wait_for(run_task, timeout=15)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
