#!/usr/bin/env python3
"""Offline test of TrailingSilenceTrimmer: real Pipecat frame flow via
pipecat.tests.utils.run_test, no service, no restart.
Case 1: speech + mid pause + speech + trailing silence + stop
        -> mid pause preserved, trailing silence dropped.
Case 2: speech + long (0.9 s) pause + speech + stop
        -> long pause fully preserved (hold window only eats the tail)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "workspace/hermes-voice"))

from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame  # noqa: E402
from pipecat.tests.utils import run_test  # noqa: E402

from hermes_voice import TrailingSilenceTrimmer  # noqa: E402

SR = 24000
CHUNK = int(SR * 0.1)  # 100 ms frames


def loud(n_chunks):
    return [
        TTSAudioRawFrame(
            audio=(8000).to_bytes(2, "little", signed=True) * CHUNK,
            sample_rate=SR,
            num_channels=1,
        )
        for _ in range(n_chunks)
    ]


def silent(n_chunks):
    return [
        TTSAudioRawFrame(
            audio=b"\x01\x00" * CHUNK, sample_rate=SR, num_channels=1
        )
        for _ in range(n_chunks)
    ]


def total_ms(frames):
    return sum(
        len(f.audio) // 2 / f.sample_rate * 1000
        for f in frames
        if isinstance(f, TTSAudioRawFrame)
    )


async def main():
    # Case 1: 0.3 loud, 0.3 silence, 0.2 loud, 0.4 silence, stop -> 0.8 s out
    frames = loud(3) + silent(3) + loud(2) + silent(4) + [TTSStoppedFrame()]
    down, _ = await run_test(TrailingSilenceTrimmer(), frames_to_send=frames)
    got = total_ms(down)
    assert abs(got - 800) < 1, f"case1 expected 800 ms, got {got}"
    assert any(isinstance(f, TTSStoppedFrame) for f in down), "stop frame lost"
    print(f"case1 PASS: {got:.0f} ms out (trailing 400 ms dropped, mid pause kept)")

    # Case 2: 0.2 loud, 0.9 silence, 0.2 loud, stop -> 1.3 s out (all kept)
    frames = loud(2) + silent(9) + loud(2) + [TTSStoppedFrame()]
    down, _ = await run_test(TrailingSilenceTrimmer(), frames_to_send=frames)
    got = total_ms(down)
    assert abs(got - 1300) < 1, f"case2 expected 1300 ms, got {got}"
    print(f"case2 PASS: {got:.0f} ms out (long mid pause fully preserved)")


if __name__ == "__main__":
    asyncio.run(main())
