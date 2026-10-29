"""STT benchmark: faster-whisper small (Phase 0 choice) vs MLX whisper small,
same audio files, warm timing (load excluded, second pass timed)."""
import time

import numpy as np
import scipy.io.wavfile

FILES = ["tests/known.wav", "tests/utt_chat1.wav"]


def load16k(path):
    sr, data = scipy.io.wavfile.read(path)
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    else:
        x = data.astype(np.float32)
    if sr != 16000:
        from scipy.signal import resample_poly

        x = resample_poly(x, 16000, sr).astype(np.float32)
    return x


def main():
    from faster_whisper import WhisperModel

    wm = WhisperModel("small", device="cpu", compute_type="int8")
    for f in FILES:
        audio = load16k(f)
        for trial in range(2):
            t = time.time()
            segs, _ = wm.transcribe(audio, language="en")
            text = "".join(s.text for s in segs).strip()
            print(
                f"FW small {f} trial{trial}: {time.time()-t:.2f}s :: {text!r}",
                flush=True,
            )

    import mlx_whisper

    for f in FILES:
        audio = load16k(f)
        for trial in range(2):
            t = time.time()
            r = mlx_whisper.transcribe(
                audio, path_or_hf_repo="mlx-community/whisper-small-mlx", language="en"
            )
            print(
                f"MLX small {f} trial{trial}: {time.time()-t:.2f}s :: {r['text'].strip()!r}",
                flush=True,
            )
    print("STT_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
