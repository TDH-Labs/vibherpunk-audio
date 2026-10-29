import time
import numpy as np, scipy.io.wavfile
from faster_whisper import WhisperModel
sr, data = scipy.io.wavfile.read("/path/to/hermes-voice/tests/known.wav")
audio = (data.astype(np.float32) / 32768.0) if data.dtype == np.int16 else data.astype(np.float32)
t = time.time()
wm = WhisperModel("small", device="cpu", compute_type="int8")
print(f"SMALL_LOAD {time.time()-t:.1f}s", flush=True)
t = time.time()
segments, info = wm.transcribe(audio, language="en")
text = "".join(s.text for s in segments).strip()
print(f"SMALL_TRANSCRIPT in {time.time()-t:.2f}s: {text!r}", flush=True)
print("SMALL_DONE", flush=True)
