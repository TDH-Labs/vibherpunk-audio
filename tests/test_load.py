"""Headless load + TTFB proof: Pocket TTS (HF gate check) and faster-whisper STT round-trip."""
import time, sys, os

t0 = time.time()
print("== pocket_tts load ==", flush=True)
try:
    from pocket_tts import TTSModel
    t = time.time()
    model = TTSModel.load_model()
    print(f"POCKET_LOAD_OK in {time.time()-t:.1f}s sample_rate={model.sample_rate}", flush=True)
except Exception as e:
    print(f"POCKET_LOAD_FAIL {type(e).__name__}: {str(e)[:400]}", flush=True)
    sys.exit(2)

print("== voice state (catalog name 'eve') ==", flush=True)
t = time.time()
state = model.get_state_for_audio_prompt("eve")
print(f"VOICE_STATE_OK (catalog) in {time.time()-t:.2f}s", flush=True)

TEXT = "Hello. This is Hermes speaking through your local voice now."
print("== streaming generate ==", flush=True)
t = time.time()
first = None
chunks = []
for chunk in model.generate_audio_stream(state, TEXT):
    if first is None:
        first = time.time() - t
    chunks.append(chunk)
total = time.time() - t
import torch
audio = torch.cat(chunks) if chunks else torch.zeros(1)
dur = audio.shape[-1] / model.sample_rate
print(f"STREAM_OK ttfb={first*1000:.0f}ms total_gen={total:.2f}s audio_dur={dur:.2f}s rtf={total/dur:.2f}", flush=True)

import scipy.io.wavfile
os.makedirs("/path/to/hermes-voice/tests", exist_ok=True)
scipy.io.wavfile.write("/path/to/hermes-voice/tests/known.wav", model.sample_rate, audio.numpy())
print("WAV_SAVED", flush=True)

print("== faster-whisper transcribe known.wav ==", flush=True)
from faster_whisper import WhisperModel
t = time.time()
wm = WhisperModel("base", device="cpu", compute_type="int8")
print(f"WHISPER_LOAD_OK in {time.time()-t:.1f}s", flush=True)
import numpy as np
sr_w, data_w = scipy.io.wavfile.read("/path/to/hermes-voice/tests/known.wav")
audio_in = (data_w.astype(np.float32) / 32768.0) if data_w.dtype == np.int16 else data_w.astype(np.float32)
print(f"whisper input: sr={sr_w} samples={len(audio_in)} (numpy path, bypasses PyAV)", flush=True)
t = time.time()
segments, info = wm.transcribe(audio_in, language="en")
text = "".join(s.text for s in segments).strip()
print(f"TRANSCRIPT in {time.time()-t:.2f}s: {text!r}", flush=True)
print(f"EXPECTED: {TEXT!r}", flush=True)
print(f"TOTAL_WALL {time.time()-t0:.1f}s", flush=True)
print("LOAD_TEST_DONE", flush=True)
