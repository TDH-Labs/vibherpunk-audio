"""Synthesize the sleep-command utterances (Pocket george, as synth_utts)."""
import scipy.io.wavfile
from pocket_tts import TTSModel

UTTS = {
    "utt_sleep": "Go to sleep.",
    "utt_sleep2": "Stop listening.",
}

model = TTSModel.load_model()
state = model.get_state_for_audio_prompt("george")
for name, text in UTTS.items():
    audio = model.generate_audio(state, text)
    arr = audio.numpy()
    pcm = (arr.clip(-1, 1) * 32767).astype("int16")
    scipy.io.wavfile.write(f"tests/{name}.wav", model.sample_rate, pcm)
    print(f"SYNTH {name}: {len(pcm)/model.sample_rate:.2f}s", flush=True)
print("SYNTH_SLEEP_DONE", flush=True)
