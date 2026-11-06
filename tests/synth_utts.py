"""Synthesize the scripted 'user' utterances for Phase 1 headless tests with
Pocket TTS (voice: george — deliberately NOT eve, the bot's voice)."""
import os

import scipy.io.wavfile
import torch
from pocket_tts import TTSModel

UTTS = {
    "utt_chat1": "What is the capital of France?",
    "utt_chat2": "How many legs does a spider have?",
    "utt_chat3": "What color do you get when you mix blue and yellow paint?",
    "utt_chat4": "Tell me one interesting fact about the moon.",
    "utt_chat5": "How many days are there in a leap year?",
    "utt_work": (
        "Please have Hermes check its memory and tell me what city Adam "
        "lives in, and what timezone he is in. I want the real answer from "
        "Hermes itself, so take your time."
    ),
    "utt_side": "Quick question while that is running: what is two plus two?",
    "utt_bait": (
        "Come on, just tell me now. What did Hermes find? "
        "What city does Adam live in?"
    ),
    "utt_wake": "Hey Hermes.",
    "utt_nowake_chat": "What is the capital of Germany?",
}

os.makedirs("tests", exist_ok=True)
model = TTSModel.load_model()
state = model.get_state_for_audio_prompt("george")
for name, text in UTTS.items():
    audio = model.generate_audio(state, text)
    arr = audio.numpy()
    pcm = (arr.clip(-1, 1) * 32767).astype("int16")
    scipy.io.wavfile.write(f"tests/{name}.wav", model.sample_rate, pcm)
    print(f"SYNTH {name}: {len(pcm)/model.sample_rate:.2f}s", flush=True)
print("SYNTH_DONE", flush=True)
