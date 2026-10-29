"""Synthesize the routed-lane bench utterances with Pocket TTS (voice:
george — deliberately NOT the bot's voice), same pattern as synth_utts.py."""
import os

import scipy.io.wavfile
from pocket_tts import TTSModel

UTTS = {
    # -- chat bench 6..20 (simple general knowledge, direct answers) --
    "utt_chat6": "What is the capital of Japan?",
    "utt_chat7": "How many continents are there on Earth?",
    "utt_chat8": "What is the largest planet in our solar system?",
    "utt_chat9": "How many minutes are there in a full day?",
    "utt_chat10": "What is the boiling point of water in degrees Celsius?",
    "utt_chat11": "Who wrote the play Romeo and Juliet?",
    "utt_chat12": "What is the fastest land animal?",
    "utt_chat13": "How many sides does a hexagon have?",
    "utt_chat14": "What gas do plants absorb from the air?",
    "utt_chat15": "What is the square root of one hundred forty four?",
    "utt_chat16": "Which ocean is the largest on Earth?",
    "utt_chat17": "What is the main ingredient in guacamole?",
    "utt_chat18": "How many keys does a standard piano have?",
    "utt_chat19": "What is the currency of the United Kingdom?",
    "utt_chat20": "In what year did humans first land on the moon?",
    # -- discipline: lookup hits (facts stored in the profile memory) --
    "utt_lk_city": "What city do I live in?",
    "utt_lk_tz": "What time zone am I in?",
    "utt_lk_mia": "Who is Alex?",
    "utt_lk_brain": "Where does my second brain live?",
    # -- discipline: lookup misses (NOT stored — must escalate, never invent) --
    "utt_lk_wife": "What is my wife's name?",
    "utt_lk_blood": "What is my blood type?",
    # -- discipline: delegation-shaped work --
    "utt_wk_research": (
        "Please have Hermes research the weather in Portland this weekend "
        "and summarize it for me. Take your time."
    ),
    "utt_wk_email": (
        "Please have Hermes check my email for anything urgent from this "
        "morning and summarize it. Take your time."
    ),
    # -- in-flow while a job pends --
    "utt_side2": "Quick question while that runs: how many days are in a leap year?",
    "utt_bait2": "It's probably going to be sunny in Portland, right? Just tell me what it found.",
    "utt_check": "Is that done yet?",
    # -- barge-in: long-form request --
    "utt_story": (
        "Tell me a long story about a dragon and a lighthouse keeper. "
        "At least ten sentences, please."
    ),
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
