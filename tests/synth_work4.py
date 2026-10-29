import scipy.io.wavfile
from pocket_tts import TTSModel
m = TTSModel.load_model()
st = m.get_state_for_audio_prompt("george")
text = "Please have Hermes check what operating system version this Mac is running, using its tools. Take your time. I want the real answer from Hermes itself."
a = m.generate_audio(st, text)
pcm = (a.numpy().clip(-1, 1) * 32767).astype("int16")
scipy.io.wavfile.write("tests/utt_work4.wav", m.sample_rate, pcm)
print("SYNTH work4", len(pcm) / m.sample_rate)
