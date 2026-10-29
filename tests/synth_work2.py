import scipy.io.wavfile
from pocket_tts import TTSModel
m = TTSModel.load_model()
st = m.get_state_for_audio_prompt("george")
text = "Please have Hermes read the first line of its SOUL file and tell me the exact words it finds there. Take your time. I want the real text from Hermes itself."
a = m.generate_audio(st, text)
pcm = (a.numpy().clip(-1, 1) * 32767).astype("int16")
scipy.io.wavfile.write("tests/utt_work2.wav", m.sample_rate, pcm)
print("SYNTH work2", len(pcm) / m.sample_rate)
