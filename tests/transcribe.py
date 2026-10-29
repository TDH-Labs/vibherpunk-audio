"""Transcribe a captured audio file with local faster-whisper (small).
Usage: transcribe.py <audio-file> — prints the transcript."""
import subprocess
import sys
import tempfile

from faster_whisper import WhisperModel


def main():
    src = sys.argv[1]
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", "16000", "-ac", "1", wav],
        check=True,
    )
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(wav)
    print(" ".join(s.text.strip() for s in segments))


if __name__ == "__main__":
    main()
