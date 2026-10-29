#!/bin/zsh
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
cd $HOME/workspace/hermes-voice
echo "=== install start $(date) ===" >> logs/install.log
if brew list --versions portaudio >/dev/null 2>&1; then
  echo "portaudio present: $(brew list --versions portaudio)" >> logs/install.log
else
  echo "portaudio absent - installing" >> logs/install.log
  brew install portaudio >> logs/install.log 2>&1
  echo "brew portaudio exit: $?" >> logs/install.log
fi
uv venv --python 3.12 .venv >> logs/install.log 2>&1
echo "venv exit: $?" >> logs/install.log
uv pip install --python .venv/bin/python "pipecat-ai[pocket-tts,whisper,silero,local-smart-turn-v3,openai]" pyaudio >> logs/install.log 2>&1
echo "pip exit: $?" >> logs/install.log
.venv/bin/python -c "import pipecat; print('pipecat version:', pipecat.__version__)" >> logs/install.log 2>&1
echo "=== INSTALL_DONE $(date) ===" >> logs/install.log
