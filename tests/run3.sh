#!/bin/zsh
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$HOME/.bun/bin:$HOME/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin
cd $HOME/workspace/hermes-voice
.venv/bin/python tests/test_pipeline.py "In one short sentence, what is the capital of France?" tests/out1.wav
.venv/bin/python tests/test_pipeline.py "Tell me one short interesting fact about the moon." tests/out2.wav
.venv/bin/python tests/test_pipeline.py "In one short sentence, how many legs does a spider have?" tests/out3.wav
echo RUN3_DONE
