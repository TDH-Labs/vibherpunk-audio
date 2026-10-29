#!/bin/zsh
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export HERMES_GATEWAY_URL=http://127.0.0.1:9999/v1
cd $HOME/workspace/hermes-voice
.venv/bin/python tests/test_pipeline.py "In one short sentence, what is the capital of France?" tests/out_deadport.wav
echo DEADPORT_DONE
