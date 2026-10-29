#!/bin/zsh
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
cd $HOME/workspace/hermes-voice
.venv/bin/python tests/test_warm.py
echo WARM_SCRIPT_DONE
