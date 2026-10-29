#!/bin/zsh
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
cd $HOME/workspace/hermes-voice
.venv/bin/python tests/probe_tools.py
echo PROBE_SCRIPT_DONE
