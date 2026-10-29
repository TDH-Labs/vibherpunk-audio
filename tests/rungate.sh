#!/bin/zsh
export API_SERVER_KEY=$(grep "^API_SERVER_KEY" ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep "^OPENCODE_GO_API_KEY" ~/.hermes/.env | head -1 | cut -d= -f2-)
export HERMES_FOLLOWUP_SECS=6
export HERMES_VOICE_LOGDIR=$HOME/workspace/hermes-voice/logs/p1-gate
cd $HOME/workspace/hermes-voice
.venv/bin/python tests/test_gate.py
echo GATE_EXIT=$?
