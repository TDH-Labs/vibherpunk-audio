#!/bin/zsh
export API_SERVER_KEY=$(grep "^API_SERVER_KEY" ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep "^OPENCODE_GO_API_KEY" ~/.hermes/.env | head -1 | cut -d= -f2-)
export HERMES_WAKE=off
export HERMES_VOICE_LOGDIR=$HOME/workspace/hermes-voice/logs/p1-acceptance1c
cd $HOME/workspace/hermes-voice
.venv/bin/python tests/bench_lane2.py 2>/dev/null | grep LANE2
.venv/bin/python tests/test_phase1c.py
echo P1B_EXIT=$?
