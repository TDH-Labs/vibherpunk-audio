#!/bin/zsh
export API_SERVER_KEY=$(grep "^API_SERVER_KEY" ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep "^OPENCODE_GO_API_KEY" ~/.hermes/.env | head -1 | cut -d= -f2-)
export HERMES_WAKE=off
cd $HOME/workspace/hermes-voice
if [ "$MODE" = "worker" ]; then
  export HERMES_WORKER_URL=http://127.0.0.1:9/v1
  export HERMES_VOICE_LOGDIR=$HOME/workspace/hermes-voice/logs/p1-down-worker
else
  export LANE_BASE_URL=http://127.0.0.1:9/v1
  export HERMES_VOICE_LOGDIR=$HOME/workspace/hermes-voice/logs/p1-down-lane
fi
.venv/bin/python tests/test_down.py
echo DOWN_EXIT=$?
