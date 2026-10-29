#!/bin/zsh
# Barge-in regression x3 for one backend: runbarge_routed.sh relay|local
BACKEND="${1:-local}"
MODEL="mlx-community/Qwen3-4B-Instruct-2507-4bit"
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
cd /path/to/hermes-voice
if [ "$BACKEND" = "local" ]; then
  if ! curl -s http://127.0.0.1:8123/v1/models 2>/dev/null | grep -qF "$MODEL"; then
    pkill -f "mlx_lm server" 2>/dev/null; sleep 2
    nohup .venv/bin/python -m mlx_lm server --model "$MODEL" --host 127.0.0.1 --port 8123 > /tmp/laneserver_barge.log 2>&1 &
    for i in {1..60}; do
      if curl -s http://127.0.0.1:8123/v1/models 2>/dev/null | grep -qF "$MODEL"; then echo SERVER_UP; break; fi
      sleep 2
    done
  fi
fi
export HERMES_WAKE=off
export HERMES_LANE_BACKEND="$BACKEND"
export LOCAL_LANE_MODEL="$MODEL"
export HERMES_VOICE_LOGDIR=/path/to/hermes-voice/logs-routed-barge-$BACKEND
for run in 1 2 3; do
  .venv/bin/python tests/test_bargein_routed.py 2>/dev/null | grep BARGE_JSON
done
echo BARGE_ROUTED_DONE
