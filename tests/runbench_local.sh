#!/bin/zsh
# Routed-lane bench — LOCAL backend (full mode). Starts the MLX lane server
# with the chosen model, waits for health, runs the harness headless.
# Keys are extracted from ~/.hermes/.env at runtime; never printed.
MODEL="${1:-mlx-community/Qwen3-4B-Instruct-2507-4bit}"
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
cd /path/to/hermes-voice
pkill -f "mlx_lm server" 2>/dev/null; sleep 2
nohup .venv/bin/python -m mlx_lm server --model "$MODEL" --host 127.0.0.1 --port 8123 > /tmp/laneserver_bench.log 2>&1 &
for i in {1..60}; do
  if curl -s http://127.0.0.1:8123/v1/models 2>/dev/null | grep -qF "$MODEL"; then echo SERVER_UP; break; fi
  sleep 2
done
export HERMES_WAKE=off
export HERMES_LANE_BACKEND=local
export LOCAL_LANE_MODEL="$MODEL"
export BENCH_MODE=full
export HERMES_VOICE_LOGDIR=/path/to/hermes-voice/logs-routed-local
.venv/bin/python tests/test_routed_lane.py 2>/tmp/bench_local.err | grep ROUTED_JSON
echo BENCH_LOCAL_DONE
