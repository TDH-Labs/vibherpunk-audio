#!/bin/zsh
# Routed-lane bench — LOCAL backend (Qwen3-4B MLX) + ENCODER routing + FUSED lookup.
# Local-fused round (2026-10-28). Per-run isolation via HERMES_VOICE_LOGDIR.
MODEL="mlx-community/Qwen3-4B-Instruct-2507-4bit"
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
cd "$(dirname "$0")/.."
if ! curl -s http://127.0.0.1:8123/v1/models 2>/dev/null | grep -qF "$MODEL"; then
  pkill -f "mlx_lm server" 2>/dev/null; sleep 2
  nohup .venv/bin/python -m mlx_lm server --model "$MODEL" --host 127.0.0.1 --port 8123 > /tmp/laneserver_bench_fused.log 2>&1 &
  for i in {1..60}; do
    if curl -s http://127.0.0.1:8123/v1/models 2>/dev/null | grep -qF "$MODEL"; then echo SERVER_UP; break; fi
    sleep 2
  done
fi
export HERMES_WAKE=off
export HERMES_LANE_BACKEND=local
export HERMES_ROUTING=encoder
export HERMES_FUSED_LOOKUP=on
export LOCAL_LANE_MODEL="$MODEL"
export BENCH_MODE="${BENCH_MODE:-full}"
export HERMES_VOICE_LOGDIR="${HERMES_VOICE_LOGDIR:-./logs-routed-local-fused}"
.venv/bin/python tests/test_routed_lane.py 2>"${HERMES_VOICE_LOGDIR}.err" | grep ROUTED_JSON > "${HERMES_VOICE_LOGDIR}.json"
echo BENCH_LOCAL_FUSED_DONE
