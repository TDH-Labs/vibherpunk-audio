#!/bin/zsh
# Routed-lane bench — CLOUD backend (GLM-5.3-Flash via Zen) + ENCODER routing.
# Per-run isolation via HERMES_VOICE_LOGDIR (metrics + ROUTED_JSON land there).
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
cd "$(dirname "$0")/.."
export HERMES_WAKE=off
export HERMES_LANE_BACKEND=cloud
export HERMES_ROUTING=encoder
export BENCH_MODE="${BENCH_MODE:-full}"
export HERMES_VOICE_LOGDIR="${HERMES_VOICE_LOGDIR:-./logs-routed-cloud}"
.venv/bin/python tests/test_routed_lane.py 2>"${HERMES_VOICE_LOGDIR}.err" | grep ROUTED_JSON > "${HERMES_VOICE_LOGDIR}.json"
echo BENCH_CLOUD_DONE
