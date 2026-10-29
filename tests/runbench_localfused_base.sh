#!/bin/zsh
# Same-window RELAY baseline for the local-fused round (chat + ack only,
# identical harness/method), own logdir.
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
cd "$(dirname "$0")/.."
export HERMES_WAKE=off
export HERMES_LANE_BACKEND=relay
export HERMES_FUSED_LOOKUP=off
export BENCH_MODE=baseline
export HERMES_VOICE_LOGDIR=./logs-routed-localfused-base
.venv/bin/python tests/test_routed_lane.py 2>"${HERMES_VOICE_LOGDIR}.err" | grep ROUTED_JSON > "${HERMES_VOICE_LOGDIR}.json"
echo BENCH_BASE_DONE
