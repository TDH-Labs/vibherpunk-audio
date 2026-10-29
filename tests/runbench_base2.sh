#!/bin/zsh
# Same-window RELAY baseline (chat + ack only, identical harness/method),
# own logdir so round 1's baseline record is not polluted.
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
cd "$(dirname "$0")/.."
export HERMES_WAKE=off
export HERMES_LANE_BACKEND=relay
export BENCH_MODE=baseline
export HERMES_VOICE_LOGDIR=./logs-routed-cloud-base
.venv/bin/python tests/test_routed_lane.py 2>"${HERMES_VOICE_LOGDIR}.err" | grep ROUTED_JSON > "${HERMES_VOICE_LOGDIR}.json"
echo BENCH_BASE_DONE
