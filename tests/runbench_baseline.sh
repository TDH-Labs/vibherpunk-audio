#!/bin/zsh
# Routed-lane bench — RELAY baseline (chat + ack only, same harness/method).
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
cd /path/to/hermes-voice
export HERMES_WAKE=off
export HERMES_LANE_BACKEND=relay
export BENCH_MODE=baseline
export HERMES_VOICE_LOGDIR=/path/to/hermes-voice/logs-routed-baseline
.venv/bin/python tests/test_routed_lane.py 2>/tmp/bench_baseline.err | grep ROUTED_JSON
echo BENCH_BASELINE_DONE
