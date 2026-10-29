#!/bin/zsh
export OPENCODE_GO_API_KEY=$(grep "^OPENCODE_GO_API_KEY" ~/.hermes/.env | head -1 | cut -d= -f2-)
cd $HOME/workspace/hermes-voice
for M in qwen3.8-flash glm-5.3-flash mimo-v2.6-flash claude-haiku-5-5 minimax-m3.1-flash; do
  echo "MODEL $M"
  LANE_MODEL=$M LANE_SESSION="hermes-voice-lane-bench-$M" .venv/bin/python tests/bench_lane.py 2>/dev/null | grep LANE_SUMMARY
done
echo BENCH2_DONE
