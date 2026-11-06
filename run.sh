#!/bin/zsh
# Vibherpunk Audio (formerly hermes-voice) — Phase 1 two-tier front-end (daily driver entry point).
# Wired to the Open Design API (https://amr-link.open-design.ai/v1), lane = deepseek-v4.1-flash, reasoning off. Worker stays on the Hermes API server.
# Exports the keys the bot needs from ~/.gemini/antigravity/voice/.env (values are
# never printed), then runs the bot from the project venv.
VOICE_ENV="${HOME}/.gemini/antigravity/voice/.env"
if [ ! -f "$VOICE_ENV" ]; then
  VOICE_ENV="${HOME}/.hermes/.env"
fi
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' "$VOICE_ENV" | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' "$VOICE_ENV" | head -1 | cut -d= -f2-)
export OPENDESIGN_API_KEY=$(grep '^CUSTOM_PROVIDER_OPENDESIGN_KEY=' "${HOME}/.hermes/.env" | head -1 | cut -d= -f2- | tr -d "\"'")
export LANE_API_KEY="${LANE_API_KEY:-$OPENDESIGN_API_KEY}"
export LANE_BASE_URL="${LANE_BASE_URL:-https://amr-link.open-design.ai/v1}"
export LANE_MODEL="${LANE_MODEL:-deepseek-v4.1-flash}"
export LANE_REASONING_EFFORT="${LANE_REASONING_EFFORT:-none}"
export AGY_WORKER_URL="${AGY_WORKER_URL:-http://127.0.0.1:8642/v1}"
export AGY_WORKER_MODEL="${AGY_WORKER_MODEL:-claude-sonnet-5-5}"
export HERMES_WORKER_URL="$AGY_WORKER_URL"
export HERMES_WORKER_MODEL="$AGY_WORKER_MODEL"
export HERMES_VOICE=/Users/ai/workspace/hermes-voice/voices/aussie-clone-v2.wav
export HERMES_TTS_TEMP=0.9
export HERMES_WAKE_ENGINE=sherpa
cd /Users/ai/workspace/hermes-voice
exec .venv/bin/python hermes_voice.py
