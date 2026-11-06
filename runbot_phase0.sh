#!/bin/zsh
# Vibherpunk Audio — Phase 0 single-tier entry point (wired to Antigravity bridge).
VOICE_ENV="${HOME}/.gemini/antigravity/voice/.env"
if [ ! -f "$VOICE_ENV" ]; then
  VOICE_ENV="${HOME}/.hermes/profiles/voice/.env"
fi
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' "$VOICE_ENV" | head -1 | cut -d= -f2-)
export AGY_GATEWAY_URL="${AGY_GATEWAY_URL:-http://127.0.0.1:8011/v1}"
export HERMES_GATEWAY_URL="$AGY_GATEWAY_URL"
cd /Users/ai/workspace/hermes-voice
exec .venv/bin/python hermes_voice_phase0.py
