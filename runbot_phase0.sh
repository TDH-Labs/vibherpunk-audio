#!/bin/zsh
# Phase 0 bot (single-tier, gateway voice profile) — kept runnable for
# comparison. Its gateway route is the trimmed "voice" profile, whose key
# lives in ~/.hermes/profiles/voice/.env.
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/profiles/voice/.env | head -1 | cut -d= -f2-)
cd $HOME/workspace/hermes-voice
exec .venv/bin/python hermes_voice_phase0.py
