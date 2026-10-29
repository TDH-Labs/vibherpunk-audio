#!/bin/zsh
# Hermes live voice — Phase 1 two-tier front-end (daily driver entry point).
# Exports the keys the bot needs from the Hermes .env files (values are
# never printed), then runs the bot from the project venv.
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export HERMES_VOICE=azelma
export HERMES_WAKE_ENGINE=sherpa
cd $HOME/workspace/hermes-voice
exec .venv/bin/python hermes_voice.py
