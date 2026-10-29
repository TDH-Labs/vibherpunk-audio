#!/bin/zsh
# Vibherpunk Audio — desk bot entry point (template).
# Keys are read from ~/.hermes/.env at runtime and never printed.
# Substitute {{APP_DIR}} and {{VOICE}} before installing as run.sh
# in the app directory. (The reference build inlines its OpenCode key
# in this script — do NOT copy that pattern; one source: the .env.)
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export HERMES_VOICE={{VOICE}}
export HERMES_WAKE_ENGINE=sherpa
cd {{APP_DIR}}
exec .venv/bin/python hermes_voice.py
