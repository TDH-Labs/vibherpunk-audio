#!/bin/zsh
# Hermes live voice — MOBILE transport server entry point.
# Same key discipline as run.sh: values are read from local files at runtime
# and never printed. The mobile access token lives in
# ~/.hermes/mobile-voice-token (created 0600 on first run if absent).
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
if [ -z "$OPENCODE_GO_API_KEY" ]; then
  # Fallback: the desk run.sh carries it (same local-only discipline).
  export OPENCODE_GO_API_KEY=$(grep '^export OPENCODE_GO_API_KEY=' $HOME/workspace/hermes-voice/run.sh | head -1 | cut -d= -f2-)
fi
TOKEN_FILE=~/.hermes/mobile-voice-token
if [ ! -f "$TOKEN_FILE" ]; then
  umask 077
  head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"
export MOBILE_ACCESS_TOKEN=$(cat "$TOKEN_FILE")
export HERMES_VOICE=azelma
cd $HOME/workspace/hermes-voice
exec .venv/bin/python mobile_server.py
