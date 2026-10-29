#!/bin/zsh
# Vibherpunk Audio — mobile transport entry point (template).
# Same key discipline as run.sh. The mobile access token lives in
# ~/.hermes/mobile-voice-token and is created (0600) on first run if
# absent: 24 random bytes as 48 hex characters. It is never printed.
# Substitute {{APP_DIR}}, {{VOICE}}, and optionally {{TURN_HOST_IP}}
# (this machine's Tailscale IPv4 address) before installing as
# run_mobile.sh in the app directory.
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
export API_SERVER_KEY=$(grep '^API_SERVER_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
export OPENCODE_GO_API_KEY=$(grep '^OPENCODE_GO_API_KEY' ~/.hermes/.env | head -1 | cut -d= -f2-)
TOKEN_FILE=~/.hermes/mobile-voice-token
if [ ! -f "$TOKEN_FILE" ]; then
  umask 077
  head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"
export MOBILE_ACCESS_TOKEN=$(cat "$TOKEN_FILE")
export MOBILE_TURN_HOST={{TURN_HOST_IP}}
export HERMES_VOICE={{VOICE}}
cd {{APP_DIR}}
exec .venv/bin/python mobile_server.py
