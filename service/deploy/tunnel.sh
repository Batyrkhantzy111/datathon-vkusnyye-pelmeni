#!/usr/bin/env bash
# Cloudflare quick tunnel to the API; writes the current public URL to ~/forte/service/public_url.txt.
# Run by forte-tunnel.service. A quick tunnel gets a new random *.trycloudflare.com URL on every start.
set -uo pipefail
URL_FILE="${FORTE_URL_FILE:-$HOME/forte/service/public_url.txt}"
TARGET="${FORTE_TUNNEL_TARGET:-http://127.0.0.1:8000}"
: > "$URL_FILE.tmp"
cloudflared tunnel --no-autoupdate --url "$TARGET" 2>&1 | while IFS= read -r line; do
  echo "$line"
  if [[ "$line" =~ (https://[a-z0-9-]+\.trycloudflare\.com) ]]; then
    echo "${BASH_REMATCH[1]}" > "$URL_FILE.tmp" && mv "$URL_FILE.tmp" "$URL_FILE"
    echo "public URL: ${BASH_REMATCH[1]} -> $URL_FILE"
  fi
done
exit "${PIPESTATUS[0]}"
