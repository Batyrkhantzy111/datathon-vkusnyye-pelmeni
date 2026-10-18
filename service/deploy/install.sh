#!/usr/bin/env bash
# Install/refresh the systemd units on the GPU host (idempotent). Run: bash ~/forte/service/deploy/install.sh
# Keeps an existing fake.conf drop-in decision: pass --fake to (re)install it, --real to remove it.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
sed -i 's/\r$//' "$D"/*.sh "$D"/*.service "$D"/*.conf "$D/Caddyfile"
/usr/local/bin/caddy validate --config "$D/Caddyfile" --adapter caddyfile
chmod +x "$D/tunnel.sh"
sudo install -m 644 "$D/forte-api.service" "$D/forte-tunnel.service" "$D/forte-proxy.service" /etc/systemd/system/
case "${1:-}" in
  --fake) sudo install -D -m 644 "$D/fake.conf" /etc/systemd/system/forte-api.service.d/fake.conf ;;
  --real) sudo rm -f /etc/systemd/system/forte-api.service.d/fake.conf ;;
esac
sudo systemctl daemon-reload
sudo systemctl enable forte-api.service forte-tunnel.service forte-proxy.service
sudo systemctl restart forte-api.service
sudo systemctl is-active --quiet forte-tunnel.service || sudo systemctl start forte-tunnel.service
sudo systemctl restart forte-proxy.service
