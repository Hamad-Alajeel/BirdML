#!/bin/sh
set -e

# Forward SIGTERM/SIGINT to all child processes so ECS can stop us cleanly.
trap 'kill $(jobs -p) 2>/dev/null' TERM INT

# Caddy: public entrypoint on :3000, routes to internal frontend/backend.
caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &

# Reflex Python backend (WebSocket state server).
reflex run --env prod --backend-only --backend-port 8001 &

# Reflex Next.js frontend (production build).
reflex run --env prod --frontend-only --frontend-port 3001 &

wait
