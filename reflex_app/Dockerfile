FROM python:3.11-slim

WORKDIR /app

# System deps for Bun (Reflex toolchain) + Caddy (in-container reverse proxy).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl unzip ca-certificates \
    && curl -L https://github.com/caddyserver/caddy/releases/download/v2.7.6/caddy_2.7.6_linux_amd64.tar.gz \
        | tar xz -C /usr/local/bin caddy \
    && chmod +x /usr/local/bin/caddy \
    && rm -rf /var/lib/apt/lists/*

COPY reflex_app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole Reflex project (rxconfig.py, the reflex_app package,
# and the assets/ tree with the species JSON + thumbnail images).
COPY reflex_app/ /app/

# Pre-initialize Reflex so Bun is downloaded and the Next.js project is
# scaffolded at build time. Without this the first container start
# spends 2-5 minutes setting up node_modules.
RUN reflex init --template blank --loglevel info || true

# Caddy listens on the single public port (3000) and routes:
#   /_event/*, /_upload/*, /ping  -> Reflex backend on internal :8001
#   everything else               -> Reflex frontend on internal :3001
COPY reflex_app/Caddyfile /etc/caddy/Caddyfile
COPY reflex_app/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 3000

CMD ["/entrypoint.sh"]
