FROM python:3.11-slim

WORKDIR /app

# Reflex bootstraps Bun on first run to compile the Next.js frontend, so
# we need curl + unzip available even after the build step.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl unzip ca-certificates \
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

# Frontend (Next.js) → 3000. Backend (state + websocket) → 8001.
# FastAPI is reached separately via BIRD_API_URL (set at run time).
EXPOSE 3000 8001

CMD ["reflex", "run", \
     "--env", "dev", \
     "--frontend-port", "3000", \
     "--backend-port", "8001"]
