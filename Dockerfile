# Odysseus (odyssai.eu) — distributed MLX inference engine
# API + Dashboard. Runs on any Docker host with SSH access to the cluster
# nodes. Does NO MLX work — pure orchestration: SSHes into the cluster nodes
# (one or more Apple Silicon Macs) and spawns runner.py on each rank.

FROM python:3.11-slim

# OpenSSH client for spawning runners on cluster nodes
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code + assets
COPY scripts/api.py /app/api.py
COPY scripts/topology.py /app/topology.py
COPY scripts/persistence.py /app/persistence.py
COPY scripts/dashboard.html /app/dashboard.html
# Brand logo — cyclops eye from the Odyssey
COPY logo/odysseus.png /app/odysseus.png

# Persisted state lives in a mounted volume. ALL state files go to /app/data/
# (per-cluster state + cluster config) so a `docker compose up --build`
# doesn't wipe load_history and last-loaded model.
ENV STATE_FILE=/app/data/state.json \
    ARGO_STATE_FILE=/app/data/state-argo.json \
    CLUSTER_CONFIG_FILE=/app/data/cluster-config.json \
    DASHBOARD_FILE=/app/dashboard.html \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PYTHONUNBUFFERED=1

# Pre-create data dir (the volume mount will overlay it)
RUN mkdir -p /app/data

EXPOSE 8000

# No --model: the API auto-reloads from STATE_FILE if present, otherwise
# stays idle until /admin/load is called from the dashboard.
CMD ["python", "/app/api.py"]
