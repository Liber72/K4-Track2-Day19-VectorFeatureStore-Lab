#!/usr/bin/env bash
# Full Docker path: Qdrant server + Redis online store + Postgres offline store.
# Same Python venv as lite + the docker extras. ~3-5 min on first run (image pulls).

set -euo pipefail

echo "[docker] Day 19 full Docker setup"
echo "[docker] Stack: Qdrant (server) + Redis + Postgres + bge-m3 embeddings"
echo "[docker] Note: bge-m3 is multilingual (much better on Vietnamese) but"
echo "[docker]       downloads ~2.2 GB on first use. Set EMBEDDING_BACKEND=fastembed"
echo "[docker]       in .env to keep the light 384-dim English model."
echo

# ── 1. Runtime preflight ────────────────────────────────────────────────
# Three runtimes are plausible now. Apple's `container` has no compose
# implementation, so it gets its own start path (scripts/container-up.sh).
# `bash scripts/runtime-check.sh` prints the full picture.
RUNTIME=""
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 \
   && docker info >/dev/null 2>&1; then
  RUNTIME="docker"
elif command -v container >/dev/null 2>&1; then
  RUNTIME="apple"
fi

case "$RUNTIME" in
  docker)
    echo "[docker] using docker compose"
    docker compose up -d
    echo "[docker] Waiting up to 30s for services to become healthy..."
    for i in $(seq 1 30); do
      if docker compose ps --format json | grep -q '"Health":"healthy"'; then break; fi
      sleep 1
    done
    ;;
  apple)
    echo "[docker] Docker unavailable; using Apple container (github.com/apple/container)"
    bash scripts/container-up.sh
    ;;
  *)
    echo "[docker] No usable container runtime found."
    echo "[docker] Run 'bash scripts/runtime-check.sh' for details, or use the lite path:"
    echo "[docker]   bash setup-lite.sh"
    exit 1
    ;;
esac

# The closing hint must match the runtime actually used -- `docker compose down`
# does nothing to containers started by Apple's `container`.
if [ "$RUNTIME" = "apple" ]; then
  STOP_HINT="Stop the stack later: make container-down            (state persists)
                  or  make container-down ARGS=--wipe  (full reset)"
else
  STOP_HINT="Stop the stack later: docker compose down    (state persists)
                  or  docker compose down -v (full reset)"
fi

# ── 3. Python venv (same as lite) ───────────────────────────────────────
if [ ! -d ".venv" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv
  else
    python3 -m venv .venv
  fi
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# ── 4. Install lite + docker extras ─────────────────────────────────────
NEED_DILL_OVERRIDE=$(python -c 'import sys; print(1 if sys.version_info >= (3,14) else 0)')
if [ "$NEED_DILL_OVERRIDE" = "1" ]; then
  echo "[docker] venv Python >= 3.14 -> applying dill>=0.4 override (see requirements.txt)"
fi
if command -v uv >/dev/null 2>&1; then
  if [ "$NEED_DILL_OVERRIDE" = "1" ]; then
    uv pip install --overrides overrides-py314.txt -r requirements.txt -r requirements-full.txt
  else
    uv pip install -r requirements.txt -r requirements-full.txt
  fi
else
  pip install -q -U pip
  pip install -q -r requirements.txt -r requirements-full.txt
  if [ "$NEED_DILL_OVERRIDE" = "1" ]; then
    pip install -q --upgrade 'dill>=0.4,<1.0'
  fi
fi

# `_setup.py` is a helper module, not a notebook (see setup-lite.sh).
python -m ipykernel install --sys-prefix --name lab19 --display-name "Python (Lab 19)"
jupytext --to notebook --update notebooks/[0-9]*.py 2>/dev/null || jupytext --to notebook notebooks/[0-9]*.py

# ── 5. .env for docker mode ─────────────────────────────────────────────
if [ ! -f .env ]; then
  cp .env.docker.example .env
fi

# ── 6. Seed corpus + smoke test ─────────────────────────────────────────
python scripts/seed_corpus.py

# Advanced missions (NB6 compound queries, NB8 spend parquet). setup-lite.sh
# does this too; without it NB6 dies on a missing data/agent_queries.jsonl.
echo "  · seeding advanced-mission data (NB6 + NB8)…"
python scripts/gen_agent_queries.py
python scripts/gen_spend.py
python scripts/verify_docker.py

cat <<EOF

[docker] Done. Services running:

  Qdrant   → http://localhost:6333  (dashboard)
  Redis    → redis://localhost:6379
  Postgres → postgresql://feast:feast@localhost:5433/feast_offline

Activate the venv and continue:

    source .venv/bin/activate
    make api       # start FastAPI on :8000
    make lab       # open Jupyter on :8888

${STOP_HINT}
EOF
