#!/bin/sh
# Used by docker-compose service train-init.
# Skips training if lr1_model.pkl already exists; otherwise trains LR1 (needs Postgres + MLflow).

set -e
TARGET=/app/models/lr1_model.pkl
mkdir -p /app/models

if [ -f "$TARGET" ]; then
  echo "train-init: $TARGET already present; skipping LR1 training."
  exit 0
fi

echo "train-init: $TARGET not found; training LR1..."
exec python /app/train-service/src/lr1/lr1_model_train.py
