#!/bin/bash
set -e

# Start Postgres in the background via the official entrypoint.
docker-entrypoint.sh postgres &

echo "Waiting for Postgres..."
until pg_isready -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
  sleep 2
done
echo "Postgres is ready"

echo "Loading train/test data into Postgres..."
# Seed runs inside the same container, so connect to local postgres socket/port.
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
python3 /seed/load_data_to_postgres.py
echo "Data load complete"

# Keep postgres process in the foreground.
wait
