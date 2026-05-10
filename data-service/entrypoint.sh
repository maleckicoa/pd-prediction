#!/bin/bash
set -e

# Start Postgres in the background via the official entrypoint.
docker-entrypoint.sh postgres &

echo "Waiting for Postgres..."
until pg_isready -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
  sleep 2
done
echo "Postgres is ready"

# Ensure schema init scripts have finished before checking data state.
until [ "$(psql -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
  "SELECT to_regclass('public.train_loans') IS NOT NULL
      AND to_regclass('public.test_loans') IS NOT NULL;")" = "t" ]; do
  echo "Waiting for init scripts to finish..."
  sleep 1
done

SHOULD_SEED="$(psql -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "
  SELECT CASE
    WHEN EXISTS (SELECT 1 FROM train_loans LIMIT 1)
     AND EXISTS (SELECT 1 FROM test_loans LIMIT 1)
    THEN 'false'
    ELSE 'true'
  END;
")"

if [ "$SHOULD_SEED" = "true" ]; then
  echo "Loading train/test data into Postgres..."
  # Seed runs inside the same container, so connect to local postgres socket/port.
  export POSTGRES_HOST=localhost
  export POSTGRES_PORT=5432
  python3 /seed/load_data_to_postgres.py
  echo "Data load complete"
else
  echo "Seed data already present; skipping load_data_to_postgres.py"
fi

# Keep postgres process in the foreground.
wait
