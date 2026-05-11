FROM postgres:16

# Install Python runtime for seed script execution.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Schema init scripts used by official Postgres entrypoint.
COPY init/ /docker-entrypoint-initdb.d/

# Seed script and env used by loader.
COPY load_data_to_postgres.py /seed/load_data_to_postgres.py
COPY feature_dist_profile.py /seed/feature_dist_profile.py
COPY dataset.csv /seed/dataset.csv

# Custom entrypoint to bootstrap DB + load data in one process.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
