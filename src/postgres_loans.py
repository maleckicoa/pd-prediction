"""Helpers to read rows from the demo `loans` Postgres table (see ../docker-compose.yml)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def get_engine():
    _load_dotenv(PROJECT_ROOT / ".env")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    dbname = os.environ["POSTGRES_DB"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5434"))
    return create_engine(
        URL.create(
            drivername="postgresql+psycopg",
            username=user,
            password=password,
            host=host,
            port=port,
            database=dbname,
        ),
        pool_pre_ping=True,
    )


def fetch_loan_by_uuid(engine, loan_uuid: str) -> pd.DataFrame:
    stmt = text("SELECT * FROM loans WHERE uuid = CAST(:uid AS uuid)")
    df = pd.read_sql(stmt, engine, params={"uid": loan_uuid})
    if df.empty:
        raise ValueError(f"No row in loans for uuid={loan_uuid!r}")
    return df
