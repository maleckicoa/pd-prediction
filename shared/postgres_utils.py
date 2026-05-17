"""Helpers to read rows from the demo `loans` Postgres table (see ../docker-compose.yml)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
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

def fetch_all_ids(engine):
    stmt = text('SELECT uuid, "default" FROM test_loans ORDER BY uuid')
    return pd.read_sql(stmt, engine)


def fetch_loan_by_uuid(engine, loan_uuid: str) -> pd.DataFrame:
    stmt = text("SELECT * FROM test_loans WHERE uuid = CAST(:uid AS uuid)")
    df = pd.read_sql(stmt, engine, params={"uid": loan_uuid})
    if df.empty:
        raise ValueError(f"No row in loans for uuid={loan_uuid!r}")
    return df


def fetch_train_dataset(engine):
    stmt = text("SELECT * FROM train_loans")
    df = pd.read_sql(stmt, engine)

    y = df["default"].astype(int)
    X = df.drop(columns=["default"]).copy()
    return X, y

def fetch_full_dataset(engine, table_name: str):
    stmt = text(f"SELECT * FROM {table_name} ORDER BY uuid")
    df = pd.read_sql(stmt, engine)
    y = df["default"].astype(int)
    X = df.drop(columns=["default"]).copy()
    return X, y

def fetch_random_loan(engine) -> pd.DataFrame:
    stmt = text("SELECT * FROM test_loans ORDER BY random() LIMIT 1")
    df = pd.read_sql(stmt, engine)
    if df.empty:
        raise ValueError("No rows found in test_loans")
    return df


def fetch_loan_id_for_uuid(engine, loan_uuid: str) -> int | None:
    """Return ``loan_ids.id`` for a scored loan (row exists after predict writes ``test_defaults``)."""
    stmt = text("SELECT id FROM loan_ids WHERE uuid = CAST(:uid AS uuid)")
    df = pd.read_sql(stmt, engine, params={"uid": loan_uuid})
    if df.empty or df.iloc[0]["id"] is None:
        return None
    return int(df.iloc[0]["id"])


def fetch_last_completed_loan_uuid(engine) -> str | None:
    """Most recent loan_uuid fully written to test_feat_hit (used to resume the fetch loop)."""
    stmt = text(
        """
        SELECT loan_uuid::text AS loan_uuid
        FROM test_feat_hit
        ORDER BY loan_event_idx DESC
        LIMIT 1
        """
    )
    df = pd.read_sql(stmt, engine)
    if df.empty or df.iloc[0]["loan_uuid"] is None:
        return None
    return str(df.iloc[0]["loan_uuid"])


def fetch_next_loan(engine, last_uuid: str | None = None) -> pd.DataFrame:
    if last_uuid:
        stmt = text(
            """
            SELECT *
            FROM test_loans
            WHERE uuid > CAST(:last_uuid AS uuid)
            ORDER BY uuid ASC
            LIMIT 1
            """
        )
        df = pd.read_sql(stmt, engine, params={"last_uuid": last_uuid})
    else:
        df = pd.DataFrame()

    if df.empty:
        # First iteration or wrap-around when reaching the largest UUID.
        stmt = text(
            """
            SELECT *
            FROM test_loans
            ORDER BY uuid ASC
            LIMIT 1
            """
        )
        df = pd.read_sql(stmt, engine)

    if df.empty:
        raise ValueError("No rows found in test_loans")
    return df


def write_default_prob(
    engine,
    loan_uuid: str,
    prob: float,
    model_name: str = "unknown",
    threshold_applied: float = 0.5,
    loan_default: int | None = None,
    predicted_at_utc: datetime | None = None,
) -> None:
    if predicted_at_utc is None:
        predicted_at_utc = datetime.now(timezone.utc)

    stmt = text(
        """
        WITH existing AS (
            SELECT id
            FROM loan_ids
            WHERE uuid = CAST(:uid AS uuid)
        ),
        inserted AS (
            INSERT INTO loan_ids (uuid)
            SELECT CAST(:uid AS uuid)
            WHERE NOT EXISTS (SELECT 1 FROM existing)
            RETURNING id
        ),
        loan AS (
            SELECT id FROM existing
            UNION ALL
            SELECT id FROM inserted
        )
        INSERT INTO test_defaults (
            id,
            uuid,
            pd,
            threshold_applied,
            loan_default,
            model_name,
            predicted_at_utc
        )
        SELECT
            loan.id,
            CAST(:uid AS uuid),
            :pd,
            :threshold_applied,
            :loan_default,
            :model_name,
            :predicted_at_utc
        FROM loan
        """
    )
    with engine.begin() as conn:
        conn.execute(
            stmt,
            {
                "uid": loan_uuid,
                "pd": prob,
                "model_name": model_name,
                "predicted_at_utc": predicted_at_utc,
                "threshold_applied": threshold_applied,
                "loan_default": loan_default,
            },
        )


def reset_test_cycle_tables(engine) -> None:
    """Clear per-cycle outputs after one full pass through test loans."""
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE test_defaults, test_feat_hit RESTART IDENTITY"))
        conn.execute(text("SELECT setval('test_feat_hit_loan_event_idx_seq', 1, false)"))