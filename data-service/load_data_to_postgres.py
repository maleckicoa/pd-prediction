"""
Load the same train/test split as model.ipynb (10% test stratified,
random_state=42) into Postgres `train_loans` and `test_loans` tables.

Import and call `load_test_split_to_postgres()`. Requires pandas, scikit-learn,
sqlalchemy, and psycopg (binary extra; driver URL postgresql+psycopg).

Connection: reads `pd-prediction/.env` (project root) for POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB.
Optional: POSTGRES_HOST (default localhost), POSTGRES_PORT (default 5434).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine import URL
from feature_dist_profile import build_feature_dist_profile

DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent

LOAN_COLUMNS: list[str] = [
    "uuid",
    "default",
    "account_amount_added_12_24m",
    "account_days_in_dc_12_24m",
    "account_days_in_rem_12_24m",
    "account_days_in_term_12_24m",
    "account_incoming_debt_vs_paid_0_24m",
    "account_status",
    "account_worst_status_0_3m",
    "account_worst_status_12_24m",
    "account_worst_status_3_6m",
    "account_worst_status_6_12m",
    "age",
    "avg_payment_span_0_12m",
    "avg_payment_span_0_3m",
    "merchant_category",
    "merchant_group",
    "has_paid",
    "max_paid_inv_0_12m",
    "max_paid_inv_0_24m",
    "name_in_email",
    "num_active_div_by_paid_inv_0_12m",
    "num_active_inv",
    "num_arch_dc_0_12m",
    "num_arch_dc_12_24m",
    "num_arch_ok_0_12m",
    "num_arch_ok_12_24m",
    "num_arch_rem_0_12m",
    "num_arch_written_off_0_12m",
    "num_arch_written_off_12_24m",
    "num_unpaid_bills",
    "status_last_archived_0_24m",
    "status_2nd_last_archived_0_24m",
    "status_3rd_last_archived_0_24m",
    "status_max_archived_0_6_months",
    "status_max_archived_0_12_months",
    "status_max_archived_0_24_months",
    "recovery_debt",
    "sum_capital_paid_account_0_12m",
    "sum_capital_paid_account_12_24m",
    "sum_paid_inv_0_12m",
    "time_hours",
    "worst_status_active_inv",
]

def load_dotenv(path: Path) -> None:
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


def _read_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", na_values=["NA"])
    df["has_paid"] = df["has_paid"].replace(
        {"TRUE": True, "FALSE": False, "true": True, "false": False}
    )
    df["has_paid"] = df["has_paid"].astype(int)
    return df


def _train_test_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_nna = df[df["default"].notna()]
    X = df_nna.drop(columns=["default"])
    y = df_nna["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        stratify=y,
        random_state=42,
    )

    train_out = X_train.copy()
    train_out["default"] = y_train
    assert y_train.index.equals(train_out.index)
    train_out["has_paid"] = train_out["has_paid"].astype(bool)

    test_out = X_test.copy()
    test_out["default"] = y_test
    assert y_test.index.equals(test_out.index)
    test_out["has_paid"] = test_out["has_paid"].astype(bool)
    return train_out, test_out


def _engine(user: str, password: str, host: str, port: int, dbname: str):
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


def _write_feature_dist_profile(conn, profile_df: pd.DataFrame) -> None:
    conn.execute(text("TRUNCATE train_feat_dist"))
    if profile_df.empty:
        return
    profile_df.to_sql(
        "train_feat_dist",
        conn,
        if_exists="append",
        index=False,
        chunksize=500,
        method="multi",
    )


def load_test_split_to_postgres(
    *,
    csv_path: Path | None = None,
    env_path: Path | None = None,
    truncate_before_insert: bool = True,
) -> tuple[int, int]:
    """
    Load notebook train/test split into `train_loans` and `test_loans`.
    Returns (train_rows_inserted, test_rows_inserted).

    Defaults: `dataset.csv` in this directory; `.env` in this directory
    (fallback: pd-prediction project root).
    Optional POSTGRES_HOST (localhost) and POSTGRES_PORT (5434).
    """
    csv_path = csv_path or (DATA_DIR / "dataset.csv")
    env_path = env_path or (
        (DATA_DIR / ".env") if (DATA_DIR / ".env").is_file() else (PROJECT_ROOT / ".env")
    )
    load_dotenv(env_path)

    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    dbname = os.environ["POSTGRES_DB"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))

    df = _read_dataset(csv_path)
    df_train, df_test = _train_test_frames(df)
    df_train_out = df_train[LOAN_COLUMNS].copy()
    df_test_out = df_test[LOAN_COLUMNS].copy()
    df_train_out["uuid"] = df_train_out["uuid"].map(lambda x: uuid.UUID(str(x).strip()))
    df_test_out["uuid"] = df_test_out["uuid"].map(lambda x: uuid.UUID(str(x).strip()))

    engine = _engine(user, password, host, port, dbname)
    with engine.begin() as conn:
        if truncate_before_insert:
            conn.execute(text("TRUNCATE train_loans, test_loans"))
        df_train_out.to_sql(
            "train_loans",
            conn,
            if_exists="append",
            index=False,
            chunksize=500,
            method="multi",
            dtype={"uuid": PG_UUID(as_uuid=True)},
        )
        df_test_out.to_sql(
            "test_loans",
            conn,
            if_exists="append",
            index=False,
            chunksize=500,
            method="multi",
            dtype={"uuid": PG_UUID(as_uuid=True)},
        )
        profile_df = build_feature_dist_profile(df_train_out)
        _write_feature_dist_profile(conn, profile_df)
    return len(df_train_out), len(df_test_out)


if __name__ == "__main__":
    print(load_test_split_to_postgres())
