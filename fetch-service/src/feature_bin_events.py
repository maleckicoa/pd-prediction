from __future__ import annotations

from datetime import datetime, timezone
import math

import pandas as pd
from sqlalchemy import text

MISSING_CATEGORY_LABEL = "MISSING"


def load_feature_profile(engine) -> dict[str, list[dict]]:
    stmt = text(
        """
        SELECT
            feature_name,
            feature_type,
            bin_index,
            category_value,
            bin_left,
            bin_right
        FROM train_feat_dist
        ORDER BY
            feature_name ASC,
            COALESCE(bin_index, 0) ASC,
            category_value ASC NULLS LAST
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(stmt).mappings().all()

    profile: dict[str, list[dict]] = {}
    for row in rows:
        feature_name = str(row["feature_name"])
        profile.setdefault(feature_name, []).append(
            {
                "feature_type": str(row["feature_type"]),
                "bin_index": row["bin_index"],
                "category_value": row["category_value"],
                "bin_left": row["bin_left"],
                "bin_right": row["bin_right"],
            }
        )
    return profile


def is_missing(value) -> bool:
    return value is None or pd.isna(value)


def normalize_categorical_value(value) -> str:
    if is_missing(value):
        return MISSING_CATEGORY_LABEL

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return MISSING_CATEGORY_LABEL
        value = stripped

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(numeric_value):
        return MISSING_CATEGORY_LABEL
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return format(numeric_value, "g")


def _get_missing_bucket(feature_type: str, buckets: list[dict]) -> dict:
    if feature_type == "categorical":
        for bucket in buckets:
            if normalize_categorical_value(bucket["category_value"]) == MISSING_CATEGORY_LABEL:
                return bucket
        raise ValueError("No categorical MISSING bucket configured")

    for bucket in buckets:
        if bucket["bin_index"] == 0:
            return bucket
    raise ValueError("No numerical MISSING bucket configured (bin_index=0)")


def match_feature_bucket(feature_value, buckets: list[dict]) -> dict:
    feature_type = str(buckets[0]["feature_type"])
    if feature_type == "categorical":
        category_value = normalize_categorical_value(feature_value)
        for bucket in buckets:
            if normalize_categorical_value(bucket["category_value"]) == category_value:
                return bucket
        return _get_missing_bucket(feature_type, buckets)

    if is_missing(feature_value):
        return _get_missing_bucket(feature_type, buckets)

    try:
        value = float(feature_value)
    except (TypeError, ValueError):
        return _get_missing_bucket(feature_type, buckets)

    non_missing_buckets = [b for b in buckets if b["bin_index"] is not None and b["bin_index"] > 0]
    non_missing_buckets.sort(key=lambda b: int(b["bin_index"]))
    finite_buckets = [b for b in non_missing_buckets if b["bin_right"] is not None]
    overflow_buckets = [b for b in non_missing_buckets if b["bin_right"] is None]

    for bucket in finite_buckets:
        left = float(bucket["bin_left"])
        right = float(bucket["bin_right"])
        if left == right and value == left:
            return bucket

    if finite_buckets:
        last_bin_index = int(finite_buckets[-1]["bin_index"])
    else:
        last_bin_index = -1

    for bucket in finite_buckets:
        left = float(bucket["bin_left"])
        right = float(bucket["bin_right"])
        if left == right:
            continue
        if int(bucket["bin_index"]) == last_bin_index:
            if value >= left and value <= right:
                return bucket
        elif value >= left and value < right:
            return bucket

    if overflow_buckets:
        overflow_bucket = sorted(overflow_buckets, key=lambda b: int(b["bin_index"]))[-1]
        if value >= float(overflow_bucket["bin_left"]):
            return overflow_bucket

    return _get_missing_bucket(feature_type, buckets)


def build_feature_event_rows(
    raw_loan: dict,
    profile_by_feature: dict[str, list[dict]],
    loan_event_idx: int,
) -> list[dict]:
    loan_uuid = str(raw_loan["uuid"])
    observed_at_utc = datetime.now(timezone.utc)
    event_rows: list[dict] = []

    for feature_name, buckets in profile_by_feature.items():
        if feature_name not in raw_loan:
            raise ValueError(f"Loan payload missing feature {feature_name!r}")

        matched_bucket = match_feature_bucket(raw_loan[feature_name], buckets)
        matched_key = (
            matched_bucket["bin_index"],
            matched_bucket["category_value"],
        )

        for bucket in buckets:
            bucket_key = (
                bucket["bin_index"],
                bucket["category_value"],
            )
            event_rows.append(
                {
                    "loan_event_idx": loan_event_idx,
                    "loan_uuid": loan_uuid,
                    "observed_at_utc": observed_at_utc,
                    "feature_name": feature_name,
                    "feature_type": bucket["feature_type"],
                    "bin_index": bucket["bin_index"],
                    "category_value": bucket["category_value"],
                    "hit": 1 if bucket_key == matched_key else 0,
                }
            )
    return event_rows


def write_feature_bin_events(
    engine,
    profile_by_feature: dict[str, list[dict]],
    raw_loan: dict,
) -> None:
    with engine.begin() as conn:
        loan_event_idx = int(
            conn.execute(text("SELECT nextval('test_feat_hit_loan_event_idx_seq')")).scalar_one()
        )
        event_rows = build_feature_event_rows(raw_loan, profile_by_feature, loan_event_idx)
        conn.execute(
            text(
                """
                INSERT INTO test_feat_hit (
                    loan_event_idx,
                    loan_uuid,
                    observed_at_utc,
                    feature_name,
                    feature_type,
                    bin_index,
                    category_value,
                    hit
                )
                VALUES (
                    :loan_event_idx,
                    CAST(:loan_uuid AS uuid),
                    :observed_at_utc,
                    :feature_name,
                    :feature_type,
                    :bin_index,
                    :category_value,
                    :hit
                )
                """
            ),
            event_rows,
        )
