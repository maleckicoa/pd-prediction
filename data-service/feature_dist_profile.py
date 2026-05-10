from __future__ import annotations

import math

import pandas as pd

FEATURE_COLUMNS: list[str] = [
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

CATEGORICAL_FEATURES: set[str] = {
    "account_status",
    "account_worst_status_0_3m",
    "account_worst_status_12_24m",
    "account_worst_status_3_6m",
    "account_worst_status_6_12m",
    "num_arch_written_off_0_12m",
    "num_arch_written_off_12_24m",
    "worst_status_active_inv",
    "merchant_category",
    "merchant_group",
    "has_paid",
    "name_in_email",
    "num_arch_dc_0_12m",
    "num_arch_dc_12_24m",
    "status_last_archived_0_24m",
    "status_2nd_last_archived_0_24m",
    "status_3rd_last_archived_0_24m",
    "status_max_archived_0_6_months",
    "status_max_archived_0_12_months",
    "status_max_archived_0_24_months",
}

NUMERIC_BIN_COUNT = 10
MISSING_CATEGORY_LABEL = "MISSING"

ZERO_THEN_PERCENTILE_FEATURES: set[str] = {
    "account_days_in_dc_12_24m",
    "account_amount_added_12_24m",
    "account_days_in_rem_12_24m",
    "account_days_in_term_12_24m",
    "account_incoming_debt_vs_paid_0_24m",
    "avg_payment_span_0_12m",
    "avg_payment_span_0_3m",
    "max_paid_inv_0_12m",
    "max_paid_inv_0_24m",
    "num_active_div_by_paid_inv_0_12m",
    "num_arch_ok_0_12m",
    "num_arch_ok_12_24m",
    "sum_capital_paid_account_0_12m",
    "sum_capital_paid_account_12_24m",
    "sum_paid_inv_0_12m",
}


def _normalize_categorical_value(value: object) -> str:
    if value is None or pd.isna(value):
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


def _append_profile_row(
    rows: list[dict[str, object]],
    *,
    feature_name: str,
    feature_type: str,
    bin_index: int | None,
    category_value: str | None,
    bin_left: float | None,
    bin_right: float | None,
    count: int,
    total_rows: int,
) -> None:
    rows.append(
        {
            "feature_name": feature_name,
            "feature_type": feature_type,
            "bin_index": bin_index,
            "category_value": category_value,
            "bin_left": bin_left,
            "bin_right": bin_right,
            "observation_pct": float(count) / float(total_rows),
        }
    )


def _append_numeric_missing_bin(
    rows: list[dict[str, object]],
    *,
    feature_name: str,
    missing_count: int,
    total_rows: int,
) -> None:
    _append_profile_row(
        rows,
        feature_name=feature_name,
        feature_type="numerical",
        bin_index=0,
        category_value=MISSING_CATEGORY_LABEL,
        bin_left=None,
        bin_right=None,
        count=missing_count,
        total_rows=total_rows,
    )


def _append_numeric_value_bin(
    rows: list[dict[str, object]],
    *,
    feature_name: str,
    bin_index: int,
    value: float,
    count: int,
    total_rows: int,
) -> int:
    _append_profile_row(
        rows,
        feature_name=feature_name,
        feature_type="numerical",
        bin_index=bin_index,
        category_value=None,
        bin_left=float(value),
        bin_right=float(value),
        count=count,
        total_rows=total_rows,
    )
    return bin_index + 1


def _append_numeric_range_bin(
    rows: list[dict[str, object]],
    *,
    feature_name: str,
    bin_index: int,
    left: float,
    right: float,
    count: int,
    total_rows: int,
) -> int:
    _append_profile_row(
        rows,
        feature_name=feature_name,
        feature_type="numerical",
        bin_index=bin_index,
        category_value=None,
        bin_left=float(left),
        bin_right=float(right),
        count=count,
        total_rows=total_rows,
    )
    return bin_index + 1


def _append_numeric_overflow_bin(
    rows: list[dict[str, object]],
    *,
    feature_name: str,
    bin_index: int,
    left: float,
    total_rows: int,
) -> None:
    _append_profile_row(
        rows,
        feature_name=feature_name,
        feature_type="numerical",
        bin_index=bin_index,
        category_value=None,
        bin_left=float(left),
        bin_right=None,
        count=0,
        total_rows=total_rows,
    )


def _append_overflow_for_feature(
    rows: list[dict[str, object]],
    *,
    feature_name: str,
    total_rows: int,
) -> None:
    numeric_feature_rows = [
        row
        for row in rows
        if row["feature_name"] == feature_name
        and row["feature_type"] == "numerical"
        and row["bin_index"] is not None
        and int(row["bin_index"]) > 0
    ]
    if not numeric_feature_rows:
        _append_numeric_overflow_bin(
            rows,
            feature_name=feature_name,
            bin_index=1,
            left=0.0,
            total_rows=total_rows,
        )
        return

    next_bin_index = max(int(row["bin_index"]) for row in numeric_feature_rows) + 1
    finite_rights = [float(row["bin_right"]) for row in numeric_feature_rows if row["bin_right"] is not None]
    left = max(finite_rights) if finite_rights else max(float(row["bin_left"]) for row in numeric_feature_rows)
    _append_numeric_overflow_bin(
        rows,
        feature_name=feature_name,
        bin_index=next_bin_index,
        left=left,
        total_rows=total_rows,
    )


def _append_percentile_bins(
    rows: list[dict[str, object]],
    *,
    feature_name: str,
    series: pd.Series,
    start_bin_index: int,
    total_rows: int,
) -> int:
    if series.empty:
        return start_bin_index

    unique_count = int(series.nunique())
    if unique_count == 0:
        return start_bin_index

    q = max(1, min(NUMERIC_BIN_COUNT, unique_count))
    bins = pd.qcut(series, q=q, duplicates="drop")
    counts = bins.value_counts(sort=False)
    bin_index = start_bin_index
    for interval, count in counts.items():
        bin_index = _append_numeric_range_bin(
            rows,
            feature_name=feature_name,
            bin_index=bin_index,
            left=float(interval.left),
            right=float(interval.right),
            count=int(count),
            total_rows=total_rows,
        )
    return bin_index


def build_feature_dist_profile(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    rows: list[dict[str, object]] = []
    if total_rows == 0:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "feature_type",
                "bin_index",
                "category_value",
                "bin_left",
                "bin_right",
                "observation_pct",
            ]
        )

    for feature_name in FEATURE_COLUMNS:
        feature_series = df[feature_name]

        if feature_name in CATEGORICAL_FEATURES:
            counts = feature_series.map(_normalize_categorical_value).value_counts(dropna=False)
            if MISSING_CATEGORY_LABEL not in counts.index:
                counts.loc[MISSING_CATEGORY_LABEL] = 0
            for category_value, count in counts.items():
                _append_profile_row(
                    rows,
                    feature_name=feature_name,
                    feature_type="categorical",
                    bin_index=None,
                    category_value=category_value,
                    bin_left=None,
                    bin_right=None,
                    count=int(count),
                    total_rows=total_rows,
                )
            continue

        numeric = pd.to_numeric(feature_series, errors="coerce")
        missing_count = int(numeric.isna().sum())
        _append_numeric_missing_bin(
            rows,
            feature_name=feature_name,
            missing_count=missing_count,
            total_rows=total_rows,
        )

        non_missing = numeric.dropna()
        if non_missing.empty:
            _append_overflow_for_feature(
                rows,
                feature_name=feature_name,
                total_rows=total_rows,
            )
            continue

        bin_index = 1

        if feature_name in ZERO_THEN_PERCENTILE_FEATURES:
            zero_count = int((non_missing == 0).sum())
            bin_index = _append_numeric_value_bin(
                rows,
                feature_name=feature_name,
                bin_index=bin_index,
                value=0.0,
                count=zero_count,
                total_rows=total_rows,
            )
            rest = non_missing[non_missing > 0]
            _append_percentile_bins(
                rows,
                feature_name=feature_name,
                series=rest,
                start_bin_index=bin_index,
                total_rows=total_rows,
            )
            _append_overflow_for_feature(
                rows,
                feature_name=feature_name,
                total_rows=total_rows,
            )
            continue

        if feature_name == "age":
            max_age = int(non_missing.max())
            max_edge = ((max_age // 10) + 1) * 10
            edges = list(range(0, max_edge + 10, 10))
            age_bins = pd.cut(
                non_missing,
                bins=edges,
                right=False,
                include_lowest=True,
            )
            counts = age_bins.value_counts(sort=False)
            for interval, count in counts.items():
                bin_index = _append_numeric_range_bin(
                    rows,
                    feature_name=feature_name,
                    bin_index=bin_index,
                    left=float(interval.left),
                    right=float(interval.right),
                    count=int(count),
                    total_rows=total_rows,
                )
            _append_overflow_for_feature(
                rows,
                feature_name=feature_name,
                total_rows=total_rows,
            )
            continue

        if feature_name == "num_active_inv":
            for value in (0.0, 1.0, 2.0):
                count = int((non_missing == value).sum())
                bin_index = _append_numeric_value_bin(
                    rows,
                    feature_name=feature_name,
                    bin_index=bin_index,
                    value=value,
                    count=count,
                    total_rows=total_rows,
                )
            rest = non_missing[non_missing >= 3]
            if not rest.empty:
                _append_numeric_range_bin(
                    rows,
                    feature_name=feature_name,
                    bin_index=bin_index,
                    left=3.0,
                    right=float(rest.max()),
                    count=int(rest.shape[0]),
                    total_rows=total_rows,
                )
            _append_overflow_for_feature(
                rows,
                feature_name=feature_name,
                total_rows=total_rows,
            )
            continue

        if feature_name == "num_arch_rem_0_12m":
            for value in (0.0, 1.0):
                count = int((non_missing == value).sum())
                bin_index = _append_numeric_value_bin(
                    rows,
                    feature_name=feature_name,
                    bin_index=bin_index,
                    value=value,
                    count=count,
                    total_rows=total_rows,
                )
            rest = non_missing[non_missing > 1]
            if not rest.empty:
                _append_numeric_range_bin(
                    rows,
                    feature_name=feature_name,
                    bin_index=bin_index,
                    left=2.0,
                    right=float(rest.max()),
                    count=int(rest.shape[0]),
                    total_rows=total_rows,
                )
            _append_overflow_for_feature(
                rows,
                feature_name=feature_name,
                total_rows=total_rows,
            )
            continue

        if feature_name == "num_unpaid_bills":
            for value in (0.0, 1.0, 2.0, 3.0):
                count = int((non_missing == value).sum())
                bin_index = _append_numeric_value_bin(
                    rows,
                    feature_name=feature_name,
                    bin_index=bin_index,
                    value=value,
                    count=count,
                    total_rows=total_rows,
                )
            rest = non_missing[non_missing > 3]
            if not rest.empty:
                _append_numeric_range_bin(
                    rows,
                    feature_name=feature_name,
                    bin_index=bin_index,
                    left=4.0,
                    right=float(rest.max()),
                    count=int(rest.shape[0]),
                    total_rows=total_rows,
                )
            _append_overflow_for_feature(
                rows,
                feature_name=feature_name,
                total_rows=total_rows,
            )
            continue

        if feature_name == "recovery_debt":
            for value in (0.0, 1.0):
                count = int((non_missing == value).sum())
                bin_index = _append_numeric_value_bin(
                    rows,
                    feature_name=feature_name,
                    bin_index=bin_index,
                    value=value,
                    count=count,
                    total_rows=total_rows,
                )
            rest = non_missing[non_missing > 1]
            _append_percentile_bins(
                rows,
                feature_name=feature_name,
                series=rest,
                start_bin_index=bin_index,
                total_rows=total_rows,
            )
            _append_overflow_for_feature(
                rows,
                feature_name=feature_name,
                total_rows=total_rows,
            )
            continue

        if feature_name == "time_hours":
            for left in range(0, 24):
                right = float(left + 1)
                if left < 23:
                    count = int(((non_missing >= left) & (non_missing < right)).sum())
                else:
                    count = int(((non_missing >= left) & (non_missing <= right)).sum())
                bin_index = _append_numeric_range_bin(
                    rows,
                    feature_name=feature_name,
                    bin_index=bin_index,
                    left=float(left),
                    right=right,
                    count=count,
                    total_rows=total_rows,
                )
            continue

        _append_percentile_bins(
            rows,
            feature_name=feature_name,
            series=non_missing,
            start_bin_index=bin_index,
            total_rows=total_rows,
        )
        _append_overflow_for_feature(
            rows,
            feature_name=feature_name,
            total_rows=total_rows,
        )

    return pd.DataFrame.from_records(
        rows,
        columns=[
            "feature_name",
            "feature_type",
            "bin_index",
            "category_value",
            "bin_left",
            "bin_right",
            "observation_pct",
        ],
    )
