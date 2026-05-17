import json
import os
import sys
import time
from pathlib import Path
from uuid import UUID

import pandas as pd
import requests
from pydantic import ValidationError

candidates = [
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2],
]

for candidate in candidates:
    if (candidate / "shared").exists():
        sys.path.insert(0, str(candidate))
        break

from shared.postgres_utils import (  # noqa: E402
    fetch_last_completed_loan_uuid,
    fetch_loan_id_for_uuid,
    fetch_next_loan,
    get_engine,
    reset_test_cycle_tables,
)
from shared.models import LoanData  # noqa: E402
from src.feature_bin_events import load_feature_profile, write_feature_bin_events  # noqa: E402

PSI_HIT_FEATURE_NAMES = frozenset(
    {"age",
    "account_status",
    "merchant_category",
    "merchant_group",
    "has_paid",
    "num_active_inv",
    "num_unpaid_bills",
    "recovery_debt",
    "time_hours",
    "name_in_email",
    "worst_status_active_inv",
    "status_last_archived_0_24m",
    "status_max_archived_0_24_months",
    "sum_paid_inv_0_12m",
    "avg_payment_span_0_12m",
    "max_paid_inv_0_24m",
    "account_incoming_debt_vs_paid_0_24m",
    "num_active_div_by_paid_inv_0_12m"
   }
)


def normalize_loan_row(raw_loan: dict) -> dict:
    normalized = {}
    for key, value in raw_loan.items():
        if key == "uuid" and value is not None:
            normalized[key] = str(value)
            continue

        if pd.isna(value):
            normalized[key] = None
            continue

        try:
            normalized[key] = value.item()
        except AttributeError:
            normalized[key] = value
    return normalized


def main() -> None:
    engine = get_engine()
    profile_by_feature = load_feature_profile(engine)
    if not profile_by_feature:
        raise RuntimeError("train_feat_dist is empty; cannot write PSI events")
    profile_by_feature = {
        k: v for k, v in profile_by_feature.items() if k in PSI_HIT_FEATURE_NAMES
    }
    if not profile_by_feature:
        raise RuntimeError(
            "train_feat_dist has no rows for PSI_HIT_FEATURE_NAMES "
            f"{sorted(PSI_HIT_FEATURE_NAMES)!r}"
        )
    last_uuid: str | None = fetch_last_completed_loan_uuid(engine)
    if last_uuid:
        print(
            json.dumps({"resume_from_loan_uuid": last_uuid}),
            flush=True,
        )

    interval_seconds = float(os.getenv("FETCH_INTERVAL_SECONDS", "5"))
    run_service_url = os.getenv("RUN_SERVICE_URL", "http://predict:8800/predict")
    request_timeout = float(os.getenv("RUN_SERVICE_TIMEOUT_SECONDS", "10"))
    # Write test_feat_hit only when loan_ids.id is a multiple of this stride (default 30 → 30, 60, …).
    psi_hit_loan_id_stride = max(1, int(os.getenv("FETCH_PSI_HIT_LOAN_ID_STRIDE", "30")))

    while True:
        loan_df = fetch_next_loan(engine, last_uuid=last_uuid)
        raw_loan = normalize_loan_row(loan_df.iloc[0].to_dict())
        current_uuid = raw_loan.get("uuid")

        if last_uuid is not None and current_uuid is not None:
            if UUID(str(current_uuid)) <= UUID(last_uuid):
                reset_test_cycle_tables(engine)
                print(
                    json.dumps(
                        {
                            "cycle_reset": True,
                            "reason": "wrapped_to_start_of_test_loans",
                            "previous_uuid": last_uuid,
                            "current_uuid": str(current_uuid),
                        }
                    ),
                    flush=True,
                )

        try:
            loan = LoanData(**raw_loan)
            payload = loan.dict()
            #print(json.dumps(payload, default=str), flush=True)

            response = requests.post(
                run_service_url,
                json=[payload],
                timeout=request_timeout,
            )
            response.raise_for_status()
            print(
                json.dumps(
                    {
                        "posted_to": run_service_url,
                        "status_code": response.status_code,
                        "response": response.json() if response.content else None,
                    },
                    default=str,
                ),
                flush=True,
            )
            loan_db_id = (
                fetch_loan_id_for_uuid(engine, str(current_uuid))
                if current_uuid is not None
                else None
            )
            if loan_db_id is None:
                print(
                    json.dumps(
                        {
                            "psi_hit_skipped": True,
                            "reason": "loan_ids_row_missing_after_predict",
                            "uuid": str(current_uuid) if current_uuid is not None else None,
                        }
                    ),
                    flush=True,
                )
            elif loan_db_id % psi_hit_loan_id_stride == 0:
                write_feature_bin_events(engine, profile_by_feature, raw_loan)
            else:
                print(
                    json.dumps(
                        {
                            "psi_hit_skipped": True,
                            "reason": "loan_id_not_on_stride",
                            "loan_ids_id": loan_db_id,
                            "stride": psi_hit_loan_id_stride,
                        }
                    ),
                    flush=True,
                )
            if current_uuid is not None:
                last_uuid = str(current_uuid)
        except ValidationError as exc:
            print(
                json.dumps(
                    {
                        "validation_error": exc.errors(),
                        "row": raw_loan,
                    },
                    default=str,
                ),
                flush=True,
            )
        except requests.RequestException as exc:
            print(
                json.dumps(
                    {
                        "post_error": str(exc),
                        "posted_to": run_service_url,
                        "row": raw_loan,
                    },
                    default=str,
                ),
                flush=True,
            )

        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
