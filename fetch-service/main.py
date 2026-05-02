import json
import os
import sys
import time
from pathlib import Path

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

from shared.postgres_utils import get_engine, fetch_random_loan  # noqa: E402
from shared.models import LoanData  # noqa: E402


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

    interval_seconds = float(os.getenv("FETCH_INTERVAL_SECONDS", "5"))
    run_service_url = os.getenv("RUN_SERVICE_URL", "http://predict:8800/predict")
    request_timeout = float(os.getenv("RUN_SERVICE_TIMEOUT_SECONDS", "10"))

    while True:
        loan_df = fetch_random_loan(engine)
        raw_loan = normalize_loan_row(loan_df.iloc[0].to_dict())

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
