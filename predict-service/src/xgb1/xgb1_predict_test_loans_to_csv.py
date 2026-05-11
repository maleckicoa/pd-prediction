import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sqlalchemy import text

# Ensure both imports work when executed as a script:
# - `from src...` requires /app/predict-service on sys.path
# - `from shared...` requires /app on sys.path
project_root = Path(__file__).resolve().parents[2]
workspace_root = project_root.parent
candidates = [project_root, workspace_root]

for candidate in candidates:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from shared.postgres_utils import get_engine  # noqa: E402
from src.utils import resolve_model_path, resolve_threshold  # noqa: E402
from src.xgb1.xgb1_predict import Xgb1Predictor  # noqa: E402


def normalize_row(raw_loan: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
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

    stmt = text("SELECT * FROM test_loans ORDER BY uuid ASC")
    test_df = pd.read_sql(stmt, engine)
    if test_df.empty:
        print("No rows found in test_loans. Nothing to score.")
        return

    loan_rows = [normalize_row(row) for row in test_df.to_dict(orient="records")]

    predictor = Xgb1Predictor(
        model_path=resolve_model_path("xgb1"),
        schema_path=None,
    )
    predictions = predictor.predict(loan_rows)
    pred_df = pd.DataFrame(predictions)

    output_df = test_df[["uuid", "default"]].copy()
    output_df["uuid"] = output_df["uuid"].astype(str)
    output_df = output_df.rename(columns={"default": "loan_default"})
    output_df = output_df.merge(pred_df, on="uuid", how="left")

    threshold = resolve_threshold("xgb1")
    output_df["threshold_applied"] = threshold
    output_df["predicted_default"] = (output_df["pd"] > threshold).astype(int)

    output_path = Path(
        os.getenv(
            "XGB1_TEST_PREDICTIONS_CSV_PATH",
            "/app/models/xgb1_test_predictions.csv",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Wrote {len(output_df)} rows to {output_path}")
    print(f"Columns: {', '.join(output_df.columns)}")


if __name__ == "__main__":
    main()
