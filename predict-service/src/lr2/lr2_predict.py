from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


class Lr2Predictor:
    """Sklearn pipeline (WoE + quantile binning + LR) saved by train-service lr2_model_train."""

    def __init__(self, model_path: str, schema_path: str | None = None) -> None:
        self.model_name = "lr_v2"
        self.model_path = Path(model_path)
        self.schema_path = Path(schema_path) if schema_path else None

        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.expected_features = list(bundle["features"])

    def prepare_dataframe(self, loan_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        if not loan_rows:
            raise ValueError("Empty input data")

        df = pd.DataFrame(loan_rows)
        df = df.drop(columns=["default"], errors="ignore")

        indicator_cols = [col for col in self.expected_features if col.endswith("_missing")]
        for indicator_col in indicator_cols:
            base_col = indicator_col[: -len("_missing")]
            if indicator_col in df.columns:
                continue
            if base_col in df.columns:
                df[indicator_col] = df[base_col].isna().astype(int)
            else:
                df[indicator_col] = 1

        df = df.reindex(columns=self.expected_features)
        return df

    def predict(self, loan_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raw_df = pd.DataFrame(loan_rows)
        if "uuid" not in raw_df.columns:
            raise ValueError("Missing required field: uuid")

        uuids = raw_df["uuid"].astype(str).tolist()
        df = self.prepare_dataframe(loan_rows)
        probs = self.model.predict_proba(df)[:, 1]
        return [
            {
                "uuid": loan_uuid,
                "pd": float(prob),
                "model_name": self.model_name,
            }
            for loan_uuid, prob in zip(uuids, probs)
        ]
