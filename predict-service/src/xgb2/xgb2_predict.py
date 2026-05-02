import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


class Xgb2Predictor:
    def __init__(self, model_path: str, schema_path: str) -> None:
        self.model_name = "xgb2"
        self.model_path = Path(model_path)
        self.schema_path = Path(schema_path)

        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]

        schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.expected_features = list(schema["expected_features"])

        self.all_cat_cols = list(schema["categories_map"].keys())
        self.all_num_cols = [
            col for col in self.expected_features if col not in self.all_cat_cols
        ]
        self.categories_map: Dict[str, List[str]] = {
            col: list(schema["categories_map"][col])
            for col in self.all_cat_cols
        }

    def prepare_dataframe(self, loan_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        if not loan_rows:
            raise ValueError("Empty input data")

        df = pd.DataFrame(loan_rows)
        df = df.drop(columns=["default"], errors="ignore")
        df = df.reindex(columns=self.expected_features)

        for col in self.all_num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in self.all_cat_cols:
            if col in df.columns:
                df[col] = pd.Categorical(df[col], categories=self.categories_map[col])

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
