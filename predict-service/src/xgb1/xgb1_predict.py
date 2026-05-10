from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

CAT_NA_COLS = [
    "account_status",
    "account_worst_status_0_3m",
    "account_worst_status_12_24m",
    "account_worst_status_3_6m",
    "account_worst_status_6_12m",
    "num_arch_written_off_0_12m",
    "num_arch_written_off_12_24m",
    "worst_status_active_inv",
]

CAT_NO_NA_COLS = [
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
]


class Xgb1Predictor:
    def __init__(self, model_path: str, schema_path: str | None = None) -> None:
        self.model_name = "xgb_v1"
        self.model_path = Path(model_path)
        self.schema_path = Path(schema_path) if schema_path else None

        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.expected_features = list(bundle["features"])
        raw_cat_categories = bundle.get("cat_categories", {})
        self.cat_categories = {
            str(col): list(categories)
            for col, categories in raw_cat_categories.items()
        }

        all_cat_cols = CAT_NA_COLS + CAT_NO_NA_COLS
        self.all_cat_cols = [col for col in all_cat_cols if col in self.expected_features]
        self.all_num_cols = [
            col for col in self.expected_features if col not in self.all_cat_cols
        ]

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
                categories = self.cat_categories.get(col)
                if categories:
                    df[col] = pd.Categorical(df[col], categories=categories)
                else:
                    df[col] = df[col].astype("category")

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
