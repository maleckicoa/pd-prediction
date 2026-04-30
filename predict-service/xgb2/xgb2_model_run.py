import sys
import json
import pandas as pd
import joblib

from pathlib import Path

sys.path.append("src")

from postgres_utils import get_engine

from ml_utils import (cat_na_cols, 
                    cat_no_na_cols, 
                    num_na_cols, 
                    num_no_na_cols, 
                    fetch_full_dataset,
                    fetch_random_loan,
                    write_default_prob)


engine = get_engine()

all_cat_cols = cat_na_cols + cat_no_na_cols
all_num_cols = num_na_cols + num_no_na_cols
feature_cols = all_cat_cols + all_num_cols

bundle = joblib.load('./models/xgb2_model.pkl')
model = bundle["model"]


SCHEMA_PATH = Path(__file__).resolve().parent / "xgb2_schema.json"

if SCHEMA_PATH.is_file():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    
    categories_map = {
        col: list(schema["categories_map"][col])
        for col in all_cat_cols
    }
    expected_features = list(schema["expected_features"])
else:
    X_train_full, y_train_full = fetch_full_dataset(engine, "train_loans")

    # keep only features
    X_train_full = X_train_full[feature_cols].copy()

    for col in all_num_cols:
        if col in X_train_full.columns:
            X_train_full[col] = pd.to_numeric(X_train_full[col], errors="coerce")


    for col in all_cat_cols:
        if col in X_train_full.columns:
            X_train_full[col] = X_train_full[col].astype("category")


    categories_map = {
        col: X_train_full[col].cat.categories.tolist()
        for col in all_cat_cols
    }
    expected_features = X_train_full.columns.tolist()

    SCHEMA_PATH.write_text(
        json.dumps(
            {
                "categories_map": categories_map,
                "expected_features": expected_features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


df_loan = fetch_random_loan(engine)
loan_uuid = str(df_loan["uuid"].iloc[0])

X_row = df_loan.drop(columns=["default"]).copy()
X_row = X_row.reindex(columns=expected_features)

for col in all_num_cols:
    if col in X_row.columns:
        X_row[col] = pd.to_numeric(X_row[col], errors="coerce")

for col in all_cat_cols:
    if col in X_row.columns:
        X_row[col] = pd.Categorical(
            X_row[col],
            categories=categories_map[col]
        )

# ===== PREDICT =====
prob = float(model.predict_proba(X_row)[:, 1][0])
write_default_prob(engine, loan_uuid, prob)
print({"uuid": loan_uuid, "prob": prob})



