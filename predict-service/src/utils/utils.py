import os
from typing import List


def parse_model_names() -> List[str]:
    names_csv = os.getenv("MODEL_NAMES", "").strip()
    if names_csv:
        names = [name.strip() for name in names_csv.split(",") if name.strip()]
        if names:
            return names
    return [os.getenv("MODEL_NAME", "xgb2").strip()]


def resolve_model_path(model_name: str) -> str:
    env_key = f"MODEL_PATH_{model_name.upper()}"
    return os.getenv(env_key, f"/app/models/{model_name}_model.pkl")


def resolve_schema_path(model_name: str) -> str | None:
    env_key = f"MODEL_SCHEMA_PATH_{model_name.upper()}"
    env_val = os.getenv(env_key)
    if env_val:
        return env_val

    # LR1 and XGB1 predictors do not require a schema file.
    if model_name in {"lr1", "xgb1"}:
        return None
    return f"/app/predict-service/src/{model_name}/{model_name}_schema.json"


def resolve_threshold(model_name: str) -> float:
    model_env_key = f"PREDICTION_THRESHOLD_{model_name.upper()}"
    raw_val = os.getenv(model_env_key, os.getenv("PREDICTION_THRESHOLD", "0.5"))
    return float(raw_val)
