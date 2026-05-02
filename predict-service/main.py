import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException

candidates = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2],
]

for candidate in candidates:
    if (candidate / "shared").exists():
        sys.path.insert(0, str(candidate))
        break

from shared.models import LoanData  # noqa: E402
from shared.postgres_utils import (  # noqa: E402
    get_engine,
    write_default_prob,
)
from src.lr1.lr1_predict import Lr1Predictor  # noqa: E402
from src.xgb2.xgb2_predict import Xgb2Predictor  # noqa: E402


class UnsupportedModelError(ValueError):
    pass


def build_predictor(model_name: str, model_path: str, schema_path: str) -> Any:
    predictors = {
        "lr1": Lr1Predictor,
        "xgb2": Xgb2Predictor,
    }
    predictor_class = predictors.get(model_name)
    if predictor_class is None:
        supported = ", ".join(sorted(predictors.keys()))
        raise UnsupportedModelError(
            f"Unsupported MODEL_NAME={model_name!r}. Supported models: {supported}"
        )
    return predictor_class(model_path=model_path, schema_path=schema_path)


app = FastAPI()
engine = get_engine()
model_name = os.getenv("MODEL_NAME", "xgb2")
prediction_threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
model_path = os.getenv("MODEL_PATH", f"/app/models/{model_name}_model.pkl")
schema_path = os.getenv(
    "MODEL_SCHEMA_PATH",
    f"/app/predict-service/src/{model_name}/{model_name}_schema.json",
)
predictor = build_predictor(model_name, model_path, schema_path)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "model": predictor.model_name}


@app.post("/predict")
async def predict(loans: List[LoanData]):
    try:
        loan_rows = [loan.dict() for loan in loans]
        predictions = predictor.predict(loan_rows)

        for loan_row, prediction in zip(loan_rows, predictions):
            write_default_prob(
                engine=engine,
                loan_uuid=prediction["uuid"],
                prob=prediction["pd"],
                model_name=prediction["model_name"],
                threshold_applied=prediction_threshold,
                loan_default=loan_row.get("default"),
            )
        return predictions
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800, reload=False)