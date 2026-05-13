import logging
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
from src.utils import (  # noqa: E402
    parse_model_names,
    resolve_model_path,
    resolve_schema_path,
    resolve_threshold,
)
from src.lr1.lr1_predict import Lr1Predictor  # noqa: E402
#from src.lr2.lr2_predict import Lr2Predictor  # noqa: E402
from src.xgb1.xgb1_predict import Xgb1Predictor  # noqa: E402
from src.xgb2.xgb2_predict import Xgb2Predictor  # noqa: E402

logger = logging.getLogger(__name__)


class UnsupportedModelError(ValueError):
    pass


def build_predictor(model_name: str, model_path: str, schema_path: str | None) -> Any:
    predictors = {
        "lr1": Lr1Predictor,
        #"lr2": Lr2Predictor,
        "xgb1": Xgb1Predictor,
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
model_names = parse_model_names()
predictors: Dict[str, Any] = {}
threshold_by_model: Dict[str, float] = {}

for name in model_names:
    try:
        predictors[name] = build_predictor(
            model_name=name,
            model_path=resolve_model_path(name),
            schema_path=resolve_schema_path(name),
        )
        threshold_by_model[name] = resolve_threshold(name)
    except Exception as exc:
        logger.warning("Skipping model %r: %s", name, exc)

if not predictors:
    sys.exit("No models loaded — check logs.")
if "lr1" in model_names and "lr1" not in predictors:
    sys.exit("lr1 model failed to load.")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "models": sorted(list(predictors.keys()))}


@app.post("/predict")
async def predict(loans: List[LoanData]):
    try:
        loan_rows = [loan.dict() for loan in loans]
        all_predictions: List[Dict[str, Any]] = []

        for model_name, predictor in predictors.items():
            predictions = predictor.predict(loan_rows)
            all_predictions.extend(predictions)

            loan_default_by_uuid = {
                str(loan_row.get("uuid")): loan_row.get("default") for loan_row in loan_rows
            }
            for prediction in predictions:
                loan_uuid = str(prediction["uuid"])
                write_default_prob(
                    engine=engine,
                    loan_uuid=loan_uuid,
                    prob=prediction["pd"],
                    model_name=prediction["model_name"],
                    threshold_applied=threshold_by_model[model_name],
                    loan_default=loan_default_by_uuid.get(loan_uuid),
                )

        return all_predictions
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800, reload=False)