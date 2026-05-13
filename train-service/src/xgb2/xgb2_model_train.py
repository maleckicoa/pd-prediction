import warnings
import joblib
import numpy as np
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import category_encoders as ce

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# =========================
# FIX IMPORT PATH (shared/)
# =========================
candidates = [
    Path(__file__).resolve().parents[2],
    Path(__file__).resolve().parents[3],
]

for candidate in candidates:
    if (candidate / "shared").exists():
        sys.path.insert(0, str(candidate))
        break

# =========================
# IMPORT SHARED MODULES
# =========================
from shared.preprocessing import QuantileBinner  # noqa: E402
from shared.postgres_utils import get_engine, fetch_full_dataset  # noqa: E402
from shared.ml_utils import (  # noqa: E402
    cat_na_cols,
    cat_no_na_cols,
    num_na_cols,
    num_no_na_cols,
    evaluate_and_report,
    evaluate_and_report_loaded_model,
    plot_confusion_matrix,
)

warnings.filterwarnings("ignore")

# =========================
# GLOBAL FEATURE DEFINITIONS
# =========================
all_cat_cols = cat_na_cols + cat_no_na_cols
all_num_cols = num_na_cols + num_no_na_cols
feature_cols = all_cat_cols + all_num_cols


# =========================
# PREPROCESSOR
# =========================
def build_preprocessor():

    num_na_pipeline = Pipeline(
        [
            ("binning", QuantileBinner(n_bins=10)),
            (
                "woe",
                ce.WOEEncoder(
                    handle_unknown="value",
                    handle_missing="value",
                ),
            ),
        ]
    )

    num_pipeline = Pipeline(
        [
            ("passthrough", "passthrough"),
        ]
    )

    cat_pipeline = Pipeline(
        [
            (
                "encoder",
                ce.WOEEncoder(
                    handle_unknown="value",
                    handle_missing="value",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num_na_woe", num_na_pipeline, num_na_cols),
            ("num", num_pipeline, num_no_na_cols),
            ("cat_woe", cat_pipeline, all_cat_cols),
        ]
    )

    return preprocessor


# =========================
# DATA PREPARATION
# =========================
def prepare_datasets(engine):

    X_train_pool, y_train_pool = fetch_full_dataset(
        engine,
        "train_loans",
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=0.2222,
        stratify=y_train_pool,
        random_state=42,
    )

    X_train = X_train[feature_cols].copy()
    X_val = X_val[feature_cols].copy()

    return X_train, X_val, y_train, y_val


# =========================
# MODEL TRAINING + MLFLOW
# =========================
def train_and_log_model(
    X_train,
    y_train,
    X_val,
    y_val,
    preprocessor,
    output_path="models/xgb2_model.pkl",
    n_iter=10,
    random_state=42,
):

    # CLASS IMBALANCE
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # MODEL
    model = Pipeline(
        [
            ("preprocessing", preprocessor),
            (
                "clf",
                XGBClassifier(
                    eval_metric="aucpr",
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    use_label_encoder=False,
                ),
            ),
        ]
    )

    # HYPERPARAMS
    param_dist = {
        "clf__n_estimators": [200, 300, 500, 1000],
        "clf__max_depth": [3, 5, 8],
        "clf__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__gamma": np.linspace(0, 2, 5),
        "clf__min_child_weight": [1, 3, 5, 10],
        "clf__reg_lambda": np.linspace(0, 3, 5),
        "clf__reg_alpha": np.linspace(0, 3, 5),
    }

    # CV
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
        error_score="raise",
    )

    # =========================
    # MLFLOW RUN
    # =========================
    with mlflow.start_run(run_name="xgb_v2"):

        # TRAIN
        random_search.fit(X_train, y_train)

        # EVALUATE
        results = evaluate_and_report(
            random_search,
            X_val,
            y_val,
            metric="average_precision",
        )

        # INPUT SAMPLE
        mlflow.log_input(
            mlflow.data.from_pandas(
                X_val.head(1000),
                name="val_sample",
            ),
            context="validation",
        )

        # PARAMS
        mlflow.log_params(random_search.best_params_)
        mlflow.log_param("model_type", "xgboost_woe")
        mlflow.log_param("eval_metric", "aucpr")
        mlflow.log_param(
            "scale_pos_weight",
            float(scale_pos_weight),
        )
        mlflow.log_param("scoring", "average_precision")
        mlflow.log_param("threshold", 0.5)
        mlflow.log_param("encoder", "woe")
        mlflow.log_param("binning", "quantile")

        # METRICS
        results_without_cm = {
            k: float(v)
            for k, v in results.items()
            if k != "confusion_matrix"
        }

        mlflow.log_metrics(results_without_cm)

        # CONFUSION MATRIX
        cm = results["confusion_matrix"]

        fig = plot_confusion_matrix(cm)

        mlflow.log_figure(
            fig,
            "confusion_matrix_normalized.png",
        )

        # MODEL LOG
        mlflow.sklearn.log_model(
            random_search.best_estimator_,
            artifact_path="xgb_v2",
            input_example=X_val.head(5),
        )

    # =========================
    # SAVE LOCALLY
    # =========================
    best_model = random_search.best_estimator_

    joblib.dump(
        {
            "model": best_model,
            "features": X_train.columns.tolist(),
        },
        output_path,
    )

    return best_model


# =========================
# MAIN
# =========================
def main():

    # MLFLOW CONFIG
    mlflow.set_tracking_uri("http://mlflow:5000/mlflow")
    experiment_name = "credit-risk-models"
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is not None:
        for run in client.search_runs(experiment_ids=[exp.experiment_id], max_results=500):
            if run.info.run_name == "xgb_v2":
                client.delete_run(run.info.run_id)
    mlflow.set_experiment(experiment_name)

    engine = get_engine()

    X_train, X_val, y_train, y_val = prepare_datasets(engine)

    preprocessor = build_preprocessor()

    best_model = train_and_log_model(
        X_train,
        y_train,
        X_val,
        y_val,
        preprocessor,
    )

    # Evaluate the trained model on the held-out test dataset.
    X_test, y_test = fetch_full_dataset(engine, "test_loans")
    X_test = X_test[feature_cols].copy()

    print("\n=== XGB2 Test-set evaluation ===")
    evaluate_and_report_loaded_model(
        best_model,
        X_test,
        y_test,
        metric="average_precision",
    )


if __name__ == "__main__":
    main()