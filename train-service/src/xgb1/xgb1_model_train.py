import joblib
import numpy as np
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
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
from shared.postgres_utils import get_engine,  fetch_full_dataset  # noqa: E402
from shared.ml_utils import (  # noqa: E402
    cat_na_cols,
    cat_no_na_cols,
    num_na_cols,
    num_no_na_cols,
    evaluate_and_report,
    plot_confusion_matrix,
)

# =========================
# GLOBAL FEATURE DEFINITIONS
# =========================
all_cat_cols = cat_na_cols + cat_no_na_cols
all_num_cols = num_na_cols + num_no_na_cols
feature_cols = all_cat_cols + all_num_cols


# =========================
# DATA PREPARATION
# =========================
def prepare_datasets(engine):
    X_train_pool, y_train_pool = fetch_full_dataset(engine, "train_loans")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=0.2222,
        stratify=y_train_pool,
        random_state=42,
    )

    X_train = X_train[feature_cols].copy()
    X_val = X_val[feature_cols].copy()

    # CATEGORICAL HANDLING (same logic as your current code)
    for col in all_cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")
        X_val[col] = X_val[col].cat.set_categories(
            X_train[col].cat.categories
        )

    return X_train, X_val, y_train, y_val


# =========================
# MODEL TRAINING + MLFLOW
# =========================
def train_and_log_model(
    X_train,
    y_train,
    X_val,
    y_val,
    output_path="models/xgb1_model.pkl",
    n_iter=10,
    random_state=42,
):
    # CLASS IMBALANCE
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # MODEL
    model = XGBClassifier(
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        enable_categorical=True,
        tree_method="hist",
    )

    # HYPERPARAMS
    param_dist = {
        "n_estimators": [200, 300, 500, 1000],
        "max_depth": [3, 5, 8],
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": np.linspace(0, 2, 5),
        "min_child_weight": [1, 3, 5, 10],
        "reg_lambda": np.linspace(0, 3, 5),
        "reg_alpha": np.linspace(0, 3, 5),
    }

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

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
    with mlflow.start_run(run_name="xgb_v1"):

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
            mlflow.data.from_pandas(X_val.head(1000), name="val_sample"),
            context="validation",
        )

        # PARAMS (aligned with LR style)
        mlflow.log_params(random_search.best_params_)
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("eval_metric", "aucpr")
        mlflow.log_param("scale_pos_weight", float(scale_pos_weight))
        mlflow.log_param("scoring", "average_precision")
        mlflow.log_param("threshold", 0.5)

        # METRICS
        results_without_cm = {
            k: float(v) for k, v in results.items() if k != "confusion_matrix"
        }
        mlflow.log_metrics(results_without_cm)

        # CONFUSION MATRIX
        cm = results["confusion_matrix"]
        fig = plot_confusion_matrix(cm)
        mlflow.log_figure(fig, "confusion_matrix_normalized.png")

        # MODEL LOG (same as LR)
        mlflow.sklearn.log_model(
            random_search.best_estimator_,
            artifact_path="model",
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
    # MLFLOW CONFIG (same as LR)
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("credit-risk-models")

    engine = get_engine()

    X_train, X_val, y_train, y_val = prepare_datasets(engine)

    train_and_log_model(
        X_train,
        y_train,
        X_val,
        y_val,
    )


if __name__ == "__main__":
    main()