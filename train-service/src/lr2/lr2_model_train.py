import warnings
import joblib
import numpy as np
import category_encoders as ce
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

candidates = [
    Path(__file__).resolve().parents[2],
    Path(__file__).resolve().parents[3],
]

for candidate in candidates:
    if (candidate / "shared").exists():
        sys.path.insert(0, str(candidate))
        break

from shared.postgres_utils import get_engine, fetch_full_dataset  # noqa: E402
from shared.preprocessing import add_missing_indicators, QuantileBinner  # noqa: E402
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

all_cat_cols = cat_na_cols + cat_no_na_cols
all_num_cols = num_na_cols + num_no_na_cols
feature_cols = all_cat_cols + all_num_cols


def prepare_datasets(engine, random_state=42):
    X_train_pool, y_train_pool = fetch_full_dataset(engine, "train_loans")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=0.2222,
        stratify=y_train_pool,
        random_state=random_state,
    )

    X_train = X_train[feature_cols].copy()
    X_val = X_val[feature_cols].copy()

    X_train = add_missing_indicators(X_train, num_na_cols)
    X_val = add_missing_indicators(X_val, num_na_cols)

    num_indicator_columns = [f"{col}_missing" for col in num_na_cols]
    return X_train, X_val, y_train, y_val, num_indicator_columns


def build_preprocessor(num_indicator_columns):
    num_na_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("binning", QuantileBinner(n_bins=5)),
        ("woe", ce.WOEEncoder(handle_missing="value", handle_unknown="value")),
    ])

    num_no_na_pipeline = Pipeline([
        ("binning", QuantileBinner(n_bins=5)),
        ("woe", ce.WOEEncoder(handle_missing="value", handle_unknown="value")),
    ])

    indicator_pipeline = "passthrough"

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("woe", ce.WOEEncoder(handle_missing="value", handle_unknown="value")),
    ])

    preprocessor = ColumnTransformer([
        ("num_na", num_na_pipeline, num_na_cols),
        ("num_no_na", num_no_na_pipeline, num_no_na_cols),
        ("indicators", indicator_pipeline, num_indicator_columns),
        ("cat_woe", cat_pipeline, all_cat_cols),
    ])
    return preprocessor


def train_and_log_model(
    X_train,
    y_train,
    X_val,
    y_val,
    preprocessor,
    output_path="models/lr2_model.pkl",
    n_iter=10,
    random_state=42,
):
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    param_dist = {"clf__C": np.logspace(-3, 2, 10)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        error_score="raise",
    )

    with mlflow.start_run(run_name="lr_v2"):
        random_search.fit(X_train, y_train.to_numpy())
        results = evaluate_and_report(
            random_search, X_val, y_val, metric="average_precision"
        )

        mlflow.set_tag("encoding", "woe + quantile_binning")
        mlflow.set_tag(
            "model_summary",
            "Logistic regression with WoE encoding and quantile binning (credit risk)",
        )

        mlflow.log_input(
            mlflow.data.from_pandas(X_val.head(1000), name="val_sample"),
            context="validation",
        )
        mlflow.log_params(random_search.best_params_)
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("penalty", "l1")
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("scoring", "average_precision")
        mlflow.log_param("threshold", float(results["threshold"]))

        results_without_cm = {
            k: float(v) for k, v in results.items() if k != "confusion_matrix"
        }
        mlflow.log_metrics(results_without_cm)

        cm = results["confusion_matrix"]
        fig = plot_confusion_matrix(cm)
        mlflow.log_figure(fig, "confusion_matrix_normalized.png")

        mlflow.sklearn.log_model(
            random_search.best_estimator_,
            artifact_path="model",
            input_example=X_val.head(5),
        )

    best_model = random_search.best_estimator_
    joblib.dump(
        {"model": best_model, "features": X_train.columns.tolist()},
        output_path,
    )
    return best_model


def main():
    mlflow.set_tracking_uri("http://mlflow:5000/mlflow")
    mlflow.set_experiment("credit-risk-models")
    engine = get_engine()
    X_train, X_val, y_train, y_val, num_indicator_columns = prepare_datasets(engine)
    preprocessor = build_preprocessor(num_indicator_columns)
    best_model = train_and_log_model(X_train, y_train, X_val, y_val, preprocessor)

    X_test, y_test = fetch_full_dataset(engine, "test_loans")
    X_test = X_test[feature_cols].copy()
    X_test = add_missing_indicators(X_test, num_na_cols)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print("\n=== LR2 Test-set evaluation ===")
    evaluate_and_report_loaded_model(
        best_model,
        X_test,
        y_test,
        metric="average_precision",
    )


if __name__ == "__main__":
    main()
