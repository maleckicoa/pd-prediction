import warnings

import joblib
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import QuantileBinner

warnings.filterwarnings("ignore")


def load_and_split_data(csv_path="./dataset.csv", sep=";", random_state=42):
    """Load dataset, drop rows without default, stratified train/test split."""
    df = pd.read_csv(csv_path, sep=sep)
    df['has_paid'] = df['has_paid'].astype(int)

    df_nna = df[df['default'].notna()]
    X = df_nna.drop(columns=["default"])
    y = df_nna["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    """Build ColumnTransformer: numeric NA → bin+WoE, numeric clean → passthrough, cat → WoE."""
    na_columns = X_train.columns[X_train.isna().any()].tolist()

    numeric_columns = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_columns = [col for col in X_train.columns if col not in numeric_columns]

    num_na_columns = [col for col in na_columns if col in numeric_columns]
    num_no_na_columns = [col for col in numeric_columns if col not in num_na_columns]

    num_na_pipeline = Pipeline([
        ("binning", QuantileBinner(n_bins=10)),
        ("woe", ce.WOEEncoder(handle_unknown='value', handle_missing='value'))
    ])

    num_pipeline = Pipeline([
        ("passthrough", "passthrough")
    ])

    cat_pipeline = Pipeline([
        ("encoder", ce.WOEEncoder(handle_unknown='value', handle_missing='value'))
    ])

    preprocessor = ColumnTransformer([
        ("num_na_woe", num_na_pipeline, num_na_columns),
        ("num", num_pipeline, num_no_na_columns),
        ("cat_woe", cat_pipeline, cat_columns),
    ])

    return preprocessor


def train_and_save_bundle(X_train, y_train, preprocessor, output_path="model_bundle.pkl",
                          n_iter=10, random_state=42):
    """Hyperparameter search, fit best pipeline, persist model + feature list."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("clf", XGBClassifier(
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            use_label_encoder=False
        ))
    ])

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    joblib.dump({
        "model": best_model,
        "features": X_train.columns.tolist()
    }, output_path)
    return best_model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    preprocessor = build_preprocessor(X_train)
    train_and_save_bundle(X_train, y_train, preprocessor)
