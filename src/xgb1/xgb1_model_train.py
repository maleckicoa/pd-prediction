import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================
# FEATURE DEFINITIONS
# =========================

def get_feature_columns():
    cat_na_cols = [
        'account_status','account_worst_status_0_3m','account_worst_status_12_24m',
        'account_worst_status_3_6m','account_worst_status_6_12m',
        'num_arch_written_off_0_12m','num_arch_written_off_12_24m',
        'worst_status_active_inv'
    ]

    cat_no_na_cols = [
        'merchant_category','merchant_group','has_paid','name_in_email',
        'num_arch_dc_0_12m','num_arch_dc_12_24m',
        'status_last_archived_0_24m','status_2nd_last_archived_0_24m',
        'status_3rd_last_archived_0_24m',
        'status_max_archived_0_6_months','status_max_archived_0_12_months',
        'status_max_archived_0_24_months'
    ]

    num_na_cols = [
        'account_days_in_dc_12_24m','account_days_in_rem_12_24m',
        'account_days_in_term_12_24m','account_incoming_debt_vs_paid_0_24m',
        'avg_payment_span_0_12m','avg_payment_span_0_3m',
        'num_active_div_by_paid_inv_0_12m'
    ]

    num_no_na_cols = [
        'account_amount_added_12_24m','age','max_paid_inv_0_12m',
        'max_paid_inv_0_24m','num_active_inv','num_arch_ok_0_12m',
        'num_arch_ok_12_24m','num_arch_rem_0_12m','num_unpaid_bills',
        'recovery_debt','sum_capital_paid_account_0_12m',
        'sum_capital_paid_account_12_24m','sum_paid_inv_0_12m','time_hours'
    ]

    all_cat_cols = cat_na_cols + cat_no_na_cols
    all_num_cols = num_na_cols + num_no_na_cols
    feature_cols = all_cat_cols + all_num_cols

    return feature_cols, all_cat_cols, all_num_cols


# =========================
# DATA LOADING
# =========================

def load_and_split_data(csv_path="./data/dataset.csv", sep=";", random_state=42):
    df = pd.read_csv(csv_path, sep=sep)
    df['has_paid'] = df['has_paid'].astype(int)

    df_nna = df[df['default'].notna()]

    feature_cols, all_cat_cols, _ = get_feature_columns()

    X = df_nna.drop(columns=["default"])
    X = X[feature_cols]
    y = df_nna["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=random_state
    )

    # =========================
    # CAST CATEGORICAL
    # =========================
    for col in all_cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

        # align categories (important!)
        X_test[col] = X_test[col].cat.set_categories(
            X_train[col].cat.categories
        )

    return X_train, X_test, y_train, y_test


# =========================
# MODEL TRAINING
# =========================

def train_and_save_model(
    X_train,
    y_train,
    output_path="models/xgb1_model.pkl",
    n_iter=10,
    random_state=42
):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        enable_categorical=True,
        tree_method="hist",
    )

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

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    joblib.dump({
        "model": best_model,
        "features": X_train.columns.tolist()
    }, output_path)

    return best_model


# =========================
# MAIN
# =========================

def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    train_and_save_model(X_train, y_train)


if __name__ == "__main__":
    main()