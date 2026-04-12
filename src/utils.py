import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sqlalchemy import text
from sklearn.metrics import roc_auc_score, recall_score, precision_score, average_precision_score, confusion_matrix
import seaborn as sns


cat_na_cols = [
    'account_status', 
    'account_worst_status_0_3m', 
    'account_worst_status_12_24m',
    'account_worst_status_3_6m', 
    'account_worst_status_6_12m',
    'num_arch_written_off_0_12m', 
    'num_arch_written_off_12_24m',
    'worst_status_active_inv'
]

cat_no_na_cols = [
    'merchant_category', 
    'merchant_group', 
    'has_paid', 
    'name_in_email',
    'num_arch_dc_0_12m', 
    'num_arch_dc_12_24m',
    'status_last_archived_0_24m', 
    'status_2nd_last_archived_0_24m',
    'status_3rd_last_archived_0_24m',
    'status_max_archived_0_6_months', 
    'status_max_archived_0_12_months',
    'status_max_archived_0_24_months'
]

num_na_cols = [
    'account_days_in_dc_12_24m', 
    'account_days_in_rem_12_24m',
    'account_days_in_term_12_24m', 
    'account_incoming_debt_vs_paid_0_24m',
    'avg_payment_span_0_12m', 
    'avg_payment_span_0_3m',
    'num_active_div_by_paid_inv_0_12m'
]

num_no_na_cols = [
    'account_amount_added_12_24m', 
    'age', 
    'max_paid_inv_0_12m',
    'max_paid_inv_0_24m', 
    'num_active_inv', 
    'num_arch_ok_0_12m',
    'num_arch_ok_12_24m', 
    'num_arch_rem_0_12m', 
    'num_unpaid_bills',
    'recovery_debt', 
    'sum_capital_paid_account_0_12m',
    'sum_capital_paid_account_12_24m', 
    'sum_paid_inv_0_12m',
    'time_hours'
]

def plot_confusion_matrix(cm, normalize=False):
 
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"
    
    plt.figure(figsize=(4, 4))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"]
    )
    
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    return plt.gcf()  # return figure (important for MLflow)


def evaluate_and_report(random_search, X_test, y_test, threshold=0.5, metric="average_precision"):

    print(f"Best CV {metric}:", random_search.best_score_)
    print("\nBest Parameters:")
    for k, v in random_search.best_params_.items():
        print(f"{k}: {v}")

    # =========================
    # TEST EVALUATION
    # =========================
    best_model = random_search.best_estimator_

    # probabilities
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # threshold (can tune later)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nTest ROC AUC:", roc_auc)
    print("Test PR AUC:", pr_auc)
    print("Precision:", precision)
    print("Recall:", recall)

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"]
    ))

    return {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "precision": precision,
    "recall": recall,
    "confusion_matrix": cm
}

def evaluate_and_report_loaded_model(best_model, X_test, y_test, threshold=0.5, metric="average_precision"):

    # probabilities
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # threshold (can tune later)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nTest ROC AUC:", roc_auc)
    print("Test PR AUC:", pr_auc)
    print("Precision:", precision)
    print("Recall:", recall)

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"]
    ))

    return {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "precision": precision,
    "recall": recall,
    "confusion_matrix": cm
}

def evaluate_nn(model, loader, threshold=0.5):
    """
    Evaluate PyTorch model on a DataLoader and print metrics.
    """

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_score,
        recall_score,
        confusion_matrix
    )

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cat, x_num, y_batch in loader:
            logits = model(x_cat, x_num)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.numpy())
            all_targets.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # threshold
    y_pred = (all_preds >= threshold).astype(int)

    # metrics
    roc_auc = roc_auc_score(all_targets, all_preds)
    pr_auc = average_precision_score(all_targets, all_preds)
    precision = precision_score(all_targets, y_pred)
    recall = recall_score(all_targets, y_pred)
    cm = confusion_matrix(all_targets, y_pred)

    print("\nEvaluation Results")
    print("------------------")
    print("ROC AUC:", roc_auc)
    print("PR AUC:", pr_auc)
    print("Precision:", precision)
    print("Recall:", recall)

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"]
    ))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall
    }


# def predict_single(model, X, y, idx, threshold = 0.5):
#     x = X.iloc[[idx]]
#     y_true = y.iloc[idx]
    
#     prob = model.predict_proba(x)[0, 1]
#     pred = int(prob >= threshold)
    
#     print(f"Prediction: {pred}")
#     print(f"Probability: {prob:.4f}")
#     print(f"Actual: {y_true}")
    
#     return prob, pred, y_true


# def predict_single_row(model, x_one_row: pd.DataFrame, y_true=None, threshold: float = 0.5):
#     """One-row feature frame (same columns as training `X` passed to the model)."""
#     if len(x_one_row) != 1:
#         raise ValueError("x_one_row must have exactly one row")

#     prob = model.predict_proba(x_one_row)[0, 1]
#     pred = int(prob >= threshold)

#     print(f"Prediction: {pred}")
#     print(f"Probability: {prob:.4f}")
#     if y_true is None:
#         print("Actual: (not provided)")
#     else:
#         print(f"Actual: {y_true}")

#     return prob, pred, y_true

def fetch_full_dataset(engine, table_name: str):
    stmt = text(f"SELECT * FROM {table_name} ORDER BY uuid")
    df = pd.read_sql(stmt, engine)
    y = df["default"].astype(int)
    X = df.drop(columns=["default"]).copy()
    return X, y

def fetch_random_loan(engine) -> pd.DataFrame:
    stmt = text("SELECT * FROM test_loans ORDER BY random() LIMIT 1")
    df = pd.read_sql(stmt, engine)
    if df.empty:
        raise ValueError("No rows found in test_loans")
    return df

def write_default_prob(engine, loan_uuid: str, prob: float) -> None:
    stmt = text(
        """
        INSERT INTO default_probs (uuid, pd)
        VALUES (CAST(:uid AS uuid), :pd)
        ON CONFLICT (uuid) DO UPDATE
        SET pd = EXCLUDED.pd
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, {"uid": loan_uuid, "pd": prob})
