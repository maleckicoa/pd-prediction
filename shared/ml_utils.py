import pandas as pd
import numpy as np
from sqlalchemy import text
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    roc_auc_score, 
    recall_score, 
    precision_score, 
    average_precision_score, 
    confusion_matrix, 
    precision_recall_curve
)

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


def evaluate_and_report1(random_search, X_test, y_test, threshold=0.5, metric="average_precision"):

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


def evaluate_and_report(
    random_search,
    X_test,
    y_test,
    min_precision=0.15,
    metric="average_precision",
):

    print(f"Best CV {metric}:", random_search.best_score_)
    print("\nBest Parameters:")
    for k, v in random_search.best_params_.items():
        print(f"{k}: {v}")

    best_model = random_search.best_estimator_

    # probabilities
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # thresholds has len = n-1 compared to precision/recall
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    valid = precisions >= min_precision

    if np.any(valid):
        best_idx = np.argmax(recalls[valid])
        best_threshold = thresholds[valid][best_idx]
    else:
        print("\nWARNING: No threshold satisfies precision constraint.")
        best_threshold = 0.5

    print(f"\nChosen threshold (precision >= {min_precision}): {best_threshold:.4f}")

    y_pred = (y_pred_proba >= best_threshold).astype(int)

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
        "threshold": best_threshold,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm
    }


def evaluate_and_report_loaded_model(
    best_model,
    X_test,
    y_test,
    min_precision=0.15,
    metric="average_precision",
):

    # =========================
    # PROBABILITIES
    # =========================
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # =========================
    # THRESHOLD SELECTION (same logic)
    # =========================
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # align shapes (thresholds = n-1)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    valid = precisions >= min_precision

    if np.any(valid):
        best_idx = np.argmax(recalls[valid])
        best_threshold = thresholds[valid][best_idx]
    else:
        print("\nWARNING: No threshold satisfies precision constraint.")
        best_threshold = 0.5

    print(f"\nChosen threshold (precision >= {min_precision}): {best_threshold:.4f}")

    # =========================
    # PREDICTIONS
    # =========================
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    # =========================
    # METRICS
    # =========================
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
        "threshold": best_threshold,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm
    }



def evaluate_nn(
    model,
    loader,
    min_precision=0.15,
):
    """
    Evaluate PyTorch model with threshold selection:
    maximize recall subject to precision >= min_precision
    """

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_score,
        recall_score,
        confusion_matrix,
        precision_recall_curve,
    )

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cat, x_num, y_batch in loader:
            logits = model(x_cat, x_num)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # =========================
    # THRESHOLD SELECTION
    # =========================
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_preds)

    # align shapes (thresholds = n-1)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    valid = precisions >= min_precision

    if np.any(valid):
        best_idx = np.argmax(recalls[valid])
        best_threshold = thresholds[valid][best_idx]
    else:
        print("\nWARNING: No threshold satisfies precision constraint.")
        best_threshold = 0.5

    print(f"\nChosen threshold (precision >= {min_precision}): {best_threshold:.4f}")

    # =========================
    # APPLY THRESHOLD
    # =========================
    y_pred = (all_preds >= best_threshold).astype(int)

    # =========================
    # METRICS
    # =========================
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
        "threshold": best_threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }
