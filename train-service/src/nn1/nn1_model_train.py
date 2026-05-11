import copy
import random
import sys
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Match import behavior from other train scripts.
candidates = [
    Path(__file__).resolve().parents[2],
    Path(__file__).resolve().parents[3],
]

for candidate in candidates:
    if (candidate / "shared").exists():
        sys.path.insert(0, str(candidate))
        break

from shared.postgres_utils import get_engine, fetch_full_dataset  # noqa: E402
from shared.ml_utils import (  # noqa: E402
    cat_na_cols,
    cat_no_na_cols,
    num_na_cols,
    num_no_na_cols,
    evaluate_nn,
    plot_confusion_matrix,
)

warnings.filterwarnings("ignore")

SEED = 42
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 256
LEARNING_RATE = 1e-4

cat_columns = cat_na_cols + cat_no_na_cols
num_columns = num_na_cols + num_no_na_cols
feature_cols = cat_columns + num_columns


class TabularDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, cat_cols: list[str], num_cols: list[str]):
        self.X_cat = torch.tensor(X[cat_cols].values, dtype=torch.long)
        self.X_num = torch.tensor(X[num_cols].values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class TabularNN(nn.Module):
    def __init__(self, cat_dims: list[int], num_dim: int):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_dim, min(16, int(cat_dim**0.5)))
                for cat_dim in cat_dims
            ]
        )

        emb_dim = sum(emb.embedding_dim for emb in self.embeddings)

        self.model = nn.Sequential(
            nn.Linear(emb_dim + num_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat(emb + [x_num], dim=1)
        return self.model(x).squeeze(1)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_datasets(engine, random_state: int = SEED):
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

    return X_train, X_val, y_train, y_val


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    cat_cols: list[str],
):
    X_train = X_train.copy()
    X_val = X_val.copy()
    cat_encoders: dict[str, dict[str, int]] = {}

    for col in cat_cols:
        X_train[col] = X_train[col].astype(str).fillna("MISSING")
        X_val[col] = X_val[col].astype(str).fillna("MISSING")

        unique_vals = sorted(X_train[col].unique().tolist())
        if "MISSING" not in unique_vals:
            unique_vals.append("MISSING")

        encoder = {val: i for i, val in enumerate(unique_vals)}
        X_train[col] = X_train[col].map(encoder).astype(int)
        X_val[col] = X_val[col].map(lambda x: encoder.get(x, encoder["MISSING"])).astype(int)
        cat_encoders[col] = encoder

    return X_train, X_val, cat_encoders


def preprocess_numeric_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    num_cols: list[str],
):
    X_train = X_train.copy()
    X_val = X_val.copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_val[num_cols] = imputer.transform(X_val[num_cols])

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])

    return X_train, X_val, imputer, scaler


def transform_with_fitted_artifacts(
    X: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    cat_encoders: dict[str, dict[str, int]],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> pd.DataFrame:
    X = X[feature_cols].copy()

    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("MISSING")
        encoder = cat_encoders[col]
        X[col] = X[col].map(lambda x: encoder.get(x, encoder["MISSING"])).astype(int)

    X[num_cols] = imputer.transform(X[num_cols])
    X[num_cols] = scaler.transform(X[num_cols])

    return X


def train_and_log_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    output_path: str = "models/dnn1_model.pkl",
):
    X_train, X_val, cat_encoders = encode_categorical_features(X_train, X_val, cat_columns)
    X_train, X_val, imputer, scaler = preprocess_numeric_features(X_train, X_val, num_columns)

    train_dataset = TabularDataset(X_train, y_train, cat_columns, num_columns)
    val_dataset = TabularDataset(X_val, y_val, cat_columns, num_columns)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    cat_dims = [len(cat_encoders[col]) for col in cat_columns]
    num_dim = len(num_columns)
    model = TabularNN(cat_dims, num_dim)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()],
        dtype=torch.float32,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_pr = 0.0
    counter = 0
    best_state = None
    best_epoch = 0

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="dnn_v1"):
        mlflow.set_tag("model_type", "dnn")
        mlflow.set_tag("encoding", "median_imputation + standardization")
        mlflow.log_input(
            mlflow.data.from_pandas(X_val.head(1000), name="val_sample"),
            context="validation",
        )
        mlflow.log_params(
            {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "patience": PATIENCE,
                "scoring": "average_precision",
            }
        )

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            for x_cat, x_num, y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(x_cat, x_num)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for x_cat, x_num, y_batch in val_loader:
                    logits = model(x_cat, x_num)
                    val_loss += criterion(logits, y_batch).item()

                    probs = torch.sigmoid(logits)
                    all_preds.append(probs.cpu())
                    all_targets.append(y_batch.cpu())

            all_preds_np = torch.cat(all_preds).numpy()
            all_targets_np = torch.cat(all_targets).numpy()

            val_loss /= len(val_loader)
            val_auc = roc_auc_score(all_targets_np, all_preds_np)
            val_pr = average_precision_score(all_targets_np, all_preds_np)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "val_pr": val_pr,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"AUC: {val_auc:.4f} | "
                f"PR: {val_pr:.4f}"
            )

            improved = val_pr > best_pr
            best_pr = val_pr if improved else best_pr
            counter = 0 if improved else counter + 1

            if improved:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            if counter >= PATIENCE:
                print("Early stopping")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        results = evaluate_nn(model, val_loader, min_precision=0.15)
        results.update(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_pr": val_pr,
                "best_val_pr": best_pr,
                "best_epoch": best_epoch,
            }
        )
        mlflow.log_param("threshold", float(results["threshold"]))
        ordered_metrics = {
            "threshold": float(results["threshold"]),
            "precision": float(results["precision"]),
            "recall": float(results["recall"]),
            "roc_auc": float(results["roc_auc"]),
            "pr_auc": float(results["pr_auc"]),
            "train_loss": float(results["train_loss"]),
            "val_loss": float(results["val_loss"]),
            "val_auc": float(results["val_auc"]),
            "val_pr": float(results["val_pr"]),
            "best_val_pr": float(results["best_val_pr"]),
            "best_epoch": float(results["best_epoch"]),
        }
        mlflow.log_metrics(ordered_metrics)

        fig = plot_confusion_matrix(results["confusion_matrix"])
        mlflow.log_figure(fig, "confusion_matrix_normalized.png")
        mlflow.pytorch.log_model(model, artifact_path="model")

    joblib.dump(
        {
            "model_state_dict": model.state_dict(),
            "cat_encoders": cat_encoders,
            "imputer": imputer,
            "scaler": scaler,
            "cat_columns": cat_columns,
            "num_columns": num_columns,
            "feature_cols": feature_cols,
            "cat_dims": cat_dims,
            "num_dim": num_dim,
        },
        output_path,
    )

    return model, cat_encoders, imputer, scaler


def main():
    set_seed(SEED)

    mlflow.set_tracking_uri("http://mlflow:5000/mlflow")
    mlflow.set_experiment("credit-risk-models")

    engine = get_engine()
    X_train, X_val, y_train, y_val = prepare_datasets(engine)

    model, cat_encoders, imputer, scaler = train_and_log_model(
        X_train,
        X_val,
        y_train,
        y_val,
    )

    X_test, y_test = fetch_full_dataset(engine, "test_loans")
    X_test = transform_with_fitted_artifacts(
        X_test,
        cat_columns,
        num_columns,
        cat_encoders,
        imputer,
        scaler,
    )

    test_dataset = TabularDataset(X_test, y_test, cat_columns, num_columns)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n=== NN1 Test-set evaluation ===")
    evaluate_nn(model, test_loader)


if __name__ == "__main__":
    main()
