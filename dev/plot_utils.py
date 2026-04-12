import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sqlalchemy import text
from sklearn.metrics import roc_auc_score, recall_score, precision_score, average_precision_score, confusion_matrix
import seaborn as sns



def plot_binary_distribution(
    df,
    target="default",
    labels=("0", "1"),
    colors=("gray", "orange"),
    figsize=(5, 2),
    bar_width=0.45,
    label_fontsize=8,
    ylim_factor=1.3
):
    """
    Plots the binary distribution of the target variable
    """

    counts = df[target].value_counts().sort_index()
    total = counts.sum()

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(labels, counts, color=colors, width=bar_width)

    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=label_fontsize
        )

    # increase vertical headroom so labels don't overflow
    ax.set_ylim(0, counts.max() * ylim_factor)

    ax.set_title(f"Distribution of {target}")

    ax.margins(x=0.2)
    plt.tight_layout()
    plt.show()


def plot_bars_rel(
    df,
    feature='Credit amount',
    feature_type="numeric",
    amount_limit=10000,
    bucket_size=100,
    figsize=(12,5),
    variable_bar_width=True,
    hline=None,
    short_x_labels=False,
    max_categories=15
):
    """
    Bar plot of bad rate:
    - numeric → bucketed (NaN feature → "(missing)" bar)
    - categorical → per category (NaN → "(missing)" bar)
    """

    data = df[[feature, "default"]].dropna(subset=["default"])

    # --- numeric ---
    if feature_type == "numeric":
        miss = data[feature].isna()
        data_ok = data[~miss].copy()
        if amount_limit is not None:
            data_ok = data_ok[data_ok[feature] <= amount_limit]
        data_ok = data_ok.assign(bucket=(data_ok[feature] // bucket_size).astype(int))

        if miss.any():
            data_miss = data[miss].copy().assign(bucket="(missing)")
            data = pd.concat([data_ok, data_miss], axis=0, ignore_index=True)
        else:
            data = data_ok

    # --- categorical ---
    elif feature_type == "categorical":
        miss = data[feature].isna()
        bucket = data[feature].astype(str).where(~miss, "(missing)")
        # normalize string form of missing if column was already object/str NaN
        bucket = bucket.mask(bucket.str.lower().isin(("nan", "nat", "none")), "(missing)")
        data = data.assign(bucket=bucket)

    else:
        raise ValueError("feature_type must be 'numeric' or 'categorical'")

    # --- aggregation ---
    grouped = (
        data
        .groupby("bucket", observed=True)
        .agg(
            total=("default", "count"),
            bad=("default", "sum")
        )
        .reset_index()
    )

    # --- limit categories (always keep "(missing)" if present) ---
    if feature_type == "categorical":
        miss_g = grouped[grouped["bucket"] == "(missing)"]
        rest = grouped[grouped["bucket"] != "(missing)"].sort_values(
            "total", ascending=False
        ).head(max_categories)
        grouped = pd.concat([rest, miss_g], ignore_index=True)

    grouped["bad_rate"] = grouped["bad"] / grouped["total"]

    if feature_type == "numeric":
        miss_g = grouped[grouped["bucket"] == "(missing)"]
        num_g = grouped[grouped["bucket"] != "(missing)"].sort_values("bucket")
        grouped = pd.concat([num_g, miss_g], ignore_index=True)

    total_obs = grouped["total"].sum()
    total_bad = grouped["bad"].sum()

    x = np.arange(len(grouped))

    fig, ax = plt.subplots(figsize=figsize)

    # --- widths ---
    if variable_bar_width:
        widths = grouped["total"] / grouped["total"].max()
        widths = widths * 0.8
    else:
        widths = np.full(len(grouped), 0.6)

    bars = ax.bar(x, grouped["bad_rate"], width=widths, color="#f2e274")

    max_rate = grouped["bad_rate"].max()
    ax.set_ylim(0, max_rate * 1.25)

    # --- annotations (FIXED) ---
    for bar, row in zip(bars, grouped.itertuples(index=False)):

        total = row.total
        bad = row.bad
        bad_rate = row.bad_rate

        label = (
            f"n={int(total)}\n"
            f"{total/total_obs:.1%} of all\n"
            f"{(bad/total_bad if total_bad > 0 else 0):.1%} of all bad\n"
            f"{bad_rate:.1%} Bad rate"
        )

        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max_rate * 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=6
        )

    # --- horizontal line ---
    if hline is not None:
        ax.axhline(y=hline, color="red", linestyle="--", linewidth=1, alpha=0.7)

    # --- x labels ---
    if feature_type == "numeric":
        labels = []
        for b in grouped["bucket"]:
            if b == "(missing)":
                labels.append("(missing)")
            elif short_x_labels:
                i = int(b)
                labels.append(f"{int(i * bucket_size)}")
            else:
                i = int(b)
                labels.append(f"{int(i * bucket_size)}-{int((i + 1) * bucket_size)}")
    else:
        labels = grouped["bucket"].astype(str).tolist()

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)

    ax.set_title(f"Bad Loans Rate by {feature}", fontsize=10)
    ax.set_xlabel(feature, fontsize=9)
    ax.set_ylabel("Bad Loans Rate", fontsize=9)

    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()