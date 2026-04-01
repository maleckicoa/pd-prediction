import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class QuantileBinner(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_edges_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        for col in X.columns:
            non_na = X[col].dropna()

            _, bins = pd.qcut(
                non_na,
                q=self.n_bins,
                retbins=True,
                duplicates='drop'
            )

            self.bin_edges_[col] = bins

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in X.columns:
            bins = self.bin_edges_[col]

            X[col] = pd.cut(
                X[col],
                bins=bins,
                include_lowest=True
            ).astype(str)

            X[col] = X[col].replace("nan", "MISSING")

        return X