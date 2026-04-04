import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class QuantileBinner(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5):
        self.n_bins = n_bins
        self.bin_edges_ = {}
        self.columns_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        self.bin_edges_ = {}

        for col in self.columns_:
            non_na = X[col].dropna()

            if non_na.nunique() <= 1:
                self.bin_edges_[col] = None
                continue

            try:
                _, bins = pd.qcut(
                    non_na,
                    q=self.n_bins,
                    retbins=True,
                    duplicates="drop"
                )
                self.bin_edges_[col] = bins
            except ValueError:
                self.bin_edges_[col] = None

        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()

        for col in self.columns_:
            bins = self.bin_edges_[col]

            if bins is None:
                X[col] = "ALL"
                continue

            binned = pd.cut(
                X[col],
                bins=bins,
                include_lowest=True
            )

            X[col] = binned.astype(object)
            X.loc[X[col].isna(), col] = "MISSING"
            X[col] = X[col].astype(str)

        return X