import numpy as np
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin


class SmartPreprocessor(BaseEstimator, TransformerMixin):
    """
    Build a scikit-learn compatible transformer that removes
    columns with > 50% missing values and log-transforms skewed numerical features.
    """

    def __init__(self, missing_threshold=0.5, skew_threshold=1.0):
        self.missing_threshold = missing_threshold
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        missing_rate = X.isnull().mean()
        self.col_to_drop = missing_rate[
            missing_rate > self.missing_threshold
        ].index.tolist()

        remaining = X.drop(columns=self.col_to_drop)
        numeric = remaining.select_dtypes(include=[np.number])
        skewness = numeric.apply(lambda c: skew(c.dropna()))
        self.skew_cols = skewness[skewness.abs() > self.skew_threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.drop(columns=self.col_to_drop).copy()
        for col in self.skew_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))

        return X
