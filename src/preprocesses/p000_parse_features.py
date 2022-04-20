import numpy as np
import pandas as pd

from .base import BaseTransformer


class ToFloat(BaseTransformer):
    def transform(self, X):
        self.n_features_in_ = 1

        if isinstance(X, pd.DataFrame):
            return X.astype(np.float32)
        if isinstance(X, np.ndarray):
            def func(x):
                try:
                    return float(x)
                except ValueError:
                    return None
            return np.vectorize(func)(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return np.array(["float_0"])


class SplitUnderBarZero(BaseTransformer):
    def transform(self, X):
        self.n_features_in_ = 1

        if isinstance(X, pd.DataFrame):
            return X.apply(lambda x: x.str.split("_").str[0])
        if isinstance(X, np.ndarray):
            func = lambda x: x.split("_")[0] if x is not None else x
            return np.vectorize(func)(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return np.array(["split_under_bar_zero_0"])


class SplitSlashZero(BaseTransformer):
    def transform(self, X):
        self.n_features_in_ = 1

        if isinstance(X, pd.DataFrame):
            return X.apply(lambda x: x.str.split("/").str[0])
        if isinstance(X, np.ndarray):
            func = lambda x: x.split("/")[0] if x is not None else x
            return np.vectorize(func)(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return np.array(["split_slash_zero_0"])


class SplitSlashOne(BaseTransformer):
    def transform(self, X):
        self.n_features_in_ = 1

        if isinstance(X, pd.DataFrame):
            return X.apply(lambda x: x.str.split("/").str[1])
        if isinstance(X, np.ndarray):
            func = lambda x: x.split("/")[1] if x is not None else x
            return np.vectorize(func)(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return np.array(["split_slash_one_0"])


class SplitSlashLast(BaseTransformer):
    def transform(self, X):
        self.n_features_in_ = 1

        if isinstance(X, pd.DataFrame):
            return X.apply(lambda x: x.str.split("/").str[-1])
        if isinstance(X, np.ndarray):
            func = lambda x: x.split("/")[-1] if x is not None else x
            return np.vectorize(func)(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return np.array(["split_slash_last_0"])
