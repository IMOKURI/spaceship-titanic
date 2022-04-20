import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocesses.p000_parse_features import SplitSlashZero, SplitSlashOne, SplitUnderBarZero

pd.set_option("display.max_columns", None)


def test_split_under_bar_zero():
    train = pd.read_csv("../inputs/train.csv")

    preprocessor = ColumnTransformer([("split_under_bar_zero_0", SplitUnderBarZero(), ["PassengerId"])])

    res = preprocessor.fit_transform(train)

    assert res.shape == (len(train), 1)

    # u, counts = np.unique(res, return_counts=True)
    # warnings.warn(f"unique values -> {u}: {counts}")

    assert np.array_equal(res.squeeze()[:3], np.array(["0001", "0002", "0003"], dtype=object))


def test_split_slash_zero():
    train = pd.read_csv("../inputs/train.csv")

    preprocessor = ColumnTransformer([("split_slash_zero_0", SplitSlashZero(), ["Cabin"])])

    res = preprocessor.fit_transform(train)

    assert res.shape == (len(train), 1)

    # u, counts = np.unique(res, return_counts=True)
    # warnings.warn(f"unique values -> {u}: {counts}")

    assert np.array_equal(res.squeeze()[:3], np.array(["B", "F", "A"], dtype=object))
    assert np.isnan(res.squeeze()[15])


def test_split_slash_one():
    # train = pd.read_csv("../inputs/train.csv")
    train = pd.read_csv("../inputs/test.csv")

    preprocessor = ColumnTransformer([("split_slash_one_0", SplitSlashOne(), ["Cabin"])])

    res = preprocessor.fit_transform(train)

    assert res.shape == (len(train), 1)

    # u, counts = np.unique(res, return_counts=True)
    # warnings.warn(f"unique values -> {u}: {counts}")

    # assert np.array_equal(res.squeeze()[:3], np.array(["0", "0", "0"], dtype=object))
    # assert np.isnan(res.squeeze()[15])
    assert np.array_equal(res.squeeze()[:3], np.array(["3", "4", "0"], dtype=object))
    assert np.isnan(res.squeeze()[4273])
