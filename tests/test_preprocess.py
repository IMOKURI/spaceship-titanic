import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

pd.set_option("display.max_columns", None)


def test_simple_imputer():
    train = pd.read_csv("../inputs/train.csv")

    assert type(train) == pd.DataFrame
    assert train["HomePlanet"].isnull().sum() != 0

    preprocessor = ColumnTransformer(
        [("simple_imputer", SimpleImputer(strategy="constant", fill_value="unknown"), ["HomePlanet"])]
    )

    res = preprocessor.fit_transform(train)

    assert res.shape == (len(train), 1)

    # u, counts = np.unique(res, return_counts=True)
    # warnings.warn(f"unique values -> {u}: {counts}")

    assert np.all(pd.isnull(res) == False)


def test_ordinal_encoder():
    train = pd.read_csv("../inputs/train.csv")

    preprocessor = ColumnTransformer([("ordinal", OrdinalEncoder(), ["HomePlanet"])])

    res = preprocessor.fit_transform(train)

    assert res.shape == (len(train), 1)
    assert np.array_equal(res.squeeze()[:3], np.array([1, 0, 1], dtype=np.float32))
    assert np.isnan(res.squeeze()[59])


def test_one_hot_encoder():
    train = pd.read_csv("../inputs/train.csv")

    preprocessor = ColumnTransformer([("one_hot", OneHotEncoder(), ["HomePlanet"])])

    res = preprocessor.fit_transform(train)

    assert res.shape == (len(train), 4)
    assert np.array_equal(
        preprocessor.get_feature_names_out(),
        np.array(
            [
                "one_hot__HomePlanet_Earth",
                "one_hot__HomePlanet_Europa",
                "one_hot__HomePlanet_Mars",
                "one_hot__HomePlanet_nan",
            ]
        ),
    )
