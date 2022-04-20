import warnings

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.preprocess import preprocess
from src.preprocesses.p000_parse_features import *

pd.set_option("display.max_columns", None)


def test_base_preprocess():
    train = pd.read_csv("../inputs/train.csv")

    c = OmegaConf.load("config/main.yaml")
    c.settings.dirs.preprocess = "./tests/preprocess"

    assert c.params.seed == 440

    new_df = preprocess(c, train)

    assert new_df.shape == (len(train), 24)
    assert new_df.isnull().to_numpy().sum() == 0

    # warnings.warn(f"{new_df.iloc[:3, :]}")


def test_simple_imputer():
    train = pd.read_csv("../inputs/train.csv")

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


def test_one_hot_encoder_multi_cols():
    train = pd.read_csv("../inputs/train.csv")

    preprocessor = ColumnTransformer([("one_hot", OneHotEncoder(), ["HomePlanet", "Destination"])])

    res = preprocessor.fit_transform(train)

    assert res.shape == (len(train), 8)
    assert np.array_equal(
        preprocessor.get_feature_names_out(),
        np.array(
            [
                "one_hot__HomePlanet_Earth",
                "one_hot__HomePlanet_Europa",
                "one_hot__HomePlanet_Mars",
                "one_hot__HomePlanet_nan",
                "one_hot__Destination_55 Cancri e",
                "one_hot__Destination_PSO J318.5-22",
                "one_hot__Destination_TRAPPIST-1e",
                "one_hot__Destination_nan",
            ]
        ),
    )


def test_pipeline_1():
    train = pd.read_csv("../inputs/train.csv")

    transformer = Pipeline(
        steps=[
            ("to_float", ToFloat()),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )

    preprocessor = ColumnTransformer([("transformer", transformer, ["VIP"])])

    res = preprocessor.fit_transform(train)

    assert res.shape == (len(train), 1)
    assert np.array_equal(res.squeeze()[:3], np.array([0, 0, 1], dtype=np.float32))
    assert not np.isnan(res.squeeze()[38])
