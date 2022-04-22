import warnings

import numpy as np
import pandas as pd

from src.preprocesses.p000_lightgbm_imputer import LGBMImputer
from src.preprocesses.p001_dist_transformer import DistTransformer

pd.set_option("display.max_columns", None)


def test_lightgbm_imputer_categorical():
    train = pd.read_csv("../inputs/train.csv")

    preprocessor = DistTransformer("ordinal", verbose=True)
    res = preprocessor.fit_transform(train[["HomePlanet", "Destination", "VIP"]])

    assert np.any(pd.isnull(res))

    preprocessor = LGBMImputer(verbose=True)
    res = preprocessor.fit_transform(res)

    # warnings.warn(f"LightGBM Imputer (categorical) result. ->\n{res.iloc[:3, :]}")

    assert res.shape == (len(train), 3)
    assert res.isnull().to_numpy().sum() == 0

    # warnings.warn(f"LightGBM Imputer (categorical) count null. ->\n{res.isnull().sum()}")



def test_lightgbm_imputer_numeric():
    train = pd.read_csv("../inputs/train.csv")

    preprocessor = DistTransformer("standard", verbose=True)
    res = preprocessor.fit_transform(train[["Age", "RoomService"]])

    assert np.any(pd.isnull(res))

    preprocessor = LGBMImputer(verbose=True)
    res = preprocessor.fit_transform(res)

    # warnings.warn(f"LightGBM Imputer (numeric) result. ->\n{res.iloc[:3, :]}")

    assert res.shape == (len(train), 2)
    assert res.isnull().to_numpy().sum() == 0

    # warnings.warn(f"LightGBM Imputer (numeric) count null. ->\n{res.isnull().sum()}")
