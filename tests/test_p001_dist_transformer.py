import warnings

import numpy as np
import pandas as pd

from src.preprocesses.p001_dist_transformer import DistTransformer

pd.set_option("display.max_columns", None)


def test_dist_transformer_categorical():
    train = pd.read_csv("../inputs/train.csv")

    preprocessor = DistTransformer("ordinal", verbose=True)
    res = preprocessor.fit_transform(train[["HomePlanet", "Destination", "VIP"]])

    # warnings.warn(f"Dist transformer (ordinal) view null. ->\n{train[~train['HomePlanet'].isin(['Earth', 'Europa', 'Mars'])]}")
    # warnings.warn(f"Dist transformer (ordinal) result. ->\n{train['HomePlanet'].value_counts()}")
    # warnings.warn(f"Dist transformer (ordinal) result. ->\n{res['HomePlanet'].value_counts()}")
    # warnings.warn(f"Dist transformer (ordinal) result. ->\n{res.iloc[:3, :]}")

    assert res.shape == (len(train), 3)
    assert np.any(pd.isnull(res))

    # warnings.warn(f"Dist transformer (ordinal) count null. ->\n{res.isnull().sum()}")


def test_dist_transformer_numeric():
    train = pd.read_csv("../inputs/train.csv")

    preprocessor = DistTransformer("standard", verbose=True)
    res = preprocessor.fit_transform(train[["Age", "RoomService"]])

    # warnings.warn(f"Dist transformer (standard) result. ->\n{res.iloc[:3, :]}")

    assert res.shape == (len(train), 2)
    assert np.any(pd.isnull(res))

    # warnings.warn(f"Dist transformer (standard) count null. ->\n{res.isnull().sum()}")
