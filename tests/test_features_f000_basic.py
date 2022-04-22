import warnings

import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.feature_store import Store
from src.features.base import Context
from src.features.f000_basic import *
from src.features.f900_target import *
from src.preprocess import preprocess

pd.set_option("display.max_columns", None)


def test_features_f000_basic():
    train = pd.read_csv("../inputs/train.csv")

    c = OmegaConf.load("config/main.yaml")
    c.settings.dirs.preprocess = "./tests/preprocess"
    assert c.params.seed == 440

    new_df = preprocess(c, train)
    store = Store.empty()

    ctx = Context(new_df, store, 0)

    features = f000_passenger_id(ctx)

    assert type(features) == dict
    assert features == {"PassengerId": "0001_01"}


@pytest.mark.skipif(True, reason="Need to update to use Store.train()")
def test_features_f900_target():
    train = pd.read_csv("../inputs/train.csv")

    c = OmegaConf.load("config/main.yaml")
    c.settings.dirs.preprocess = "./tests/preprocess"
    assert c.params.seed == 440

    new_df = preprocess(c, train)
    store = Store.empty()

    ctx = Context(new_df, store, 0)

    features = f900_target(ctx)

    assert type(features) == dict
    assert features == {"Transported": 0.0}
