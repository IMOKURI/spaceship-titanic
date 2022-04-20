import warnings

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

    features = f000_initial_features(ctx)

    assert type(features) == dict
    assert features == {
        "PassengerId": "0001_01",
        "Age": 0.49367088079452515,
        "Cabin_Deck": 0.25,
        "Cabin_Num": 0.0,
        "Cabin_Side_P": 1.0,
        "Cabin_Side_S": 0.0,
        "Cabin_Side_nan": 0.0,
        "CryoSleep": 0.0,
        "Destination_55 Cancri e": 0.0,
        "Destination_PSO J318.5-22": 0.0,
        "Destination_TRAPPIST-1e": 1.0,
        "Destination_nan": 0.0,
        "FoodCourt": 0.0,
        "HomePlanet_Earth": 0.0,
        "HomePlanet_Europa": 1.0,
        "HomePlanet_Mars": 0.0,
        "HomePlanet_nan": 0.0,
        "RoomService": 0.0,
        "ShoppingMall": 0.0,
        "Spa": 0.0,
        "VIP": 0.0,
        "VRDeck": 0.0,
    }


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
    assert features == {
        "Transported": 0.0,
    }
