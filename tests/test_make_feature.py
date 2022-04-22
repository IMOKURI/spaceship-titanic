import warnings

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.feature_store import Store
from src.make_feature import make_feature
from src.preprocess import preprocess
import src.utils as utils

pd.set_option("display.max_columns", None)


def test_make_feature():
    train = pd.read_csv("../inputs/train.csv")

    c = OmegaConf.load("config/main.yaml")
    c.settings.dirs.preprocess = "./tests/preprocess"
    c.params.feature_set = ["f000"]
    assert c.params.seed == 440

    base_df = preprocess(c, train)
    store = Store.empty()

    with utils.timer("make_feature"):
        new_df = make_feature(
            base_df,
            store,
            c.params.feature_set,
            feature_store="./tests/features",
            load_from_store=True,
            save_to_store=True,
            # with_target=True,
        )

    assert type(new_df) == pd.DataFrame
    assert new_df.shape == (len(train), 1)
