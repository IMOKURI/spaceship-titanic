import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from nptyping import NDArray
from omegaconf.dictconfig import DictConfig

from .preprocess import transform_data
from .preprocesses.p000_lightgbm_imputer import LGBMImputer
from .preprocesses.p001_dist_transformer import DistTransformer
from .utils import catch_everything_in_kaggle

log = logging.getLogger(__name__)


class Store:
    def __init__(self, feature_df: Optional[pd.DataFrame] = None):
        self.feature_df = feature_df

    @classmethod
    def empty(cls) -> "Store":
        return cls()

    @classmethod
    def train(
        cls,
        c: DictConfig,
        df: pd.DataFrame,
        df_name: str,
        is_training: bool = True,
        fold: Optional[int] = -1,
    ) -> "Store":
        """
        TODO: c.params.preprocess によって、ロードするものを変更する。
        """
        log.info("Setup store with preprocess.")
        instance = cls.empty()

        # Convert None to NaN
        df = df.fillna(np.nan)

        cols_categorical = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
        cols_numeric = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

        # TODO: Cabin

        preprocessor = DistTransformer("ordinal")
        features_categorical = transform_data(c, f"{df_name}-ordinal_categorical_{fold}.npy", df[cols_categorical], preprocessor)

        preprocessor = DistTransformer("standard")
        features_numeric = transform_data(c, f"{df_name}-standard_numeric_{fold}.npy", df[cols_numeric], preprocessor)

        features = pd.concat([features_categorical, features_numeric], axis=1)
        log.debug(f"Pre Encoder count null. ->\n{features.isnull().sum()}")

        preprocessor = LGBMImputer()
        features = transform_data(c, f"{df_name}-lightgbm_imputer_{fold}.f", features, preprocessor)

        if is_training:
            features["Transported"] = df["Transported"].astype(np.int8)

        log.debug(f"LightGBM Imputer count null. ->\n{features.isnull().sum()}")
        log.info(f"Columns of feature_df: {features.columns}, shape: {features.shape}")

        instance.feature_df = features
        return instance

    def update(self, array: NDArray[(Any, Any), Any]):
        for row in array:
            ...
            # row の中のデータは dtype: object であることに注意。

            # TODO: 最終的に catch_everything_in_kaggle をいれていく
            # with catch_everything_in_kaggle():
            #     self.passengers.extend(row)

    def update_post(self, array: NDArray[(Any, Any), Any]):
        for row in array:
            ...
            # row の中のデータは dtype: object であることに注意。

            # TODO: 最終的に catch_everything_in_kaggle をいれていく
            # with catch_everything_in_kaggle():
            #     self.passengers.extend_post(row)
