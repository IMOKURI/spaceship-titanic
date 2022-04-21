import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from nptyping import NDArray
from omegaconf.dictconfig import DictConfig
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

from .preprocess import transform_data
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

        features = []
        cols = []

        f_home_planet = transform_data(c, f"{df_name}-home_planet_{fold}.npy", df["HomePlanet"].to_numpy(), OrdinalEncoder())
        features.append(f_home_planet)
        cols += ["HomePlanet"]

        f_destination = transform_data(c, f"{df_name}-destination_{fold}.npy", df["Destination"].to_numpy(), OrdinalEncoder())
        features.append(f_destination)
        cols += ["Destination"]

        f_cryo_sleep = df["CryoSleep"].fillna(False).astype(np.int8).to_numpy().reshape(-1, 1)
        features.append(f_cryo_sleep)
        cols.append("CryoSleep")

        f_vip = df["VIP"].fillna(False).astype(np.int8).to_numpy().reshape(-1, 1)
        features.append(f_vip)
        cols.append("VIP")

        # TODO: Cabin
        """
        # Cabin
        steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="/")),
            ("cabin_deck", SplitSlashZero()),
            ("ordinal", OrdinalEncoder()),
            ("power", MinMaxScaler()),
        ]
        fit_column_transformer(c, steps, ["Cabin"], "transformer_cabin_deck.pkl", df)

        steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="/0")),
            ("cabin_num", SplitSlashOne()),
            ("to_float", ToFloat()),
            ("power", MinMaxScaler()),
        ]
        fit_column_transformer(c, steps, ["Cabin"], "transformer_cabin_num.pkl", df)

        steps = [
            ("cabin_side", SplitSlashLast()),
            ("onehot", OneHotEncoder()),
        ]
        fit_column_transformer(c, steps, ["Cabin"], "transformer_cabin_side.pkl", df)

        """

        f_age = transform_data(c, f"{df_name}-age_{fold}.npy", df["Age"].fillna(df["Age"].median()).to_numpy(), MinMaxScaler())
        features.append(f_age)
        cols += ["Age"]

        f_room_service = transform_data(
            c, f"{df_name}-room_service_{fold}.npy", df["RoomService"].fillna(0).to_numpy(), MinMaxScaler()
        )
        features.append(f_room_service)
        cols += ["RoomService"]

        f_food_court = transform_data(c, f"{df_name}-food_court_{fold}.npy", df["FoodCourt"].fillna(0).to_numpy(), MinMaxScaler())
        features.append(f_food_court)
        cols += ["FoodCourt"]

        f_shopping_mall = transform_data(
            c, f"{df_name}-shopping_mall_{fold}.npy", df["ShoppingMall"].fillna(0).to_numpy(), MinMaxScaler()
        )
        features.append(f_shopping_mall)
        cols += ["ShoppingMall"]

        f_spa = transform_data(c, f"{df_name}-spa_{fold}.npy", df["Spa"].fillna(0).to_numpy(), MinMaxScaler())
        features.append(f_spa)
        cols += ["Spa"]

        f_vr_deck = transform_data(c, f"{df_name}-vr_deck_{fold}.npy", df["VRDeck"].fillna(0).to_numpy(), MinMaxScaler())
        features.append(f_vr_deck)
        cols += ["VRDeck"]

        if is_training:
            f_transported = df["Transported"].astype(np.int8).to_numpy().reshape(-1, 1)
            features.append(f_transported)
            cols.append("Transported")

        for f in features:
            log.debug(f"{df_name}-feature shape: {f.shape}")
        instance.feature_df = pd.DataFrame(np.hstack(tuple(features)), columns=cols)

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
