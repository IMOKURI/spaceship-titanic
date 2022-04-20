# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import logging
import os
import pickle
import warnings
from functools import wraps
from typing import Callable, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

from .preprocesses.p000_parse_features import *
from .utils import timeSince

log = logging.getLogger(__name__)


def load_or_fit(func: Callable):
    """
    前処理を行うクラスがすでに保存されていれば、それをロードする。
    保存されていなければ、 func で生成、学習する。
    与えられたデータを、学習済みクラスで前処理する。

    Args:
        func (Callable): 前処理を行うクラスのインスタンスを生成し、学習する関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        c = args[0]
        path = os.path.join(c.settings.dirs.preprocess, args[1]) if args[1] is not None else None

        if path is not None and os.path.exists(path):
            instance = pickle.load(open(path, "rb"))

        else:
            instance = func(*args, **kwargs)

            if path is not None:
                os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
                pickle.dump(instance, open(path, "wb"), protocol=4)

        return instance

    return wrapper


def load_or_transform(func: Callable):
    """
    前処理されたデータがすでに存在すれば、それをロードする。
    存在しなければ、 func で生成する。生成したデータは保存しておく。

    Args:
        func (Callable): 前処理を行う関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        c = args[0]
        path = os.path.join(c.settings.dirs.preprocess, args[1])

        if os.path.exists(path) and os.path.splitext(path)[1] == ".npy":
            array = np.load(path, allow_pickle=True)
        elif os.path.exists(path) and os.path.splitext(path)[1] == ".f":
            array = pd.read_feather(path)

        else:
            array = func(*args, **kwargs)

            if isinstance(array, np.ndarray):
                os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
                np.save(os.path.splitext(path)[0], array)
            elif isinstance(array, pd.DataFrame):
                os.makedirs(c.settings.dirs.preprocess, exist_ok=True)
                array.to_feather(path)

        return array

    return wrapper


@load_or_fit
def fit_instance(c, _, data: Union[np.ndarray, pd.DataFrame], instance):
    log.info("Fit base preprocess.")
    instance.fit(data)
    return instance


def preprocess(c, df: pd.DataFrame, name: str = "train") -> pd.DataFrame:
    base_df = pd.concat([df, df[["Cabin"]].copy(), df[["Cabin"]].copy()], axis=1)
    base_df.columns = list(df.columns) + ["Cabin2", "Cabin3"]

    features = base_preprocess(c, f"{name}_preprocess.npy", base_df)

    cols = [
        "GroupId",
        "HomePlanet_Earth",
        "HomePlanet_Europa",
        "HomePlanet_Mars",
        "HomePlanet_nan",
        "Destination_55 Cancri e",
        "Destination_PSO J318.5-22",
        "Destination_TRAPPIST-1e",
        "Destination_nan",
        "CryoSleep",
        "VIP",
        "Cabin_Deck",
        "Cabin_Num",
        "Cabin_Side_P",
        "Cabin_Side_S",
        "Cabin_Side_nan",
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]

    if name == "train":
        cols.append("Transported")
        preprocessor = ColumnTransformer(transformers=[("transported", ToFloat(), ["Transported"])])
        targets = preprocessor.fit_transform(df[["Transported"]])
        features = np.hstack((features, targets))

    new_df = pd.concat([df["PassengerId"], pd.DataFrame(features, columns=cols, dtype=np.float32)], axis=1)

    try:
        assert new_df.isnull().to_numpy().sum() == 0
    except AssertionError:
        warnings.warn(f"DataFrame contains null. -> \n{new_df.isnull().sum()}")

    return new_df


@load_or_transform
def base_preprocess(c, _, df: pd.DataFrame) -> np.ndarray:

    # GroupId
    group_id_transformer = Pipeline(
        steps=[
            ("group_id", SplitUnderBarZero()),
            ("to_float", ToFloat()),
        ]
    )

    # Home Planet, Destination
    one_hot_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder()),
        ]
    )

    # CryoSleep, VIP, Transported
    float_transformer = Pipeline(
        steps=[
            ("to_float", ToFloat()),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )

    # Cabin
    cabin_deck_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="/")),
            ("cabin_deck", SplitSlashZero()),
            ("ordinal", OrdinalEncoder()),
            ("power", MinMaxScaler()),
        ]
    )
    cabin_num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="/0")),
            ("cabin_num", SplitSlashOne()),
            ("to_float", ToFloat()),
            ("power", MinMaxScaler()),
        ]
    )
    cabin_side_transformer = Pipeline(
        steps=[
            ("cabin_side", SplitSlashLast()),
            ("onehot", OneHotEncoder()),
        ]
    )

    # Age
    median_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("power", MinMaxScaler()),
        ]
    )

    # Billing
    zero_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("power", MinMaxScaler()),
        ]
    )

    transformers = [
        ("group_id", group_id_transformer, ["PassengerId"]),
        ("one_hot", one_hot_transformer, ["HomePlanet", "Destination"]),
        ("float", float_transformer, ["CryoSleep", "VIP"]),
        ("cabin_deck", cabin_deck_transformer, ["Cabin"]),
        ("cabin_num", cabin_num_transformer, ["Cabin2"]),
        ("cabin_side", cabin_side_transformer, ["Cabin3"]),
        ("median", median_transformer, ["Age"]),
        ("zero", zero_transformer, ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]),
    ]

    preprocessor = ColumnTransformer(transformers=transformers)
    preprocessor = fit_instance(c, "base_preprocess.pkl", df, preprocessor)

    log.info("Transform base preprocess.")
    features = preprocessor.transform(df)

    log.info(f"Base preprocess output dict: {preprocessor.output_indices_}")
    # log.warning(f"Base preprocess output feature names: {preprocessor.get_feature_names_out()}")

    return features
