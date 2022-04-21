from typing import Dict

import numpy as np

from .base import Context, feature


@feature(["PassengerId"])
def f000_passenger_id(ctx: Context) -> Dict[str, float]:
    return ctx.base_df.loc[ctx.index, ["PassengerId"]].to_dict()


@feature(["HomePlanet"])
def f001_home_planet(ctx: Context) -> Dict[str, float]:
    assert ctx.store.feature_df is not None
    return ctx.store.feature_df.loc[ctx.index, ["HomePlanet"]].to_dict()


@feature(["Destination"])
def f002_destination(ctx: Context) -> Dict[str, float]:
    assert ctx.store.feature_df is not None
    return ctx.store.feature_df.loc[ctx.index, ["Destination"]].to_dict()


@feature(["CryoSleep"])
def f003_cryo_sleep(ctx: Context) -> Dict[str, float]:
    assert ctx.store.feature_df is not None
    return ctx.store.feature_df.loc[ctx.index, ["CryoSleep"]].to_dict()


@feature(["VIP"])
def f004_vip(ctx: Context) -> Dict[str, float]:
    assert ctx.store.feature_df is not None
    return ctx.store.feature_df.loc[ctx.index, ["VIP"]].to_dict()


@feature(["Age"])
def f005_age(ctx: Context) -> Dict[str, float]:
    assert ctx.store.feature_df is not None
    return ctx.store.feature_df.loc[ctx.index, ["Age"]].to_dict()
