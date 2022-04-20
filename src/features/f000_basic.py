from typing import Dict

import numpy as np

from .base import Context, feature

FEATURE_COLS = [
    "PassengerId",
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


@feature(FEATURE_COLS)
def f000_initial_features(ctx: Context) -> Dict[str, float]:
    return ctx.base_df.loc[ctx.index, FEATURE_COLS].to_dict()
