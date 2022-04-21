import numpy as np
from typing import Dict

from .base import Context, feature


@feature(["Transported"])
def f900_target(ctx: Context) -> Dict[str, float]:
    assert ctx.store.feature_df is not None
    return ctx.store.feature_df.loc[ctx.index, ["Transported"]].to_dict()
