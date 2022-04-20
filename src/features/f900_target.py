from typing import Dict

from .base import Context, feature


@feature(["Transported"])
def f900_target(ctx: Context) -> Dict[str, float]:
    return ctx.base_df.loc[ctx.index, ["Transported"]].to_dict()


@feature(["fold"])
def f990_fold(ctx: Context) -> Dict[str, float]:
    return ctx.base_df.loc[ctx.index, ["fold"]].to_dict()
