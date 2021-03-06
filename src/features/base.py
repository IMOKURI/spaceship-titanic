import copy
import logging
import traceback
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Any

import numpy as np
import pandas as pd
from nptyping import NDArray

from ..feature_store import Store

_ALL_FEATURES = {}  # type: Dict[str, Callable]
_ALL_FEATURE_NAMES = set()
_FEATURE_COLUMNS = {}  # type: Dict[str, List[str]]

log = logging.getLogger(__name__)


@dataclass
class Context:
    base_df: pd.DataFrame  # NDArray[(Any, Any), Any]
    store: Store
    index: int
    current_feature_name: str = ""
    fallback_to_none: bool = False


def feature(columns: List[str]):
    def _feature(func):
        _ALL_FEATURE_NAMES.add(func.__name__)
        _FEATURE_COLUMNS[func.__name__] = columns

        prefix = _prefix(func.__name__)
        features_with_same_prefix = [f for f in _ALL_FEATURE_NAMES if _prefix(f) == prefix]
        assert len(features_with_same_prefix) == 1, f"feature prefix is duplicated! {features_with_same_prefix}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            ctx = args[0]
            # assert isinstance(ctx, Context)
            # assert len(args) == 1
            ctx.current_feature_name = func.__name__

            try:
                return func(ctx)
            except Exception:
                msg = f"WARNING: exception occured in feature {func.__name__}: {traceback.format_exc()}"
                warnings.warn(msg)
                log.warning(msg)

                if ctx.fallback_to_none:
                    return empty_feature(func.__name__)
                else:
                    raise

        _ALL_FEATURES[func.__name__] = wrapper

        return wrapper

    return _feature


def empty_feature(name: str) -> Dict:
    return {k: None for k in _FEATURE_COLUMNS[name]}


def get_features() -> Dict[str, Callable]:
    return _ALL_FEATURES


def get_feature(name: str) -> Callable:
    return _ALL_FEATURES[normalize_feature_name(name)]


def get_feature_schema() -> Dict[str, List[str]]:
    return copy.deepcopy(_FEATURE_COLUMNS)


def normalize_feature_name(name: str) -> str:
    if name in _ALL_FEATURES:
        return name

    for k in _ALL_FEATURES:
        if _prefix(k) == name:
            return k

    raise ValueError(f"Feature {name} not found.")


def _prefix(feature_name: str) -> str:
    return feature_name.split("_")[0]
