import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

log = logging.getLogger(__name__)


def make_fold(c, df):
    if c.params.fold == "bins_stratified":
        df = bins_stratified_kfold(c, df, c.params.label_name)
    elif c.params.fold == "stratified":
        df = stratified_kfold(c, df, c.params.label_name)
    elif c.params.fold == "group":
        df = group_kfold(c, df, c.params.group_name)
    elif c.params.fold == "stratified_group":
        df = stratified_group_kfold(c, df, c.params.label_name, c.params.group_name)

    else:
        raise Exception("Invalid fold.")

    return df


def train_test_split(c, df, fold):
    trn_idx = df[df["fold"] != fold].index
    val_idx = df[df["fold"] == fold].index

    log.info(f"Num of training data: {len(trn_idx)}, num of validation data: {len(val_idx)}")

    train_folds = df.loc[trn_idx].reset_index(drop=True)
    valid_folds = df.loc[val_idx].reset_index(drop=True)

    return train_folds, valid_folds


def bins_stratified_kfold(c, df, col):
    num_bins = int(np.floor(1 + np.log2(len(df))))
    df.loc[:, "bins"] = pd.cut(df[col], bins=num_bins, labels=False)

    fold_ = StratifiedKFold(n_splits=c.params.n_fold, shuffle=True, random_state=c.params.seed)
    for n, (_, val_index) in enumerate(fold_.split(df, df["bins"])):
        df.loc[val_index, "fold"] = int(n)

    return df


def stratified_kfold(c, df, col):
    fold_ = StratifiedKFold(n_splits=c.params.n_fold, shuffle=True, random_state=c.params.seed)
    for n, (_, val_index) in enumerate(fold_.split(df, df[col])):
        df.loc[val_index, "fold"] = int(n)

    return df


def group_kfold(c, df, col):
    fold_ = GroupKFold(n_splits=c.params.n_fold)
    for n, (_, val_index) in enumerate(fold_.split(df, groups=df[col])):
        df.loc[val_index, "fold"] = int(n)

    return df


def stratified_group_kfold(c, df, label, group):
    fold_ = StratifiedGroupKFold(n_splits=c.params.n_fold, shuffle=True, random_state=c.params.seed)
    for n, (_, val_index) in enumerate(fold_.split(df, df[label], df[group])):
        df.loc[val_index, "fold"] = int(n)

    return df
