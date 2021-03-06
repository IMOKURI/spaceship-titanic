import lightgbm as lgb
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def make_dataset_nn(c, df, label=True):
    if c.params.dataset == "ump_1":
        ds = BaseDataset(c, df, label)

    else:
        raise Exception("Invalid dataset.")
    return ds


def make_dataset(c, train_df, valid_df, is_training=True, lightgbm=False):
    if c.params.n_class == 1:
        labels = [c.params.label_name]
    else:
        labels = [f"{c.params.label_name}_{n}" for n in range(c.params.n_class)]

    if is_training:
        train_labels = train_df[labels].to_numpy()
        valid_labels = valid_df[labels].to_numpy()
    else:
        train_labels = None
        valid_labels = None


    for col in ["PassengerId", "fold", "group_fold", "time_fold", c.params.label_name] + labels:
        try:
            train_df = train_df.drop(col, axis=1)
            valid_df = valid_df.drop(col, axis=1)
        except KeyError:
            pass

    if lightgbm:
        train_ds = lgb.Dataset(data=train_df, label=train_labels)
        valid_ds = lgb.Dataset(data=valid_df, label=valid_labels)

        return train_ds, train_df.to_numpy(), valid_ds, valid_df.to_numpy()

    else:
        train_ds = train_df.to_numpy()
        valid_ds = valid_df.to_numpy()

        return train_ds, train_labels, valid_ds, valid_labels


def make_dataloader(c, ds, shuffle, drop_last):
    dataloader = DataLoader(
        ds,
        batch_size=c.params.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataloader


class BaseDataset(Dataset):
    def __init__(self, c, df, label=True):
        # self.df = df
        self.use_label = label
        if self.use_label:
            if c.params.model == "ump_ad_ae":
                labels = [f"f_{n}" for n in range(c.params.n_class)]
                self.labels = df[labels].to_numpy()
                self.features = df[labels].to_numpy()
                return
            elif c.params.n_class == 1:
                labels = [c.params.label_name]
                self.labels = df[c.params.label_name].to_numpy()
            else:
                labels = [f"{c.params.label_name}_{n}" for n in range(c.params.n_class)]
                self.labels = df[labels].to_numpy()
        else:
            labels = []

        for col in [
            "row_id",
            "investment_id",
            "time_id",
            "fold",
            "group_fold",
            "time_fold",
            c.params.label_name,
        ] + labels:
            try:
                df = df.drop(col, axis=1)
            except KeyError:
                pass

        self.features = df.to_numpy()

    def __len__(self):
        # return len(self.df)
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx]).float()
        if self.use_label:
            label = torch.tensor(self.labels[idx]).float()
            return feature, label
        return feature
