import gc
import logging
import os
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import wandb
import xgboost as xgb
from scipy.optimize import minimize, minimize_scalar

from .feature_store import Store
from .get_score import get_score, optimize_function
from .make_dataset import make_dataloader, make_dataset
from .make_feature import make_feature
from .make_fold import train_test_split
from .make_loss import make_criterion, make_optimizer, make_scheduler
from .make_model import make_model, make_model_tabnet, make_model_xgboost
from .run_epoch import inference_epoch, train_epoch, validate_epoch
from .utils import AverageMeter, timeSince

# from wandb.lightgbm import log_summary, wandb_callback


log = logging.getLogger(__name__)


def train_fold_lightgbm(c, df, fold):
    train_df, valid_df = train_test_split(c, df, fold)
    train_store = Store.train(c, train_df, "train", fold=fold)
    valid_store = Store.train(c, valid_df, "valid", fold=fold)
    train_folds = make_feature(
        train_df,
        train_store,
        feature_list=c.params.feature_set,
        feature_store=c.settings.dirs.feature,
        with_target=True,
        fallback_to_none=False,
    )
    valid_folds = make_feature(
        valid_df,
        valid_store,
        feature_list=c.params.feature_set,
        feature_store=c.settings.dirs.feature,
        with_target=True,
        fallback_to_none=False,
    )
    train_ds, _, valid_ds, valid_raw_ds = make_dataset(c, train_folds, valid_folds, lightgbm=True)
    assert type(train_ds) == lgb.basic.Dataset
    assert type(valid_ds) == lgb.basic.Dataset

    lgb_params = {
        "objective": "binary",  # "regression",
        "boosting": "gbdt",
        # "extra_trees": True,  # https://note.com/j26/n/n64d9c37167a6
        "metric": "binary_logloss",  # "rmse",
        # "learning_rate": 0.05,
        # "min_data_in_leaf": 120,
        # "feature_fraction": 0.7,
        # "bagging_fraction": 0.85,
        # "lambda_l1": 0.01,
        # "lambda_l2": 0.01,
        # "num_leaves": 96,
        # "max_depth": 12,
        # "drop_rate": 0.0,
        "seed": c.params.seed,
    }

    eval_result = {}
    callbacks = [
        lgb.log_evaluation(period=c.settings.print_freq),
        lgb.record_evaluation(eval_result),
        lgb.early_stopping(stopping_rounds=100),
        # wandb_callback(),
    ]

    booster = lgb.train(
        train_set=train_ds,
        valid_sets=[train_ds, valid_ds],
        valid_names=["train", "valid"],
        params=lgb_params,
        num_boost_round=10000,
        callbacks=callbacks,
    )

    os.makedirs(f"fold{fold}", exist_ok=True)
    joblib.dump(booster, f"fold{fold}/lightgbm.pkl")
    # booster.save_model(f"fold{fold}/lightgbm.pkl", num_iteration=booster.best_iteration)
    # log_summary(booster, save_model_checkpoint=True)

    valid_folds["base_preds"] = booster.predict(valid_raw_ds, num_iteration=booster.best_iteration)

    minimize_result = minimize(
        optimize_function(c, valid_folds[c.params.label_name].to_numpy(), valid_folds["base_preds"].to_numpy()),
        np.array([0.5]),
        method="Nelder-Mead",
    )
    log.info(f"optimize result. -> {minimize_result}")

    valid_folds["preds"] = (valid_folds["base_preds"] > minimize_result["x"].item()).astype(np.int8)

    return valid_folds, 0, booster.best_score["valid"]["binary_logloss"]  # ["rmse"]


def train_fold_xgboost(c, df, fold):
    train_folds, valid_folds = train_test_split(c, df, fold)
    train_ds, train_labels, valid_ds, valid_labels = make_dataset(c, train_folds, valid_folds)

    clf = make_model_xgboost(c, train_ds)

    clf.fit(
        train_ds,
        train_labels,
        eval_set=[(valid_ds, valid_labels)],
        verbose=100,
        early_stopping_rounds=100,
    )

    os.makedirs(f"fold{fold}", exist_ok=True)
    clf.save_model(f"fold{fold}/xgboost.pkl")

    valid_folds["preds"] = clf.predict(valid_ds)

    return valid_folds, 0, clf.best_score


def train_fold_tabnet(c, df, fold):
    train_folds, valid_folds = train_test_split(c, df, fold)
    train_ds, train_labels, valid_ds, valid_labels = make_dataset(c, train_folds, valid_folds)

    clf = make_model_tabnet(c, train_ds)

    clf.fit(
        train_ds,
        train_labels,
        eval_set=[(valid_ds, valid_labels)],
        max_epochs=10000,
        patience=100,
        batch_size=1024 * 20,
        virtual_batch_size=128 * 20,
        num_workers=4,
        drop_last=True,
    )

    os.makedirs(f"fold{fold}", exist_ok=True)
    clf.save_model(f"fold{fold}/tabnet")

    valid_folds["preds"] = clf.predict(valid_ds)

    return valid_folds, 0, clf.best_cost


def train_fold_nn(c, input, fold, device):
    df = input.train
    train_folds, valid_folds = train_test_split(c, df, fold)

    # ====================================================
    # Data Loader
    # ====================================================
    train_ds = make_dataset(c, train_folds)
    valid_ds = make_dataset(c, valid_folds)

    train_loader = make_dataloader(c, train_ds, shuffle=True, drop_last=True)
    valid_loader = make_dataloader(c, valid_ds, shuffle=False, drop_last=False)

    # ====================================================
    # Model
    # ====================================================
    model = make_model(c, device)

    criterion = make_criterion(c)
    optimizer = make_optimizer(c, model)
    scaler = amp.GradScaler(enabled=c.settings.amp)
    # scheduler = make_scheduler(c, optimizer, train_ds)
    scheduler = make_scheduler(c, optimizer, df)

    es = EarlyStopping(c=c, fold=fold)

    # ====================================================
    # Loop
    # ====================================================
    for epoch in range(c.params.epoch):
        start_time = time.time()

        # ====================================================
        # Training
        # ====================================================
        if c.params.skip_training:
            avg_train_loss = 0
        else:
            avg_train_loss = train_epoch(
                c,
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                scaler,
                epoch,
                device,
                verbose=True,
            )

        # ====================================================
        # Validation
        # ====================================================
        avg_val_loss, preds = validate_epoch(c, valid_loader, model, criterion, device, verbose=True)
        valid_labels = valid_folds[c.params.label_name].values

        if "LogitsLoss" in c.params.criterion:
            preds = 1 / (1 + np.exp(-preds))

        # scoring
        if c.params.n_class == 1 or c.params.model == "ump_ad_ae":
            score = get_score(c.params.scoring, valid_labels, preds)
        elif c.params.n_class > 1:
            score = get_score(c.params.scoring, valid_labels, preds.argmax(1))
        else:
            raise Exception("Invalid n_class.")

        elapsed = time.time() - start_time
        log.info(
            f"Epoch {epoch + 1} - "
            f"train_loss: {avg_train_loss:.4f} "
            f"valid_loss: {avg_val_loss:.4f} "
            f"score: {score:.4f} "
            f"time: {elapsed:.0f}s"
        )
        if c.wandb.enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    f"train_loss/fold{fold}": avg_train_loss,
                    f"valid_loss/fold{fold}": avg_val_loss,
                    f"score/fold{fold}": score,
                }
            )

        es(avg_val_loss, score, model, preds)

        if es.early_stop or os.path.exists(os.path.join(c.settings.dirs.working, "abort-training.flag")):
            log.info("Early stopping")
            break

    if c.params.n_class == 1:
        valid_folds["preds"] = es.best_preds
    elif c.params.n_class > 1:
        # valid_folds["preds"] = es.best_preds
        # valid_folds[[str(c) for c in range(c.params.n_class)]] = es.best_preds
        valid_folds["preds"] = 0.0  # es.best_preds.argmax(1)
    else:
        raise Exception("Invalid n_class.")

    return valid_folds, es.best_score, es.best_loss


def inference_lightgbm(df, models):
    predictions = np.zeros((len(df), len(models)), dtype=np.float64)
    feature_cols = [f"f_{n}" for n in range(300)]

    for n, model in enumerate(models):
        preds = model.predict(df[feature_cols].values)
        predictions[:, n] = preds.reshape(-1)

    return predictions


def inference(c, df, device, models):
    predictions = np.zeros((len(df), len(models)), dtype=np.float64)
    # (len(df), len(c.params.pretrained) * c.params.n_fold))

    for n, model in enumerate(models):
        inference_ds = make_dataset(c, df, label=False)
        inference_loader = make_dataloader(c, inference_ds, shuffle=False, drop_last=False)

        # c.params.model = training.model

        # for fold in range(c.params.n_fold):
        # start_time = time.time()

        # model_path = os.path.join(training.dir, f"fold{fold}")
        # model = make_model(c, device, training.dir)

        preds = inference_epoch(c, inference_loader, model, device)

        if "LogitsLoss" in c.params.criterion:
            preds = 1 / (1 + np.exp(-preds))

        # assert len(df) == len(preds), "Inference result size does not match input size."

        predictions[:, n] = preds

        # elapsed = time.time() - start_time
        # log.info(f"time: {elapsed:.0f}s")

    return predictions


# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, c, fold, delta=0):
        self.patience = c.params.es_patience
        self.dir = f"fold{fold}"
        self.path = "pytorch_model.bin"
        os.makedirs(self.dir, exist_ok=True)

        self.early_stop = False
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.best_loss = np.Inf
        self.best_preds = None

    def __call__(self, val_loss, score, model, preds, ds=None):

        if self.best_score is None:
            self.best_score = score
            self.best_preds = preds
            self.save_checkpoint(val_loss, model, ds)
        elif val_loss >= self.best_loss + self.delta:
            if self.patience <= 0:
                return
            self.counter += 1
            log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_preds = preds
            self.save_checkpoint(val_loss, model, ds)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ds):
        """Saves model when validation loss decrease."""
        log.info(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...")
        self.best_loss = val_loss
        torch.save(model.state_dict(), os.path.join(self.dir, self.path))
