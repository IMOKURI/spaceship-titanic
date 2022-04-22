import logging
import os

import hydra
import numpy as np
import pandas as pd
from omegaconf.errors import ConfigAttributeError
from scipy.optimize import minimize

import src.utils as utils
from src.get_score import record_result, optimize_function
from src.load_data import InputData
from src.features.helper import *
from src.run_loop import train_fold_lightgbm, train_fold_nn, train_fold_tabnet, train_fold_xgboost

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main")
def main(c):
    utils.debug_settings(c)
    run = utils.setup_wandb(c)

    log.info(f"Started at {os.path.basename(os.getcwd())}")

    utils.fix_seed(c.params.seed)
    device = utils.gpu_settings(c)

    input = InputData(c)

    oof_df = pd.DataFrame()
    losses = utils.AverageMeter()
    single_run = False
    num_fold = c.params.n_fold

    for fold in range(num_fold):
        try:
            fold = int(c.settings.run_fold)
            single_run = True
        except ConfigAttributeError:
            pass

        log.info(f"========== fold {fold} training start ==========")
        utils.fix_seed(c.params.seed + fold)

        if c.params.training_method == "lightgbm":
            _oof_df, loss = train_fold_lightgbm(c, input, fold)
        elif c.params.training_method == "xgboost":
            _oof_df, loss = train_fold_xgboost(c, input, fold)
        elif c.params.training_method == "tabnet":
            _oof_df, loss = train_fold_tabnet(c, input, fold)
        else:
            _oof_df, loss = train_fold_nn(c, input, fold, device)

        log.info(f"========== fold {fold} training result ==========")
        record_result(c, _oof_df, fold, loss)

        oof_df = pd.concat([oof_df, _oof_df])
        losses.update(loss)

        if c.settings.debug or single_run:
            break

    log.info("========== training result ==========")
    score = record_result(c, oof_df, c.params.n_fold, losses.avg)

    log.info("========== optimize training result ==========")
    minimize_result = minimize(
        optimize_function(c, oof_df[c.params.label_name].to_numpy(), oof_df["base_preds"].to_numpy()),
        np.array([0.5]),
        method="Nelder-Mead",
    )
    log.info(f"optimize result. -> \n{minimize_result}")
    score = record_result(c, oof_df, c.params.n_fold, losses.avg)

    oof_df["preds"] = (oof_df["base_preds"] > minimize_result["x"].item()).astype(np.int8)

    oof_df.reset_index(drop=True).to_feather("oof_df.f")
    # oof_df[["PassengerId", c.params.label_name, "preds", "fold"]].reset_index(drop=True).to_feather("oof_df.f")

    log.info(f"oof -> \n{oof_df}")

    if c.settings.inference:
        log.info("========== inference result ==========")

        cols_base_preds = [col for col in input.test.columns if "base_preds" in col]
        input.test["base_preds"] = nanmean(input.test[cols_base_preds].to_numpy(), axis=1)
        input.test["preds"] = (input.test[f"base_preds"] > minimize_result["x"].item()).astype(bool)

        input.sample_submission[c.params.label_name] = input.test["preds"]

        input.test.to_feather("inference.f")
        input.sample_submission.to_csv("submission.csv", index=False)

        log.info(f"inference -> \n{input.test}")

    log.info("Done.")

    utils.teardown_wandb(c, run, losses.avg, score)


if __name__ == "__main__":
    main()
