import logging
import os

import hydra
import pandas as pd
from omegaconf.errors import ConfigAttributeError

import src.utils as utils
from src.get_score import record_result
from src.load_data import InputData
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

        log.info(f"========== fold {fold} training ==========")
        utils.fix_seed(c.params.seed + fold)

        if c.params.training_method == "lightgbm":
            _oof_df, _, loss = train_fold_lightgbm(c, input.train, fold)
        elif c.params.training_method == "xgboost":
            _oof_df, _, loss = train_fold_xgboost(c, input.train, fold)
        elif c.params.training_method == "tabnet":
            _oof_df, _, loss = train_fold_tabnet(c, input.train, fold)
        else:
            _oof_df, _, loss = train_fold_nn(c, input, fold, device)

        log.info(f"========== fold {fold} result ==========")
        record_result(c, _oof_df, fold, loss)

        oof_df = pd.concat([oof_df, _oof_df])
        losses.update(loss)

        if c.settings.debug or single_run:
            break

    oof_df.to_feather("oof_df.f")
    # oof_df[["PassengerId", c.params.label_name, "preds", "fold"]].reset_index(drop=True).to_feather("oof_df.f")

    log.info("========== final result ==========")
    score = record_result(c, oof_df, c.params.n_fold, losses.avg)

    log.info("Done.")

    utils.teardown_wandb(c, run, losses.avg, score)


if __name__ == "__main__":
    main()
