.PHONY: help
.DEFAULT_GOAL := help
SHELL = /bin/bash

NOW = $(shell date '+%Y%m%d-%H%M%S-%N')
GROUP := $(shell date '+%Y%m%d-%H%M')


train-nn: ## Run training
	@for i in {0..4}; do nohup python train.py +settings.run_fold=$${i} wandb.group=$(GROUP) settings.gpus=\'7\' > /tmp/nohup_$(NOW).log & sleep 5; done

train-nn-seq: ## Run training by Sequential
	@nohup python train.py wandb.group=$(GROUP) > /tmp/nohup_$(NOW).log &

train-lgb: ## Run training by LightGBM
	@nohup python train.py wandb.group=$(GROUP) params.training_method="lightgbm" > /tmp/nohup_$(NOW).log &

train-xgb: ## Run training by XGBoost
	@nohup python train.py wandb.group=$(GROUP) params.training_method="xgboost" > /tmp/nohup_$(NOW).log &

train-tabnet: ## Run training by TabNet
	@nohup python train.py wandb.group=$(GROUP) params.training_method="tabnet" > /tmp/nohup_$(NOW).log &

debug: ## Run training debug mode
	@python train.py settings.debug=True hydra.verbose=True +settings.run_fold=1

debug-lgb: ## Run training by LightGBM debug mode
	@python train.py settings.debug=True hydra.verbose=True params.training_method="lightgbm"

debug-xgb: ## Run training by XGBoost debug mode
	@python train.py settings.debug=True hydra.verbose=True params.training_method="xgboost"

debug-tabnet: ## Run training by TabNet debug mode
	@python train.py settings.debug=True hydra.verbose=True params.training_method="tabnet"

early-stop: ## Abort training gracefully
	@touch abort-training.flag

push: clean-build ## Push notebook
	@rm -f ./notebooks/ump-inference.ipynb
	@python encode.py ./src ./config
	@cd ./notebooks/ && \
		kaggle kernels push

clean: clean-build clean-pyc clean-test clean-training ## Remove all build and python artifacts

clean-build: ## Remove build artifacts
	@rm -fr build/
	@rm -fr dist/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove python artifacts
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test artifacts
	@rm -rf .pytest_cache/
	@rm -rf tests/preprocess
	@rm -rf tests/features

clean-training: ## Remove training artifacts
	@rm -rf ../outputs ../multirun abort-training.flag

clean-preprocess:  ## Remove preprocess artifacts
	@rm -rf ../inputs/preprocess/*.{pkl,npy,f}

test: ## Run tests
	@pytest

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[38;2;98;209;150m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
