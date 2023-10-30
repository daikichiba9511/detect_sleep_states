
.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPE := child-mind-institute-detect-sleep-states
PYTHONPATH := $(shell pwd)

.PHONY: setup
setup: ## setup install packages
	@# bootstrap of rye
	@if ! command -v rye > /dev/null 2>&1; then \
		curl -sSf https://rye-up.com/get | bash; \
		echo 'source $HOME/.rye/env' > ~/.profile; \
	fi;
	@rye sync

.PHONY: download_data
download_data: ## download data from competition page
	@if [[ ! -d ./input ]]; then \
		mkdir ./input; \
	fi;
	@rye add --dev kaggle \
	&& rye run kaggle competitions download -c "${COMPE}" -p ./input;
	@unzip "./input/${COMPE}.zip" -d "./input/${COMPE}"


.PHONY: upload
upload: ## upload dataset
	@rm -rf ./src/__pycache__
	@rm -rf ./configs/__pycache__
	@rye run kaggle datasets version --dir-mode zip -p ./src -m "update"
	@rye run kaggle datasets version --dir-mode zip -p ./output/submit -m "update"

.PHONY: lint
lint: ## lint code
	@rye run ruff check scripts src

.PHONY: mypy
mypy: ## typing check
	@rye run mypy --config-file pyproject.toml scirpts src

.PHONY: fmt
fmt: ## auto format
	@rye run isort scripts src
	@rye run black scripts src

.PHONY: test
test: ## run test with pytest
	@rye run pytest -c tests

.PHONY: clean
clean: ## clean outputs
	@rm -rf ./output/*
	@rm -rf ./wandb
	@rm -rf ./debug
	@rm -rf ./.venv

.PHONY: train
train: ## train model
	@PYTHONPATH=${PYTHONPATH} rye run python scripts/train.py \
		--config $(CONFIG)

.PHONY: train2
train2: ## train model by train_v2.py
	@PYTHONPATH=${PYTHONPATH} rye run python scripts/train_v2.py \
		--config $(CONFIG)

.PHONY: train-all
train-all: ## train model with 5-fold
	@for i in {0..4}; do \
		PYTHONPATH=${PYTHONPATH} rye run python scripts/train.py \
			--config $(CONFIG) \
			--fold $$i; \
	done

.PHONY: train2-all
train2-all: ## train model 2 with 5-fold
	@for i in {0..4}; do \
		PYTHONPATH=${PYTHONPATH} rye run python scripts/train_v2.py \
			--config $(CONFIG) \
			--fold $$i; \
	done

.PHONY: train-debug
train-debug: ## debug of train.py
	@PYTHONPATH=${PYTHONPATH} rye run python scripts/train.py \
		--config $(CONFIG) \
		--debug

.PHONY: train2-debug
train2-debug: ## debug of train.py
	@PYTHONPATH=${PYTHONPATH} rye run python scripts/train_v2.py \
		--config $(CONFIG) \
		--debug

.PHONY: cv
cv: # caluculate cv score with 1-fold
	@PYTHONPATH=${PYTHONPATH} rye run python scripts/cv.py \
		--config $(CONFIG) \
		--fold $(FOLD)

.PHONY: cv
cv-all-fold: # caluculate cv score with all fold at once
	@PYTHONPATH=${PYTHONPATH} rye run python scripts/cv.py \
		--config $(CONFIG) \
		--all

.PHONY: cv
cv-all: # caluculate cv score with 5-fold
	@for i in {0..4}; do \
		PYTHONPATH=${PYTHONPATH} rye run python scripts/cv.py \
			--config $(CONFIG) \
			--fold $$i; \
	done
%:
	@echo 'command "$@" is not found.'
	@$(MAKE) help
	@exit 1

help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
