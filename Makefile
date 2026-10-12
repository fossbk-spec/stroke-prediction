.PHONY: help setup data verify eda train train-fast test lint clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create a virtual environment and install dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:  ## Download the Kaggle dataset into data/raw/
	.venv/bin/python scripts/download_data.py

verify:  ## Verify the raw data matches expectations (5,110 / 249 / 201 / 1,544)
	.venv/bin/python scripts/download_data.py --verify-only

eda:  ## Run EDA and export tables + figures into reports/
	.venv/bin/python scripts/run_eda.py

train:  ## Full training run (4 models + cross-validation)
	.venv/bin/python -m src.train

train-fast:  ## Fast training run, skipping cross-validation
	.venv/bin/python -m src.train --models logistic_regression random_forest --no-cv

test:  ## Run the full unit test suite
	.venv/bin/python -m pytest -v

lint:  ## Check source code quality
	.venv/bin/python -m ruff check src tests scripts

clean:  ## Remove temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
