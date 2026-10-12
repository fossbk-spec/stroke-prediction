.PHONY: help setup data verify eda train train-fast test lint clean

help:  ## Hien thi danh sach lenh
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Tao moi truong ao va cai dat phu thuoc
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:  ## Tai bo du lieu Kaggle vao data/raw/
	.venv/bin/python scripts/download_data.py

verify:  ## Kiem tra du lieu goc dung ky vong (5,110 / 249 / 201 / 1,544)
	.venv/bin/python scripts/download_data.py --verify-only

eda:  ## Chay EDA va xuat bang + bieu do vao reports/
	.venv/bin/python scripts/run_eda.py

train:  ## Huan luyen day du (4 mo hinh + cross-validation)
	.venv/bin/python -m src.train

train-fast:  ## Huan luyen nhanh, bo qua cross-validation
	.venv/bin/python -m src.train --models logistic_regression random_forest --no-cv

test:  ## Chay toan bo unit test
	.venv/bin/python -m pytest -v

lint:  ## Kiem tra chat luong ma nguon
	.venv/bin/python -m ruff check src tests scripts

clean:  ## Xoa file tam
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
