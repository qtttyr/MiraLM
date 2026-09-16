PY ?= .venv/bin/python
SAMPLE ?= configs/model_sparsemind.yaml

.PHONY: setup params test

## bootstrap: create venv and install requirements
setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

## run the parameter-budget gate (both configs)
params:
	$(PY) scripts/param_budget.py

## run the test suite
test:
	$(PY) -m pytest -q