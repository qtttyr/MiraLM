PY ?= .venv/bin/python
SAMPLE ?= configs/model_sparsemind.yaml

.PHONY: setup params check test

## bootstrap: create venv and install requirements
setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

## run the parameter-budget gate (static projection)
params:
	$(PY) scripts/param_budget.py

## run the authoritative gate on the REAL model (projection must equal numel)
check:
	$(PY) scripts/check_params.py

## run the test suite
test:
	$(PY) -m pytest -q