.PHONY: install test smoke env
install:
	pip install -r requirements.txt && pip install -e . --no-deps
test:
	python -m pytest -q
smoke:
	python scripts/smoke_test.py --tag local
env:
	python scripts/check_env.py
