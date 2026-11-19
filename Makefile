PYTHON ?= python
SOURCES = src scripts tests

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check demo experiments clean

help:
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-10s %s\n", $$1, $$2}'

install:  ## Installe le projet
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## Vérifie le style
	$(PYTHON) -m ruff check $(SOURCES)
	$(PYTHON) -m ruff format --check $(SOURCES)

format:  ## Reformate le code
	$(PYTHON) -m ruff format $(SOURCES)
	$(PYTHON) -m ruff check --fix $(SOURCES)

typecheck:  ## Vérifie les types
	$(PYTHON) -m pyright

test:  ## Lance les tests
	$(PYTHON) -m pytest

check: lint typecheck test

demo:  ## Lance la démo interactive
	$(PYTHON) -m scripts.demo

experiments:
	$(PYTHON) -m scripts.run_experiments --conditions adaptive sine --seeds 3

speciation:
	$(PYTHON) -m scripts.run_experiments --bodies worm quadruped --scenarios walk swim food

analyze:  ## Courbes moyennes et tableau récapitulatif des évolutions comparées
	$(PYTHON) -m scripts.analyze_experiments

clean:  ## Supprime les fichiers générés
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
