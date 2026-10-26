# Project commands use the active Python environment; no Poetry is required.
PYTHON ?= python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
MKDOCS := $(PYTHON) -m mkdocs
BUILD := $(PYTHON) -m build
TWINE := $(PYTHON) -m twine

.DEFAULT_GOAL := help

.PHONY: help install install-dev install-doc test test-mpl coverage lint docs-serve docs-build build publish clean clean-notebooks clean-all install_dev cov_run serve clean_notebooks clean_all

help: ## show available commands
	@$(PYTHON) --version
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## install Flowdyn from this checkout
	$(PIP) install .

install-dev: ## install Flowdyn and development dependencies in editable mode
	$(PIP) install --editable '.[dev]'

install-doc: ## install Flowdyn and documentation dependencies in editable mode
	$(PIP) install --editable '.[doc]'
	$(PIP) install -r docs/requirements.txt

test: ## run the test suite
	$(PYTEST)

test-mpl: ## run Matplotlib image-regression tests
	$(PYTEST) --mpl --mpl-generate-summary=html

coverage: ## run tests and write coverage.xml
	$(PYTEST) --cov=flowdyn --cov-report=xml

lint: ## report Flake8 findings (syntax errors remain fatal)
	$(PYTHON) -m flake8 flowdyn tests --count --select=E9,F63,F7,F82 --show-source --statistics
	$(PYTHON) -m flake8 flowdyn tests --count --exit-zero --statistics

docs-serve: install-doc ## serve the MkDocs site locally
	$(MKDOCS) serve

docs-build: install-doc ## build the MkDocs site
	$(MKDOCS) build

build: ## build source and wheel distributions
	$(PIP) install build
	$(BUILD)

publish: build ## upload existing distributions to PyPI (requires credentials)
	$(PIP) install twine
	$(TWINE) upload dist/*

clean: ## remove Python test, coverage, and documentation build artefacts
	find . -type d \( -name '__pycache__' -o -name '.mypy_cache' -o -name '.pytest_cache' \) -prune -exec rm -rf {} +
	find . -type f \( -name '.coverage' -o -name 'coverage.xml' \) -delete
	rm -rf build dist htmlcov mpl-results

clean-notebooks: ## clear notebook outputs from lessons
	find lessons -name '*.ipynb' -exec $(PYTHON) scripts/remove_output.py {} +

clean-all: clean clean-notebooks ## remove all generated local artefacts

# Compatibility aliases for the previous snake_case targets.
install_dev: install-dev
cov_run: coverage
serve: docs-serve
clean_notebooks: clean-notebooks
clean_all: clean-all
