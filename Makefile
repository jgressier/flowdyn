SRC := .
PKG := flowdyn

#env:
#	mkvirtualenv --python=$(which python3.7) $(PKG)

.PHONY: help
help: ## print this help
	@echo "\n$$(poetry version): use target ; available Makefile following targets"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## install minimum required packages and flowdyn to local
	pip install -r $(SRC)/requirements.txt
	pip install $(SRC)

install_dev: install ## install package for development and testing
	pip install -r $(SRC)/requirements-dev.txt
	pip install -r $(SRC)/docs/requirements.txt

poetry.lock: pyproject.toml
	poetry update

check_pyproject: ## check all requirements are defined in pyproject.toml
	cat requirements.txt | grep -E '^[^# ]' | cut -d= -f1 | xargs -n 1 poetry add
	cat requirements-dev.txt | grep -E '^[^# ]' | cut -d= -f1 | xargs -n 1 poetry add -D
	cat docs/requirements.txt | grep -E '^[^# ]' | cut -d= -f1 | xargs -n 1 poetry add -D

test: install_dev ## run tests with pytest
	pytest

poetry_test:  ## run tests with poetry run pytest
	poetry update
	poetry run pytest

serve:
	poetry run mkdocs serve

build: ## build package
	poetry build

publish: build ## package publishing to pypi with poetry
	poetry publish

cov_run:
	poetry run pytest --cov-report=xml

cov_publish: .codecov_token
	CODECOV_TOKEN=$$(cat .codecov_token)  bash <(curl -s https://codecov.io/bash)

clean: ## clean all unnecessary files
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name ".mypy_cache" -exec rm -rf {} +
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".coverage" -exec rm -f {} +
	find . -name ".ipynb_checkpoints" -exec rm -f {} +

clean_notebooks: ## remove ouputs in Jupyter notebooks files
	find lessons -name \*.ipynb -exec python3 scripts/remove_output.py {} +

clean_all: clean_notebooks clean ## run clean and clean_notebooks