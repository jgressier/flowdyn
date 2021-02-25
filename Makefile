SRC := .
PKG := flowdyn

.PHONY: clean install

#env:
#	mkvirtualenv --python=$(which python3.7) $(PKG)

help:
	@echo "\n$$(poetry version): use target ; available Makefile following targets\n"
	@echo "  for USERS:"
	@echo "    install: install local package"
	@echo "  for DEVELOPERS (poetry based):"
	@echo "    install_dev: install local package and requirements"

install:
	pip install -e $(SRC)

install_pipdev:
	pip install -r $(SRC)/requirements.txt
	pip install -e $(SRC)
	pip install -r $(SRC)/requirements-dev.txt
	pip install -r $(SRC)/docs/requirements.txt

poetry.lock: pyproject.toml
	poetry update

install_dev:
	poetry install

check_pyproject:
	cat requirements.txt | grep -E '^[^# ]' | cut -d= -f1 | xargs -n 1 poetry add
	cat requirements-dev.txt | grep -E '^[^# ]' | cut -d= -f1 | xargs -n 1 poetry add -D
	cat docs/requirements.txt | grep -E '^[^# ]' | cut -d= -f1 | xargs -n 1 poetry add -D

test: install_dev
	poetry run pytest

serve:
	poetry run mkdocs serve

build:
	poetry build

publish: build
	poetry publish

cov_run:
	poetry run pytest --cov-report=xml

cov_publish: .codecov_token
	CODECOV_TOKEN=$(cat .codecov_token) bash <(curl -s https://codecov.io/bash)

clean:
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name ".mypy_cache" -exec rm -rf {} +
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".coverage" -exec rm -f {} +
	find . -name ".ipynb_checkpoints" -exec rm -f {} +

clean_notebooks:
	find lessons -name \*.ipynb -exec python3 scripts/remove_output.py {} +

clean_all: clean_notebooks clean