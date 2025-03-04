.PHONY: build install-packages test style bumpver bucket docker clean

install:
	pip3 install -U pip
	pip3 install poetry==1.8.3
	poetry config virtualenvs.create false
	poetry install --no-interaction --no-ansi --with dev --with optional --verbose

build:
	poetry build -f wheel


# Utilities

test:
	pytest -vs tests/ -m "not slow"

test-full:
	pytest -vs tests/

style:
	poetry run black planet

type: 
	poetry run mypy planet --config-file pyproject.toml



