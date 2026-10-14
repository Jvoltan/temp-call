MAIN := -m src
UV_USE := uv run
USER := $(shell whoami)
HF_HOME_DIR := /sgoinfre/$(USER)/Call_me_maybe/.cache/huggingface
UV_CACHE_PATH := /sgoinfre/$(USER)/uv-cache
UV_ENV_PATH := /sgoinfre/$(USER)/Call_me_maybe/.venv
ENTRY_POINT := src/

FLAGS = --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

.PHONY: install run debug clean lint lint-strict

install: export HF_HOME := $(HF_HOME_DIR)
install: export UV_CACHE_DIR := $(UV_CACHE_PATH)
install: export UV_PROJECT_ENVIRONMENT := $(UV_ENV_PATH)
install:
	uv sync

run: export HF_HOME := $(HF_HOME_DIR)
run: export UV_CACHE_DIR := $(UV_CACHE_PATH)
run: export UV_PROJECT_ENVIRONMENT := $(UV_ENV_PATH)
run:
	$(UV_USE) python3 $(MAIN)

debug: export HF_HOME := $(HF_HOME_DIR)
debug: export UV_CACHE_DIR := $(UV_CACHE_PATH)
debug: export UV_PROJECT_ENVIRONMENT := $(UV_ENV_PATH)
debug:
	$(UV_USE) python3 -m pdb src/__main__.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint: export UV_PROJECT_ENVIRONMENT := $(UV_ENV_PATH)
lint:
	$(UV_USE) flake8 $(ENTRY_POINT)
	$(UV_USE) mypy $(ENTRY_POINT) $(FLAGS)

lint-strict: export UV_PROJECT_ENVIRONMENT := $(UV_ENV_PATH)
lint-strict:
	$(UV_USE) flake8 $(ENTRY_POINT)
	$(UV_USE) mypy --strict $(ENTRY_POINT)

docstr: export HF_HOME := $(HF_HOME_DIR)
docstr: export UV_CACHE_DIR := $(UV_CACHE_PATH)
docstr: export UV_PROJECT_ENVIRONMENT := $(UV_ENV_PATH)
docstr:
	uv sync
	uv pip install --python "$(UV_ENV_PATH)/bin/python" flake8-docstrings
	@echo "********** Docstring testing in progress ***********"
	$(UV_USE) flake8 --docstring-convention=google $(ENTRY_POINT)
	@echo "********** Testing done ************"
	uv pip uninstall --python "$(UV_ENV_PATH)/bin/python" flake8-docstrings
