# Use bash for a couple of niceties
SHELL := /bin/bash
PY := python
NAME := $(notdir $(CURDIR))     # project dir name
KERNEL := Python ($(NAME))

.PHONY: ensure env sync add shell setup lint fmt lab clean

# Ensure .venv exists and dependencies are installed
ensure:
	@if [ ! -d ".venv" ]; then echo "==> Creating venv"; uv venv; fi
	@echo "==> Syncing dependencies"
	@uv sync

# Explicit targets if you want to call them separately
env:
	uv venv

sync:
	uv sync

# Add a dependency, e.g. make add PKG=pydantic
add: ensure
	uv add $(PKG)

# Handy interactive shell *inside* the env
shell: ensure
	uv run $(SHELL)

# Register Jupyter kernel for this project
setup: ensure
	uv run $(PY) -m ipykernel install --user --name $(NAME) --display-name "$(KERNEL)"

lint: ensure
	uv run ruff check src notebooks

fmt: ensure
	uv run ruff check --fix src notebooks
	uv run ruff format src notebooks

# Robust Jupyter launch: invoke via module to avoid PATH issues
lab: ensure
	uv run $(PY) -m jupyterlab

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \; ; \
	find . -type f -name "*.pyc" -delete
