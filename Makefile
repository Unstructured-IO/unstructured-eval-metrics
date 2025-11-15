.PHONY: install
install:
	uv sync

.PHONY: check
check:
	uv run ruff check .

.PHONY: tidy
tidy:
	uv run ruff check . --fix-only || true
