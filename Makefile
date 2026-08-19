.PHONY: lint format typecheck doccheck test docs ci ci-docker clean install

lint:
	uv run ruff check src/ tests/ examples/
	uv run ruff format --check src/ tests/ examples/

format:
	uv run ruff check --fix src/ tests/ examples/
	uv run ruff format src/ tests/ examples/

typecheck:
	uv run pyright

doccheck:
	uv run pydoclint src/preclink/

test:
	uv run pytest --cov -q

docs:
	uv run --group docs sphinx-build -W -b html docs/ docs/_build/

ci: lint typecheck doccheck test docs

ci-docker:
	docker run --rm -v $(PWD):/app -w /app python:3.12-slim sh -c "\
		pip install uv && \
		uv sync --all-groups && \
		make ci"

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/
	rm -rf docs/_build/ htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

install:
	uv sync --all-groups
