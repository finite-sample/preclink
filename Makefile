.PHONY: lint format typecheck deadcode doccheck test docs ci ci-docker clean install

lint:
	ruff check preclink/ tests/
	ruff format --check preclink/ tests/

format:
	ruff check --fix preclink/ tests/
	ruff format preclink/ tests/

typecheck:
	mypy preclink/

deadcode:
	vulture preclink/

doccheck:
	pydoclint preclink/

test:
	pytest

docs:
	sphinx-build -W -b html docs/ docs/_build/

ci: lint typecheck deadcode doccheck test docs

ci-docker:
	docker run --rm -v $(PWD):/app -w /app python:3.12-slim sh -c "\
		pip install uv && \
		uv pip install --system -e '.[dev,docs]' && \
		make ci"

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf docs/_build/ htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

install:
	uv pip install -e '.[dev,docs]'
