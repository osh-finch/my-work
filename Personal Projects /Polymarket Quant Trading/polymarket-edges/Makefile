.PHONY: help install dev-install format lint clean pipeline dashboard test

help:
	@echo "Polymarket Edges - Makefile Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install         Install package in editable mode"
	@echo "  make dev-install     Install with development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format          Format code with black"
	@echo "  make lint            Run ruff linter"
	@echo ""
	@echo "Pipeline:"
	@echo "  make pipeline        Run full data pipeline"
	@echo "  make ingest          Fetch markets from Gamma API"
	@echo "  make quotes          Update order book quotes"
	@echo "  make parse           Parse rules with local provider"
	@echo "  make score           Compute scores"
	@echo "  make show-top        Display top ranked markets"
	@echo ""
	@echo "Dashboard:"
	@echo "  make dashboard       Launch Streamlit dashboard"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean           Remove build artefacts and cache"
	@echo "  make clean-db        Remove local database"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

format:
	black src/ apps/

lint:
	ruff check src/ apps/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete

clean-db:
	rm -f data/*.duckdb
	rm -f data/*.duckdb.wal

pipeline:
	polymarket-edges pipeline --max-pages 10 --provider local

ingest:
	polymarket-edges ingest --max-pages 10

quotes:
	polymarket-edges update-quotes

parse:
	polymarket-edges parse-rules --provider local --limit 50

parse-openai:
	polymarket-edges parse-rules --provider openai --limit 50

score:
	polymarket-edges score

show-top:
	polymarket-edges show-top --limit 20

dashboard:
	polymarket-edges serve

test:
	@echo "Tests not yet implemented"
	@echo "Run: pytest tests/ when test suite is added"
