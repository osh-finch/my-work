# Polymarket Edges

**Production-ready analytics for Polymarket prediction markets**

⚠️ **IMPORTANT**: This is an **informational analytics tool only**. It does not provide financial advice and contains **no trade execution code**. See [LEGAL.md](LEGAL.md) for full disclaimer.

## Overview

Polymarket Edges ingests public Polymarket data via official APIs, stores it locally, and ranks markets using tradability and rules ambiguity signals. This helps analysts identify markets with good liquidity and clear resolution criteria.

### Features

- ✅ Fetches active markets from Polymarket Gamma API (paginated)
- ✅ Retrieves current order book quotes from CLOB API
- ✅ Stores raw responses and normalised data in local DuckDB database
- ✅ LLM-powered rules extraction (OpenAI or local placeholder mode)
- ✅ Scoring system combining tradability and rules risk
- ✅ Rich CLI with multiple commands
- ✅ Interactive Streamlit dashboard for filtering and exploration
- ✅ Fully async HTTP with rate limiting and retries
- ✅ Structured logging and error handling
- ✅ No HTML scraping, API-first design

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager
- (Optional) OpenAI API key for rules extraction

### Installation

1. Clone the repository:

```bash
cd polymarket-edges
```

2. Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

3. Copy the example environment file:

```bash
cp .env.example .env
```

4. (Optional) Edit `.env` to add your OpenAI API key if you want to use GPT for rules analysis:

```bash
OPENAI_API_KEY=sk-your-key-here
```

### Usage

#### Run the Full Pipeline

The easiest way to get started:

```bash
polymarket-edges pipeline
```

This runs all steps in sequence: ingest → update-quotes → parse-rules → score

#### Individual Commands

Run each step separately for more control:

```bash
# 1. Fetch active markets from Gamma API
polymarket-edges ingest --max-pages 10

# 2. Fetch current order book quotes from CLOB API
polymarket-edges update-quotes

# 3. Parse rules using LLM (use --provider openai if you have API key)
polymarket-edges parse-rules --provider local --limit 50

# 4. Compute tradability and rules risk scores
polymarket-edges score

# 5. View top ranked markets in terminal
polymarket-edges show-top --limit 20

# 6. Launch interactive dashboard
polymarket-edges serve
```

### Dashboard

The Streamlit dashboard provides:

- Interactive filtering by score, spread, and risk
- Search functionality
- Detailed market breakdowns with ambiguity analysis
- Real-time metrics

Access at `http://localhost:8501` after running `polymarket-edges serve`

## Architecture

```
polymarket-edges/
├── src/polymarket_edges/
│   ├── clients/          # API clients (Gamma, CLOB)
│   ├── llm/              # LLM provider abstraction
│   ├── config.py         # Configuration management
│   ├── db.py             # DuckDB database layer
│   ├── models.py         # Pydantic data models
│   ├── ingest.py         # Ingestion logic
│   ├── scoring.py        # Scoring algorithms
│   └── cli.py            # Typer CLI interface
├── apps/
│   └── dashboard.py      # Streamlit dashboard
├── data/                 # Local database storage
└── pyproject.toml        # Project dependencies
```

## Database Schema

### Tables

- **raw_gamma**: Raw JSON responses from Gamma API
- **raw_clob**: Raw JSON responses from CLOB API
- **markets**: Normalised market records
- **outcomes**: Normalised outcome/token records
- **quotes**: Time-series quote data (best bid/ask)
- **rules_structured**: LLM-extracted rules analysis
- **scores**: Computed tradability and risk scores

## Scoring Methodology

### Tradability Score (0-100)

Starts at 100, penalised by:
- `4000 × spread` (1% spread = 40 points deduction)
- `-20` if best bid or ask missing
- Clamped to [0, 100]

### Rules Risk Score (0-100)

Starts at 0, increased by:
- `100 × ambiguity_score` (from LLM analysis)
- `+30` if market is unfalsifiable
- Clamped to [0, 100]

### Combined Score (0-100)

```
combined = tradability_score × 0.6 + (100 - rules_risk_score) × 0.4
```

Higher scores indicate better markets (good liquidity + clear rules).

## LLM Providers

### Local Provider (Default)

No API calls, returns placeholder data. Use this for:
- Testing the pipeline
- Running without API costs
- Understanding the data flow

```bash
polymarket-edges parse-rules --provider local
```

### OpenAI Provider

Uses GPT models for sophisticated rules analysis. Requires:
- `OPENAI_API_KEY` in `.env`
- Model defaults to `gpt-4o-mini` (configurable)

```bash
polymarket-edges parse-rules --provider openai
```

The LLM extracts:
- Resolution source
- Yes/No conditions
- Key dates
- Ambiguity score and reasons
- Unfalsifiable flag

## Configuration

All settings can be configured via `.env` file:

```bash
# OpenAI (optional)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini

# Database
DATABASE_URL=data/polymarket.duckdb

# Logging
LOG_LEVEL=INFO

# Rate limiting (requests per second)
GAMMA_RATE_LIMIT=10
CLOB_RATE_LIMIT=10
```

## API Rate Limiting

The tool implements:
- Configurable rate limits per API
- Exponential backoff on failures
- Automatic retries (up to 3 attempts)
- Concurrent request limiting

Default rates are conservative (10 req/s). Adjust in `.env` if needed.

## Development

### Installing Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
black src/ apps/

# Lint
ruff check src/ apps/
```

## Makefile

Common commands are available via make:

```bash
make install        # Install package
make dev-install    # Install with dev dependencies
make format         # Format code with black
make lint           # Run ruff linter
make clean          # Clean build artefacts
make pipeline       # Run full pipeline
make dashboard      # Start dashboard
```

## Troubleshooting

### "No markets found"

- Check your internet connection
- Verify Polymarket APIs are accessible
- Try reducing `--max-pages` if timeout occurs

### "OpenAI API error"

- Verify `OPENAI_API_KEY` is set correctly
- Check API key has sufficient credits
- Try `--provider local` as fallback

### "Database locked"

- Close any other processes using the database
- Delete `data/polymarket.duckdb.wal` if safe to do so

## Legal and Compliance

⚠️ **READ BEFORE USE**: See [LEGAL.md](LEGAL.md) for important legal information including:

- Terms of service compliance
- Rate limiting obligations
- Jurisdictional restrictions
- No financial advice disclaimer
- Data usage policies

## Contributing

This is a starter repository for personal use. Feel free to fork and adapt to your needs.

## Licence

MIT Licence - see LICENCE file for details.

## Disclaimer

This tool is provided "as is" for informational purposes only. It is not financial advice. The authors are not responsible for any losses incurred from using this software. Always comply with Polymarket's terms of service and applicable laws in your jurisdiction.

**NO TRADING AUTOMATION IS INCLUDED OR SUPPORTED.**
