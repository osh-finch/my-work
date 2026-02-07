# Poly Top - Polymarket Market Rankings CLI

A production-quality CLI tool for ranking Polymarket markets by volume, liquidity, competitiveness, and composite scores.

## Features

- 📊 **Multiple ranking metrics**: 24h volume, total volume, liquidity, competitiveness, tight spreads, composite score
- 🔄 **Automatic retries**: Exponential backoff for network resilience
- 📄 **Multiple output formats**: Rich tables, JSON, CSV
- 🎯 **Flexible filtering**: Min liquidity/volume thresholds
- 📖 **Pagination support**: Fetch multiple pages for comprehensive analysis
- 🛡️ **Robust error handling**: Graceful handling of missing/malformed data

## Installation

```bash
cd poly_top
pip install -e .
```

Or install dependencies directly:

```bash
pip install httpx tenacity rich
```

## Usage

### Basic Command

```bash
python -m poly_top [options]
```

### CLI Options

```
--metric          Ranking metric (default: volume24hr)
                  Choices: volume24hr, volumeNum, liquidityNum, competitive,
                          tight_spread, composite

--limit           Max markets to display (default: 50)
--min-liquidity   Minimum liquidity threshold (default: 0)
--min-volume      Minimum total volume threshold (default: 0)
--active-only     Only include active markets (default: True)
--include-closed  Include closed markets (default: False)
--pages           Number of API pages to fetch (default: 1)
--format          Output format: table, json, csv (default: table)
--timeout         API timeout in seconds (default: 30)
-v, --verbose     Enable verbose logging
```

## Examples

### Example 1: Top Markets by 24h Volume

```bash
python -m poly_top --metric volume24hr --limit 20
```

**Sample Output:**
```
                          Top Markets by volume24hr
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Rank┃ Question                                     ┃  24h Vol┃ Total Vol┃ Liquidity ┃ Spread ┃ Competitive┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│    1│ Presidential Election Winner 2024             │ $2.45M  │ $125.30M │ $45.20K   │ 1.20%  │       0.85 │
│    2│ Will Trump win popular vote?                  │ $1.82M  │ $89.40M  │ $32.10K   │ 1.50%  │       0.78 │
│    3│ Super Bowl LIX Winner                         │ $987.30K│ $23.50M  │ $18.90K   │ 2.30%  │       0.71 │
│    4│ Will Biden run for re-election?               │ $654.20K│ $45.20M  │ $25.60K   │ 1.80%  │       0.69 │
│    5│ Fed Rate Decision March 2024                  │ $543.10K│ $12.80M  │ $14.30K   │ 3.10%  │       0.62 │
...
```

### Example 2: Most Liquid Markets with Thresholds

```bash
python -m poly_top --metric liquidityNum --min-liquidity 10000 --limit 30
```

**Sample Output:**
```
                         Top Markets by liquidityNum
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Rank┃ Question                                     ┃  24h Vol┃ Total Vol┃ Liquidity ┃ Spread ┃ Competitive┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│    1│ Presidential Election Winner 2024             │ $2.45M  │ $125.30M │ $45.20K   │ 1.20%  │       0.85 │
│    2│ Will Trump win popular vote?                  │ $1.82M  │ $89.40M  │ $32.10K   │ 1.50%  │       0.78 │
│    3│ Will Biden run for re-election?               │ $654.20K│ $45.20M  │ $25.60K   │ 1.80%  │       0.69 │
│    4│ Super Bowl LIX Winner                         │ $987.30K│ $23.50M  │ $18.90K   │ 2.30%  │       0.71 │
...

Showing 30 markets
```

### Example 3: Tightest Spreads Among Liquid Markets

```bash
python -m poly_top --metric tight_spread --min-liquidity 5000 --limit 15
```

**Sample Output:**
```
                        Top Markets by tight_spread
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ Rank┃ Question                                     ┃  24h Vol┃ Total Vol┃ Liquidity ┃ Spread ┃ Competitive┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│    1│ Presidential Election Winner 2024             │ $2.45M  │ $125.30M │ $45.20K   │ 0.80%  │       0.92 │
│    2│ Fed Rate Decision March 2024                  │ $543.10K│ $12.80M  │ $14.30K   │ 1.10%  │       0.87 │
│    3│ Will Trump win popular vote?                  │ $1.82M  │ $89.40M  │ $32.10K   │ 1.20%  │       0.84 │
│    4│ Super Bowl LIX Winner                         │ $987.30K│ $23.50M  │ $18.90K   │ 1.50%  │       0.79 │
...

Showing 15 markets
```

### Example 4: Composite Ranking

```bash
python -m poly_top --metric composite --pages 3 --limit 50
```

**Sample Output:**
```
                          Top Markets by composite
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
┃ Rank┃ Question                                     ┃  24h Vol┃ Total Vol┃ Liquidity ┃ Spread ┃ Competitive┃  Score ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
│    1│ Presidential Election Winner 2024             │ $2.45M  │ $125.30M │ $45.20K   │ 1.20%  │       0.85 │ 0.9234 │
│    2│ Will Trump win popular vote?                  │ $1.82M  │ $89.40M  │ $32.10K   │ 1.50%  │       0.78 │ 0.8891 │
│    3│ Super Bowl LIX Winner                         │ $987.30K│ $23.50M  │ $18.90K   │ 2.30%  │       0.71 │ 0.7645 │
│    4│ Fed Rate Decision March 2024                  │ $543.10K│ $12.80M  │ $14.30K   │ 3.10%  │       0.62 │ 0.6982 │
...

Showing 50 markets
```

### Example 5: JSON Output

```bash
python -m poly_top --metric volume24hr --format json --limit 10 > markets.json
```

### Example 6: CSV Export

```bash
python -m poly_top --metric composite --format csv --limit 100 > markets.csv
```

## Composite Score Formula

The composite metric combines multiple factors into a single score:

```
score = 0.35 × norm(volume24hr)         # 35% weight: higher 24h volume
      + 0.30 × norm(liquidityNum)       # 30% weight: higher liquidity
      + 0.25 × (1 - norm(spread))       # 25% weight: lower spread (inverted)
      + 0.10 × norm(competitive)        # 10% weight: higher competitiveness
```

Where `norm()` normalizes values to [0, 1] range relative to the fetched market set.

**Weights can be adjusted** in `poly_top/rank.py`:
```python
WEIGHT_VOLUME_24H = 0.35
WEIGHT_LIQUIDITY = 0.30
WEIGHT_SPREAD = 0.25
WEIGHT_COMPETITIVE = 0.10
```

## Architecture

```
poly_top/
├── __init__.py          # Package metadata
├── __main__.py          # CLI interface (argparse, output formatting)
├── gamma.py             # Gamma API client (httpx, retries, pagination)
├── rank.py              # Ranking logic (filtering, sorting, composite scoring)
├── pyproject.toml       # Dependencies and build config
└── README.md            # This file
```

## Error Handling

The tool handles common error scenarios gracefully:

- **Network errors**: Automatic retries with exponential backoff
- **Timeouts**: Configurable timeout with clear error messages
- **Missing data**: Safe parsing with sensible defaults
- **API errors**: HTTP status codes translated to user-friendly messages
- **Empty results**: Helpful message when no markets match criteria

## Development

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black poly_top/
```

### Type Checking
```bash
mypy poly_top/
```

## API Reference

### Gamma Markets API Endpoint

```
GET https://gamma-api.polymarket.com/markets
```

**Query Parameters:**
- `limit`: Number of results (default: 100)
- `offset`: Pagination offset
- `active`: Filter active markets (true/false)
- `closed`: Filter closed markets (true/false)
- `order`: Sort field (volume24hr, liquidityNum, etc.)
- `ascending`: Sort direction (true/false)

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- Type hints on all functions
- Docstrings for public APIs
- Tests for new features
- Code formatted with Black
