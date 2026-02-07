# Poly Top - Complete Runthrough Guide

A production-quality CLI tool for discovering and ranking the best trading opportunities on Polymarket by analyzing volume, liquidity, spreads, and competitiveness.

---

## Quick Start

```bash
# Install the package
cd poly_top
pip install -e .

# Run your first ranking
python -m poly_top --metric volume24hr --limit 10
```

---

## Why Use This Tool?

**Problem**: Polymarket has thousands of markets, most with poor liquidity and wide spreads. Finding tradable opportunities manually is time-consuming.

**Solution**: This tool automatically:
- Fetches market data from Polymarket's Gamma API
- Calculates execution quality metrics (spreads, liquidity)
- Ranks markets by multiple strategies
- Shows you the best opportunities in seconds

---

## Installation

### Method 1: Development Install (Recommended)
```bash
cd /path/to/poly_top
pip install -e .
```

### Method 2: Direct Dependencies
```bash
pip install httpx tenacity rich
cd /path/to/poly_top
python -m poly_top --help
```

### Verify Installation
```bash
python -m poly_top --metric volume24hr --limit 5
```

You should see a table with the top 5 markets by 24-hour volume.

---

## Core Concepts

### Ranking Metrics

1. **volume24hr** - 24-hour trading volume
   - Use case: Find most active markets right now
   - Good for: Momentum trading, news-driven events

2. **volumeNum** - Total lifetime volume
   - Use case: Find established, popular markets
   - Good for: Long-term positions, major events

3. **liquidityNum** - Available market depth
   - Use case: Find markets where you can trade size
   - Good for: Large positions, institutional trading

4. **competitive** - Market competitiveness score (0-1)
   - Use case: Find markets with active competition
   - Good for: Efficient pricing, arbitrage opportunities

5. **tight_spread** - Bid-ask spread (lower is better)
   - Use case: Find cheapest markets to trade
   - Good for: Minimizing execution costs, scalping

6. **composite** - Weighted combination of all metrics
   - Formula: `0.35×volume24hr + 0.30×liquidity + 0.25×(1-spread) + 0.10×competitive`
   - Use case: Find overall best trading opportunities
   - Good for: Balanced risk/reward, systematic trading

### Key Filters

- **--min-liquidity**: Exclude thin markets (recommended: 10000+)
- **--min-volume**: Exclude low-volume markets
- **--pages**: Fetch more markets for comprehensive analysis
- **--active-only**: Only show currently active markets
- **--include-closed**: Include resolved markets for backtesting

---

## Common Use Cases

### 1. Finding Today's Hot Markets

**Goal**: Discover what traders are excited about right now

```bash
python -m poly_top --metric volume24hr --limit 20
```

**What you'll see**:
- Markets with highest 24h volume
- Current price, spread, and liquidity for each
- Competitiveness score

**Example output**:
```
┃ Rank ┃ Question                              ┃  24h Vol ┃ Spread ┃
├──────┼───────────────────────────────────────┼──────────┼────────┤
│    1 │ US government shutdown Saturday?      │ $20.83M  │  0.10% │
│    2 │ Will Trump nominate Judy Shelton...   │ $16.99M  │  0.10% │
```

**When to use**: Morning routine, breaking news, market opens

---

### 2. Finding Best Execution Markets

**Goal**: Minimize trading costs by finding tight spreads

```bash
python -m poly_top --metric tight_spread --min-liquidity 50000 --limit 15
```

**What this does**:
1. Filters to markets with $50k+ liquidity
2. Sorts by tightest spread first
3. Shows top 15 markets

**Why liquidity filter matters**: A 0.1% spread on a $100 market is useless if you can't get filled.

**Example output**:
```
┃ Rank ┃ Question                              ┃ Liquidity ┃ Spread ┃
├──────┼───────────────────────────────────────┼───────────┼────────┤
│    1 │ US government shutdown Saturday?      │   $1.56M  │  0.10% │
│    2 │ Will Trump nominate Jerome Powell...  │   $3.10M  │  0.10% │
```

**When to use**: Before entering any trade, arbitrage opportunities

---

### 3. Finding Markets for Large Positions

**Goal**: Trade size without moving the market

```bash
python -m poly_top --metric liquidityNum --min-liquidity 100000 --limit 25
```

**What you'll see**:
- Markets with deepest order books
- Markets where you can trade $10k-$100k+ without slippage

**Example scenario**:
```
Market A: $1.5M liquidity, 0.1% spread → Can trade $50k position
Market B: $20k liquidity, 0.1% spread → $5k position will move market 5%+
```

**When to use**: Portfolio-sized positions, institutional trading, risk management

---

### 4. Balanced Opportunity Ranking (Composite)

**Goal**: Find overall best risk/reward opportunities

```bash
python -m poly_top --metric composite --pages 3 --limit 50
```

**What composite scoring does**:
```
Score = 35% volume24hr     (activity)
      + 30% liquidityNum   (depth)
      + 25% (1 - spread)   (execution cost, inverted)
      + 10% competitive    (market efficiency)
```

**Why it's useful**: Balances multiple factors instead of optimizing just one.

**Example comparison**:
```
Market A: High volume, low liquidity → Good for momentum, bad for size
Market B: High liquidity, low volume → Good for size, bad for momentum
Market C: Balanced (high composite)  → Good for both
```

**When to use**:
- Building a watchlist of tradable markets
- Systematic trading strategies
- General opportunity discovery

**Adjust weights**: Edit `poly_top/rank.py` lines 10-13 to customize scoring

---

### 5. Data Export for Analysis

**Goal**: Pull market data into spreadsheets or scripts

#### CSV Export
```bash
python -m poly_top --metric composite --limit 100 --format csv > markets.csv
```

**Use cases**:
- Excel analysis
- Backtesting frameworks
- Portfolio tracking
- Data pipelines

#### JSON Export
```bash
python -m poly_top --metric volume24hr --limit 50 --format json > markets.json
```

**Use cases**:
- Programmatic access
- API integrations
- Custom analysis scripts
- Database ingestion

**Example JSON structure**:
```json
[
  {
    "question": "US government shutdown Saturday?",
    "volume24hr": 20830000,
    "volumeNum": 59470000,
    "liquidityNum": 1560000,
    "spread": 0.001,
    "competitive": 0.80,
    "endDateIso": "2026-02-01",
    "slug": "us-government-shutdown-saturday",
    ...
  }
]
```

---

### 6. Comprehensive Market Scan

**Goal**: Analyze a large universe of markets for hidden gems

```bash
python -m poly_top --metric composite --pages 5 --min-liquidity 10000 --min-volume 50000 --limit 100
```

**What this does**:
1. Fetches 500 markets (5 pages × 100 per page)
2. Filters to markets with $10k+ liquidity AND $50k+ volume
3. Ranks by composite score
4. Returns top 100

**Runtime**: ~5-10 seconds (with retry logic)

**When to use**:
- Weekly market research
- Building trading universe
- Discovering new market categories
- Competitive intelligence

---

## Advanced Workflows

### Morning Trading Routine

```bash
# 1. What's hot today?
python -m poly_top --metric volume24hr --limit 10

# 2. What's tradable today?
python -m poly_top --metric tight_spread --min-liquidity 50000 --limit 20

# 3. Export for tracking
python -m poly_top --metric composite --limit 100 --format csv > ~/watchlist_$(date +%Y%m%d).csv
```

### Pre-Trade Checklist

Before entering a position on a market:

```bash
# Find the market and check multiple metrics
python -m poly_top --metric composite --limit 100 | grep "your market name"
```

Look for:
- ✅ Spread < 1% (preferably < 0.5%)
- ✅ Liquidity > 10× your position size
- ✅ Volume24hr > $100k (shows active interest)
- ✅ Competitive > 0.7 (efficient pricing)

### Building a Trading Universe

```bash
# Step 1: Get comprehensive data
python -m poly_top --metric composite --pages 10 --min-liquidity 5000 --format json > universe.json

# Step 2: Filter by category (example: Python script)
cat universe.json | jq '.[] | select(.question | contains("Trump")) | {question, volume24hr, spread}'

# Step 3: Monitor daily
python -m poly_top --metric volume24hr --format csv > daily_$(date +%Y%m%d).csv
```

---

## Output Formats

### Table (Default)

```bash
python -m poly_top --metric composite --limit 10
```

**Features**:
- Color-coded columns
- Formatted numbers ($1.5M, 0.10%)
- Truncated questions (50 chars)
- Rank numbers
- Composite score column (when using --metric composite)

**Best for**: Quick visual scanning, terminal use, human readability

---

### CSV

```bash
python -m poly_top --metric liquidityNum --limit 100 --format csv
```

**Columns**:
- question
- volume24hr
- volumeNum
- liquidityNum
- spread
- competitive
- endDateIso

**Best for**: Spreadsheet analysis, data imports, time series tracking

**Example workflow**:
```bash
# Daily export
python -m poly_top --metric composite --limit 200 --format csv >> markets_history.csv

# Analyze in Python
import pandas as pd
df = pd.read_csv('markets_history.csv')
df.groupby('question').agg({'volume24hr': 'mean', 'spread': 'min'})
```

---

### JSON

```bash
python -m poly_top --metric volume24hr --limit 50 --format json
```

**Full market objects** including:
- All metadata (id, slug, conditionId)
- Descriptions
- Image URLs
- Dates (start, end, created, updated)
- CLOB token IDs
- Market maker addresses
- Resolution data

**Best for**:
- API integrations
- Custom analysis tools
- Database ingestion
- Building trading bots

---

## Filtering Strategies

### Conservative (High Quality Only)

```bash
python -m poly_top \
  --metric composite \
  --min-liquidity 100000 \
  --min-volume 500000 \
  --limit 25
```

**Result**: Only institutional-grade markets

---

### Aggressive (Find Hidden Gems)

```bash
python -m poly_top \
  --metric composite \
  --min-liquidity 5000 \
  --pages 5 \
  --limit 100
```

**Result**: Includes smaller but potentially profitable markets

---

### Execution-Focused

```bash
python -m poly_top \
  --metric tight_spread \
  --min-liquidity 25000 \
  --limit 50
```

**Result**: Best markets for minimizing transaction costs

---

### Volume-Focused (Momentum)

```bash
python -m poly_top \
  --metric volume24hr \
  --pages 3 \
  --limit 30
```

**Result**: Markets with strongest current momentum

---

## Understanding the Metrics

### Spread

**Definition**: Difference between best bid and best ask as percentage
```
spread = (ask - bid) / mid_price
```

**Interpretation**:
- 0.1% = Excellent (institutional quality)
- 0.5% = Good (retail tradable)
- 1.0% = Fair (costs add up)
- 5%+ = Poor (avoid unless unavoidable)

**Real cost example**:
```
Position: $10,000
Spread: 0.5%
Round-trip cost: $10,000 × 0.005 × 2 = $100

If you trade 10 times/month: $1,000/month in spreads
Over 1 year: $12,000 in costs
```

---

### Liquidity

**Definition**: Total value available to trade at current prices

**Interpretation**:
- $1M+ = Can trade $50k-100k positions
- $100k+ = Can trade $5k-10k positions
- $10k+ = Can trade $500-1k positions
- < $10k = Scalping only

**Rule of thumb**: Don't trade more than 5-10% of available liquidity in a single order

---

### Competitive Score

**Definition**: Polymarket's measure of market efficiency (0 to 1)

**Interpretation**:
- 0.9+ = Highly competitive, efficient pricing
- 0.8-0.9 = Good competition
- 0.7-0.8 = Moderate competition
- < 0.7 = May have pricing inefficiencies (opportunity OR risk)

**Use case**: High competitive scores suggest harder to find edge, but more reliable pricing

---

### Volume

**24h vs Total**:
- **volume24hr**: Current interest (momentum, news flow)
- **volumeNum**: Historical interest (established market, liquidity)

**What volume tells you**:
- High volume = Active trading, easier to enter/exit
- Low volume = Patience required, wider spreads
- Increasing volume = Growing interest, momentum
- Decreasing volume = Losing interest, market resolving soon

---

## Troubleshooting

### "No module named poly_top"

**Solution**:
```bash
cd /path/to/poly_top
pip install -e .
```

---

### "No markets returned from API"

**Causes**:
1. Network issues
2. Polymarket API down
3. Filters too restrictive

**Solutions**:
```bash
# Test without filters
python -m poly_top --metric volume24hr --limit 10

# Increase timeout
python -m poly_top --metric volume24hr --timeout 60

# Check with verbose logging
python -m poly_top --metric volume24hr -v
```

---

### "Empty results after filtering"

**Cause**: Filters too restrictive

**Solution**: Relax filters or increase pages
```bash
# Before (no results)
python -m poly_top --min-liquidity 1000000 --limit 10

# After (found results)
python -m poly_top --min-liquidity 50000 --pages 3 --limit 10
```

---

### Slow performance

**Cause**: Fetching many pages

**Solutions**:
```bash
# Reduce pages (fastest)
python -m poly_top --pages 1 --limit 50

# Or increase timeout if network is slow
python -m poly_top --pages 5 --timeout 60
```

**Note**: Tool includes automatic retry with exponential backoff for network reliability

---

## Customization

### Adjust Composite Scoring Weights

Edit `poly_top/rank.py`:

```python
# Default weights (must sum to 1.0)
WEIGHT_VOLUME_24H = 0.35    # Higher 24h volume is better
WEIGHT_LIQUIDITY = 0.30     # Higher liquidity is better
WEIGHT_SPREAD = 0.25        # Lower spread is better (inverted)
WEIGHT_COMPETITIVE = 0.10   # Higher competitiveness is better
```

**Example customizations**:

```python
# Execution-focused (minimize costs)
WEIGHT_VOLUME_24H = 0.15
WEIGHT_LIQUIDITY = 0.35
WEIGHT_SPREAD = 0.45
WEIGHT_COMPETITIVE = 0.05

# Momentum-focused (follow the action)
WEIGHT_VOLUME_24H = 0.60
WEIGHT_LIQUIDITY = 0.20
WEIGHT_SPREAD = 0.10
WEIGHT_COMPETITIVE = 0.10

# Liquidity-focused (trade size)
WEIGHT_VOLUME_24H = 0.20
WEIGHT_LIQUIDITY = 0.50
WEIGHT_SPREAD = 0.25
WEIGHT_COMPETITIVE = 0.05
```

After editing, reinstall:
```bash
pip install -e .
```

---

### Add Custom Filters

Edit `poly_top/__main__.py` to add new arguments:

```python
parser.add_argument(
    "--min-competitive",
    type=float,
    default=0.0,
    help="Minimum competitive score threshold",
)
```

Then update `rank_markets()` in `poly_top/rank.py` to use the new filter.

---

## Integration Examples

### Python Script Integration

```python
import subprocess
import json

# Fetch top markets
result = subprocess.run(
    ["python", "-m", "poly_top", "--metric", "composite",
     "--limit", "50", "--format", "json"],
    capture_output=True,
    text=True
)

markets = json.loads(result.stdout)

# Filter to specific criteria
tradable = [
    m for m in markets
    if m['liquidityNum'] > 50000
    and m['spread'] < 0.01
]

print(f"Found {len(tradable)} tradable markets")
```

---

### Shell Script Monitoring

```bash
#!/bin/bash
# monitor_markets.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./market_data"

mkdir -p $OUTPUT_DIR

# Fetch and save
python -m poly_top \
  --metric composite \
  --pages 5 \
  --limit 200 \
  --format csv > "$OUTPUT_DIR/markets_$TIMESTAMP.csv"

echo "Saved to $OUTPUT_DIR/markets_$TIMESTAMP.csv"

# Alert on new high-volume markets
python -m poly_top --metric volume24hr --limit 5
```

Run with cron:
```bash
# Every hour
0 * * * * /path/to/monitor_markets.sh
```

---

### Database Ingestion

```python
import subprocess
import json
import sqlite3

# Fetch markets
result = subprocess.run(
    ["python", "-m", "poly_top", "--metric", "composite",
     "--pages", "10", "--format", "json"],
    capture_output=True,
    text=True
)

markets = json.loads(result.stdout)

# Store in database
conn = sqlite3.connect('markets.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS markets
             (question TEXT, volume24hr REAL, liquidity REAL,
              spread REAL, competitive REAL, timestamp TEXT)''')

from datetime import datetime
timestamp = datetime.now().isoformat()

for m in markets:
    c.execute(
        "INSERT INTO markets VALUES (?, ?, ?, ?, ?, ?)",
        (m['question'], m['volume24hr'], m['liquidityNum'],
         m['spread'], m['competitive'], timestamp)
    )

conn.commit()
conn.close()
```

---

## API Reference

### Gamma Markets API

**Endpoint**: `https://gamma-api.polymarket.com/markets`

**Parameters** (handled by tool):
- `limit`: Results per page (default: 100)
- `offset`: Pagination offset
- `active`: Filter active markets
- `closed`: Filter closed markets
- `order`: Sort field
- `ascending`: Sort direction

**Retry logic**: 3 attempts with exponential backoff (2-10 seconds)

**Timeout**: Configurable (default: 30s, max: 600s)

---

## Performance

### Typical Runtimes

- Single page (100 markets): ~1-2 seconds
- 3 pages (300 markets): ~3-5 seconds
- 10 pages (1000 markets): ~10-15 seconds

### Memory Usage

- Minimal (<50MB for 1000 markets)
- Suitable for resource-constrained environments

### Rate Limits

- No known rate limits on Gamma API
- Built-in retry handles transient failures

---

## Best Practices

### 1. Daily Market Research
```bash
# Morning: What's hot?
python -m poly_top --metric volume24hr --limit 20

# Afternoon: Build watchlist
python -m poly_top --metric composite --pages 5 --format csv > watchlist.csv
```

### 2. Pre-Trade Checks
- Always check spread and liquidity before trading
- Use `--min-liquidity` to filter untradable markets
- Compare multiple metrics (volume, spread, liquidity)

### 3. Data Collection
- Export daily snapshots for historical analysis
- Use CSV format for easy time series analysis
- Tag files with timestamps

### 4. Risk Management
- Don't trade more than 5-10% of available liquidity
- Wider spreads = higher minimum edge required
- Low volume markets = harder to exit

---

## Example Outputs

### High-Quality Market
```
Question: US government shutdown Saturday?
24h Vol: $20.83M
Liquidity: $1.56M
Spread: 0.10%
Competitive: 0.80

→ Excellent for trading (tight spread, deep liquidity)
```

### Medium-Quality Market
```
Question: Will Trump nominate X as Fed chair?
24h Vol: $2.5M
Liquidity: $200K
Spread: 0.30%
Competitive: 0.80

→ Good for positions < $10K
```

### Poor-Quality Market
```
Question: Will Y happen by 2027?
24h Vol: $15K
Liquidity: $8K
Spread: 5.00%
Competitive: 0.45

→ Avoid unless you have strong edge
```

---

## Next Steps

1. **Installation**: Get the tool running
2. **Exploration**: Try different metrics and filters
3. **Customization**: Adjust weights for your strategy
4. **Integration**: Build into your workflow
5. **Automation**: Set up daily exports and monitoring

---

## Support

**Issues**: Open issues at your project repository

**Documentation**: See README.md for architecture details

**API Docs**: https://docs.polymarket.com/

---

## Summary

This tool solves the fundamental problem of finding tradable opportunities on Polymarket:

- ✅ **Automated discovery** - No manual browsing
- ✅ **Multiple strategies** - Volume, liquidity, spread, composite
- ✅ **Quality filters** - Exclude untradable markets
- ✅ **Multiple formats** - Table, CSV, JSON
- ✅ **Production-ready** - Retries, error handling, logging
- ✅ **Customizable** - Adjust weights and filters

**The goal**: Spend less time finding opportunities, more time analyzing and trading them.
