# Polymarket Edges v2: Execution-Aware Analytics

**Production-ready execution-aware analytics for Polymarket prediction markets**

⚠️ **IMPORTANT**: This is an **informational analytics tool only**. It does not provide financial advice and contains **no trade execution code**. See [LEGAL.md](LEGAL.md) for full disclaimer.

## What's New in v2

v2 is a major upgrade focused on **execution-aware analysis** and **research-grade infrastructure**:

✅ **Order Book Depth Capture** - 30 levels of bids/asks, not just top-of-book
✅ **Execution Simulation** - VWAP, slippage, liquidity tax at multiple trade sizes
✅ **Fee Model** - Parameterised taker fees and maker rebates
✅ **Constraint Detection** - Complete set arbitrage and cross-market consistency
✅ **Regime Features** - Spread trends, volatility, market age, time-to-resolution
✅ **Bayesian Belief Filter** - Kalman-style de-noising of market prices
✅ **Enhanced LLM Analysis** - Structured JSON extraction + narrative reports
✅ **Multi-Component Scoring** - Execution quality + rules clarity + constraint edge + regime
✅ **Full CLI** - Commands for every stage of the v2 pipeline
✅ **Enhanced Dashboard** - Execution curves, constraint panels, regime signals

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Data Pipeline](#data-pipeline)
4. [Execution Simulation](#execution-simulation)
5. [Constraint Detection](#constraint-detection)
6. [Regime Features](#regime-features)
7. [Bayesian Belief Filter](#bayesian-belief-filter)
8. [LLM Analysis](#llm-analysis)
9. [Scoring v2](#scoring-v2)
10. [CLI Reference](#cli-reference)
11. [Configuration](#configuration)
12. [Dashboard](#dashboard)
13. [Assumptions and Limitations](#assumptions-and-limitations)
14. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager
- (Optional) OpenAI API key for LLM-based analysis

### Installation

```bash
# Clone and navigate to repo
cd polymarket-edges

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Copy environment template
cp .env.example .env

# (Optional) Edit .env to add OpenAI API key
nano .env
```

### Run the v2 Pipeline

```bash
# Full pipeline (all steps)
polymarket-edges ingest
polymarket-edges update-orderbooks --levels 30
polymarket-edges compute-execution --sizes 25,100,250,1000
polymarket-edges detect-constraints
polymarket-edges compute-features
polymarket-edges compute-beliefs
polymarket-edges parse-rules --provider openai  # or 'local' for placeholders
polymarket-edges build-reports --provider openai
polymarket-edges score-v2

# View top ranked markets
polymarket-edges show-top-v2 --limit 20

# Launch dashboard
polymarket-edges serve
```

---

## Architecture

```
polymarket-edges/
├── src/polymarket_edges/
│   ├── clients/          # API clients (Gamma, CLOB)
│   ├── execution/        # Order book simulator, fee calculator
│   ├── constraints/      # Arbitrage detection
│   ├── features/         # Regime analysis, belief filter
│   ├── scoring/          # v2 multi-component scorer
│   ├── llm/              # LLM providers (OpenAI, local)
│   ├── workflows/        # v2 pipeline orchestration
│   ├── database.py       # DuckDB with extended v2 schema
│   ├── config.py         # Configuration management
│   ├── models.py         # Pydantic data models
│   └── cli.py            # Typer CLI interface
├── apps/
│   └── dashboard.py      # Streamlit dashboard
├── data/
│   └── polymarket.duckdb # Local DuckDB database
└── pyproject.toml        # Dependencies
```

### Database Schema (v2)

**Core Tables:**
- `markets` - Market metadata
- `outcomes` - Token IDs and outcome labels
- `quotes` - Top-of-book time series

**v2 Tables:**
- `orderbook_snapshots` - Full order book snapshots with timestamp
- `orderbook_levels` - Individual bid/ask levels (snapshot_id, side, level_index, price, size)
- `execution_metrics` - Simulated execution results per size bucket
- `rules_structured` - LLM-extracted resolution criteria
- `constraint_violations` - Detected arbitrage opportunities
- `time_series_features` - Regime and lifecycle features
- `belief_estimates` - Bayesian filtered beliefs
- `reports` - Human-readable markdown reports
- `scores_v2` - Multi-component scores

---

## Data Pipeline

### 1. Gamma Ingestion

Fetches all active markets from the Gamma Markets API:

```bash
polymarket-edges ingest --max-pages 10
```

Stores:
- Raw JSON responses in `raw_gamma`
- Normalised market and outcome records

### 2. Order Book Update (v2)

Captures **30 levels of depth** on each side:

```bash
polymarket-edges update-orderbooks --levels 30
```

For each outcome:
- Fetches bids and asks up to N levels
- Stores snapshot with timestamp
- Stores each level individually

**Key difference from v1:** Previously captured only top-of-book. Now captures full depth for realistic execution simulation.

---

## Execution Simulation

### Overview

The execution simulator walks through order book levels to compute realistic execution prices at different trade sizes.

### Simulator Functions

**`simulate_buy_yes(asks, target_notional)`**
- Walks through ask levels to simulate buying YES tokens
- Computes VWAP, fill ratio, slippage
- Accounts for partial fills

**`simulate_sell_yes(bids, target_quantity)`**
- Walks through bid levels to simulate selling YES tokens
- Computes VWAP, proceeds after fees

**`compute_liquidity_tax(bids, asks, size)`**
- Entry: Buy YES at asks → `entry_vwap`
- Exit: Sell YES at bids → `exit_vwap`
- Liquidity tax = `entry_vwap - exit_vwap`

### Trade Size Buckets

Default sizes: `[25, 100, 250, 1000]` USD notional

Configure in `.env`:
```bash
TRADE_SIZE_BUCKETS=[25.0,100.0,250.0,1000.0]
REFERENCE_SIZE_BUCKET=100.0  # Used for scoring
```

### Fee Model

All execution metrics account for taker fees:

```bash
TAKER_FEE_BPS=0.0      # 0 basis points = 0.00%
MAKER_REBATE_BPS=0.0
```

To model fees:
```bash
TAKER_FEE_BPS=20.0  # 20 bps = 0.20%
```

**Important:** Polymarket currently has 0% fees, but the system is built to handle non-zero fees.

### Stored Metrics

For each outcome and size bucket:
- `entry_vwap` - Average price when buying
- `exit_vwap` - Average price when selling
- `liquidity_tax` - Round-trip cost
- `fill_ratio` - Fraction of order filled (0-1)
- `effective_spread` - Entry VWAP - Exit VWAP

### Command

```bash
polymarket-edges compute-execution --sizes 25,100,250,1000
```

### Assumptions

1. **Taker model:** All trades take liquidity (cross the spread)
2. **Instantaneous execution:** No market impact beyond consuming levels
3. **Static order book:** Snapshot does not change during simulation
4. **No fees currently:** Polymarket has 0% taker fees (as of v2 build)
5. **Notional sizing:** Sizes in USD equivalent, not token units

---

## Constraint Detection

### Complete Set Arbitrage

For binary markets (YES/NO), buying a complete set should cost $1.00 exactly (since one will pay out $1.00).

**Underpriced:**
```
cost = entry_vwap(YES) + entry_vwap(NO)
if cost < 1.0: arbitrage opportunity
```

**Overpriced:**
```
proceeds = exit_vwap(YES) + exit_vwap(NO)
if proceeds > 1.0: arbitrage opportunity
```

### Detection Threshold

```bash
COMPLETE_SET_THRESHOLD=0.005  # 0.5% = 50 bps
```

Only flag violations exceeding this threshold to avoid noise.

### Cross-Market Consistency (Future)

v2 lays groundwork for cross-market constraints:
- If market A implies market B, then P(A) ≤ P(B)
- For mutually exclusive outcomes, sum ≤ 1.0

**Current implementation:** Focuses on complete set violations. Cross-market links require LLM-based market matching (high confidence threshold).

### Command

```bash
polymarket-edges detect-constraints --size 100
```

### Stored Violations

`constraint_violations` table:
- `violation_type` - complete_set_under_1, complete_set_over_1, etc.
- `magnitude` - Size of opportunity
- `evidence` - Prices used in detection
- `detected_at` - Timestamp

---

## Regime Features

### Overview

Time series features that characterise the market's "regime":

**Lifecycle:**
- `market_age_days` - Time since first seen
- `time_to_resolution_days` - Days until market expires

**Price dynamics:**
- `spread_trend_short` - Slope of spread over 24h
- `spread_trend_long` - Slope over 7 days
- `price_volatility_short` - Std dev of mid over 24h
- `price_volatility_long` - Std dev over 7 days

**Liquidity:**
- `liquidity_depth_short` - Average depth over 24h
- `liquidity_depth_long` - Average depth over 7 days

### Time Windows

```bash
REGIME_WINDOW_HOURS_SHORT=24
REGIME_WINDOW_HOURS_LONG=168  # 7 days
```

### Command

```bash
polymarket-edges compute-features --window 24h
```

### Usage in Scoring

Regime features feed into the **regime opportunity score**:
- Higher score when time-to-resolution is short
- Lower score when volatility is exploding or spreads widening
- Combined with belief uncertainty

---

## Bayesian Belief Filter

### Overview

Markets can be noisy. The belief filter estimates the "true" latent probability by treating:
- **Latent belief θ_t** - Random walk (slow drift)
- **Observed mid price p_t** - Noisy observation

Observation noise inversely related to liquidity (low liquidity → high noise).

### Algorithm

Kalman-style filter:
1. **Predict:** `θ_{t+1} ~ N(θ_t, Q)` where Q = process variance
2. **Update:** Incorporate new observation using Kalman gain

### Parameters

```bash
BELIEF_PROCESS_VARIANCE=0.0001
BELIEF_MIN_LIQUIDITY=100.0
```

### Output

Per outcome:
- `posterior_mean` - Filtered belief estimate
- `posterior_std` - Uncertainty

### Command

```bash
polymarket-edges compute-beliefs
```

### When to Use

Use `posterior_mean` instead of raw `mid_price` when:
- Liquidity is sufficient (`>= BELIEF_MIN_LIQUIDITY`)
- You want a de-noised estimate

---

## LLM Analysis

### Two-Step Process

**Step 1: Structured Extraction** (strict JSON)

```bash
polymarket-edges parse-rules --provider openai --limit 50
```

Extracts:
- `resolution_source` - Who decides?
- `primary_measurement` - What's being measured?
- `yes_conditions` - What makes YES true?
- `no_conditions` - What makes NO true?
- `key_dates` - Important timestamps
- `edge_cases` - Unusual scenarios
- `ambiguity_score` - 0 (clear) to 1 (ambiguous)
- `unfalsifiable_flag` - Can this be objectively verified?
- `dispute_risk_notes` - Potential controversies
- `recommended_evidence_to_monitor` - Sources to track

**Step 2: Report Generation** (grounded narrative)

```bash
polymarket-edges build-reports --provider openai --limit 20
```

Generates markdown reports with:
- Payout conditions summary
- Key numbers table
- Execution analysis
- Constraint signals
- Regime characteristics
- Risk assessment

**Grounding:** LLM is given a structured facts payload (JSON) with ALL quantitative data. It must not invent numbers.

### Providers

**OpenAI:**
- Uses GPT-4o-mini (configurable)
- Requires `OPENAI_API_KEY`

**Local:**
- No API calls
- Returns placeholder data
- Useful for testing pipeline

### Prompts

Prompts are in `src/polymarket_edges/llm/provider.py`:
- `RULES_EXTRACTION_PROMPT_V2` - Structured extraction
- `REPORT_GENERATION_PROMPT` - Narrative generation

---

## Scoring v2

### Multi-Component System

Score = weighted combination of 4 components:

1. **Execution Quality (0-100)**
2. **Rules Clarity (0-100)** - Inverted risk
3. **Constraint Edge (0-100)**
4. **Regime Opportunity (0-100)**

### Component Formulas

**1. Execution Quality**

```
score = 100
score -= 4000 × effective_spread
score -= (1 - fill_ratio) × 100
clamp to [0, 100]
```

Example: 1% effective spread = 40 point deduction

**2. Rules Risk** (then inverted)

```
risk = 100 × ambiguity_score
if unfalsifiable: risk += 30
clamp to [0, 100]
clarity = 100 - risk
```

**3. Constraint Edge**

```
if complete_set_buy_cost < 1:
    edge = 1 - cost
    score = min(100, edge × 20000)
else:
    score = 0
```

Example: 0.5% edge → score = 100

**4. Regime Opportunity**

Heuristic based on:
- Time to resolution (prefer < 30 days, not past due)
- Belief uncertainty (prefer low)
- Volatility (prefer moderate, not exploding)
- Spread trend (prefer tightening, not widening)

### Weights

```bash
SCORE_WEIGHT_EXECUTION=0.45
SCORE_WEIGHT_RULES=0.25
SCORE_WEIGHT_CONSTRAINT=0.20
SCORE_WEIGHT_REGIME=0.10
```

### Combined Score

```
combined = 0.45 × execution_quality
         + 0.25 × rules_clarity
         + 0.20 × constraint_edge
         + 0.10 × regime_opportunity
```

### Command

```bash
polymarket-edges score-v2
```

### Interpretation

- **90-100:** Excellent market (tight execution, clear rules, potential edge)
- **70-90:** Good market (decent execution, moderate clarity)
- **50-70:** Mediocre market (wide spreads or ambiguous rules)
- **0-50:** Poor market (illiquid or high-risk rules)

---

## CLI Reference

### Ingestion

```bash
# Fetch markets from Gamma API
polymarket-edges ingest --max-pages 10
```

### v2 Pipeline

```bash
# Update order books with depth
polymarket-edges update-orderbooks --levels 30

# Compute execution metrics
polymarket-edges compute-execution --sizes 25,100,250,1000

# Detect constraints
polymarket-edges detect-constraints --size 100

# Compute regime features
polymarket-edges compute-features --window 24h

# Compute Bayesian beliefs
polymarket-edges compute-beliefs

# Parse rules (structured JSON)
polymarket-edges parse-rules --provider openai --limit 50

# Build reports (grounded narratives)
polymarket-edges build-reports --provider openai --limit 20

# Score v2
polymarket-edges score-v2
```

### Display

```bash
# Show top v2 scores
polymarket-edges show-top-v2 --limit 20

# Launch dashboard
polymarket-edges serve --port 8501
```

### Legacy Commands (v1 compatibility)

```bash
# Update top-of-book quotes (v1)
polymarket-edges update-quotes

# Score v1
polymarket-edges score

# Show top v1
polymarket-edges show-top
```

---

## Configuration

All settings in `.env`:

```bash
# API
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=data/polymarket.duckdb
LOG_LEVEL=INFO

# Rate limiting
GAMMA_RATE_LIMIT=10.0
CLOB_RATE_LIMIT=10.0

# Execution simulation
ORDERBOOK_DEPTH_LEVELS=30
TRADE_SIZE_BUCKETS=[25.0,100.0,250.0,1000.0]
REFERENCE_SIZE_BUCKET=100.0

# Fee model
TAKER_FEE_BPS=0.0
MAKER_REBATE_BPS=0.0

# Constraints
CONSTRAINT_CONFIDENCE_THRESHOLD=0.8
COMPLETE_SET_THRESHOLD=0.005

# Regime features
REGIME_WINDOW_HOURS_SHORT=24
REGIME_WINDOW_HOURS_LONG=168

# Belief filter
BELIEF_PROCESS_VARIANCE=0.0001
BELIEF_MIN_LIQUIDITY=100.0

# Scoring weights
SCORE_WEIGHT_EXECUTION=0.45
SCORE_WEIGHT_RULES=0.25
SCORE_WEIGHT_CONSTRAINT=0.20
SCORE_WEIGHT_REGIME=0.10
```

---

## Dashboard

### Features

**v2 Dashboard includes:**

1. **Market List** - Ranked by combined score
2. **Filters:**
   - Min execution quality
   - Max rules risk
   - Has constraint edge
   - Time-to-resolution range
3. **Market Detail View:**
   - Key numbers table
   - Execution curves by size bucket
   - Constraint violation panel
   - Payout conditions summary
   - Full markdown report

### Launch

```bash
polymarket-edges serve
```

Access at `http://localhost:8501`

---

## Assumptions and Limitations

### Execution Simulator

**Assumptions:**
1. **Taker model** - All trades cross the spread
2. **Static snapshot** - Order book doesn't change during simulation
3. **No market impact** - Beyond consuming visible levels
4. **Notional sizing** - Sizes in USD, not token units
5. **Zero fees** - Polymarket currently has no taker fees

**Limitations:**
- Does not model maker strategies
- Does not account for order book regeneration
- Does not model adverse selection
- Assumes instantaneous execution

### Constraint Detection

**Limitations:**
- Only detects complete set violations in binary markets
- Cross-market links not yet implemented
- Does not account for execution costs in arbitrage P&L
- Threshold (0.5%) may miss smaller opportunities

### Belief Filter

**Limitations:**
- Simple random walk model
- Observation variance inversely related to liquidity (simple proxy)
- Not a prediction model, just a de-noising filter
- Requires sufficient history (at least 2 observations)

### LLM Analysis

**Limitations:**
- Depends on LLM quality (gpt-4o-mini vs opus)
- May misinterpret complex rules
- Ambiguity score is subjective
- Report generation can hallucinate if not properly grounded

### Scoring

**Limitations:**
- Weights are fixed (not learned)
- Regime scoring is heuristic, not optimised
- Does not account for correlations between markets
- Not a prediction of market direction

### Safety Rails

- **No trade execution** - This is analytics only
- **Read-only APIs** - Only fetches public data
- **Rate limiting** - Respects API limits
- **Local database** - No data sent to third parties (except OpenAI if used)

---

## Troubleshooting

### "No markets found"

- Check internet connection
- Verify Polymarket APIs are accessible
- Try reducing `--max-pages`

### "OpenAI API error"

- Verify `OPENAI_API_KEY` is set correctly in `.env`
- Check API key has credits
- Use `--provider local` as fallback

### "Database locked"

- Close any other processes using the database
- Delete `data/polymarket.duckdb.wal` if safe

### "Execution metrics not computed"

- Ensure order books were updated first with `update-orderbooks`
- Check that order book snapshots exist in database

### "Scores v2 are all zero"

- Verify execution metrics were computed
- Check that rules were parsed
- Ensure features and beliefs were computed

---

## Legal and Compliance

⚠️ **READ BEFORE USE**: See [LEGAL.md](LEGAL.md) for important legal information including:

- Terms of service compliance
- Rate limiting obligations
- Jurisdictional restrictions
- No financial advice disclaimer
- Data usage policies

---

## Contributing

This is a research-grade starter repository. Feel free to fork and adapt.

---

## Licence

MIT Licence - see LICENCE file

---

## Disclaimer

This tool is provided "as is" for informational purposes only. It is not financial advice. The authors are not responsible for any losses incurred from using this software. Always comply with Polymarket's terms of service and applicable laws in your jurisdiction.

**NO TRADING AUTOMATION IS INCLUDED OR SUPPORTED.**

---

## Version History

**v2.0.0** (2026-01)
- Execution-aware analytics with order book depth
- Multi-component scoring system
- Bayesian belief filter
- Enhanced LLM analysis with grounded reports
- Comprehensive constraint detection

**v1.0.0** (2025-12)
- Initial release with top-of-book quotes
- Basic scoring (tradability + rules risk)
- LLM rules extraction

---

**Built with care for Polymarket analysts and researchers 🔍**
