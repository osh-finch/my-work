# Poly Top - Quick Reference Card

## Installation
```bash
cd poly_top && pip install -e .
```

## Basic Usage
```bash
python -m poly_top --metric [METRIC] --limit [N]
```

---

## Metrics Cheat Sheet

| Metric | Use Case | Command |
|--------|----------|---------|
| **volume24hr** | What's hot today? | `--metric volume24hr` |
| **volumeNum** | Established markets | `--metric volumeNum` |
| **liquidityNum** | Deep order books | `--metric liquidityNum` |
| **competitive** | Market efficiency | `--metric competitive` |
| **tight_spread** | Best execution | `--metric tight_spread` |
| **composite** | Balanced opportunity | `--metric composite` |

---

## Common Commands

### Morning Routine
```bash
# What's trading today?
python -m poly_top --metric volume24hr --limit 20

# What's tradable?
python -m poly_top --metric tight_spread --min-liquidity 50000 --limit 15
```

### Find Best Opportunities
```bash
# Balanced ranking
python -m poly_top --metric composite --pages 3 --min-liquidity 10000 --limit 50
```

### Export Data
```bash
# CSV for Excel
python -m poly_top --metric composite --limit 100 --format csv > markets.csv

# JSON for scripts
python -m poly_top --metric volume24hr --format json > markets.json
```

### Pre-Trade Check
```bash
# Check a specific market
python -m poly_top --metric composite --limit 100 | grep "market name"
```

---

## Filters

| Filter | Purpose | Example |
|--------|---------|---------|
| `--min-liquidity` | Minimum $ liquidity | `--min-liquidity 50000` |
| `--min-volume` | Minimum $ volume | `--min-volume 100000` |
| `--pages` | Fetch more markets | `--pages 5` |
| `--limit` | Max results | `--limit 25` |
| `--active-only` | Active markets only | `--active-only` |
| `--include-closed` | Include resolved | `--include-closed` |

---

## Output Formats

```bash
# Pretty table (default)
python -m poly_top --metric volume24hr

# CSV
python -m poly_top --format csv > data.csv

# JSON
python -m poly_top --format json > data.json
```

---

## Metric Interpretation

### Spread
- **0.1%** = Excellent (institutional)
- **0.5%** = Good (tradable)
- **1.0%** = Fair (costs add up)
- **5%+** = Poor (avoid)

### Liquidity
- **$1M+** = Trade $50k-100k
- **$100k+** = Trade $5k-10k
- **$10k+** = Trade $500-1k
- **<$10k** = Scalping only

### Volume24hr
- **$10M+** = Major event
- **$1M+** = Active market
- **$100k+** = Decent activity
- **<$100k** = Low activity

### Competitive
- **0.9+** = Highly efficient
- **0.8-0.9** = Good
- **0.7-0.8** = Moderate
- **<0.7** = Inefficient (opportunity OR risk)

---

## Composite Score Weights

Default formula (edit `poly_top/rank.py` to customize):
```
Score = 35% volume24hr
      + 30% liquidity
      + 25% (1 - spread)    ← inverted
      + 10% competitive
```

---

## Trade Quality Checklist

Before entering a position:

- ✅ Spread < 0.5%
- ✅ Liquidity > 10× position size
- ✅ Volume24hr > $100k
- ✅ Competitive > 0.7

---

## Filtering Strategies

### Conservative (High Quality)
```bash
python -m poly_top --metric composite --min-liquidity 100000 --min-volume 500000 --limit 25
```

### Aggressive (Find Gems)
```bash
python -m poly_top --metric composite --min-liquidity 5000 --pages 5 --limit 100
```

### Execution-Focused
```bash
python -m poly_top --metric tight_spread --min-liquidity 25000 --limit 50
```

### Momentum
```bash
python -m poly_top --metric volume24hr --pages 3 --limit 30
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named poly_top" | `pip install -e .` |
| "No markets returned" | Check network, reduce filters |
| "Empty results" | Lower `--min-liquidity`, increase `--pages` |
| Slow performance | Reduce `--pages` or increase `--timeout` |

---

## Daily Workflow

```bash
# 1. Morning scan
python -m poly_top --metric volume24hr --limit 10

# 2. Build watchlist
python -m poly_top --metric composite --pages 5 --min-liquidity 10000 --format csv > watchlist_$(date +%Y%m%d).csv

# 3. Check execution quality
python -m poly_top --metric tight_spread --min-liquidity 50000 --limit 20
```

---

## Options Reference

```
--metric {volume24hr,volumeNum,liquidityNum,competitive,tight_spread,composite}
--limit LIMIT                  Maximum results (default: 50)
--min-liquidity MIN           Minimum liquidity threshold
--min-volume MIN              Minimum volume threshold
--active-only                  Only active markets (default: True)
--include-closed              Include closed markets
--pages N                      Fetch N pages (default: 1)
--format {table,json,csv}     Output format (default: table)
--timeout SECONDS             API timeout (default: 30)
-v, --verbose                 Enable debug logging
```

---

## Performance

- **1 page**: ~1-2 seconds
- **3 pages**: ~3-5 seconds
- **10 pages**: ~10-15 seconds

---

## Help

```bash
python -m poly_top --help
```

Full documentation: `RUNTHROUGH.md`
