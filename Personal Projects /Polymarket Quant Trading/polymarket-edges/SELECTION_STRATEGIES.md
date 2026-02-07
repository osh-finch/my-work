# Market Selection Strategies

## Problem: API Liquidity ≠ Real Liquidity

The Gamma API's `liquidityNum` field doesn't always reflect actual tradable depth in CLOB order books. Many markets show high "liquidity" in the API but have terrible execution quality (wide spreads, high slippage) when you try to trade.

## Solution: Filter by Activity Metrics

Focus on markets with **proven trading activity** (high 24h volume) rather than just API liquidity numbers.

---

## Recommended Configurations

### 1. **High Volume Strategy** (Default, Best Results)

**Goal**: Find the most actively traded markets with real order book depth

```bash
LIMIT=25
PAGES=1
MIN_LIQ=150000      # $150k minimum liquidity
MIN_VOL=500000      # $500k minimum 24h volume
METRIC="volume24hr"
```

**Why it works**:
- High 24h volume = active trading = real liquidity
- $500k daily volume filters to top ~25-30 markets
- These markets have proven execution quality

**Typical results**:
- US government shutdown ($33M volume, 0.1% spread)
- Major political events (Fed nominations, elections)
- High-profile sports events
- Breaking news markets

**Use when**: Daily trading, looking for actionable opportunities

---

### 2. **Execution Quality Strategy**

**Goal**: Prioritize tightest spreads among liquid markets

```bash
LIMIT=30
PAGES=2
MIN_LIQ=100000      # $100k minimum liquidity
MIN_VOL=100000      # $100k minimum 24h volume
METRIC="tight_spread"
```

**Why it works**:
- Sorts by spread first, then filters by volume
- Finds markets with best execution costs
- Good for larger position sizes

**Use when**: Minimizing transaction costs, trading size

---

### 3. **Conservative Strategy** (Ultra High Quality)

**Goal**: Only the absolute best markets

```bash
LIMIT=15
PAGES=1
MIN_LIQ=250000      # $250k minimum liquidity
MIN_VOL=1000000     # $1M minimum 24h volume
METRIC="volume24hr"
```

**Why it works**:
- $1M daily volume = top 10-15 markets only
- Highest quality execution
- Zero noise

**Typical results**:
- 5-10 major political/news events
- Top sports markets
- Institutional-grade liquidity

**Use when**: Large positions, low risk tolerance, institutional trading

---

### 4. **Balanced Strategy**

**Goal**: Mix of volume and liquidity

```bash
LIMIT=40
PAGES=2
MIN_LIQ=75000       # $75k minimum liquidity
MIN_VOL=200000      # $200k minimum 24h volume
METRIC="composite"
```

**Why it works**:
- Composite score balances multiple factors
- Broader universe than high volume strategy
- Still filters out illiquid markets

**Use when**: Building a diverse watchlist, research

---

## How to Choose

### Start Here (Recommended for Most Users)

```bash
METRIC="volume24hr"
MIN_LIQ=150000
MIN_VOL=500000
LIMIT=25
```

This gives you **proven liquid markets** with real execution quality.

---

### If You See Poor Results (ExecQ = 0, LiqTax > 0.95)

**Increase thresholds**:
```bash
MIN_VOL=1000000     # $1M instead of $500k
MIN_LIQ=250000      # $250k instead of $150k
```

**Or switch to ultra-conservative**:
```bash
LIMIT=10            # Top 10 only
PAGES=1
```

---

### If You Want More Opportunities

**Lower thresholds carefully**:
```bash
MIN_VOL=200000      # Down from $500k
MIN_LIQ=75000       # Down from $150k
LIMIT=40            # More markets
```

**Warning**: Below $100k 24h volume, execution quality degrades rapidly.

---

## Metrics Explained

### volume24hr (Recommended)
- **What**: Total $ traded in last 24 hours
- **Why best**: Proves real trading activity
- **Use when**: Daily trading, finding opportunities

### tight_spread
- **What**: Bid-ask spread (lower is better)
- **Why useful**: Minimizes transaction costs
- **Limitation**: API spread ≠ actual CLOB spread sometimes

### composite
- **What**: Weighted combination (35% volume, 30% liquidity, 25% spread, 10% competitive)
- **Why balanced**: Considers multiple factors
- **Limitation**: Can still select markets with poor real liquidity

### liquidityNum
- **What**: API's liquidity measure
- **Why**: Useful for large positions
- **Limitation**: Doesn't guarantee real CLOB depth (least reliable)

---

## Real-World Examples

### ❌ Bad Configuration (Original)
```bash
METRIC="composite"
MIN_LIQ=1000        # Too low!
MIN_VOL=0           # No volume filter!
LIMIT=50
```

**Result**: 50 markets, but 90% have:
- ExecQ = 0.0
- LiqTax > 0.95
- No tradable depth

---

### ✅ Good Configuration (Updated)
```bash
METRIC="volume24hr"
MIN_LIQ=150000
MIN_VOL=500000
LIMIT=25
```

**Result**: 25 markets, most have:
- ExecQ > 50
- LiqTax < 0.05
- Real execution quality

---

## Customizing run_selected_pipeline.sh

Edit these lines at the top of the script:

```bash
# ---- SETTINGS YOU CAN TWEAK ----
LIMIT=25           # How many markets
PAGES=1            # API pages to scan
MIN_LIQ=150000     # Minimum liquidity ($)
MIN_VOL=500000     # Minimum 24h volume ($)
METRIC="volume24hr"  # Ranking metric
```

Then run:
```bash
bash run_selected_pipeline.sh
```

---

## Quick Reference

| Strategy | MIN_LIQ | MIN_VOL | METRIC | Markets | Quality |
|----------|---------|---------|--------|---------|---------|
| **Default** | 150k | 500k | volume24hr | 25 | High |
| **Conservative** | 250k | 1M | volume24hr | 10-15 | Very High |
| **Balanced** | 75k | 200k | composite | 40 | Medium-High |
| **Execution** | 100k | 100k | tight_spread | 30 | High |

---

## Troubleshooting

### "Selected 0 markets"

**Cause**: Thresholds too high

**Solution**:
```bash
# Lower thresholds
MIN_VOL=200000
MIN_LIQ=50000
```

---

### "All markets have ExecQ = 0"

**Cause**: Volume threshold too low

**Solution**:
```bash
# Increase volume requirement
MIN_VOL=500000  # or 1000000
```

---

### "Not enough markets for analysis"

**Cause**: Too restrictive filters

**Solution**:
```bash
# Relax slightly
MIN_VOL=300000
LIMIT=30
PAGES=2
```

---

## Summary

**Best practice**: Start with **volume24hr + high thresholds**, then adjust based on results.

```bash
# This works well for 95% of use cases
METRIC="volume24hr"
MIN_LIQ=150000
MIN_VOL=500000
LIMIT=25
```

**Key insight**: $500k+ daily volume is the sweet spot for real liquidity on Polymarket.
