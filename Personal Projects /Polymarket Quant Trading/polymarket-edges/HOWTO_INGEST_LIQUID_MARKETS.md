# How to Ingest Liquid Markets from Polymarket

## The Problem
The Gamma API returns markets in an arbitrary order, not by liquidity/volume. The first 1000-2000 markets are mostly illiquid long-tail markets.

## Solution: Manual Curation

### Step 1: Visit Polymarket Trending Page
Go to: https://polymarket.com

Look at the trending/popular markets. These typically have:
- Presidential elections
- Major sports events  
- Current news events
- High volume indicators

### Step 2: Copy Market Slugs
From each liquid market URL, copy the slug:

**Example URLs:**
```
https://polymarket.com/event/presidential-election-winner-2024
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              This is the slug

https://polymarket.com/event/trump-popular-vote-2024
                              ^^^^^^^^^^^^^^^^^^^^^
                              This is the slug
```

### Step 3: Ingest Specific Markets

**Option A: Using the ingest script (recommended)**
```bash
cd scripts
python3 ingest_by_slug.py \
    presidential-election-winner-2024 \
    trump-popular-vote-2024 \
    super-bowl-winner-2025
```

**Option B: Get condition IDs from browser**
1. Open browser DevTools (F12)
2. Go to Network tab
3. Visit a liquid market on Polymarket
4. Look for API calls to gamma-api.polymarket.com
5. Find the `condition_id` in the response
6. Manually query that market

### Step 4: Run Full Pipeline
```bash
polymarket-edges update-orderbooks
polymarket-edges compute-execution
polymarket-edges compute-features
polymarket-edges compute-beliefs
polymarket-edges score-v2
polymarket-edges show-top-v2
```

## Quick Test: Ingest Known Liquid Markets

Here are some typically liquid market types to search for:
1. **Presidential Elections** - Always most liquid
2. **NFL Playoffs** - High volume during season
3. **NBA Championships** - During playoffs
4. **Major Economic Events** - Fed decisions, etc.

## Alternative: Wait for Automated Solution

We could build a proper trending markets fetcher, but it requires:
- Web scraping (fragile, may break)
- Or Polymarket API key for advanced features
- Or waiting for them to add sort-by-volume to public API

For now, manual curation of 10-20 liquid markets is the most reliable approach.
