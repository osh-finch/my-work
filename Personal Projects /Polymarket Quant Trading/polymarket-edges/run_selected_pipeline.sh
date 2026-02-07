#!/usr/bin/env bash
# Integration script: poly_top → polymarket-edges pipeline
#
# This script:
# 1. Uses poly_top to select the best liquid markets
# 2. Runs the full polymarket-edges v2 pipeline on ONLY those markets
# 3. Dramatically improves results by focusing on tradable opportunities
#
# Usage:
#   bash run_selected_pipeline.sh

set -euo pipefail

# ---- SETTINGS YOU CAN TWEAK ----
LIMIT=200           # Number of top markets to analyze
PAGES=1             # Pages to fetch from Gamma API (100 markets/page)
MIN_LIQ=5000       # Minimum liquidity threshold ($)
MIN_VOL=10000       # Minimum 24h volume threshold ($)
MIN_PROB=0.05      # Minimum Yes probability (5% = filter out >95% certain No)
MAX_PROB=0.95      # Maximum Yes probability (95% = filter out >95% certain Yes)
METRIC="composite"  # Use volume24hr to find most active markets
SIZES="25,100,250,1000"  # Trade sizes for execution analysis
LEVELS=30          # Order book depth levels
RULES_LIMIT=30     # Max markets for LLM rules parsing
REPORTS_LIMIT=20   # Max markets for LLM report generation

echo "=================================================="
echo "Poly Top → Polymarket Edges Integration Pipeline"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Selection metric: ${METRIC}"
echo "  Probability range: ${MIN_PROB} - ${MAX_PROB} (filters extreme outcomes)"
echo "  Markets to analyze: ${LIMIT}"
echo "  API pages to scan: ${PAGES}"
echo "  Min liquidity: \$${MIN_LIQ}"
echo "  Min 24h volume: \$${MIN_VOL}"
echo "  Trade sizes: ${SIZES}"
echo ""

# ---- 1) SELECT BEST MARKETS (writes JSON file) ----
echo "[1/9] Selecting best markets with poly_top..."
echo "  Strategy: ${METRIC} metric, probabilities ${MIN_PROB}-${MAX_PROB}"
echo "  Min liquidity: \$${MIN_LIQ}, Min volume: \$${MIN_VOL}"
mkdir -p data
python -m poly_top \
  --metric "$METRIC" \
  --limit "$LIMIT" \
  --pages "$PAGES" \
  --min-liquidity "$MIN_LIQ" \
  --min-volume "$MIN_VOL" \
  --min-prob "$MIN_PROB" \
  --max-prob "$MAX_PROB" \
  --format json \
  2> >(tee /tmp/poly_top_error.log >&2) > data/selected_markets.json

SELECTED_COUNT=$(jq length data/selected_markets.json 2>/dev/null || echo "0")
echo "✓ Selected ${SELECTED_COUNT} markets → data/selected_markets.json"
echo ""

if [ "$SELECTED_COUNT" -eq 0 ]; then
  echo "ERROR: No markets selected. Try adjusting filters (lower MIN_LIQ or increase PAGES)"
  exit 1
fi

# ---- 2) RUN V2 PIPELINE ON SELECTED MARKETS ONLY ----
# The --selected flag tells each command to only process these markets

echo "[2/9] Ingesting market metadata..."
polymarket-edges ingest
echo ""

echo "[3/9] Updating order books (${LEVELS} levels)..."
polymarket-edges update-orderbooks \
  --levels "$LEVELS" \
  --selected data/selected_markets.json
echo ""

echo "[4/9] Computing execution metrics (sizes: ${SIZES})..."
polymarket-edges compute-execution \
  --sizes "$SIZES" \
  --selected data/selected_markets.json
echo ""

echo "[5/9] Detecting constraint violations..."
polymarket-edges detect-constraints \
  --size 100 \
  --selected data/selected_markets.json
echo ""

echo "[6/9] Computing regime features..."
polymarket-edges compute-features \
  --window 24h \
  --selected data/selected_markets.json
echo ""

echo "[7/9] Computing Bayesian belief estimates..."
polymarket-edges compute-beliefs \
  --selected data/selected_markets.json
echo ""

echo "[8/9] Parsing market rules (LLM)..."
polymarket-edges parse-rules \
  --provider openai \
  --limit "$RULES_LIMIT" \
  --selected data/selected_markets.json
echo ""

echo "[9/9] Building reports (LLM)..."
polymarket-edges build-reports \
  --provider openai \
  --limit "$REPORTS_LIMIT" \
  --selected data/selected_markets.json
echo ""

echo "[FINAL] Computing v2 scores..."
polymarket-edges score-v2 \
  --selected data/selected_markets.json
echo ""

# ---- 3) DISPLAY RESULTS ----
echo "=================================================="
echo "Pipeline Complete! Top 20 Results:"
echo "=================================================="
echo ""

polymarket-edges show-top-v2 --limit 20

echo ""
echo "Next steps:"
echo "  • Review top markets above"
echo "  • Launch dashboard: polymarket-edges serve"
echo "  • Adjust parameters and re-run this script"
echo ""