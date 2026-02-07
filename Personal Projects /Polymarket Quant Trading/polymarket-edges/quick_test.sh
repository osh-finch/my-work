#!/bin/bash

echo "=========================================="
echo "Polymarket Edges v2 - Quick Test Pipeline"
echo "=========================================="
echo ""

# Clean start
echo "Step 1: Cleaning database..."
rm -f data/polymarket.duckdb*
echo "✓ Database cleaned"
echo ""

# Ingest a small sample
echo "Step 2: Ingesting 2 pages of markets (~200 markets)..."
polymarket-edges ingest --max-pages 2 2>&1 | grep -E "(Success|markets)"
echo ""

# Capture order book depth
echo "Step 3: Capturing order book depth (5 levels)..."
polymarket-edges update-orderbooks --levels 5 2>&1 | grep -E "(Success|order books)"
echo ""

# Compute execution metrics
echo "Step 4: Computing execution metrics (size: $100)..."
polymarket-edges compute-execution --sizes 100 2>&1 | grep -E "(Success|metrics)"
echo ""

# Detect constraints
echo "Step 5: Detecting arbitrage opportunities..."
polymarket-edges detect-constraints --size 100 2>&1 | grep -E "(Success|violations)"
echo ""

# Score outcomes
echo "Step 6: Scoring outcomes with v2 system..."
polymarket-edges score-v2 2>&1 | grep -E "(Success|outcomes)"
echo ""

# Show top results
echo "Step 7: Top 5 ranked markets:"
echo "=========================================="
polymarket-edges show-top-v2 --limit 5
echo ""

# Database stats
echo "=========================================="
echo "Database Statistics:"
echo "=========================================="
python3 -c "
from polymarket_edges.database import Database
db = Database()
print(f'Markets:              {db.conn.execute(\"SELECT COUNT(*) FROM markets\").fetchone()[0]}')
print(f'Outcomes:             {db.conn.execute(\"SELECT COUNT(*) FROM outcomes\").fetchone()[0]}')
print(f'Order Book Snapshots: {db.conn.execute(\"SELECT COUNT(*) FROM orderbook_snapshots\").fetchone()[0]}')
print(f'Order Book Levels:    {db.conn.execute(\"SELECT COUNT(*) FROM orderbook_levels\").fetchone()[0]}')
print(f'Execution Metrics:    {db.conn.execute(\"SELECT COUNT(*) FROM execution_metrics\").fetchone()[0]}')
print(f'Scores (v2):          {db.conn.execute(\"SELECT COUNT(*) FROM scores_v2\").fetchone()[0]}')
print()
print('Sample Score Data:')
result = db.conn.execute('''
    SELECT
        outcome,
        ROUND(mid_price, 3) as mid,
        ROUND(spread, 4) as spread,
        ROUND(liquidity_tax, 4) as liq_tax,
        ROUND(combined_score, 1) as score
    FROM scores_v2
    WHERE mid_price IS NOT NULL
    ORDER BY combined_score DESC
    LIMIT 3
''').fetchall()
if result:
    print('Top 3 with valid prices:')
    print('Outcome | Mid Price | Spread  | Liq Tax | Score')
    print('-' * 55)
    for row in result:
        print(f'{row[0]:7} | {row[1] if row[1] else \"N/A\":9} | {row[2] if row[2] else \"N/A\":7} | {row[3] if row[3] else \"N/A\":7} | {row[4]}')
else:
    print('No scores with valid mid prices found')
db.close()
"
echo ""

echo "=========================================="
echo "✅ Quick Test Complete!"
echo "=========================================="
echo ""
echo "If you see mid prices and spreads above (not N/A), the fix worked!"
echo ""
echo "Next steps:"
echo "  - Run full pipeline: bash validate_v2.sh"
echo "  - View dashboard: polymarket-edges serve"
echo "  - Parse rules: polymarket-edges parse-rules --provider openai --limit 10"
