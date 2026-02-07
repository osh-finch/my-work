#!/bin/bash

echo "=============================================="
echo "Polymarket Edges v2 - Full Pipeline"
echo "=============================================="
echo ""
echo "This will run the complete v2 pipeline with all features."
echo "Estimated time: 5-10 minutes (depending on market count)"
echo ""

# Configuration
MAX_PAGES=${1:-5}  # Default to 5 pages (~500 markets) if not specified
SIZES="100,250"    # Trade sizes to simulate

echo "Configuration:"
echo "  Max pages: $MAX_PAGES (~$((MAX_PAGES * 100)) markets)"
echo "  Trade sizes: \$$SIZES"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Clean start
echo "==== Step 1: Cleaning database ===="
rm -f data/polymarket.duckdb*
echo "✓ Database cleaned"
echo ""

# Ingest markets
echo "==== Step 2: Ingesting markets ===="
polymarket-edges ingest --max-pages $MAX_PAGES
if [ $? -ne 0 ]; then
    echo "❌ Ingestion failed!"
    exit 1
fi
echo ""

# Capture order book depth (v2)
echo "==== Step 3: Capturing order book depth ===="
polymarket-edges update-orderbooks --levels 30
if [ $? -ne 0 ]; then
    echo "❌ Order book capture failed!"
    exit 1
fi
echo ""

# Compute execution metrics (v2)
echo "==== Step 4: Computing execution metrics ===="
polymarket-edges compute-execution --sizes $SIZES
if [ $? -ne 0 ]; then
    echo "❌ Execution metrics failed!"
    exit 1
fi
echo ""

# Detect constraints (v2)
echo "==== Step 5: Detecting constraint violations ===="
polymarket-edges detect-constraints --size 100
if [ $? -ne 0 ]; then
    echo "❌ Constraint detection failed!"
    exit 1
fi
echo ""

# Extract regime features (v2)
echo "==== Step 6: Extracting regime features ===="
echo "Note: Requires quotes table - populating it first..."
polymarket-edges update-quotes > /dev/null 2>&1
polymarket-edges compute-features --window 24h
if [ $? -ne 0 ]; then
    echo "❌ Feature extraction failed!"
    exit 1
fi
echo ""

# Apply Bayesian filter (v2)
echo "==== Step 7: Computing Bayesian beliefs ===="
polymarket-edges compute-beliefs
if [ $? -ne 0 ]; then
    echo "❌ Belief computation failed!"
    exit 1
fi
echo ""

# Score outcomes (v2)
echo "==== Step 8: Scoring with v2 multi-component system ===="
polymarket-edges score-v2
if [ $? -ne 0 ]; then
    echo "❌ Scoring failed!"
    exit 1
fi
echo ""

# Display results
echo "=============================================="
echo "              Pipeline Complete!"
echo "=============================================="
echo ""

# Database statistics
echo "==== Database Statistics ===="
python3 -c "
from polymarket_edges.database import Database
db = Database()

markets = db.conn.execute('SELECT COUNT(*) FROM markets').fetchone()[0]
outcomes = db.conn.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]
snapshots = db.conn.execute('SELECT COUNT(*) FROM orderbook_snapshots').fetchone()[0]
levels = db.conn.execute('SELECT COUNT(*) FROM orderbook_levels').fetchone()[0]
exec_metrics = db.conn.execute('SELECT COUNT(*) FROM execution_metrics').fetchone()[0]
constraints = db.conn.execute('SELECT COUNT(*) FROM constraint_violations').fetchone()[0]
features = db.conn.execute('SELECT COUNT(*) FROM time_series_features').fetchone()[0]
beliefs = db.conn.execute('SELECT COUNT(*) FROM belief_estimates').fetchone()[0]
scores = db.conn.execute('SELECT COUNT(*) FROM scores_v2').fetchone()[0]

print(f'Markets:                 {markets:>6}')
print(f'Outcomes:                {outcomes:>6}')
print(f'Order Book Snapshots:    {snapshots:>6}')
print(f'Order Book Levels:       {levels:>6}')
print(f'Execution Metrics:       {exec_metrics:>6}')
print(f'Constraint Violations:   {constraints:>6}')
print(f'Regime Features:         {features:>6}')
print(f'Belief Estimates:        {beliefs:>6}')
print(f'Scores (v2):             {scores:>6}')

db.close()
"
echo ""

# Show top markets
echo "==== Top 10 Ranked Markets ===="
polymarket-edges show-top-v2 --limit 10
echo ""

# Score breakdown
echo "==== Score Component Analysis ===="
python3 -c "
from polymarket_edges.database import Database
db = Database()

result = db.conn.execute('''
    SELECT
        ROUND(AVG(execution_quality_score), 1) as avg_exec,
        ROUND(AVG(rules_risk_score), 1) as avg_rules,
        ROUND(AVG(constraint_edge_score), 1) as avg_constraint,
        ROUND(AVG(regime_opportunity_score), 1) as avg_regime,
        ROUND(AVG(combined_score), 1) as avg_combined,
        COUNT(*) as total
    FROM scores_v2
''').fetchone()

if result:
    print(f'Average Scores (across {result[5]} outcomes):')
    print(f'  Execution Quality:   {result[0]}/100')
    print(f'  Rules Risk:          {result[1]}/100')
    print(f'  Constraint Edge:     {result[2]}/100')
    print(f'  Regime Opportunity:  {result[3]}/100')
    print(f'  Combined:            {result[4]}/100')

db.close()
"
echo ""

# Mid price check
echo "==== Mid Price & Spread Check ===="
python3 -c "
from polymarket_edges.database import Database
db = Database()

with_prices = db.conn.execute('''
    SELECT COUNT(*) FROM scores_v2 WHERE mid_price IS NOT NULL
''').fetchone()[0]

total = db.conn.execute('SELECT COUNT(*) FROM scores_v2').fetchone()[0]

print(f'Outcomes with valid mid prices: {with_prices}/{total} ({100*with_prices/total if total > 0 else 0:.1f}%)')

if with_prices > 0:
    sample = db.conn.execute('''
        SELECT
            outcome,
            ROUND(mid_price, 4) as mid,
            ROUND(spread, 4) as spread,
            ROUND(liquidity_tax, 4) as liq_tax
        FROM scores_v2
        WHERE mid_price IS NOT NULL
        ORDER BY combined_score DESC
        LIMIT 5
    ''').fetchall()

    print()
    print('Sample (top 5 by score):')
    print('Outcome | Mid Price | Spread  | Liq Tax')
    print('-' * 45)
    for row in sample:
        print(f'{row[0]:7} | {row[1]:9.4f} | {row[2]:7.4f} | {row[3]:7.4f}')

db.close()
"
echo ""

echo "=============================================="
echo "✅ All v2 Components Working!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. View dashboard:      polymarket-edges serve"
echo "  2. Parse rules (LLM):   polymarket-edges parse-rules --provider openai --limit 20"
echo "  3. Generate reports:    polymarket-edges build-reports --provider openai --limit 5"
echo "  4. Re-score after LLM:  polymarket-edges score-v2"
echo ""
echo "Optional enhancements:"
echo "  - Run update-orderbooks periodically to track changes"
echo "  - Build up quote history for better belief estimates"
echo "  - Parse more rules to improve risk scoring"
