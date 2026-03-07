# Production Game Ops Cockpit (MVP)

Offline web app to help with:
- order selection and recommendation (`profit/min`)
- production queue scheduling
- procurement planning under game constraints
- live accounts / P&L tracking
- photo-to-orders parsing via LLM vision
- game countdown timer (start/pause/reset)
- adaptive duration learning from completed orders
- explicit COGS in profitability and procurement views
- per-order fulfillment countdown after order is taken
- procurement-order arrival countdown with custom placed-time input
- parsed-order table visibility toggle (shown/hidden)
- top-score-only order recommendations with stock-aware sourcing guidance

## Run

From the project root:

```bash
cd app
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Data Model (localStorage)

- `settings`: `gameMinute`, `operatorCount`
- `orders`: SODN-like entries with status lifecycle (`board`, `accepted`, `in_progress`, `delivered`, `rejected`)
- `stock`: starter pack + updates from procurement and production consumption
- `procurementLog`: orders with 10-minute lead handling and receipt state
- `ledger`: accounting entries (starter pack, purchases, sales, penalties, adjustments)

## Recommendation Engine

For each order:
1. estimate cycle time from quantity, size, verse lines and operator count
2. estimate material cost from colour paper sheets + envelope
3. estimate `P(late/reject)` from complexity + deadline pressure
4. compute expected penalty (`0.2 * price * P`)
5. compute expected profit and score (`expectedProfit / expectedMinutes`)

Hard fail recommendation if projected finish exceeds deadline.

## Order Photo Parsing (LLM)

In the `Orders` tab:
1. paste your API key
2. choose model/endpoint (defaults to OpenAI Responses endpoint)
3. upload a photo of the live order board
4. click `Parse Photo + Score Orders`

The app asks the LLM to extract structured orders, then computes the same score and recommendation used by the main order board.

## Technical Settings (Hidden From Ops Tabs)

Technical values are in the `Settings` tab (not shown in operational dashboards):
- API key / model / endpoint for photo parsing
- COGS is based on material purchase prices from admin price list

You can also preconfigure these before game day in:
- `config.js` (baseline defaults)
- `config.local.js` (local overrides, recommended)

## Current Constraints Enforced

- procurement max order value `£1000`
- minimum `5` minutes between procurement orders
- procurement lead time `10` minutes before stock is receivable
- rejection/missed order penalty `20%` of order value
- projected end-game asset recovery `30%` of remaining stock value
- Procurement:
  - Admin: designated lot sizes enforced (e.g., paper/envelopes in lots of 10), max `£1000` per order, min `5` mins between admin orders, `10` minute lead
  - Other Team: no max value and no lead time (received immediately), logged per item (not batched)

## Inventory Timing Rules

- Sales order materials are deducted when order is started (not when accepted).
- Each taken sales order shows a countdown to deadline.
- Each procurement order shows a countdown to arrival.
- Procurement `placed minute` is editable and defaults to current game minute, so you can record orders retroactively or in advance.

## Adaptive Prediction

When an order is started and completed/rejected, the app stores:
- predicted minutes at start
- actual minutes from start-to-complete (or manual entry)

It then updates a learned time factor (global + size-specific A5/A6/A7) so future order duration and feasibility predictions adapt during the game.

## Notes

This is an MVP for game-day operations speed, not a perfect simulator.
You can tune time/cost/risk assumptions in `app.js` constants.
