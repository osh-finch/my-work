"""Market ranking and composite scoring logic."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Composite scoring weights (must sum to 1.0)
# Adjust these to change relative importance of each factor
WEIGHT_VOLUME_24H = 0.10  # Higher 24h volume is better
WEIGHT_LIQUIDITY = 0.10  # Higher liquidity is better
WEIGHT_SPREAD = 0.80  # Lower spread is better (inverted)
WEIGHT_COMPETITIVE = 0.00  # Higher competitiveness is better


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def extract_market_metrics(market: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric metrics from market dict with safe parsing.

    Args:
        market: Market data dict

    Returns:
        Dict of metric name to float value
    """
    return {
        "volume24hr": safe_float(market.get("volume24hr")),
        "volumeNum": safe_float(market.get("volumeNum")),
        "liquidityNum": safe_float(market.get("liquidityNum")),
        "spread": safe_float(market.get("spread"), default=1.0),  # Default to worst spread
        "competitive": safe_float(market.get("competitive"), default=0.0),
    }


def normalize_metrics(
    markets: List[Dict[str, Any]],
    metrics: List[str],
) -> Dict[str, Dict[str, float]]:
    """Normalize metrics to [0, 1] range for composite scoring.

    Args:
        markets: List of market dicts
        metrics: List of metric names to normalize

    Returns:
        Dict mapping market ID to dict of normalized metrics
    """
    if not markets:
        return {}

    # Extract all values for each metric
    metric_values = {metric: [] for metric in metrics}
    market_ids = []

    for market in markets:
        market_id = market.get("id") or market.get("condition_id") or market.get("conditionId")
        if not market_id:
            continue

        market_ids.append(market_id)
        extracted = extract_market_metrics(market)

        for metric in metrics:
            metric_values[metric].append(extracted.get(metric, 0.0))

    # Calculate min/max for normalization
    normalized = {}

    for i, market_id in enumerate(market_ids):
        normalized[market_id] = {}

        for metric in metrics:
            values = metric_values[metric]
            value = values[i]

            # Normalize to [0, 1]
            min_val = min(values) if values else 0.0
            max_val = max(values) if values else 1.0

            if max_val - min_val > 0:
                normalized_value = (value - min_val) / (max_val - min_val)
            else:
                normalized_value = 0.0 if value == 0 else 1.0

            normalized[market_id][metric] = normalized_value

    return normalized


def compute_composite_score(
    market: Dict[str, Any],
    normalized: Dict[str, float],
) -> float:
    """Compute composite score for a market.

    Formula:
        score = w1 * norm(volume24hr)
              + w2 * norm(liquidityNum)
              + w3 * (1 - norm(spread))      # Inverted: lower spread is better
              + w4 * norm(competitive)

    Where weights sum to 1.0.

    Args:
        market: Market data dict
        normalized: Normalized metric values for this market

    Returns:
        Composite score in [0, 1]
    """
    # For spread, we want lower to be better, so invert the normalized value
    spread_score = 1.0 - normalized.get("spread", 0.0)

    score = (
        WEIGHT_VOLUME_24H * normalized.get("volume24hr", 0.0)
        + WEIGHT_LIQUIDITY * normalized.get("liquidityNum", 0.0)
        + WEIGHT_SPREAD * spread_score
        + WEIGHT_COMPETITIVE * normalized.get("competitive", 0.0)
    )

    return score


def rank_markets(
    markets: List[Dict[str, Any]],
    metric: str,
    limit: int,
    min_liquidity: float = 0.0,
    min_volume: float = 0.0,
    min_prob: float = 0.0,
    max_prob: float = 1.0,
) -> List[Dict[str, Any]]:
    """Rank and filter markets by specified metric.

    Args:
        markets: List of market dicts
        metric: Ranking metric (volume24hr, volumeNum, liquidityNum, competitive,
                tight_spread, composite)
        limit: Maximum number of results
        min_liquidity: Minimum liquidityNum threshold
        min_volume: Minimum volumeNum threshold
        min_prob: Minimum probability for Yes outcome (filters extreme markets)
        max_prob: Maximum probability for Yes outcome (filters extreme markets)

    Returns:
        Sorted and filtered list of markets
    """
    if not markets:
        return []

    # Apply filters
    filtered = []
    for market in markets:
        metrics = extract_market_metrics(market)

        if metrics["liquidityNum"] < min_liquidity:
            continue
        if metrics["volumeNum"] < min_volume:
            continue

        # Probability filter (only for binary markets)
        if min_prob > 0.0 or max_prob < 1.0:
            try:
                outcome_prices_str = market.get("outcomePrices", "[]")
                if isinstance(outcome_prices_str, str):
                    import json
                    outcome_prices = json.loads(outcome_prices_str)
                else:
                    outcome_prices = outcome_prices_str

                if outcome_prices and len(outcome_prices) >= 1:
                    yes_prob = float(outcome_prices[0])
                    if yes_prob < min_prob or yes_prob > max_prob:
                        continue
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                # If we can't parse probabilities, skip this filter
                pass

        filtered.append(market)

    if not filtered:
        logger.warning("No markets passed filters")
        return []

    # Sort based on metric
    if metric == "composite":
        # Compute composite scores
        normalized = normalize_metrics(
            filtered,
            ["volume24hr", "liquidityNum", "spread", "competitive"],
        )

        # Add composite score to each market
        for market in filtered:
            market_id = market.get("id") or market.get("condition_id") or market.get("conditionId")
            if market_id and market_id in normalized:
                market["_composite_score"] = compute_composite_score(
                    market, normalized[market_id]
                )
            else:
                market["_composite_score"] = 0.0

        # Sort by composite score descending
        sorted_markets = sorted(
            filtered,
            key=lambda m: m.get("_composite_score", 0.0),
            reverse=True,
        )

    elif metric == "tight_spread":
        # Sort by spread ascending (tightest first)
        sorted_markets = sorted(
            filtered,
            key=lambda m: extract_market_metrics(m)["spread"],
            reverse=False,  # Ascending for spread
        )

    else:
        # Direct metric sorting (descending for all except spread)
        reverse = True  # Higher is better for volume, liquidity, competitive

        sorted_markets = sorted(
            filtered,
            key=lambda m: extract_market_metrics(m).get(metric, 0.0),
            reverse=reverse,
        )

    return sorted_markets[:limit]
