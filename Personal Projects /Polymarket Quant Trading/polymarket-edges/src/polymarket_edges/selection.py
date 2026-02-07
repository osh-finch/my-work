"""Utilities for loading selected markets from poly_top output."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


def load_selected_markets(selected_file: Optional[str]) -> Optional[Set[str]]:
    """Load selected market condition IDs from poly_top JSON output.

    Args:
        selected_file: Path to JSON file from poly_top --format json

    Returns:
        Set of condition IDs to process, or None if no selection file

    Raises:
        FileNotFoundError: If selected_file doesn't exist
        ValueError: If JSON is malformed or missing required fields
    """
    if not selected_file:
        return None

    path = Path(selected_file)
    if not path.exists():
        raise FileNotFoundError(f"Selected markets file not found: {selected_file}")

    try:
        with open(path) as f:
            markets = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {selected_file}: {e}")

    if not isinstance(markets, list):
        raise ValueError(f"Expected JSON array in {selected_file}, got {type(markets)}")

    # Extract condition IDs from markets
    condition_ids = set()
    for market in markets:
        if not isinstance(market, dict):
            logger.warning(f"Skipping non-dict market entry: {market}")
            continue

        # Try multiple field names for condition ID
        cond_id = (
            market.get("conditionId")
            or market.get("condition_id")
            or market.get("id")
        )

        if cond_id:
            condition_ids.add(str(cond_id))
        else:
            logger.warning(f"Market missing condition ID: {market.get('question', 'unknown')}")

    if not condition_ids:
        raise ValueError(f"No valid condition IDs found in {selected_file}")

    logger.info(f"Loaded {len(condition_ids)} selected market IDs from {selected_file}")
    return condition_ids


def filter_token_ids(
    token_ids: List[str],
    db,
    selected_conditions: Optional[Set[str]],
) -> List[str]:
    """Filter token IDs to only those from selected condition IDs.

    Args:
        token_ids: List of token IDs to filter
        db: Database instance
        selected_conditions: Set of condition IDs from load_selected_markets()

    Returns:
        Filtered list of token IDs
    """
    if selected_conditions is None:
        return token_ids

    # Query database to map tokens to conditions
    filtered = []
    for token_id in token_ids:
        # Get condition_id for this token
        result = db.conn.execute(
            "SELECT condition_id FROM outcomes WHERE token_id = ?",
            [token_id]
        ).fetchone()

        if result and result[0] in selected_conditions:
            filtered.append(token_id)

    logger.info(f"Filtered {len(token_ids)} tokens to {len(filtered)} from selected markets")
    return filtered


def filter_condition_ids(
    condition_ids: List[str],
    selected_conditions: Optional[Set[str]],
) -> List[str]:
    """Filter condition IDs to only selected ones.

    Args:
        condition_ids: List of condition IDs to filter
        selected_conditions: Set of condition IDs from load_selected_markets()

    Returns:
        Filtered list of condition IDs
    """
    if selected_conditions is None:
        return condition_ids

    filtered = [cid for cid in condition_ids if cid in selected_conditions]
    logger.info(f"Filtered {len(condition_ids)} conditions to {len(filtered)} from selected markets")
    return filtered
