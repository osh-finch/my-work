"""Scoring logic for tradability and rules risk assessment."""

import logging
from datetime import datetime

from rich.console import Console

from polymarket_edges.db import Database
from polymarket_edges.models import Score

logger = logging.getLogger(__name__)
console = Console()


def calculate_tradability_score(
    best_bid: float | None,
    best_ask: float | None,
) -> float:
    """Calculate tradability score based on spread.

    Score starts at 100 and is penalised by:
    - 4000 * spread (so 1% spread = 40 points off)
    - 20 points if either side missing

    Args:
        best_bid: Best bid price (0 to 1)
        best_ask: Best ask price (0 to 1)

    Returns:
        Tradability score (0 to 100)
    """
    score = 100.0

    if best_bid is None or best_ask is None:
        score -= 20.0
        if best_bid is None and best_ask is None:
            score -= 20.0  # Extra penalty if both missing
    else:
        spread = best_ask - best_bid
        spread_penalty = 4000 * spread
        score -= spread_penalty

    return max(0.0, min(100.0, score))


def calculate_rules_risk_score(
    ambiguity_score: float | None,
    unfalsifiable_flag: bool | None,
) -> float:
    """Calculate rules risk score.

    Score starts at 0 (no risk) and increases by:
    - 100 * ambiguity_score
    - 30 if unfalsifiable_flag is true

    Args:
        ambiguity_score: Ambiguity score (0 to 1)
        unfalsifiable_flag: Whether market is unfalsifiable

    Returns:
        Rules risk score (0 to 100)
    """
    score = 0.0

    if ambiguity_score is not None:
        score += 100 * ambiguity_score

    if unfalsifiable_flag:
        score += 30.0

    return max(0.0, min(100.0, score))


def calculate_combined_score(
    tradability_score: float,
    rules_risk_score: float,
) -> float:
    """Calculate combined score.

    Combined = tradability * 0.6 + (100 - rules_risk) * 0.4

    Args:
        tradability_score: Tradability score (0 to 100)
        rules_risk_score: Rules risk score (0 to 100)

    Returns:
        Combined score (0 to 100)
    """
    combined = tradability_score * 0.6 + (100 - rules_risk_score) * 0.4
    return max(0.0, min(100.0, combined))


def compute_scores(db: Database) -> int:
    """Compute and store scores for all outcomes.

    Args:
        db: Database instance

    Returns:
        Number of scores computed
    """
    logger.info("Computing scores")

    # Get scoring data
    data = db.get_scoring_data()

    if data.empty:
        console.print("[yellow]No data available for scoring. Run ingest and update-quotes first.[/yellow]")
        return 0

    scores_computed = 0

    for _, row in data.iterrows():
        try:
            # Calculate mid price
            mid_price = None
            if row["best_bid"] is not None and row["best_ask"] is not None:
                mid_price = (row["best_bid"] + row["best_ask"]) / 2.0

            # Calculate spread
            spread = None
            if row["best_bid"] is not None and row["best_ask"] is not None:
                spread = row["best_ask"] - row["best_bid"]

            # Calculate scores
            tradability_score = calculate_tradability_score(
                row["best_bid"],
                row["best_ask"],
            )

            rules_risk_score = calculate_rules_risk_score(
                row.get("ambiguity_score"),
                row.get("unfalsifiable_flag", False),
            )

            combined_score = calculate_combined_score(
                tradability_score,
                rules_risk_score,
            )

            # Create score record
            score = Score(
                token_id=row["token_id"],
                condition_id=row["condition_id"],
                outcome=row["outcome"],
                mid_price=mid_price,
                spread=spread,
                depth_proxy=None,  # Not implemented yet
                tradability_score=tradability_score,
                rules_risk_score=rules_risk_score,
                combined_score=combined_score,
                scored_at=datetime.utcnow(),
            )

            db.insert_score(score)
            scores_computed += 1

        except Exception as e:
            logger.error(f"Failed to compute score for {row['token_id']}: {e}")
            continue

    logger.info(f"Computed {scores_computed} scores successfully")
    console.print(f"[green]✓[/green] Computed {scores_computed} scores")
    return scores_computed
