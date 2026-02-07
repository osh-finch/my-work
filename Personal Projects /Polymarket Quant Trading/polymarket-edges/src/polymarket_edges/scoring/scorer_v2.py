"""v2 Scoring system with execution-aware multi-component ranking."""

import logging
from dataclasses import dataclass
from typing import Any

from polymarket_edges.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ScoreComponents:
    """Individual score components for an outcome."""

    execution_quality: float  # 0-100
    rules_risk: float  # 0-100
    constraint_edge: float  # 0-100
    regime_opportunity: float  # 0-100
    combined: float  # 0-100


class ScorerV2:
    """v2 scoring system with multiple components."""

    def __init__(
        self,
        weight_execution: float | None = None,
        weight_rules: float | None = None,
        weight_constraint: float | None = None,
        weight_regime: float | None = None,
    ):
        """Initialise scorer with weights.

        Args:
            weight_execution: Weight for execution quality (defaults to config)
            weight_rules: Weight for rules clarity (defaults to config)
            weight_constraint: Weight for constraint edge (defaults to config)
            weight_regime: Weight for regime opportunity (defaults to config)
        """
        self.weight_execution = weight_execution or settings.score_weight_execution
        self.weight_rules = weight_rules or settings.score_weight_rules
        self.weight_constraint = weight_constraint or settings.score_weight_constraint
        self.weight_regime = weight_regime or settings.score_weight_regime

        # Normalise weights
        total_weight = (
            self.weight_execution
            + self.weight_rules
            + self.weight_constraint
            + self.weight_regime
        )
        if total_weight > 0:
            self.weight_execution /= total_weight
            self.weight_rules /= total_weight
            self.weight_constraint /= total_weight
            self.weight_regime /= total_weight

    def compute_execution_quality_score(
        self,
        effective_spread: float | None,
        fill_ratio: float | None,
    ) -> float:
        """Compute execution quality score (0-100).

        Formula:
        - If no data available, return 0 (cannot assess tradability)
        - Otherwise start at 100
        - Subtract 4000 * effective_spread
        - Subtract (1 - fill_ratio) * 100 as fill penalty
        - Clamp to [0, 100]

        Args:
            effective_spread: Entry VWAP - Exit VWAP
            fill_ratio: Fraction of order filled (0-1)

        Returns:
            Score 0-100 (higher is better)
        """
        # If no execution data available, we cannot assess tradability
        # Return 0 to push these markets down in rankings
        if effective_spread is None and fill_ratio is None:
            return 0.0

        score = 100.0

        if effective_spread is not None:
            spread_penalty = 1000.0 * effective_spread
            score -= spread_penalty

        if fill_ratio is not None:
            fill_penalty = (1.0 - fill_ratio) * 100.0
            score -= fill_penalty

        return max(0.0, min(100.0, score))

    def compute_rules_risk_score(
        self,
        ambiguity_score: float | None,
        unfalsifiable_flag: bool,
    ) -> float:
        """Compute rules risk score (0-100).

        Formula:
        - Start at 0
        - Add 100 * ambiguity_score
        - Add 30 if unfalsifiable_flag
        - Clamp to [0, 100]

        Higher score = higher risk (worse)

        Args:
            ambiguity_score: Ambiguity from LLM (0-1)
            unfalsifiable_flag: Whether market is unfalsifiable

        Returns:
            Risk score 0-100 (higher is worse)
        """
        score = 0.0

        if ambiguity_score is not None:
            score += 100.0 * ambiguity_score

        if unfalsifiable_flag:
            score += 30.0

        return max(0.0, min(100.0, score))

    def compute_constraint_edge_score(
        self,
        complete_set_buy_cost: float | None,
    ) -> float:
        """Compute constraint edge score (0-100).

        If complete set can be bought for < 1.0, this is an edge.

        Formula:
        - If cost < 1: edge = (1 - cost), score = min(100, edge * 20000)
        - Else: score = 0

        Example: 0.5% edge (cost = 0.995) -> score = 100

        Args:
            complete_set_buy_cost: Cost to buy YES + NO

        Returns:
            Score 0-100 (higher is better)
        """
        if complete_set_buy_cost is None:
            return 0.0

        if complete_set_buy_cost < 1.0:
            edge = 1.0 - complete_set_buy_cost
            score = min(100.0, edge * 20000.0)
            return score

        return 0.0

    def compute_regime_opportunity_score(
        self,
        time_to_resolution_days: float | None,
        price_volatility: float | None,
        spread_trend: float | None,
        belief_std: float | None,
    ) -> float:
        """Compute regime opportunity score (0-100).

        Heuristic scoring based on regime features.

        Higher scores when:
        - Time to resolution is short but not past due
        - Belief uncertainty is low
        - Volatility is moderate
        - Spreads are not widening

        Args:
            time_to_resolution_days: Days until market resolution
            price_volatility: Recent price volatility
            spread_trend: Spread slope (positive = widening)
            belief_std: Belief uncertainty

        Returns:
            Score 0-100
        """
        score = 50.0  # Start neutral

        # Time component
        if time_to_resolution_days is not None:
            if time_to_resolution_days < 0:
                score -= 30  # Past due
            elif time_to_resolution_days < 7:
                score += 25  # Very short term
            elif time_to_resolution_days < 30:
                score += 15  # Short term
            elif time_to_resolution_days > 180:
                score -= 15  # Too far out

        # Belief uncertainty component
        if belief_std is not None:
            if belief_std < 0.05:
                score += 20  # Low uncertainty
            elif belief_std < 0.1:
                score += 10  # Moderate
            elif belief_std > 0.2:
                score -= 15  # High uncertainty

        # Volatility component
        if price_volatility is not None:
            if price_volatility < 0.02:
                score += 10  # Stable
            elif price_volatility > 0.1:
                score -= 20  # Too volatile

        # Spread trend component
        if spread_trend is not None:
            if spread_trend > 0.0001:
                score -= 15  # Widening spreads
            elif spread_trend < -0.0001:
                score += 10  # Tightening spreads

        return max(0.0, min(100.0, score))

    def compute_combined_score(
        self,
        execution_quality: float,
        rules_risk: float,
        constraint_edge: float,
        regime_opportunity: float,
    ) -> float:
        """Compute weighted combined score.

        Formula:
        combined = w_ex * execution_quality
                 + w_ru * (100 - rules_risk)
                 + w_co * constraint_edge
                 + w_re * regime_opportunity

        Args:
            execution_quality: Execution quality score (0-100)
            rules_risk: Rules risk score (0-100, higher is worse)
            constraint_edge: Constraint edge score (0-100)
            regime_opportunity: Regime opportunity score (0-100)

        Returns:
            Combined score 0-100
        """
        # Invert rules_risk so higher is better
        rules_clarity = 100.0 - rules_risk

        combined = (
            self.weight_execution * execution_quality
            + self.weight_rules * rules_clarity
            + self.weight_constraint * constraint_edge
            + self.weight_regime * regime_opportunity
        )

        return max(0.0, min(100.0, combined))

    def score_outcome(
        self,
        execution_metrics: dict[str, Any] | None,
        ambiguity_score: float | None,
        unfalsifiable_flag: bool,
        complete_set_buy_cost: float | None,
        regime_features: dict[str, float | None] | None,
    ) -> ScoreComponents:
        """Score an outcome with all components.

        Args:
            execution_metrics: Dict with effective_spread, fill_ratio
            ambiguity_score: Ambiguity from LLM (0-1)
            unfalsifiable_flag: Whether market is unfalsifiable
            complete_set_buy_cost: Cost to buy complete set
            regime_features: Dict with time_to_resolution_days, price_volatility, etc.

        Returns:
            ScoreComponents with all score values
        """
        # Extract execution metrics
        effective_spread = None
        fill_ratio = None
        if execution_metrics:
            effective_spread = execution_metrics.get("effective_spread")
            fill_ratio = execution_metrics.get("fill_ratio")

        # Compute execution quality
        exec_quality = self.compute_execution_quality_score(effective_spread, fill_ratio)

        # Compute rules risk
        rules_risk = self.compute_rules_risk_score(ambiguity_score, unfalsifiable_flag)

        # Compute constraint edge
        constraint = self.compute_constraint_edge_score(complete_set_buy_cost)

        # Compute regime opportunity (default to 0 if no features - cannot assess opportunity)
        regime = 0.0
        if regime_features:
            regime = self.compute_regime_opportunity_score(
                time_to_resolution_days=regime_features.get("time_to_resolution_days"),
                price_volatility=regime_features.get("price_volatility_short"),
                spread_trend=regime_features.get("spread_trend_short"),
                belief_std=regime_features.get("belief_std"),
            )

        # Compute combined
        combined = self.compute_combined_score(exec_quality, rules_risk, constraint, regime)

        return ScoreComponents(
            execution_quality=exec_quality,
            rules_risk=rules_risk,
            constraint_edge=constraint,
            regime_opportunity=regime,
            combined=combined,
        )
