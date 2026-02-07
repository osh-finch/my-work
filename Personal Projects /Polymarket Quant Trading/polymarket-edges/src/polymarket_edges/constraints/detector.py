"""Constraint violation detection for complete set and cross-market consistency."""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from polymarket_edges.config import settings
from polymarket_edges.execution.simulator import OrderBookSimulator

logger = logging.getLogger(__name__)


@dataclass
class ViolationResult:
    """Result of constraint violation detection."""

    violation_type: str
    magnitude: float
    evidence: dict[str, Any]
    condition_id: str
    size_bucket: float


class ConstraintDetector:
    """Detects constraint violations and arbitrage opportunities."""

    def __init__(self, simulator: OrderBookSimulator | None = None):
        """Initialise constraint detector.

        Args:
            simulator: Order book simulator instance
        """
        self.simulator = simulator or OrderBookSimulator()
        self.threshold = settings.complete_set_threshold

    def check_complete_set(
        self,
        yes_bids: list[dict[str, Any]],
        yes_asks: list[dict[str, Any]],
        no_bids: list[dict[str, Any]],
        no_asks: list[dict[str, Any]],
        size_bucket: float,
        condition_id: str,
    ) -> list[ViolationResult]:
        """Check for complete set pricing violations.

        In binary markets, buying YES + NO should cost ~$1.00 (complete set).
        Selling YES + NO should yield ~$1.00.

        Args:
            yes_bids: YES bid levels
            yes_asks: YES ask levels
            no_bids: NO bid levels
            no_asks: NO ask levels
            size_bucket: Trade size to simulate
            condition_id: Market condition ID

        Returns:
            List of detected violations
        """
        violations = []

        # Check underpriced complete set (buy opportunity)
        yes_buy = self.simulator.simulate_buy_yes(yes_asks, size_bucket)
        no_buy = self.simulator.simulate_buy_yes(no_asks, size_bucket)

        if yes_buy.vwap is not None and no_buy.vwap is not None:
            complete_set_buy_cost = yes_buy.vwap + no_buy.vwap

            if complete_set_buy_cost < (1.0 - self.threshold):
                magnitude = 1.0 - complete_set_buy_cost
                violations.append(
                    ViolationResult(
                        violation_type="complete_set_under_1",
                        magnitude=magnitude,
                        evidence={
                            "yes_vwap": yes_buy.vwap,
                            "no_vwap": no_buy.vwap,
                            "total_cost": complete_set_buy_cost,
                            "yes_fill_ratio": yes_buy.fill_ratio,
                            "no_fill_ratio": no_buy.fill_ratio,
                        },
                        condition_id=condition_id,
                        size_bucket=size_bucket,
                    )
                )
                logger.info(
                    f"Complete set underpriced at {condition_id}: "
                    f"{complete_set_buy_cost:.4f} (edge: {magnitude:.4f})"
                )

        # Check overpriced complete set (sell opportunity)
        # Estimate quantity from size bucket
        if yes_buy.vwap:
            quantity = size_bucket / yes_buy.vwap
        elif yes_asks:
            quantity = size_bucket / float(yes_asks[0]["price"])
        else:
            quantity = size_bucket

        yes_sell = self.simulator.simulate_sell_yes(yes_bids, quantity)
        no_sell = self.simulator.simulate_sell_yes(no_bids, quantity)

        if yes_sell.vwap is not None and no_sell.vwap is not None:
            complete_set_sell_proceeds = yes_sell.vwap + no_sell.vwap

            if complete_set_sell_proceeds > (1.0 + self.threshold):
                magnitude = complete_set_sell_proceeds - 1.0
                violations.append(
                    ViolationResult(
                        violation_type="complete_set_over_1",
                        magnitude=magnitude,
                        evidence={
                            "yes_vwap": yes_sell.vwap,
                            "no_vwap": no_sell.vwap,
                            "total_proceeds": complete_set_sell_proceeds,
                            "yes_fill_ratio": yes_sell.fill_ratio,
                            "no_fill_ratio": no_sell.fill_ratio,
                        },
                        condition_id=condition_id,
                        size_bucket=size_bucket,
                    )
                )
                logger.info(
                    f"Complete set overpriced at {condition_id}: "
                    f"{complete_set_sell_proceeds:.4f} (edge: {magnitude:.4f})"
                )

        return violations

    def check_sum_probabilities(
        self,
        outcomes: list[tuple[str, float]],  # (outcome_id, mid_price)
        condition_id: str,
    ) -> list[ViolationResult]:
        """Check if sum of probabilities for mutually exclusive outcomes exceeds 1.

        Args:
            outcomes: List of (outcome_id, mid_price) tuples
            condition_id: Market condition ID

        Returns:
            List of violations
        """
        violations = []

        if len(outcomes) < 2:
            return violations

        total_prob = sum(price for _, price in outcomes)

        # For mutually exclusive outcomes, sum should be ≤ 1.0
        if total_prob > (1.0 + self.threshold):
            magnitude = total_prob - 1.0
            violations.append(
                ViolationResult(
                    violation_type="sum_prob_violation",
                    magnitude=magnitude,
                    evidence={
                        "total_probability": total_prob,
                        "outcomes": [
                            {"outcome_id": oid, "price": price} for oid, price in outcomes
                        ],
                    },
                    condition_id=condition_id,
                    size_bucket=0.0,  # Not size-dependent
                )
            )
            logger.info(
                f"Sum probability violation at {condition_id}: "
                f"{total_prob:.4f} (excess: {magnitude:.4f})"
            )

        return violations

    def check_inequality_constraint(
        self,
        market_a_price: float,
        market_b_price: float,
        market_a_id: str,
        market_b_id: str,
        link_type: str,
    ) -> ViolationResult | None:
        """Check if logical constraint is violated between two markets.

        Args:
            market_a_price: Price of market A
            market_b_price: Price of market B
            market_a_id: Condition ID of market A
            market_b_id: Condition ID of market B
            link_type: Type of link ('implies', 'exclusive', 'same_event')

        Returns:
            Violation if detected, else None
        """
        if link_type == "implies":
            # If A implies B, then P(A) <= P(B)
            if market_a_price > market_b_price + self.threshold:
                magnitude = market_a_price - market_b_price
                return ViolationResult(
                    violation_type="inequality_violation",
                    magnitude=magnitude,
                    evidence={
                        "market_a": market_a_id,
                        "market_b": market_b_id,
                        "market_a_price": market_a_price,
                        "market_b_price": market_b_price,
                        "link_type": link_type,
                    },
                    condition_id=market_a_id,
                    size_bucket=0.0,
                )

        return None

    def generate_violation_id(self) -> str:
        """Generate unique violation ID."""
        return str(uuid.uuid4())
