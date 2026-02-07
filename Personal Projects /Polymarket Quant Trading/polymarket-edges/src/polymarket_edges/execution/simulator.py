"""Order book simulator for execution analysis."""

import logging
from dataclasses import dataclass
from typing import Any

from polymarket_edges.execution.fees import FeeCalculator

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of order book execution simulation."""

    vwap: float | None  # Volume-weighted average price
    filled_quantity: float  # Amount filled
    requested_quantity: float  # Amount requested
    fill_ratio: float  # filled / requested
    total_cost: float | None  # Total cost including fees (for buy)
    total_proceeds: float | None  # Total proceeds after fees (for sell)
    slippage: float | None  # Difference from best price
    levels_consumed: int  # Number of order book levels used


class OrderBookSimulator:
    """Simulates trade execution through order book levels."""

    def __init__(self, fee_calculator: FeeCalculator | None = None):
        """Initialise order book simulator.

        Args:
            fee_calculator: Fee calculator instance (defaults to new instance)
        """
        self.fee_calculator = fee_calculator or FeeCalculator()

    def simulate_buy_yes(
        self,
        asks: list[dict[str, Any]],
        target_notional: float,
    ) -> SimulationResult:
        """Simulate buying YES tokens (taking asks).

        Args:
            asks: List of ask levels [{"price": str, "size": str}, ...]
            target_notional: Target USD notional to spend

        Returns:
            SimulationResult with execution details
        """
        if not asks:
            return SimulationResult(
                vwap=None,
                filled_quantity=0.0,
                requested_quantity=target_notional,
                fill_ratio=0.0,
                total_cost=None,
                total_proceeds=None,
                slippage=None,
                levels_consumed=0,
            )

        best_price = float(asks[0]["price"])
        total_quantity = 0.0
        total_cost_raw = 0.0
        levels_consumed = 0

        remaining_notional = target_notional

        for level in asks:
            if remaining_notional <= 0:
                break

            price = float(level["price"])
            size = float(level["size"])

            # Calculate how much we can buy at this level
            max_notional_at_level = price * size
            notional_to_use = min(remaining_notional, max_notional_at_level)
            quantity_to_buy = notional_to_use / price

            total_quantity += quantity_to_buy
            total_cost_raw += notional_to_use
            remaining_notional -= notional_to_use
            levels_consumed += 1

        if total_quantity == 0:
            return SimulationResult(
                vwap=None,
                filled_quantity=0.0,
                requested_quantity=target_notional,
                fill_ratio=0.0,
                total_cost=None,
                total_proceeds=None,
                slippage=None,
                levels_consumed=0,
            )

        vwap = total_cost_raw / total_quantity
        fill_ratio = (target_notional - remaining_notional) / target_notional

        # Apply fees
        total_cost_with_fees = total_cost_raw + self.fee_calculator.apply_taker_fee(
            total_cost_raw
        )
        slippage = vwap - best_price

        return SimulationResult(
            vwap=vwap,
            filled_quantity=total_quantity,
            requested_quantity=target_notional / vwap if vwap else target_notional,
            fill_ratio=fill_ratio,
            total_cost=total_cost_with_fees,
            total_proceeds=None,
            slippage=slippage,
            levels_consumed=levels_consumed,
        )

    def simulate_sell_yes(
        self,
        bids: list[dict[str, Any]],
        target_quantity: float,
    ) -> SimulationResult:
        """Simulate selling YES tokens (hitting bids).

        Args:
            bids: List of bid levels [{"price": str, "size": str}, ...]
            target_quantity: Target quantity to sell

        Returns:
            SimulationResult with execution details
        """
        if not bids:
            return SimulationResult(
                vwap=None,
                filled_quantity=0.0,
                requested_quantity=target_quantity,
                fill_ratio=0.0,
                total_cost=None,
                total_proceeds=None,
                slippage=None,
                levels_consumed=0,
            )

        best_price = float(bids[0]["price"])
        total_quantity = 0.0
        total_proceeds_raw = 0.0
        levels_consumed = 0

        remaining_quantity = target_quantity

        for level in bids:
            if remaining_quantity <= 0:
                break

            price = float(level["price"])
            size = float(level["size"])

            quantity_to_sell = min(remaining_quantity, size)

            total_quantity += quantity_to_sell
            total_proceeds_raw += quantity_to_sell * price
            remaining_quantity -= quantity_to_sell
            levels_consumed += 1

        if total_quantity == 0:
            return SimulationResult(
                vwap=None,
                filled_quantity=0.0,
                requested_quantity=target_quantity,
                fill_ratio=0.0,
                total_cost=None,
                total_proceeds=None,
                slippage=None,
                levels_consumed=0,
            )

        vwap = total_proceeds_raw / total_quantity
        fill_ratio = total_quantity / target_quantity

        # Apply fees
        total_proceeds_with_fees = total_proceeds_raw - self.fee_calculator.apply_taker_fee(
            total_proceeds_raw
        )
        slippage = best_price - vwap  # Negative slippage = worse price

        return SimulationResult(
            vwap=vwap,
            filled_quantity=total_quantity,
            requested_quantity=target_quantity,
            fill_ratio=fill_ratio,
            total_cost=None,
            total_proceeds=total_proceeds_with_fees,
            slippage=slippage,
            levels_consumed=levels_consumed,
        )

    def vwap(
        self,
        side: str,
        levels: list[dict[str, Any]],
        target_size: float,
    ) -> float | None:
        """Generic VWAP calculation for any side.

        Args:
            side: 'bid' or 'ask'
            levels: Order book levels
            target_size: Target size to execute

        Returns:
            VWAP price or None if cannot execute
        """
        if side == "ask":
            result = self.simulate_buy_yes(levels, target_size)
        elif side == "bid":
            # For bids, target_size is quantity not notional
            # Estimate quantity from notional assuming mid-level pricing
            if levels:
                avg_price = sum(float(level["price"]) for level in levels[:3]) / min(3, len(levels))
                estimated_quantity = target_size / avg_price if avg_price > 0 else target_size
            else:
                estimated_quantity = target_size
            result = self.simulate_sell_yes(levels, estimated_quantity)
        else:
            raise ValueError(f"Invalid side: {side}")

        return result.vwap

    def compute_liquidity_tax(
        self,
        bids: list[dict[str, Any]],
        asks: list[dict[str, Any]],
        target_notional: float,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute liquidity tax: difference between entry and exit VWAP.

        Args:
            bids: Bid levels
            asks: Ask levels
            target_notional: Trade size in USD notional

        Returns:
            Tuple of (entry_vwap, exit_vwap, liquidity_tax)
        """
        # Entry: buy YES at asks
        entry_result = self.simulate_buy_yes(asks, target_notional)
        entry_vwap = entry_result.vwap

        if entry_vwap is None:
            return None, None, None

        # Exit: sell YES at bids
        # Convert notional to quantity using entry VWAP
        quantity_bought = target_notional / entry_vwap
        exit_result = self.simulate_sell_yes(bids, quantity_bought)
        exit_vwap = exit_result.vwap

        if exit_vwap is None:
            return entry_vwap, None, None

        # Liquidity tax = entry - exit
        liquidity_tax = entry_vwap - exit_vwap

        return entry_vwap, exit_vwap, liquidity_tax

    def effective_spread(
        self,
        bids: list[dict[str, Any]],
        asks: list[dict[str, Any]],
        target_notional: float,
    ) -> float | None:
        """Compute effective spread including execution costs.

        Args:
            bids: Bid levels
            asks: Ask levels
            target_notional: Trade size

        Returns:
            Effective spread or None
        """
        entry_vwap, exit_vwap, _ = self.compute_liquidity_tax(bids, asks, target_notional)

        if entry_vwap is None or exit_vwap is None:
            return None

        return entry_vwap - exit_vwap
