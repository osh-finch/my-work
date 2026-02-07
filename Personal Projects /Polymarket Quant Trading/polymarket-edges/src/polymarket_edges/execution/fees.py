"""Fee calculation module for execution costs."""

import logging
from dataclasses import dataclass

from polymarket_edges.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FeeConfig:
    """Fee configuration with basis points."""

    taker_fee_bps: float = 0.0
    maker_rebate_bps: float = 0.0

    def taker_fee_rate(self) -> float:
        """Convert taker fee from bps to rate."""
        return self.taker_fee_bps / 10000.0

    def maker_rebate_rate(self) -> float:
        """Convert maker rebate from bps to rate."""
        return self.maker_rebate_bps / 10000.0


class FeeCalculator:
    """Calculate fees for trades based on configuration."""

    def __init__(self, config: FeeConfig | None = None):
        """Initialise fee calculator.

        Args:
            config: Fee configuration (defaults to global settings)
        """
        if config is None:
            config = FeeConfig(
                taker_fee_bps=settings.taker_fee_bps,
                maker_rebate_bps=settings.maker_rebate_bps,
            )
        self.config = config

    def apply_taker_fee(self, notional: float) -> float:
        """Apply taker fee to notional amount.

        Args:
            notional: Trade notional in USD

        Returns:
            Fee amount (positive = cost)
        """
        return notional * self.config.taker_fee_rate()

    def apply_maker_rebate(self, notional: float) -> float:
        """Apply maker rebate to notional amount.

        Args:
            notional: Trade notional in USD

        Returns:
            Rebate amount (positive = earnings)
        """
        return notional * self.config.maker_rebate_rate()

    def effective_entry_price(self, price: float, is_buy: bool) -> float:
        """Adjust price for taker fees on entry.

        Args:
            price: Raw execution price
            is_buy: True if buying YES

        Returns:
            Effective price after fees
        """
        if is_buy:
            # Buying: pay more due to fees
            return price * (1.0 + self.config.taker_fee_rate())
        else:
            # Selling: receive less due to fees
            return price * (1.0 - self.config.taker_fee_rate())

    def effective_exit_price(self, price: float, is_sell: bool) -> float:
        """Adjust price for taker fees on exit.

        Args:
            price: Raw execution price
            is_sell: True if selling YES

        Returns:
            Effective price after fees
        """
        if is_sell:
            # Selling: receive less due to fees
            return price * (1.0 - self.config.taker_fee_rate())
        else:
            # Buying back: pay more due to fees
            return price * (1.0 + self.config.taker_fee_rate())

    def get_summary(self) -> dict[str, float]:
        """Get fee configuration summary.

        Returns:
            Dictionary with fee parameters
        """
        return {
            "taker_fee_bps": self.config.taker_fee_bps,
            "maker_rebate_bps": self.config.maker_rebate_bps,
            "taker_fee_rate": self.config.taker_fee_rate(),
            "maker_rebate_rate": self.config.maker_rebate_rate(),
        }
