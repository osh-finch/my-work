"""Regime and lifecycle feature extraction for markets."""

import logging
from datetime import datetime, timedelta

import pandas as pd
from scipy import stats

from polymarket_edges.config import settings

logger = logging.getLogger(__name__)


class RegimeFeatureExtractor:
    """Extract time series features for regime detection."""

    def __init__(
        self,
        short_window_hours: int | None = None,
        long_window_hours: int | None = None,
    ):
        """Initialise regime feature extractor.

        Args:
            short_window_hours: Short time window (defaults to config)
            long_window_hours: Long time window (defaults to config)
        """
        self.short_window = timedelta(
            hours=short_window_hours or settings.regime_window_hours_short
        )
        self.long_window = timedelta(
            hours=long_window_hours or settings.regime_window_hours_long
        )

    def compute_market_age(
        self,
        first_seen: datetime,
        current_time: datetime,
    ) -> float:
        """Compute market age in days.

        Args:
            first_seen: When market was first ingested
            current_time: Current timestamp

        Returns:
            Age in days
        """
        age = current_time - first_seen
        return age.total_seconds() / 86400.0

    def compute_time_to_resolution(
        self,
        end_time: datetime,
        current_time: datetime,
    ) -> float:
        """Compute time until market resolution in days.

        Args:
            end_time: Market end date
            current_time: Current timestamp

        Returns:
            Days until resolution (negative if past due)
        """
        delta = end_time - current_time
        return delta.total_seconds() / 86400.0

    def compute_spread_trend(
        self,
        spreads: pd.Series,
        timestamps: pd.Series,
    ) -> float | None:
        """Compute spread trend (slope) over recent history.

        Args:
            spreads: Series of spread values
            timestamps: Series of timestamps

        Returns:
            Slope of spread trend (positive = widening)
        """
        if len(spreads) < 2:
            return None

        # Convert timestamps to numeric (seconds since first)
        ts_numeric = (timestamps - timestamps.min()).dt.total_seconds()

        # Remove NaN values
        valid_mask = spreads.notna() & ts_numeric.notna()
        if valid_mask.sum() < 2:
            return None

        spreads_clean = spreads[valid_mask].values
        ts_clean = ts_numeric[valid_mask].values

        # Linear regression
        try:
            slope, _, _, _, _ = stats.linregress(ts_clean, spreads_clean)
            return float(slope)
        except Exception as e:
            logger.warning(f"Failed to compute spread trend: {e}")
            return None

    def compute_price_volatility(
        self,
        mid_prices: pd.Series,
    ) -> float | None:
        """Compute volatility of mid price over recent history.

        Args:
            mid_prices: Series of mid price values

        Returns:
            Standard deviation of mid prices
        """
        if len(mid_prices) < 2:
            return None

        # Remove NaN
        mid_prices_clean = mid_prices.dropna()
        if len(mid_prices_clean) < 2:
            return None

        return float(mid_prices_clean.std())

    def compute_liquidity_depth(
        self,
        bid_sizes: pd.Series,
        ask_sizes: pd.Series,
        prices: pd.Series,
        depth_percent: float = 0.02,
    ) -> float | None:
        """Compute liquidity depth within X% of mid price.

        Args:
            bid_sizes: Bid sizes at best
            ask_sizes: Ask sizes at best
            prices: Mid prices
            depth_percent: Percentage band around mid

        Returns:
            Average depth (sum of bid and ask sizes)
        """
        if len(bid_sizes) == 0 or len(ask_sizes) == 0:
            return None

        # Simple proxy: average of bid and ask sizes
        total_depth = bid_sizes.fillna(0) + ask_sizes.fillna(0)
        avg_depth = total_depth.mean()

        return float(avg_depth) if not pd.isna(avg_depth) else None

    def extract_features(
        self,
        outcome_id: str,
        quotes_history: pd.DataFrame,
        market_first_seen: datetime,
        market_end_time: datetime | None,
        current_time: datetime,
    ) -> dict[str, float | None]:
        """Extract all regime features for an outcome.

        Args:
            outcome_id: Outcome token ID
            quotes_history: DataFrame with columns [timestamp, best_bid, best_ask, bid_size, ask_size]
            market_first_seen: When market was first seen
            market_end_time: When market expires
            current_time: Current timestamp

        Returns:
            Dictionary of feature name to value
        """
        features = {}

        # Market age and time-to-resolution
        features["market_age_days"] = self.compute_market_age(market_first_seen, current_time)

        if market_end_time:
            features["time_to_resolution_days"] = self.compute_time_to_resolution(
                market_end_time, current_time
            )
        else:
            features["time_to_resolution_days"] = None

        # Filter history to short and long windows
        short_cutoff = current_time - self.short_window
        long_cutoff = current_time - self.long_window

        short_history = quotes_history[quotes_history["timestamp"] >= short_cutoff]
        long_history = quotes_history[quotes_history["timestamp"] >= long_cutoff]

        # Compute mid and spread
        for df, window_name in [(short_history, "short"), (long_history, "long")]:
            if len(df) == 0:
                features[f"spread_trend_{window_name}"] = None
                features[f"price_volatility_{window_name}"] = None
                features[f"liquidity_depth_{window_name}"] = None
                continue

            mid = (df["best_bid"].fillna(0) + df["best_ask"].fillna(0)) / 2
            spread = df["best_ask"] - df["best_bid"]

            features[f"spread_trend_{window_name}"] = self.compute_spread_trend(
                spread, df["timestamp"]
            )
            features[f"price_volatility_{window_name}"] = self.compute_price_volatility(mid)
            features[f"liquidity_depth_{window_name}"] = self.compute_liquidity_depth(
                df["bid_size"], df["ask_size"], mid
            )

        return features

    def score_regime_opportunity(
        self,
        time_to_resolution_days: float | None,
        price_volatility: float | None,
        spread_trend: float | None,
        belief_std: float | None,
    ) -> float:
        """Score regime opportunity from 0-100.

        Higher scores when:
        - Time to resolution is short (< 30 days)
        - Belief uncertainty is low
        - Volatility is moderate (not exploding)
        - Spreads are not widening rapidly

        Args:
            time_to_resolution_days: Days until resolution
            price_volatility: Price volatility
            spread_trend: Spread slope
            belief_std: Belief standard deviation

        Returns:
            Score 0-100
        """
        score = 50.0  # Start neutral

        # Time to resolution component
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

        # Clamp to 0-100
        return max(0.0, min(100.0, score))
