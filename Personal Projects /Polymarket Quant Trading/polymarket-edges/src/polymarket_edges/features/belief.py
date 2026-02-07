"""Bayesian belief filter for latent probability estimation."""

import logging
from dataclasses import dataclass

import numpy as np

from polymarket_edges.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BeliefState:
    """State estimate from belief filter."""

    mean: float  # Posterior mean
    variance: float  # Posterior variance
    timestamp: float  # Time of estimate


class BeliefFilter:
    """Kalman-style filter for latent belief estimation.

    Models latent belief as random walk with noisy observations.
    Observation noise inversely related to liquidity.
    """

    def __init__(
        self,
        process_variance: float | None = None,
        min_liquidity: float | None = None,
    ):
        """Initialise belief filter.

        Args:
            process_variance: Variance of belief random walk
            min_liquidity: Minimum liquidity for reliable estimates
        """
        self.process_variance = process_variance or settings.belief_process_variance
        self.min_liquidity = min_liquidity or settings.belief_min_liquidity

    def observation_variance(self, liquidity: float) -> float:
        """Compute observation variance based on liquidity.

        Lower liquidity -> higher observation noise

        Args:
            liquidity: Market liquidity proxy (depth or inverse spread)

        Returns:
            Observation variance
        """
        if liquidity <= 0:
            return 1.0  # Maximum uncertainty

        # Inverse relationship: var ~ 1 / liquidity
        # Scale so that reference liquidity gives reasonable variance
        reference_liquidity = 1000.0
        reference_variance = 0.001

        variance = reference_variance * (reference_liquidity / liquidity)
        return min(variance, 1.0)  # Cap at 1.0

    def predict(self, state: BeliefState) -> BeliefState:
        """Prediction step: advance belief state.

        Args:
            state: Current belief state

        Returns:
            Predicted belief state
        """
        # Random walk: mean stays same, variance increases
        return BeliefState(
            mean=state.mean,
            variance=state.variance + self.process_variance,
            timestamp=state.timestamp,
        )

    def update(
        self,
        prior: BeliefState,
        observation: float,
        liquidity: float,
    ) -> BeliefState:
        """Update step: incorporate new observation.

        Args:
            prior: Prior belief state
            observation: Observed mid price
            liquidity: Liquidity proxy

        Returns:
            Updated belief state
        """
        obs_var = self.observation_variance(liquidity)

        # Kalman gain
        kalman_gain = prior.variance / (prior.variance + obs_var)

        # Update mean and variance
        posterior_mean = prior.mean + kalman_gain * (observation - prior.mean)
        posterior_variance = (1 - kalman_gain) * prior.variance

        return BeliefState(
            mean=posterior_mean,
            variance=posterior_variance,
            timestamp=prior.timestamp,
        )

    def filter_sequence(
        self,
        observations: list[tuple[float, float, float]],  # (timestamp, mid_price, liquidity)
        initial_mean: float | None = None,
        initial_variance: float = 0.01,
    ) -> list[BeliefState]:
        """Filter a sequence of observations.

        Args:
            observations: List of (timestamp, mid_price, liquidity) tuples
            initial_mean: Initial belief mean (defaults to first observation)
            initial_variance: Initial belief variance

        Returns:
            List of belief states
        """
        if not observations:
            return []

        # Sort by timestamp
        observations = sorted(observations, key=lambda x: x[0])

        # Initialise
        first_ts, first_mid, _ = observations[0]
        if initial_mean is None:
            initial_mean = first_mid

        state = BeliefState(mean=initial_mean, variance=initial_variance, timestamp=first_ts)
        states = [state]

        # Process each observation
        for ts, mid, liq in observations[1:]:
            # Predict
            state = self.predict(state)

            # Update
            state = self.update(state, mid, liq)
            state.timestamp = ts

            states.append(state)

        return states

    def get_latest_estimate(
        self,
        observations: list[tuple[float, float, float]],
    ) -> tuple[float, float] | None:
        """Get latest belief estimate from observations.

        Args:
            observations: List of (timestamp, mid_price, liquidity) tuples

        Returns:
            Tuple of (posterior_mean, posterior_std) or None
        """
        if not observations:
            return None

        states = self.filter_sequence(observations)
        if not states:
            return None

        latest = states[-1]
        posterior_std = np.sqrt(latest.variance)

        return latest.mean, posterior_std

    def should_use_belief(self, liquidity: float) -> bool:
        """Check if liquidity is sufficient for reliable belief estimation.

        Args:
            liquidity: Market liquidity proxy

        Returns:
            True if liquidity is sufficient
        """
        return liquidity >= self.min_liquidity
