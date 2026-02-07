"""Configuration management for Polymarket Edges v2."""

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")

    # Database
    database_url: str = Field(
        default="data/polymarket.duckdb",
        description="Database connection string or file path",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    # Rate limiting (requests per second)
    gamma_rate_limit: float = Field(default=10.0, description="Gamma API rate limit (req/s)")
    clob_rate_limit: float = Field(default=10.0, description="CLOB API rate limit (req/s)")

    # API endpoints
    gamma_base_url: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Gamma Markets API base URL",
    )
    clob_base_url: str = Field(
        default="https://clob.polymarket.com",
        description="CLOB API base URL",
    )

    # Cache settings
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")

    # v2: Execution simulation parameters
    orderbook_depth_levels: int = Field(
        default=30, description="Number of order book levels to capture per side"
    )
    trade_size_buckets: list[float] = Field(
        default=[25.0, 100.0, 250.0, 1000.0],
        description="Trade sizes in USD notional for execution simulation",
    )
    liquidity_spread_threshold: float = Field(
        default=0.9, description="Maximum spread to consider a market liquid"
    )
    reference_size_bucket: float = Field(
        default=100.0, description="Reference trade size for scoring"
    )

    # v2: Fee model (basis points)
    taker_fee_bps: float = Field(default=0.0, description="Taker fee in basis points")
    maker_rebate_bps: float = Field(default=0.0, description="Maker rebate in basis points")

    # v2: Constraint detection
    constraint_confidence_threshold: float = Field(
        default=0.8, description="Minimum confidence for cross-market link acceptance"
    )
    complete_set_threshold: float = Field(
        default=0.005, description="Minimum deviation (0.5%) to flag complete set violations"
    )

    # v2: Regime features
    regime_window_hours_short: int = Field(
        default=24, description="Short time window for regime features (hours)"
    )
    regime_window_hours_long: int = Field(
        default=168, description="Long time window for regime features (hours, 7 days)"
    )

    # v2: Belief filter
    belief_process_variance: float = Field(
        default=0.0001, description="Random walk variance for belief state-space model"
    )
    belief_min_liquidity: float = Field(
        default=100.0, description="Minimum liquidity for reliable belief estimation"
    )

    # v2: Scoring weights
    score_weight_execution: float = Field(default=0.45, description="Weight for execution quality")
    score_weight_rules: float = Field(default=0.25, description="Weight for rules clarity")
    score_weight_constraint: float = Field(default=0.20, description="Weight for constraint edge")
    score_weight_regime: float = Field(default=0.10, description="Weight for regime opportunity")

    @property
    def database_path(self) -> Path:
        """Return database path as Path object."""
        return Path(self.database_url)

    def setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# Global settings instance
settings = Settings()
