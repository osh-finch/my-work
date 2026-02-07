"""Pydantic models for Polymarket data structures v2."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# API Response Models
# ============================================================================


class GammaMarket(BaseModel):
    """Market data from Gamma API."""

    model_config = {"populate_by_name": True}

    condition_id: str | None = Field(default=None, alias="conditionId")
    id: str | None = None  # Some markets use 'id' instead
    question: str
    description: str | None = None
    market_slug: str | None = Field(default=None, alias="marketSlug")
    end_date_iso: str | None = Field(default=None, alias="endDateIso")
    game_start_time: str | None = Field(default=None, alias="gameStartTime")
    question_id: str | None = Field(default=None, alias="questionID")
    tokens: list[dict[str, Any]] = Field(default_factory=list)
    outcomes: list[str] | None = None  # ["Yes", "No"]
    clob_token_ids: str | None = Field(default=None, alias="clobTokenIds")  # JSON string of token IDs
    active: bool = True
    closed: bool = False
    archived: bool = False
    rewards_min_size: float | None = Field(default=None, alias="rewardsMinSize")
    rewards_max_spread: float | None = Field(default=None, alias="rewardsMaxSpread")
    enable_order_book: bool = Field(default=True, alias="enableOrderBook")

    @field_validator("outcomes", mode="before")
    @classmethod
    def parse_outcomes(cls, v):
        """Parse outcomes if it's a JSON string."""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return v

    @model_validator(mode="after")
    def ensure_condition_id(self) -> "GammaMarket":
        """Ensure condition_id is set, using id if necessary."""
        if not self.condition_id and self.id:
            self.condition_id = self.id
        if not self.condition_id:
            raise ValueError("Either condition_id or id must be provided")
        return self


class CLOBOrderBookSummary(BaseModel):
    """Order book snapshot from CLOB API with depth."""

    market: str  # Market condition ID
    asset_id: str  # Token ID
    timestamp: int
    hash: str | None = None
    bids: list[dict[str, Any]] = Field(default_factory=list)  # [{"price": "0.5", "size": "100"}]
    asks: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def best_bid(self) -> float | None:
        """Extract best bid price."""
        if self.bids:
            return float(self.bids[0].get("price", 0))
        return None

    @property
    def best_ask(self) -> float | None:
        """Extract best ask price."""
        if self.asks:
            return float(self.asks[0].get("price", 0))
        return None

    @property
    def bid_size(self) -> float | None:
        """Extract best bid size."""
        if self.bids:
            return float(self.bids[0].get("size", 0))
        return None

    @property
    def ask_size(self) -> float | None:
        """Extract best ask size."""
        if self.asks:
            return float(self.asks[0].get("size", 0))
        return None


# ============================================================================
# Normalised Database Models
# ============================================================================


class Market(BaseModel):
    """Normalised market record."""

    condition_id: str
    question: str
    description: str | None = None
    market_slug: str | None = None
    end_date_iso: str | None = None
    active: bool = True
    closed: bool = False
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class Outcome(BaseModel):
    """Normalised outcome (token) record."""

    token_id: str
    condition_id: str
    outcome: str  # "Yes" or "No" typically
    winner: bool = False


class Quote(BaseModel):
    """Normalised quote record (top-of-book)."""

    token_id: str
    condition_id: str
    timestamp: datetime
    best_bid: float | None = None
    best_ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None


class OrderBookSnapshot(BaseModel):
    """v2: Full order book snapshot with depth."""

    snapshot_id: str  # UUID for this snapshot
    token_id: str
    condition_id: str
    timestamp: datetime
    raw_data: dict[str, Any]  # Store raw JSON for audit


class OrderBookLevel(BaseModel):
    """v2: Individual order book level."""

    snapshot_id: str
    side: str  # 'bid' or 'ask'
    level_index: int  # 0 = best, 1 = second best, etc.
    price: float
    size: float


class ExecutionMetric(BaseModel):
    """v2: Execution simulation results for a given size."""

    outcome_id: str  # token_id
    snapshot_id: str
    size_bucket: float
    entry_vwap: float | None = None
    exit_vwap: float | None = None
    liquidity_tax: float | None = None
    fill_ratio: float = 0.0  # Fraction of order filled
    effective_spread: float | None = None


class RulesStructured(BaseModel):
    """Structured rules extraction from LLM."""

    model_config = {"protected_namespaces": ()}

    condition_id: str
    resolution_source: str
    yes_conditions: list[str] = Field(default_factory=list)
    no_conditions: list[str] = Field(default_factory=list)
    key_dates: list[str] = Field(default_factory=list)  # ISO date strings
    ambiguity_score: float = Field(ge=0.0, le=1.0)
    ambiguity_reasons: list[str] = Field(default_factory=list)
    unfalsifiable_flag: bool = False
    # v2: Additional fields
    edge_cases: list[str] = Field(default_factory=list)
    dispute_risk_notes: list[str] = Field(default_factory=list)
    recommended_evidence_to_monitor: list[str] = Field(default_factory=list)
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str | None = None


class ConstraintViolation(BaseModel):
    """v2: Detected constraint violation (arbitrage opportunity)."""

    violation_id: str
    condition_id: str
    violation_type: str  # complete_set_under_1, complete_set_over_1, sum_prob_violation, inequality_violation
    size_bucket: float
    magnitude: float  # Size of violation
    evidence: dict[str, Any]  # Prices and details
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class MarketLink(BaseModel):
    """v2: Proposed logical link between markets."""

    link_id: str
    market_a: str  # condition_id
    market_b: str  # condition_id
    link_type: str  # implies, exclusive, same_event
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TimeSeriesFeature(BaseModel):
    """v2: Regime and lifecycle features."""

    outcome_id: str
    asof_ts: datetime
    feature_name: str
    value: float


class BeliefEstimate(BaseModel):
    """v2: Bayesian filtered belief estimate."""

    outcome_id: str
    asof_ts: datetime
    posterior_mean: float
    posterior_std: float


class Report(BaseModel):
    """v2: Human-readable markdown report per outcome."""

    outcome_id: str
    condition_id: str
    asof_ts: datetime
    markdown_report: str


class ScoreV2(BaseModel):
    """v2: Enhanced scoring record with multiple components."""

    token_id: str
    condition_id: str
    outcome: str
    # Basic market data
    mid_price: float | None = None
    spread: float | None = None
    # v2: Execution metrics (at reference size)
    entry_vwap: float | None = None
    exit_vwap: float | None = None
    liquidity_tax: float | None = None
    fill_ratio: float | None = None
    # v2: Score components (0-100 each)
    execution_quality_score: float = Field(ge=0.0, le=100.0)
    rules_risk_score: float = Field(ge=0.0, le=100.0)
    constraint_edge_score: float = Field(ge=0.0, le=100.0)
    regime_opportunity_score: float = Field(ge=0.0, le=100.0)
    # v2: Combined score
    combined_score: float = Field(ge=0.0, le=100.0)
    scored_at: datetime = Field(default_factory=datetime.utcnow)


# Legacy Score model for backward compatibility
class Score(BaseModel):
    """Legacy scoring record (v1)."""

    token_id: str
    condition_id: str
    outcome: str
    mid_price: float | None = None
    spread: float | None = None
    depth_proxy: float | None = None
    tradability_score: float = Field(ge=0.0, le=100.0)
    rules_risk_score: float = Field(ge=0.0, le=100.0)
    combined_score: float = Field(ge=0.0, le=100.0)
    scored_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# LLM Extraction Schemas
# ============================================================================


class RulesExtractionOutput(BaseModel):
    """Expected JSON output from LLM rules extraction (v2)."""

    resolution_source: str = Field(
        description="The authoritative source that will determine market resolution"
    )
    primary_measurement: str = Field(
        default="", description="The key metric or event being measured"
    )
    yes_conditions: list[str] = Field(
        description="Specific conditions that would result in 'Yes' resolution",
        default_factory=list,
    )
    no_conditions: list[str] = Field(
        description="Specific conditions that would result in 'No' resolution",
        default_factory=list,
    )
    key_dates: list[str] = Field(
        description="Important dates in ISO format (YYYY-MM-DD) relevant to resolution",
        default_factory=list,
    )
    edge_cases: list[str] = Field(
        description="Unusual scenarios or edge cases that might affect resolution",
        default_factory=list,
    )
    ambiguity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score from 0 (crystal clear) to 1 (highly ambiguous)",
    )
    unfalsifiable_flag: bool = Field(
        description="True if the market cannot be objectively resolved or verified",
        default=False,
    )
    dispute_risk_notes: list[str] = Field(
        description="Potential points of dispute or controversy",
        default_factory=list,
    )
    recommended_evidence_to_monitor: list[str] = Field(
        description="Sources or events to track for resolution",
        default_factory=list,
    )

    @field_validator("key_dates")
    @classmethod
    def validate_iso_dates(cls, v: list[str]) -> list[str]:
        """Ensure dates are in valid ISO format."""
        validated = []
        for date_str in v:
            try:
                # Try to parse to validate format
                datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                validated.append(date_str)
            except (ValueError, AttributeError):
                # Skip invalid dates
                continue
        return validated


class ReportFactsPayload(BaseModel):
    """Facts provided to LLM for report generation."""

    market_title: str
    market_description: str
    outcome: str
    end_time: str | None
    current_best_bid: float | None
    current_best_ask: float | None
    current_mid: float | None
    current_spread: float | None
    execution_metrics: dict[str, dict[str, float]]  # size_bucket -> metrics
    fee_parameters: dict[str, float]
    belief_posterior_mean: float | None
    belief_posterior_std: float | None
    constraint_violations: list[dict[str, Any]]
    regime_features: dict[str, float]
    rules_structured: dict[str, Any]
