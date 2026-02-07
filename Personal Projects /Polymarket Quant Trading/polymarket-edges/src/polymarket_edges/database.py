"""Database layer for DuckDB persistence (v2 with extended schema)."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from polymarket_edges.config import settings
from polymarket_edges.models import (
    Market,
    Outcome,
    Quote,
    RulesStructured,
    ScoreV2,
    ConstraintViolation,
    TimeSeriesFeature,
    BeliefEstimate,
    Report,
)

logger = logging.getLogger(__name__)


class Database:
    """DuckDB database manager with v2 schema."""

    def __init__(self, db_path: str | Path | None = None):
        """Initialise database connection."""
        self.db_path = Path(db_path or settings.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._create_tables()
        logger.info(f"Connected to database at {self.db_path}")

    def _create_tables(self) -> None:
        """Create all required tables if they don't exist."""

        # Create sequences
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS raw_clob_seq START 1")
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS quotes_seq START 1")
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS scores_v2_seq START 1")
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS orderbook_snapshot_seq START 1")

        # Raw data tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_gamma (
                condition_id VARCHAR PRIMARY KEY,
                response JSON NOT NULL,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_clob (
                id INTEGER PRIMARY KEY DEFAULT nextval('raw_clob_seq'),
                token_id VARCHAR NOT NULL,
                condition_id VARCHAR NOT NULL,
                response JSON NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Normalised tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                condition_id VARCHAR PRIMARY KEY,
                question VARCHAR NOT NULL,
                description VARCHAR,
                market_slug VARCHAR,
                end_date_iso VARCHAR,
                active BOOLEAN DEFAULT TRUE,
                closed BOOLEAN DEFAULT FALSE,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                token_id VARCHAR PRIMARY KEY,
                condition_id VARCHAR NOT NULL,
                outcome VARCHAR NOT NULL,
                winner BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (condition_id) REFERENCES markets(condition_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY DEFAULT nextval('quotes_seq'),
                token_id VARCHAR NOT NULL,
                condition_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                best_bid DOUBLE,
                best_ask DOUBLE,
                bid_size DOUBLE,
                ask_size DOUBLE,
                FOREIGN KEY (token_id) REFERENCES outcomes(token_id),
                FOREIGN KEY (condition_id) REFERENCES markets(condition_id)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_quotes_token ON quotes(token_id, timestamp DESC)"
        )

        # v2: Order book depth tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                snapshot_id VARCHAR PRIMARY KEY,
                token_id VARCHAR NOT NULL,
                condition_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                raw_data JSON NOT NULL,
                FOREIGN KEY (token_id) REFERENCES outcomes(token_id)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshots_token ON orderbook_snapshots(token_id, timestamp DESC)"
        )

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_levels (
                snapshot_id VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                level_index INTEGER NOT NULL,
                price DOUBLE NOT NULL,
                size DOUBLE NOT NULL,
                PRIMARY KEY (snapshot_id, side, level_index),
                FOREIGN KEY (snapshot_id) REFERENCES orderbook_snapshots(snapshot_id)
            )
        """)

        # v2: Execution metrics
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS execution_metrics (
                outcome_id VARCHAR NOT NULL,
                snapshot_id VARCHAR NOT NULL,
                size_bucket DOUBLE NOT NULL,
                entry_vwap DOUBLE,
                exit_vwap DOUBLE,
                liquidity_tax DOUBLE,
                fill_ratio DOUBLE NOT NULL,
                effective_spread DOUBLE,
                PRIMARY KEY (outcome_id, snapshot_id, size_bucket)
            )
        """)

        # Rules structured
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS rules_structured (
                condition_id VARCHAR PRIMARY KEY,
                resolution_source VARCHAR NOT NULL,
                primary_measurement VARCHAR,
                yes_conditions JSON,
                no_conditions JSON,
                key_dates JSON,
                edge_cases JSON,
                ambiguity_score DOUBLE NOT NULL,
                ambiguity_reasons JSON,
                unfalsifiable_flag BOOLEAN DEFAULT FALSE,
                dispute_risk_notes JSON,
                recommended_evidence_to_monitor JSON,
                parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used VARCHAR,
                FOREIGN KEY (condition_id) REFERENCES markets(condition_id)
            )
        """)

        # v2: Constraint violations
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS constraint_violations (
                violation_id VARCHAR PRIMARY KEY,
                condition_id VARCHAR NOT NULL,
                violation_type VARCHAR NOT NULL,
                size_bucket DOUBLE NOT NULL,
                magnitude DOUBLE NOT NULL,
                evidence JSON NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (condition_id) REFERENCES markets(condition_id)
            )
        """)

        # v2: Market links
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_links (
                link_id VARCHAR PRIMARY KEY,
                market_a VARCHAR NOT NULL,
                market_b VARCHAR NOT NULL,
                link_type VARCHAR NOT NULL,
                confidence DOUBLE NOT NULL,
                reasoning VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # v2: Time series features
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS time_series_features (
                outcome_id VARCHAR NOT NULL,
                asof_ts TIMESTAMP NOT NULL,
                feature_name VARCHAR NOT NULL,
                value DOUBLE,
                PRIMARY KEY (outcome_id, asof_ts, feature_name)
            )
        """)

        # v2: Belief estimates
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS belief_estimates (
                outcome_id VARCHAR NOT NULL,
                asof_ts TIMESTAMP NOT NULL,
                posterior_mean DOUBLE NOT NULL,
                posterior_std DOUBLE NOT NULL,
                PRIMARY KEY (outcome_id, asof_ts)
            )
        """)

        # v2: Reports
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                outcome_id VARCHAR NOT NULL,
                condition_id VARCHAR NOT NULL,
                asof_ts TIMESTAMP NOT NULL,
                markdown_report VARCHAR NOT NULL,
                PRIMARY KEY (outcome_id, asof_ts)
            )
        """)

        # v2: Scores
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scores_v2 (
                id INTEGER PRIMARY KEY DEFAULT nextval('scores_v2_seq'),
                token_id VARCHAR NOT NULL,
                condition_id VARCHAR NOT NULL,
                outcome VARCHAR NOT NULL,
                mid_price DOUBLE,
                spread DOUBLE,
                entry_vwap DOUBLE,
                exit_vwap DOUBLE,
                liquidity_tax DOUBLE,
                fill_ratio DOUBLE,
                execution_quality_score DOUBLE NOT NULL,
                rules_risk_score DOUBLE NOT NULL,
                constraint_edge_score DOUBLE NOT NULL,
                regime_opportunity_score DOUBLE NOT NULL,
                combined_score DOUBLE NOT NULL,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (token_id) REFERENCES outcomes(token_id),
                FOREIGN KEY (condition_id) REFERENCES markets(condition_id)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scores_v2_combined ON scores_v2(combined_score DESC)"
        )

        logger.debug("Database tables created/verified")

    # Original methods (kept for compatibility)
    def upsert_raw_gamma(self, condition_id: str, response: dict[str, Any]) -> None:
        """Insert or update raw Gamma API response."""
        self.conn.execute(
            """
            INSERT INTO raw_gamma (condition_id, response, ingested_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (condition_id) DO UPDATE SET
                response = EXCLUDED.response,
                ingested_at = EXCLUDED.ingested_at
            """,
            [condition_id, json.dumps(response)],
        )

    def insert_raw_clob(self, token_id: str, condition_id: str, response: dict[str, Any]) -> None:
        """Insert raw CLOB API response (append-only)."""
        self.conn.execute(
            """
            INSERT INTO raw_clob (token_id, condition_id, response, fetched_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [token_id, condition_id, json.dumps(response)],
        )

    def upsert_market(self, market: Market) -> None:
        """Insert or update market record."""
        self.conn.execute(
            """
            INSERT INTO markets (
                condition_id, question, description, market_slug,
                end_date_iso, active, closed, ingested_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (condition_id) DO UPDATE SET
                question = EXCLUDED.question,
                description = EXCLUDED.description,
                market_slug = EXCLUDED.market_slug,
                end_date_iso = EXCLUDED.end_date_iso,
                active = EXCLUDED.active,
                closed = EXCLUDED.closed,
                ingested_at = EXCLUDED.ingested_at
            """,
            [
                market.condition_id,
                market.question,
                market.description,
                market.market_slug,
                market.end_date_iso,
                market.active,
                market.closed,
                market.ingested_at,
            ],
        )

    def upsert_outcome(self, outcome: Outcome) -> None:
        """Insert or update outcome record."""
        self.conn.execute(
            """
            INSERT INTO outcomes (token_id, condition_id, outcome, winner)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (token_id) DO NOTHING
            """,
            [outcome.token_id, outcome.condition_id, outcome.outcome, outcome.winner],
        )

    def insert_quote(self, quote: Quote) -> None:
        """Insert quote record."""
        self.conn.execute(
            """
            INSERT INTO quotes (
                token_id, condition_id, timestamp,
                best_bid, best_ask, bid_size, ask_size
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                quote.token_id,
                quote.condition_id,
                quote.timestamp,
                quote.best_bid,
                quote.best_ask,
                quote.bid_size,
                quote.ask_size,
            ],
        )

    # v2: New methods for order book depth
    def insert_orderbook_snapshot(
        self,
        token_id: str,
        condition_id: str,
        timestamp: datetime,
        raw_data: dict[str, Any],
        bids: list[dict[str, Any]],
        asks: list[dict[str, Any]],
    ) -> str:
        """Insert order book snapshot with depth.

        Returns:
            snapshot_id
        """
        snapshot_id = str(uuid.uuid4())

        self.conn.execute(
            """
            INSERT INTO orderbook_snapshots (snapshot_id, token_id, condition_id, timestamp, raw_data)
            VALUES (?, ?, ?, ?, ?)
            """,
            [snapshot_id, token_id, condition_id, timestamp, json.dumps(raw_data)],
        )

        # Insert bid levels
        for idx, bid in enumerate(bids):
            self.conn.execute(
                """
                INSERT INTO orderbook_levels (snapshot_id, side, level_index, price, size)
                VALUES (?, 'bid', ?, ?, ?)
                """,
                [snapshot_id, idx, float(bid["price"]), float(bid["size"])],
            )

        # Insert ask levels
        for idx, ask in enumerate(asks):
            self.conn.execute(
                """
                INSERT INTO orderbook_levels (snapshot_id, side, level_index, price, size)
                VALUES (?, 'ask', ?, ?, ?)
                """,
                [snapshot_id, idx, float(ask["price"]), float(ask["size"])],
            )

        return snapshot_id

    def insert_execution_metric(
        self,
        outcome_id: str,
        snapshot_id: str,
        size_bucket: float,
        entry_vwap: float | None,
        exit_vwap: float | None,
        liquidity_tax: float | None,
        fill_ratio: float,
        effective_spread: float | None,
    ) -> None:
        """Insert execution metric."""
        self.conn.execute(
            """
            INSERT INTO execution_metrics (
                outcome_id, snapshot_id, size_bucket,
                entry_vwap, exit_vwap, liquidity_tax, fill_ratio, effective_spread
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (outcome_id, snapshot_id, size_bucket) DO UPDATE SET
                entry_vwap = EXCLUDED.entry_vwap,
                exit_vwap = EXCLUDED.exit_vwap,
                liquidity_tax = EXCLUDED.liquidity_tax,
                fill_ratio = EXCLUDED.fill_ratio,
                effective_spread = EXCLUDED.effective_spread
            """,
            [
                outcome_id,
                snapshot_id,
                size_bucket,
                entry_vwap,
                exit_vwap,
                liquidity_tax,
                fill_ratio,
                effective_spread,
            ],
        )

    def upsert_rules_structured(self, rules: RulesStructured) -> None:
        """Insert or update structured rules (v2)."""
        self.conn.execute(
            """
            INSERT INTO rules_structured (
                condition_id, resolution_source, primary_measurement,
                yes_conditions, no_conditions, key_dates, edge_cases,
                ambiguity_score, ambiguity_reasons, unfalsifiable_flag,
                dispute_risk_notes, recommended_evidence_to_monitor,
                parsed_at, model_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (condition_id) DO UPDATE SET
                resolution_source = EXCLUDED.resolution_source,
                primary_measurement = EXCLUDED.primary_measurement,
                yes_conditions = EXCLUDED.yes_conditions,
                no_conditions = EXCLUDED.no_conditions,
                key_dates = EXCLUDED.key_dates,
                edge_cases = EXCLUDED.edge_cases,
                ambiguity_score = EXCLUDED.ambiguity_score,
                ambiguity_reasons = EXCLUDED.ambiguity_reasons,
                unfalsifiable_flag = EXCLUDED.unfalsifiable_flag,
                dispute_risk_notes = EXCLUDED.dispute_risk_notes,
                recommended_evidence_to_monitor = EXCLUDED.recommended_evidence_to_monitor,
                parsed_at = EXCLUDED.parsed_at,
                model_used = EXCLUDED.model_used
            """,
            [
                rules.condition_id,
                rules.resolution_source,
                getattr(rules, "primary_measurement", ""),
                json.dumps(rules.yes_conditions),
                json.dumps(rules.no_conditions),
                json.dumps(rules.key_dates),
                json.dumps(getattr(rules, "edge_cases", [])),
                rules.ambiguity_score,
                json.dumps(rules.ambiguity_reasons),
                rules.unfalsifiable_flag,
                json.dumps(getattr(rules, "dispute_risk_notes", [])),
                json.dumps(getattr(rules, "recommended_evidence_to_monitor", [])),
                rules.parsed_at,
                rules.model_used,
            ],
        )

    def insert_constraint_violation(self, violation: ConstraintViolation) -> None:
        """Insert constraint violation."""
        self.conn.execute(
            """
            INSERT INTO constraint_violations (
                violation_id, condition_id, violation_type, size_bucket,
                magnitude, evidence, detected_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                violation.violation_id,
                violation.condition_id,
                violation.violation_type,
                violation.size_bucket,
                violation.magnitude,
                json.dumps(violation.evidence),
                violation.detected_at,
            ],
        )

    def insert_time_series_feature(self, feature: TimeSeriesFeature) -> None:
        """Insert time series feature."""
        self.conn.execute(
            """
            INSERT INTO time_series_features (outcome_id, asof_ts, feature_name, value)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (outcome_id, asof_ts, feature_name) DO UPDATE SET
                value = EXCLUDED.value
            """,
            [feature.outcome_id, feature.asof_ts, feature.feature_name, feature.value],
        )

    def insert_belief_estimate(self, estimate: BeliefEstimate) -> None:
        """Insert belief estimate."""
        self.conn.execute(
            """
            INSERT INTO belief_estimates (outcome_id, asof_ts, posterior_mean, posterior_std)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (outcome_id, asof_ts) DO UPDATE SET
                posterior_mean = EXCLUDED.posterior_mean,
                posterior_std = EXCLUDED.posterior_std
            """,
            [estimate.outcome_id, estimate.asof_ts, estimate.posterior_mean, estimate.posterior_std],
        )

    def insert_report(self, report: Report) -> None:
        """Insert outcome report."""
        self.conn.execute(
            """
            INSERT INTO reports (outcome_id, condition_id, asof_ts, markdown_report)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (outcome_id, asof_ts) DO UPDATE SET
                markdown_report = EXCLUDED.markdown_report
            """,
            [report.outcome_id, report.condition_id, report.asof_ts, report.markdown_report],
        )

    def insert_score_v2(self, score: ScoreV2) -> None:
        """Insert v2 score record."""
        self.conn.execute(
            """
            INSERT INTO scores_v2 (
                token_id, condition_id, outcome, mid_price, spread,
                entry_vwap, exit_vwap, liquidity_tax, fill_ratio,
                execution_quality_score, rules_risk_score,
                constraint_edge_score, regime_opportunity_score,
                combined_score, scored_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                score.token_id,
                score.condition_id,
                score.outcome,
                score.mid_price,
                score.spread,
                score.entry_vwap,
                score.exit_vwap,
                score.liquidity_tax,
                score.fill_ratio,
                score.execution_quality_score,
                score.rules_risk_score,
                score.constraint_edge_score,
                score.regime_opportunity_score,
                score.combined_score,
                score.scored_at,
            ],
        )

    # Query methods
    def get_active_markets(self) -> pd.DataFrame:
        """Fetch all active markets."""
        return self.conn.execute(
            "SELECT * FROM markets WHERE active = TRUE ORDER BY ingested_at DESC"
        ).df()

    def get_unparsed_markets(self, limit: int | None = None, selected_conditions: set[str] | None = None) -> pd.DataFrame:
        """Fetch markets without structured rules.

        Args:
            limit: Maximum number of markets to return
            selected_conditions: Optional set of condition IDs to filter to
        """
        query = """
            SELECT m.*
            FROM markets m
            LEFT JOIN rules_structured r ON m.condition_id = r.condition_id
            WHERE m.active = TRUE AND r.condition_id IS NULL
        """

        if selected_conditions:
            placeholders = ",".join(["?" for _ in selected_conditions])
            query += f" AND m.condition_id IN ({placeholders})"

        query += " ORDER BY m.ingested_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        if selected_conditions:
            return self.conn.execute(query, list(selected_conditions)).df()
        else:
            return self.conn.execute(query).df()

    def get_latest_quotes(self) -> pd.DataFrame:
        """Fetch most recent quote for each token."""
        return self.conn.execute("""
            WITH ranked_quotes AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY token_id ORDER BY timestamp DESC) as rn
                FROM quotes
            )
            SELECT token_id, condition_id, timestamp, best_bid, best_ask, bid_size, ask_size
            FROM ranked_quotes
            WHERE rn = 1
        """).df()

    def get_latest_v2_scores(self, limit: int = 100) -> pd.DataFrame:
        """Fetch most recent v2 scores."""
        return self.conn.execute("""
            WITH ranked_scores AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY token_id ORDER BY scored_at DESC) as rn
                FROM scores_v2
            )
            SELECT
                s.token_id,
                s.condition_id,
                s.outcome,
                m.question,
                s.mid_price,
                s.spread,
                s.entry_vwap,
                s.exit_vwap,
                s.liquidity_tax,
                s.fill_ratio,
                s.execution_quality_score,
                s.rules_risk_score,
                s.constraint_edge_score,
                s.regime_opportunity_score,
                s.combined_score,
                s.scored_at
            FROM ranked_scores s
            JOIN markets m ON s.condition_id = m.condition_id
            WHERE s.rn = 1
            ORDER BY s.combined_score DESC
            LIMIT ?
        """, [limit]).df()

    def get_orderbook_levels_for_snapshot(self, snapshot_id: str) -> dict[str, list[dict[str, float]]]:
        """Get order book levels for a snapshot.

        Returns:
            Dict with 'bids' and 'asks' lists
        """
        df = self.conn.execute("""
            SELECT side, level_index, price, size
            FROM orderbook_levels
            WHERE snapshot_id = ?
            ORDER BY side, level_index
        """, [snapshot_id]).df()

        bids = []
        asks = []

        for _, row in df.iterrows():
            level = {"price": float(row["price"]), "size": float(row["size"])}
            if row["side"] == "bid":
                bids.append(level)
            else:
                asks.append(level)

        return {"bids": bids, "asks": asks}

    def get_latest_snapshot_id(self, token_id: str) -> str | None:
        """Get latest snapshot ID for a token."""
        result = self.conn.execute("""
            SELECT snapshot_id FROM orderbook_snapshots
            WHERE token_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, [token_id]).fetchone()

        return result[0] if result else None

    def get_latest_scores(self, limit: int = 100) -> "pd.DataFrame":
        """Get latest scores for dashboard (tries v2 first, falls back to v1).

        Args:
            limit: Maximum number of scores to return

        Returns:
            DataFrame with score data
        """

        # Try v2 scores first
        v2_count = self.conn.execute("SELECT COUNT(*) FROM scores_v2").fetchone()[0]

        if v2_count > 0:
            # Return v2 scores with v1-compatible column names for dashboard
            return self.conn.execute("""
                SELECT
                    s.token_id,
                    s.condition_id,
                    m.question,
                    s.outcome,
                    s.mid_price,
                    s.spread,
                    s.execution_quality_score as tradability_score,
                    s.rules_risk_score,
                    s.combined_score,
                    s.scored_at,
                    r.resolution_source,
                    r.ambiguity_score,
                    '[]' as ambiguity_reasons
                FROM scores_v2 s
                JOIN markets m ON s.condition_id = m.condition_id
                LEFT JOIN rules_structured r ON s.condition_id = r.condition_id
                ORDER BY s.combined_score DESC
                LIMIT ?
            """, [limit]).df()
        else:
            # Fall back to v1 scores
            return self.conn.execute("""
                SELECT
                    s.token_id,
                    s.condition_id,
                    m.question,
                    s.outcome,
                    s.mid_price,
                    s.spread,
                    s.tradability_score,
                    s.rules_risk_score,
                    s.combined_score,
                    s.scored_at,
                    r.resolution_source,
                    r.ambiguity_score,
                    '[]' as ambiguity_reasons
                FROM scores s
                JOIN markets m ON s.condition_id = m.condition_id
                LEFT JOIN rules_structured r ON s.condition_id = r.condition_id
                ORDER BY s.combined_score DESC
                LIMIT ?
            """, [limit]).df()

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        logger.info("Database connection closed")
