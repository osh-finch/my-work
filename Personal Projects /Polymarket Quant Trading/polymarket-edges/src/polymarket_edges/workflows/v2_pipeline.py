"""v2 Pipeline workflows for execution-aware analysis."""

import logging
from datetime import datetime

import pandas as pd

from polymarket_edges.clients.clob import CLOBClient
from polymarket_edges.constraints.detector import ConstraintDetector
from polymarket_edges.database import Database
from polymarket_edges.execution.simulator import OrderBookSimulator
from polymarket_edges.features.regime import RegimeFeatureExtractor
from polymarket_edges.features.belief import BeliefFilter
from polymarket_edges.llm.openai_provider import OpenAIProvider
from polymarket_edges.llm.local_provider import LocalProvider
from polymarket_edges.scoring.scorer_v2 import ScorerV2
from polymarket_edges.models import (
    TimeSeriesFeature,
    BeliefEstimate,
    ConstraintViolation,
    Report,
    ScoreV2,
    ReportFactsPayload,
)
from polymarket_edges.config import settings

logger = logging.getLogger(__name__)


async def update_orderbooks_v2(
    db: Database,
    levels: int | None = None,
    max_concurrent: int = 5,
    selected_conditions: set[str] | None = None,
) -> int:
    """Update order books with depth capture.

    Args:
        db: Database instance
        levels: Number of levels to capture (defaults to config)
        max_concurrent: Max concurrent requests
        selected_conditions: Optional set of condition IDs to filter to

    Returns:
        Number of order books updated
    """
    logger.info("Starting order book update with depth capture")

    # Build query with optional condition filtering
    if selected_conditions:
        placeholders = ",".join(["?" for _ in selected_conditions])
        query = f"""
            SELECT o.token_id, o.condition_id
            FROM outcomes o
            JOIN markets m ON o.condition_id = m.condition_id
            WHERE m.active = TRUE AND o.condition_id IN ({placeholders})
        """
        outcomes_df = db.conn.execute(query, list(selected_conditions)).df()
    else:
        outcomes_df = db.conn.execute("""
            SELECT o.token_id, o.condition_id
            FROM outcomes o
            JOIN markets m ON o.condition_id = m.condition_id
            WHERE m.active = TRUE
        """).df()

    if outcomes_df.empty:
        logger.warning("No active outcomes found")
        return 0

    token_ids = outcomes_df["token_id"].tolist()

    # Filter out synthetic token IDs (they have format like "condition_id-yes")
    real_token_ids = [t for t in token_ids if "-yes" not in t.lower() and "-no" not in t.lower()]
    synthetic_count = len(token_ids) - len(real_token_ids)

    if synthetic_count > 0:
        logger.warning(f"Skipping {synthetic_count} synthetic token IDs (no CLOB data available)")

    logger.info(f"Fetching order books for {len(real_token_ids)} tokens (out of {len(token_ids)} total)")

    if not real_token_ids:
        logger.warning("No real token IDs found - all tokens appear to be synthetic. Re-ingest markets to get clobTokenIds.")
        return 0

    # Fetch order books with depth
    client = CLOBClient(depth_levels=levels or settings.orderbook_depth_levels)
    order_books = await client.get_order_books_batch(real_token_ids, max_concurrent=max_concurrent)

    logger.info(f"Successfully fetched {len(order_books)}/{len(real_token_ids)} order books from CLOB API")

    # Store snapshots and levels
    count = 0
    empty_books = 0
    illiquid_books = 0
    timestamp = datetime.utcnow()

    for token_id, order_book in order_books.items():
        # Get condition_id
        condition_id = outcomes_df[outcomes_df["token_id"] == token_id]["condition_id"].iloc[0]

        # Store snapshot with levels
        snapshot_id = db.insert_orderbook_snapshot(
            token_id=token_id,
            condition_id=condition_id,
            timestamp=timestamp,
            raw_data={"market": order_book.market, "hash": order_book.hash, "timestamp": order_book.timestamp},
            bids=order_book.bids,
            asks=order_book.asks,
        )

        logger.debug(f"Stored snapshot {snapshot_id} for token {token_id}: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
        count += 1

    logger.info(f"Stored {count} order book snapshots with depth (skipped {empty_books} empty and {illiquid_books} illiquid books)")
    return count


async def compute_execution_metrics(
    db: Database,
    sizes: list[float] | None = None,
    selected_conditions: set[str] | None = None,
) -> int:
    """Compute execution metrics for all recent snapshots.

    Args:
        db: Database instance
        sizes: Trade size buckets (defaults to config)
        selected_conditions: Optional set of condition IDs to filter to

    Returns:
        Number of execution metrics computed
    """
    logger.info("Computing execution metrics")

    size_buckets = sizes or settings.trade_size_buckets
    simulator = OrderBookSimulator()

    # Get recent snapshots with optional filtering
    if selected_conditions:
        placeholders = ",".join(["?" for _ in selected_conditions])
        query = f"""
            SELECT snapshot_id, token_id, condition_id
            FROM orderbook_snapshots
            WHERE condition_id IN ({placeholders})
            ORDER BY timestamp DESC
            LIMIT 1000
        """
        snapshots_df = db.conn.execute(query, list(selected_conditions)).df()
    else:
        snapshots_df = db.conn.execute("""
            SELECT snapshot_id, token_id, condition_id
            FROM orderbook_snapshots
            ORDER BY timestamp DESC
            LIMIT 1000
        """).df()

    if snapshots_df.empty:
        logger.warning("No snapshots found")
        return 0

    count = 0

    for _, row in snapshots_df.iterrows():
        snapshot_id = row["snapshot_id"]
        token_id = row["token_id"]

        # Get order book levels
        levels = db.get_orderbook_levels_for_snapshot(snapshot_id)
        bids = levels["bids"]
        asks = levels["asks"]

        if not bids or not asks:
            continue

        # Compute metrics for each size bucket
        for size in size_buckets:
            entry_vwap, exit_vwap, liquidity_tax = simulator.compute_liquidity_tax(bids, asks, size)
            effective_spread = simulator.effective_spread(bids, asks, size)

            # Compute fill ratio
            entry_result = simulator.simulate_buy_yes(asks, size)
            fill_ratio = entry_result.fill_ratio if entry_result else 0.0

            db.insert_execution_metric(
                outcome_id=token_id,
                snapshot_id=snapshot_id,
                size_bucket=size,
                entry_vwap=entry_vwap,
                exit_vwap=exit_vwap,
                liquidity_tax=liquidity_tax,
                fill_ratio=fill_ratio,
                effective_spread=effective_spread,
            )

            count += 1

    logger.info(f"Computed {count} execution metrics")
    return count


async def detect_constraints(
    db: Database,
    size: float | None = None,
    selected_conditions: set[str] | None = None,
) -> int:
    """Detect constraint violations (arbitrage opportunities).

    Args:
        db: Database instance
        size: Trade size for detection (defaults to reference size)
        selected_conditions: Optional set of condition IDs to filter to

    Returns:
        Number of violations detected
    """
    logger.info("Detecting constraint violations")

    size_bucket = size or settings.reference_size_bucket
    detector = ConstraintDetector()

    # Get binary markets (YES/NO pairs) with optional filtering
    if selected_conditions:
        placeholders = ",".join(["?" for _ in selected_conditions])
        query = f"""
            SELECT m.condition_id, m.question,
                   yes.token_id as yes_token_id,
                   no.token_id as no_token_id
            FROM markets m
            LEFT JOIN outcomes yes ON m.condition_id = yes.condition_id AND yes.outcome = 'Yes'
            LEFT JOIN outcomes no ON m.condition_id = no.condition_id AND no.outcome = 'No'
            WHERE m.active = TRUE AND yes.token_id IS NOT NULL AND no.token_id IS NOT NULL
              AND m.condition_id IN ({placeholders})
        """
        markets_df = db.conn.execute(query, list(selected_conditions)).df()
    else:
        markets_df = db.conn.execute("""
            SELECT m.condition_id, m.question,
                   yes.token_id as yes_token_id,
                   no.token_id as no_token_id
            FROM markets m
            LEFT JOIN outcomes yes ON m.condition_id = yes.condition_id AND yes.outcome = 'Yes'
            LEFT JOIN outcomes no ON m.condition_id = no.condition_id AND no.outcome = 'No'
            WHERE m.active = TRUE AND yes.token_id IS NOT NULL AND no.token_id IS NOT NULL
        """).df()

    if markets_df.empty:
        logger.warning("No binary markets found")
        return 0

    count = 0

    for _, row in markets_df.iterrows():
        condition_id = row["condition_id"]
        yes_token_id = row["yes_token_id"]
        no_token_id = row["no_token_id"]

        # Get latest snapshots for both
        yes_snapshot_id = db.get_latest_snapshot_id(yes_token_id)
        no_snapshot_id = db.get_latest_snapshot_id(no_token_id)

        if not yes_snapshot_id or not no_snapshot_id:
            continue

        yes_levels = db.get_orderbook_levels_for_snapshot(yes_snapshot_id)
        no_levels = db.get_orderbook_levels_for_snapshot(no_snapshot_id)

        # Check complete set violations
        violations = detector.check_complete_set(
            yes_bids=yes_levels["bids"],
            yes_asks=yes_levels["asks"],
            no_bids=no_levels["bids"],
            no_asks=no_levels["asks"],
            size_bucket=size_bucket,
            condition_id=condition_id,
        )

        for violation in violations:
            violation_record = ConstraintViolation(
                violation_id=detector.generate_violation_id(),
                condition_id=violation.condition_id,
                violation_type=violation.violation_type,
                size_bucket=violation.size_bucket,
                magnitude=violation.magnitude,
                evidence=violation.evidence,
                detected_at=datetime.utcnow(),
            )
            db.insert_constraint_violation(violation_record)
            count += 1

    logger.info(f"Detected {count} constraint violations")
    return count


async def compute_features(
    db: Database,
    window: str = "24h",
    selected_conditions: set[str] | None = None,
) -> int:
    """Compute regime and lifecycle features.

    Args:
        db: Database instance
        window: Time window ('24h' or '7d')
        selected_conditions: Optional set of condition IDs to filter to

    Returns:
        Number of features computed
    """
    logger.info(f"Computing regime features (window={window})")

    extractor = RegimeFeatureExtractor()
    current_time = datetime.utcnow()

    # Get all active outcomes with optional filtering
    if selected_conditions:
        placeholders = ",".join(["?" for _ in selected_conditions])
        query = f"""
            SELECT o.token_id, o.condition_id, m.ingested_at, m.end_date_iso
            FROM outcomes o
            JOIN markets m ON o.condition_id = m.condition_id
            WHERE m.active = TRUE AND o.condition_id IN ({placeholders})
        """
        outcomes_df = db.conn.execute(query, list(selected_conditions)).df()
    else:
        outcomes_df = db.conn.execute("""
            SELECT o.token_id, o.condition_id, m.ingested_at, m.end_date_iso
            FROM outcomes o
            JOIN markets m ON o.condition_id = m.condition_id
            WHERE m.active = TRUE
        """).df()

    if outcomes_df.empty:
        logger.warning("No active outcomes found")
        return 0

    count = 0

    for _, row in outcomes_df.iterrows():
        token_id = row["token_id"]
        first_seen = row["ingested_at"]
        end_date_iso = row["end_date_iso"]

        # Parse end date
        try:
            end_time = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00")) if end_date_iso else None
        except ValueError:
            end_time = None

        # Get quote history
        quotes_history = db.conn.execute("""
            SELECT timestamp, best_bid, best_ask, bid_size, ask_size
            FROM quotes
            WHERE token_id = ?
            ORDER BY timestamp DESC
            LIMIT 1000
        """, [token_id]).df()

        if quotes_history.empty:
            continue

        # Extract features
        features = extractor.extract_features(
            outcome_id=token_id,
            quotes_history=quotes_history,
            market_first_seen=first_seen,
            market_end_time=end_time,
            current_time=current_time,
        )

        # Store features
        for feature_name, value in features.items():
            if value is not None:
                feature_record = TimeSeriesFeature(
                    outcome_id=token_id,
                    asof_ts=current_time,
                    feature_name=feature_name,
                    value=value,
                )
                db.insert_time_series_feature(feature_record)
                count += 1

    logger.info(f"Computed {count} regime features")
    return count


async def compute_beliefs(
    db: Database,
    selected_conditions: set[str] | None = None,
) -> int:
    """Compute Bayesian belief estimates.

    Args:
        db: Database instance
        selected_conditions: Optional set of condition IDs to filter to

    Returns:
        Number of beliefs computed
    """
    logger.info("Computing Bayesian belief estimates")

    belief_filter = BeliefFilter()
    current_time = datetime.utcnow()

    # Get all active outcomes with optional filtering
    if selected_conditions:
        placeholders = ",".join(["?" for _ in selected_conditions])
        query = f"""
            SELECT o.token_id
            FROM outcomes o
            JOIN markets m ON o.condition_id = m.condition_id
            WHERE m.active = TRUE AND o.condition_id IN ({placeholders})
        """
        outcomes_df = db.conn.execute(query, list(selected_conditions)).df()
    else:
        outcomes_df = db.conn.execute("""
            SELECT o.token_id
            FROM outcomes o
            JOIN markets m ON o.condition_id = m.condition_id
            WHERE m.active = TRUE
        """).df()

    if outcomes_df.empty:
        logger.warning("No active outcomes found")
        return 0

    count = 0

    for token_id in outcomes_df["token_id"]:
        observations = []

        # First try quotes table (v1 data)
        quotes_df = db.conn.execute("""
            SELECT timestamp, best_bid, best_ask, bid_size, ask_size
            FROM quotes
            WHERE token_id = ?
            ORDER BY timestamp
        """, [token_id]).df()

        if not quotes_df.empty:
            for _, row in quotes_df.iterrows():
                if row["best_bid"] is None or row["best_ask"] is None:
                    continue

                mid = (row["best_bid"] + row["best_ask"]) / 2
                liquidity = (row["bid_size"] or 0) + (row["ask_size"] or 0)

                ts_numeric = row["timestamp"].timestamp()
                observations.append((ts_numeric, mid, liquidity))

        # If no quotes, try orderbook_snapshots (v2 data)
        if len(observations) < 2:
            snapshots_df = db.conn.execute("""
                SELECT os.snapshot_id, os.timestamp
                FROM orderbook_snapshots os
                WHERE os.token_id = ?
                ORDER BY os.timestamp
            """, [token_id]).df()

            for _, snap_row in snapshots_df.iterrows():
                levels = db.get_orderbook_levels_for_snapshot(snap_row["snapshot_id"])
                if levels["bids"] and levels["asks"]:
                    best_bid = levels["bids"][0]["price"]
                    best_ask = levels["asks"][0]["price"]
                    mid = (best_bid + best_ask) / 2

                    # Calculate total liquidity from all levels
                    bid_liquidity = sum(level["size"] for level in levels["bids"])
                    ask_liquidity = sum(level["size"] for level in levels["asks"])
                    liquidity = bid_liquidity + ask_liquidity

                    ts_numeric = snap_row["timestamp"].timestamp()
                    observations.append((ts_numeric, mid, liquidity))

        if len(observations) < 2:
            continue

        # Compute belief estimate
        result = belief_filter.get_latest_estimate(observations)
        if result:
            posterior_mean, posterior_std = result

            estimate = BeliefEstimate(
                outcome_id=token_id,
                asof_ts=current_time,
                posterior_mean=posterior_mean,
                posterior_std=posterior_std,
            )
            db.insert_belief_estimate(estimate)
            count += 1

    logger.info(f"Computed {count} belief estimates")
    return count


async def build_reports(
    db: Database,
    provider_type: str = "local",
    limit: int | None = None,
    selected_conditions: set[str] | None = None,
) -> int:
    """Build human-readable markdown reports.

    Args:
        db: Database instance
        provider_type: LLM provider ('openai' or 'local')
        limit: Maximum number of reports to generate
        selected_conditions: Optional set of condition IDs to filter to

    Returns:
        Number of reports generated
    """
    logger.info(f"Building reports with provider={provider_type}")

    # Initialize LLM provider
    if provider_type == "openai":
        provider = OpenAIProvider()
    else:
        provider = LocalProvider()

    current_time = datetime.utcnow()

    # Get outcomes that need reports with optional filtering
    query = """
        SELECT
            o.token_id, o.condition_id, o.outcome,
            m.question, m.description, m.end_date_iso,
            r.resolution_source, r.ambiguity_score, r.unfalsifiable_flag,
            r.yes_conditions, r.no_conditions, r.dispute_risk_notes
        FROM outcomes o
        JOIN markets m ON o.condition_id = m.condition_id
        LEFT JOIN rules_structured r ON o.condition_id = r.condition_id
        WHERE m.active = TRUE
    """

    if selected_conditions:
        placeholders = ",".join(["?" for _ in selected_conditions])
        query += f" AND o.condition_id IN ({placeholders})"

    if limit:
        query += f" LIMIT {limit}"

    if selected_conditions:
        outcomes_df = db.conn.execute(query, list(selected_conditions)).df()
    else:
        outcomes_df = db.conn.execute(query).df()

    if outcomes_df.empty:
        logger.warning("No outcomes found for report generation")
        return 0

    count = 0

    for _, row in outcomes_df.iterrows():
        token_id = row["token_id"]
        condition_id = row["condition_id"]

        # Build facts payload
        # Get execution metrics at reference size
        exec_metrics_df = db.conn.execute("""
            SELECT entry_vwap, exit_vwap, liquidity_tax, fill_ratio, effective_spread
            FROM execution_metrics
            WHERE outcome_id = ? AND size_bucket = ?
            ORDER BY snapshot_id DESC
            LIMIT 1
        """, [token_id, settings.reference_size_bucket]).df()

        exec_metrics_dict = {}
        if not exec_metrics_df.empty:
            exec_metrics_dict = {
                str(settings.reference_size_bucket): {
                    "entry_vwap": float(exec_metrics_df["entry_vwap"].iloc[0]) if pd.notna(exec_metrics_df["entry_vwap"].iloc[0]) else None,
                    "exit_vwap": float(exec_metrics_df["exit_vwap"].iloc[0]) if pd.notna(exec_metrics_df["exit_vwap"].iloc[0]) else None,
                    "liquidity_tax": float(exec_metrics_df["liquidity_tax"].iloc[0]) if pd.notna(exec_metrics_df["liquidity_tax"].iloc[0]) else None,
                    "fill_ratio": float(exec_metrics_df["fill_ratio"].iloc[0]),
                }
            }

        # Get regime features
        regime_features_df = db.conn.execute("""
            SELECT feature_name, value
            FROM time_series_features
            WHERE outcome_id = ?
            ORDER BY asof_ts DESC
            LIMIT 20
        """, [token_id]).df()

        regime_dict = {}
        if not regime_features_df.empty:
            regime_dict = dict(zip(regime_features_df["feature_name"], regime_features_df["value"]))

        # Get belief estimate
        belief_df = db.conn.execute("""
            SELECT posterior_mean, posterior_std
            FROM belief_estimates
            WHERE outcome_id = ?
            ORDER BY asof_ts DESC
            LIMIT 1
        """, [token_id]).df()

        belief_mean = None
        belief_std = None
        if not belief_df.empty:
            belief_mean = float(belief_df["posterior_mean"].iloc[0])
            belief_std = float(belief_df["posterior_std"].iloc[0])

        # Get constraint violations
        violations_df = db.conn.execute("""
            SELECT violation_type, magnitude, evidence
            FROM constraint_violations
            WHERE condition_id = ?
            ORDER BY detected_at DESC
            LIMIT 5
        """, [condition_id]).df()

        violations_list = violations_df.to_dict("records") if not violations_df.empty else []

        # Get best bid/ask from latest order book snapshot
        best_bid = None
        best_ask = None

        snapshot_id = db.get_latest_snapshot_id(token_id)
        if snapshot_id:
            levels = db.get_orderbook_levels_for_snapshot(snapshot_id)
            if levels["bids"]:
                best_bid = levels["bids"][0]["price"]
            if levels["asks"]:
                best_ask = levels["asks"][0]["price"]

        # Build facts payload
        mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
        spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None

        facts = ReportFactsPayload(
            market_title=row["question"],
            market_description=row["description"] or "",
            outcome=row["outcome"],
            end_time=row["end_date_iso"],
            current_best_bid=best_bid,
            current_best_ask=best_ask,
            current_mid=mid,
            current_spread=spread,
            execution_metrics=exec_metrics_dict,
            fee_parameters={"taker_fee_bps": settings.taker_fee_bps, "maker_rebate_bps": settings.maker_rebate_bps},
            belief_posterior_mean=belief_mean,
            belief_posterior_std=belief_std,
            constraint_violations=violations_list,
            regime_features=regime_dict,
            rules_structured={
                "resolution_source": row.get("resolution_source", "") if pd.notna(row.get("resolution_source")) else "",
                "ambiguity_score": float(row.get("ambiguity_score", 0.5)) if pd.notna(row.get("ambiguity_score")) else 0.5,
                "unfalsifiable_flag": bool(row.get("unfalsifiable_flag", False)) if pd.notna(row.get("unfalsifiable_flag")) else False,
            },
        )

        # Generate report
        try:
            markdown_report = await provider.generate_report(facts)

            report = Report(
                outcome_id=token_id,
                condition_id=condition_id,
                asof_ts=current_time,
                markdown_report=markdown_report,
            )
            db.insert_report(report)
            count += 1
        except Exception as e:
            logger.error(f"Failed to generate report for {token_id}: {e}")

    logger.info(f"Generated {count} reports")
    return count


async def score_v2_outcomes(
    db: Database,
    selected_conditions: set[str] | None = None,
) -> int:
    """Score all outcomes using v2 multi-component system.

    Args:
        db: Database instance
        selected_conditions: Optional set of condition IDs to filter to

    Returns:
        Number of outcomes scored
    """
    logger.info("Scoring outcomes with v2 system")

    scorer = ScorerV2()
    current_time = datetime.utcnow()

    # Get all active outcomes with all required data and optional filtering
    if selected_conditions:
        placeholders = ",".join(["?" for _ in selected_conditions])
        query = f"""
            SELECT
                o.token_id, o.condition_id, o.outcome,
                r.ambiguity_score, r.unfalsifiable_flag
            FROM outcomes o
            JOIN markets m ON o.condition_id = m.condition_id
            LEFT JOIN rules_structured r ON o.condition_id = r.condition_id
            WHERE m.active = TRUE AND o.condition_id IN ({placeholders})
        """
        outcomes_df = db.conn.execute(query, list(selected_conditions)).df()
    else:
        outcomes_df = db.conn.execute("""
            SELECT
                o.token_id, o.condition_id, o.outcome,
                r.ambiguity_score, r.unfalsifiable_flag
            FROM outcomes o
            JOIN markets m ON o.condition_id = m.condition_id
            LEFT JOIN rules_structured r ON o.condition_id = r.condition_id
            WHERE m.active = TRUE
        """).df()

    if outcomes_df.empty:
        logger.warning("No outcomes found for scoring")
        return 0

    # Track statistics
    count = 0
    snapshots_found = 0
    snapshots_with_levels = 0

    for _, row in outcomes_df.iterrows():
        token_id = row["token_id"]
        condition_id = row["condition_id"]
        outcome = row["outcome"]

        # Get execution metrics at reference size
        exec_metrics = {}
        exec_df = db.conn.execute("""
            SELECT entry_vwap, exit_vwap, liquidity_tax, fill_ratio, effective_spread
            FROM execution_metrics
            WHERE outcome_id = ? AND size_bucket = ?
            ORDER BY snapshot_id DESC
            LIMIT 1
        """, [token_id, settings.reference_size_bucket]).df()

        if not exec_df.empty:
            exec_metrics = {
                "effective_spread": float(exec_df["effective_spread"].iloc[0]) if pd.notna(exec_df["effective_spread"].iloc[0]) else None,
                "fill_ratio": float(exec_df["fill_ratio"].iloc[0]),
            }

        # Get regime features
        regime_features = {}
        regime_df = db.conn.execute("""
            SELECT feature_name, value
            FROM time_series_features
            WHERE outcome_id = ?
            ORDER BY asof_ts DESC
            LIMIT 20
        """, [token_id]).df()

        if not regime_df.empty:
            regime_features = dict(zip(regime_df["feature_name"], regime_df["value"]))

        # Add belief std to regime features
        belief_df = db.conn.execute("""
            SELECT posterior_std
            FROM belief_estimates
            WHERE outcome_id = ?
            ORDER BY asof_ts DESC
            LIMIT 1
        """, [token_id]).df()

        if not belief_df.empty:
            regime_features["belief_std"] = float(belief_df["posterior_std"].iloc[0])

        # Get complete set buy cost (if binary market)
        complete_set_cost = None
        # For simplicity, we'll skip this in the initial implementation
        # This would require matching YES/NO pairs and their execution metrics

        # Compute score
        score_components = scorer.score_outcome(
            execution_metrics=exec_metrics,
            ambiguity_score=float(row["ambiguity_score"]) if pd.notna(row["ambiguity_score"]) else 0.5,
            unfalsifiable_flag=bool(row["unfalsifiable_flag"]) if pd.notna(row["unfalsifiable_flag"]) else False,
            complete_set_buy_cost=complete_set_cost,
            regime_features=regime_features,
        )

        # Get best bid/ask from latest order book snapshot
        best_bid = None
        best_ask = None

        snapshot_id = db.get_latest_snapshot_id(token_id)
        if snapshot_id:
            snapshots_found += 1
            levels = db.get_orderbook_levels_for_snapshot(snapshot_id)
            if levels["bids"] and levels["asks"]:
                snapshots_with_levels += 1
                best_bid = levels["bids"][0]["price"]
                best_ask = levels["asks"][0]["price"]
            elif levels["bids"]:
                best_bid = levels["bids"][0]["price"]
            elif levels["asks"]:
                best_ask = levels["asks"][0]["price"]

            # Debug logging for first few outcomes
            if count < 3:
                logger.debug(f"Token {token_id[:30]}...: snapshot={snapshot_id[:8]}..., bids={len(levels['bids'])}, asks={len(levels['asks'])}, best_bid={best_bid}, best_ask={best_ask}")
        else:
            # Debug: log tokens without snapshots
            if count < 3:
                logger.debug(f"Token {token_id[:30]}...: NO SNAPSHOT FOUND")

        # Compute mid and spread
        mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
        spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None

        # Get entry/exit VWAPs
        entry_vwap = float(exec_df["entry_vwap"].iloc[0]) if not exec_df.empty and pd.notna(exec_df["entry_vwap"].iloc[0]) else None
        exit_vwap = float(exec_df["exit_vwap"].iloc[0]) if not exec_df.empty and pd.notna(exec_df["exit_vwap"].iloc[0]) else None
        liquidity_tax = float(exec_df["liquidity_tax"].iloc[0]) if not exec_df.empty and pd.notna(exec_df["liquidity_tax"].iloc[0]) else None
        fill_ratio = float(exec_df["fill_ratio"].iloc[0]) if not exec_df.empty else None

        # Insert score
        score_record = ScoreV2(
            token_id=token_id,
            condition_id=condition_id,
            outcome=outcome,
            mid_price=mid,
            spread=spread,
            entry_vwap=entry_vwap,
            exit_vwap=exit_vwap,
            liquidity_tax=liquidity_tax,
            fill_ratio=fill_ratio,
            execution_quality_score=score_components.execution_quality,
            rules_risk_score=score_components.rules_risk,
            constraint_edge_score=score_components.constraint_edge,
            regime_opportunity_score=score_components.regime_opportunity,
            combined_score=score_components.combined,
            scored_at=current_time,
        )
        db.insert_score_v2(score_record)
        count += 1

    logger.info(f"Scored {count} outcomes (snapshots found: {snapshots_found}, with bid/ask data: {snapshots_with_levels})")
    return count
