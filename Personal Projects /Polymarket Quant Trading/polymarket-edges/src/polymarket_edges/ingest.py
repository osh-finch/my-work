"""Data ingestion logic for Polymarket markets and quotes."""

import logging
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from polymarket_edges.clients.clob import CLOBClient
from polymarket_edges.clients.gamma import GammaClient
from polymarket_edges.db import Database
from polymarket_edges.llm import get_provider
from polymarket_edges.models import Market, Outcome, Quote, RulesStructured

logger = logging.getLogger(__name__)
console = Console()


async def ingest_markets(db: Database, max_pages: int = 10) -> int:
    """Ingest active markets from Gamma API.

    Args:
        db: Database instance
        max_pages: Maximum pages to fetch

    Returns:
        Number of markets ingested
    """
    logger.info("Starting market ingestion")

    client = GammaClient()
    markets = await client.get_all_active_markets(max_pages=max_pages)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting markets...", total=len(markets))

        for gamma_market in markets:
            try:
                # Store raw response
                db.upsert_raw_gamma(
                    gamma_market.condition_id,
                    gamma_market.model_dump(),
                )

                # Store normalised market
                market = Market(
                    condition_id=gamma_market.condition_id,
                    question=gamma_market.question,
                    description=gamma_market.description,
                    market_slug=gamma_market.market_slug,
                    end_date_iso=gamma_market.end_date_iso,
                    active=gamma_market.active,
                    closed=gamma_market.closed,
                    ingested_at=datetime.utcnow(),
                )
                db.upsert_market(market)

                # Store outcomes (tokens)
                # Try to use clobTokenIds first (most reliable for v2)
                if gamma_market.clob_token_ids and gamma_market.clob_token_ids != "null":
                    import json
                    try:
                        token_ids = json.loads(gamma_market.clob_token_ids)
                        outcomes = gamma_market.outcomes or ["Yes", "No"]

                        for i, token_id in enumerate(token_ids):
                            outcome_name = outcomes[i] if i < len(outcomes) else f"Outcome {i+1}"
                            outcome = Outcome(
                                token_id=str(token_id),
                                condition_id=gamma_market.condition_id,
                                outcome=outcome_name,
                                winner=False,
                            )
                            db.upsert_outcome(outcome)
                    except (json.JSONDecodeError, Exception) as e:
                        logger.debug(f"Failed to parse clobTokenIds for {gamma_market.condition_id}: {e}")
                        # Fall back to synthetic IDs
                        for outcome_name in ["Yes", "No"]:
                            outcome = Outcome(
                                token_id=f"{gamma_market.condition_id}-{outcome_name.lower()}",
                                condition_id=gamma_market.condition_id,
                                outcome=outcome_name,
                                winner=False,
                            )
                            db.upsert_outcome(outcome)

                elif gamma_market.tokens:
                    # Use tokens from API if available
                    for token_data in gamma_market.tokens:
                        outcome = Outcome(
                            token_id=token_data.get("token_id") or token_data.get("tokenId", ""),
                            condition_id=gamma_market.condition_id,
                            outcome=token_data.get("outcome", "Unknown"),
                            winner=token_data.get("winner", False),
                        )
                        db.upsert_outcome(outcome)
                else:
                    # If no tokens provided, create synthetic Yes/No outcomes
                    # Many prediction markets are binary, so this is a reasonable default
                    for outcome_name in ["Yes", "No"]:
                        # Use condition_id as base for token_id (will be replaced when we fetch from CLOB)
                        outcome = Outcome(
                            token_id=f"{gamma_market.condition_id}-{outcome_name.lower()}",
                            condition_id=gamma_market.condition_id,
                            outcome=outcome_name,
                            winner=False,
                        )
                        db.upsert_outcome(outcome)

                progress.advance(task)

            except Exception as e:
                logger.error(
                    f"Failed to ingest market {gamma_market.condition_id}: {e}"
                )
                continue

    logger.info(f"Ingested {len(markets)} markets successfully")
    console.print(f"[green]✓[/green] Ingested {len(markets)} markets")
    return len(markets)


async def update_quotes(db: Database, max_concurrent: int = 5) -> int:
    """Update quotes for all outcomes from CLOB API.

    Args:
        db: Database instance
        max_concurrent: Maximum concurrent requests

    Returns:
        Number of quotes updated
    """
    logger.info("Starting quote updates")

    # Get all outcomes
    outcomes_df = db.conn.execute("SELECT token_id, condition_id FROM outcomes").df()

    if outcomes_df.empty:
        console.print("[yellow]No outcomes found. Run 'ingest' first.[/yellow]")
        return 0

    token_ids = outcomes_df["token_id"].tolist()
    condition_ids = dict(zip(outcomes_df["token_id"], outcomes_df["condition_id"]))

    client = CLOBClient()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching quotes...", total=len(token_ids))

        # Fetch in batches
        batch_size = 50
        total_quotes = 0

        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i : i + batch_size]

            try:
                order_books = await client.get_order_books_batch(
                    batch, max_concurrent=max_concurrent
                )

                for token_id, book in order_books.items():
                    try:
                        # Store raw response
                        db.insert_raw_clob(
                            token_id,
                            condition_ids[token_id],
                            {
                                "market": book.market,
                                "asset_id": book.asset_id,
                                "timestamp": book.timestamp,
                                "bids": book.bids,
                                "asks": book.asks,
                            },
                        )

                        # Store normalised quote
                        quote = Quote(
                            token_id=token_id,
                            condition_id=condition_ids[token_id],
                            timestamp=datetime.utcnow(),
                            best_bid=book.best_bid,
                            best_ask=book.best_ask,
                            bid_size=book.bid_size,
                            ask_size=book.ask_size,
                        )
                        db.insert_quote(quote)
                        total_quotes += 1

                        progress.advance(task)

                    except Exception as e:
                        logger.error(f"Failed to store quote for {token_id}: {e}")
                        progress.advance(task)
                        continue

            except Exception as e:
                logger.error(f"Failed to fetch batch: {e}")
                progress.advance(task, advance=len(batch))
                continue

    logger.info(f"Updated {total_quotes} quotes successfully")
    console.print(f"[green]✓[/green] Updated {total_quotes} quotes")
    return total_quotes


async def parse_rules(
    db: Database,
    provider_type: str = "local",
    limit: int | None = None,
    selected_conditions: set[str] | None = None,
) -> int:
    """Parse market rules using LLM.

    Args:
        db: Database instance
        provider_type: LLM provider type ("openai" or "local")
        limit: Maximum number of markets to parse
        selected_conditions: Optional set of condition IDs to filter to

    Returns:
        Number of markets parsed
    """
    logger.info(f"Starting rules parsing with {provider_type} provider")

    # Get unparsed markets with optional filtering
    unparsed = db.get_unparsed_markets(limit=limit, selected_conditions=selected_conditions)

    if unparsed.empty:
        console.print("[yellow]No unparsed markets found.[/yellow]")
        return 0

    provider = get_provider(provider_type)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing rules...", total=len(unparsed))

        parsed_count = 0

        for _, row in unparsed.iterrows():
            try:
                # Extract rules
                result = await provider.extract_rules(
                    question=row["question"],
                    description=row.get("description"),
                    rules=row.get("description"),  # Use description as rules for now
                )

                # Store structured rules
                rules = RulesStructured(
                    condition_id=row["condition_id"],
                    resolution_source=result.resolution_source,
                    yes_conditions=result.yes_conditions,
                    no_conditions=result.no_conditions,
                    key_dates=result.key_dates,
                    ambiguity_score=result.ambiguity_score,
                    ambiguity_reasons=[],  # v1 field, kept for backwards compatibility
                    unfalsifiable_flag=result.unfalsifiable_flag,
                    edge_cases=result.edge_cases,
                    dispute_risk_notes=result.dispute_risk_notes,
                    recommended_evidence_to_monitor=result.recommended_evidence_to_monitor,
                    parsed_at=datetime.utcnow(),
                    model_used=provider.model_name,
                )
                db.upsert_rules_structured(rules)
                parsed_count += 1

                progress.advance(task)

            except Exception as e:
                logger.error(
                    f"Failed to parse rules for {row['condition_id']}: {e}"
                )
                progress.advance(task)
                continue

    logger.info(f"Parsed {parsed_count} markets successfully")
    console.print(f"[green]✓[/green] Parsed {parsed_count} markets")
    return parsed_count
