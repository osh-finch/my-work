#!/usr/bin/env python3
"""Ingest specific markets by their slugs (from Polymarket URLs)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from polymarket_edges.clients.gamma import GammaClient
from polymarket_edges.database import Database
from polymarket_edges.models import Market, Outcome
from datetime import datetime
import json

console = Console()


async def fetch_market_by_slug(slug: str) -> dict | None:
    """Fetch market data from Gamma API by slug.

    Args:
        slug: Market slug from URL (e.g., 'presidential-election-winner-2024')

    Returns:
        Market data dict or None
    """
    try:
        client = GammaClient()

        # Try fetching markets and search for matching slug
        # The Gamma API might not have a direct slug endpoint, so we try different approaches
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            # Try direct API call with slug
            response = await http_client.get(
                f"{client.base_url}/markets",
                params={"slug": slug}
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif isinstance(data, dict) and data.get('condition_id'):
                    return data

    except Exception as e:
        console.print(f"[dim]Error fetching {slug}: {e}[/dim]")

    return None


async def ingest_market_by_slug(db: Database, slug: str) -> bool:
    """Ingest a single market by its slug.

    Args:
        db: Database instance
        slug: Market slug

    Returns:
        True if successful
    """
    console.print(f"Fetching: [cyan]{slug}[/cyan]")

    market_data = await fetch_market_by_slug(slug)

    if not market_data:
        console.print(f"  [red]✗[/red] Could not fetch market")
        return False

    try:
        # Store raw response
        condition_id = market_data['condition_id'] or market_data.get('id')
        db.upsert_raw_gamma(condition_id, market_data)

        # Store normalized market
        market = Market(
            condition_id=condition_id,
            question=market_data['question'],
            description=market_data.get('description'),
            market_slug=market_data.get('market_slug') or market_data.get('marketSlug') or slug,
            end_date_iso=market_data.get('end_date_iso') or market_data.get('endDateIso'),
            active=market_data.get('active', True),
            closed=market_data.get('closed', False),
            ingested_at=datetime.utcnow(),
        )
        db.upsert_market(market)

        # Store outcomes
        clob_token_ids = market_data.get('clob_token_ids') or market_data.get('clobTokenIds')
        if clob_token_ids and clob_token_ids != "null":
            try:
                token_ids = json.loads(clob_token_ids) if isinstance(clob_token_ids, str) else clob_token_ids
                outcomes = market_data.get('outcomes', ["Yes", "No"])

                for i, token_id in enumerate(token_ids):
                    outcome_name = outcomes[i] if i < len(outcomes) else f"Outcome {i+1}"
                    outcome = Outcome(
                        token_id=str(token_id),
                        condition_id=condition_id,
                        outcome=outcome_name,
                        winner=False,
                    )
                    db.upsert_outcome(outcome)
            except Exception as e:
                console.print(f"  [yellow]⚠[/yellow] Could not parse tokens: {e}")

        console.print(f"  [green]✓[/green] Ingested: {market_data['question'][:50]}...")
        return True

    except Exception as e:
        console.print(f"  [red]✗[/red] Error: {e}")
        return False


async def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest specific markets by their slugs",
        epilog="Example: python ingest_by_slug.py presidential-election-winner-2024 trump-popular-vote"
    )
    parser.add_argument("slugs", nargs="+", help="Market slugs from Polymarket URLs")

    args = parser.parse_args()

    console.print("[bold blue]Ingesting specific markets by slug[/bold blue]\n")

    db = Database()
    success_count = 0

    for slug in args.slugs:
        if await ingest_market_by_slug(db, slug):
            success_count += 1
        await asyncio.sleep(0.5)  # Rate limiting

    console.print(f"\n[bold]Result:[/bold] Ingested {success_count}/{len(args.slugs)} markets")

    if success_count > 0:
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. polymarket-edges update-orderbooks")
        console.print("  2. polymarket-edges compute-execution")
        console.print("  3. polymarket-edges score-v2")

    db.close()


if __name__ == "__main__":
    asyncio.run(main())
