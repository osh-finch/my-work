#!/usr/bin/env python3
"""Find liquid markets by sampling order books before ingestion."""

import asyncio
import logging
from typing import List, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from polymarket_edges.clients.gamma import GammaClient
from polymarket_edges.clients.clob import CLOBClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


async def check_market_liquidity(
    clob_client: CLOBClient,
    token_id: str,
    max_spread_threshold: float = 0.10  # 10%
) -> Tuple[str, float, bool]:
    """Check if a market has acceptable liquidity.

    Args:
        clob_client: CLOB API client
        token_id: Token ID to check
        max_spread_threshold: Maximum acceptable spread (default 10%)

    Returns:
        Tuple of (token_id, spread, is_liquid)
    """
    try:
        order_book = await clob_client.get_order_book(token_id, with_depth=False)

        if not order_book or not order_book.bids or not order_book.asks:
            return (token_id, 1.0, False)

        best_bid = order_book.best_bid
        best_ask = order_book.best_ask

        if best_bid is None or best_ask is None:
            return (token_id, 1.0, False)

        spread = best_ask - best_bid
        is_liquid = spread <= max_spread_threshold

        return (token_id, spread, is_liquid)

    except Exception as e:
        logger.debug(f"Error checking {token_id}: {e}")
        return (token_id, 1.0, False)


async def find_liquid_markets(
    max_pages: int = 10,
    max_spread: float = 0.10,
    min_liquid_markets: int = 50,
) -> List[str]:
    """Find liquid markets by sampling order books.

    Args:
        max_pages: Maximum pages to scan from Gamma API
        max_spread: Maximum acceptable spread (default 10%)
        min_liquid_markets: Stop after finding this many liquid markets

    Returns:
        List of condition_ids for liquid markets
    """
    console.print(f"[bold blue]Scanning for liquid markets (spread < {max_spread*100}%)[/bold blue]\n")

    gamma_client = GammaClient()
    clob_client = CLOBClient()

    liquid_markets = []
    total_checked = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        scan_task = progress.add_task(
            f"Scanning markets (found {len(liquid_markets)} liquid)...",
            total=max_pages
        )

        for page in range(max_pages):
            # Fetch markets from Gamma API
            markets = await gamma_client.get_markets(
                active=True,
                closed=False,
                limit=100,
                offset=page * 100
            )

            if not markets:
                break

            # Extract token IDs
            token_ids = []
            market_map = {}  # token_id -> market

            for market in markets:
                if market.clob_token_ids and market.clob_token_ids != "null":
                    import json
                    try:
                        tokens = json.loads(market.clob_token_ids)
                        for token_id in tokens:
                            token_ids.append(str(token_id))
                            market_map[str(token_id)] = market
                    except:
                        pass

            # Check liquidity for these tokens
            tasks = [
                check_market_liquidity(clob_client, token_id, max_spread)
                for token_id in token_ids[:100]  # Limit to avoid rate limiting
            ]

            results = await asyncio.gather(*tasks)

            # Collect liquid markets
            for token_id, spread, is_liquid in results:
                total_checked += 1

                if is_liquid and token_id in market_map:
                    market = market_map[token_id]
                    liquid_markets.append(market.condition_id)

                    console.print(
                        f"[green]✓[/green] Found: {market.question[:60]}... "
                        f"(spread: {spread*100:.1f}%)"
                    )

                    if len(liquid_markets) >= min_liquid_markets:
                        console.print(f"\n[green]Found {len(liquid_markets)} liquid markets![/green]")
                        return list(set(liquid_markets))  # Deduplicate

            progress.update(
                scan_task,
                advance=1,
                description=f"Scanning markets (found {len(liquid_markets)} liquid, checked {total_checked})..."
            )

    console.print(f"\n[yellow]Scanned {total_checked} tokens, found {len(liquid_markets)} liquid markets[/yellow]")
    return list(set(liquid_markets))


async def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Find liquid markets on Polymarket")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to scan")
    parser.add_argument("--max-spread", type=float, default=0.10, help="Max spread threshold (default 0.10 = 10%)")
    parser.add_argument("--min-markets", type=int, default=50, help="Stop after finding this many")
    parser.add_argument("--output", type=str, help="Output file to save condition IDs")

    args = parser.parse_args()

    # Find liquid markets
    condition_ids = await find_liquid_markets(
        max_pages=args.max_pages,
        max_spread=args.max_spread,
        min_liquid_markets=args.min_markets,
    )

    # Display results
    console.print(f"\n[bold green]Found {len(condition_ids)} liquid markets[/bold green]")

    if args.output:
        with open(args.output, 'w') as f:
            for cid in condition_ids:
                f.write(f"{cid}\n")
        console.print(f"Saved to: {args.output}")

    # Show summary
    if condition_ids:
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. These markets have acceptable spreads for trading")
        console.print("  2. Re-run the full pipeline on these markets")
        console.print(f"  3. Or manually review them on Polymarket.com")


if __name__ == "__main__":
    asyncio.run(main())
