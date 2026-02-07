#!/usr/bin/env python3
"""Fetch trending markets from Polymarket's website."""

import asyncio
import re
from typing import List, Set

import httpx
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()


async def fetch_trending_market_slugs() -> List[str]:
    """Scrape Polymarket trending page for market slugs.

    Returns:
        List of market slugs (e.g., 'presidential-election-winner-2024')
    """
    console.print("[bold blue]Fetching trending markets from Polymarket.com...[/bold blue]")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch the trending page
            response = await client.get("https://polymarket.com")
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find market links (they typically contain /event/ or /market/)
            market_slugs = set()

            # Look for links to markets
            for link in soup.find_all('a', href=True):
                href = link['href']

                # Extract market slugs from various URL patterns
                # Pattern: /event/market-slug or /market/market-slug
                if '/event/' in href or '/market/' in href:
                    # Extract slug
                    match = re.search(r'/(event|market)/([a-z0-9-]+)', href)
                    if match:
                        slug = match.group(2)
                        market_slugs.add(slug)

            console.print(f"[green]Found {len(market_slugs)} trending market slugs[/green]")
            return list(market_slugs)

    except Exception as e:
        console.print(f"[red]Error fetching trending markets: {e}[/red]")
        return []


async def get_condition_id_from_slug(slug: str) -> str | None:
    """Get condition_id from market slug using Gamma API.

    Args:
        slug: Market slug (e.g., 'presidential-election-winner-2024')

    Returns:
        condition_id or None
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Try to fetch market by slug
            # The Gamma API might have an endpoint for this, or we search
            response = await client.get(
                f"https://gamma-api.polymarket.com/markets",
                params={"slug": slug}
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data:
                    return data[0].get('condition_id')
                elif isinstance(data, dict):
                    return data.get('condition_id')

    except Exception as e:
        console.print(f"[dim]Could not fetch {slug}: {e}[/dim]")

    return None


async def main():
    """Main function."""
    # Fetch trending market slugs
    slugs = await fetch_trending_market_slugs()

    if not slugs:
        console.print("[yellow]No trending markets found. Try manual approach.[/yellow]")
        console.print("\nAlternative: Visit https://polymarket.com and copy market slugs manually")
        console.print("Then use: polymarket-edges ingest --condition-ids <id1>,<id2>...")
        return

    console.print(f"\n[bold]Trending market slugs:[/bold]")
    for i, slug in enumerate(slugs[:20], 1):
        console.print(f"  {i}. {slug}")

    # Try to get condition IDs (this might not work depending on API)
    console.print("\n[bold]Attempting to fetch condition IDs...[/bold]")
    condition_ids = []

    for slug in slugs[:20]:  # Limit to first 20
        cid = await get_condition_id_from_slug(slug)
        if cid:
            condition_ids.append(cid)
            console.print(f"[green]✓[/green] {slug}: {cid}")
        await asyncio.sleep(0.1)  # Rate limiting

    if condition_ids:
        console.print(f"\n[bold green]Found {len(condition_ids)} condition IDs[/bold green]")

        # Save to file
        with open('trending_markets.txt', 'w') as f:
            for cid in condition_ids:
                f.write(f"{cid}\n")

        console.print("Saved to: trending_markets.txt")
        console.print("\n[bold]Next: Ingest these specific markets[/bold]")
        console.print("  1. Modify ingest.py to accept condition_id filter")
        console.print("  2. Or manually add them to your database")
    else:
        console.print("\n[yellow]Could not fetch condition IDs automatically[/yellow]")
        console.print("Manual approach needed: Copy market URLs from Polymarket.com")


if __name__ == "__main__":
    asyncio.run(main())
