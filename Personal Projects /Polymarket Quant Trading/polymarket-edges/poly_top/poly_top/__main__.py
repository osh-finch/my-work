"""CLI interface for poly_top."""

import argparse
import csv
import json
import logging
import sys
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from .gamma import GammaAPIError, GammaClient
from .rank import extract_market_metrics, rank_markets

console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def format_number(value: float) -> str:
    """Format number for display."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"


def format_spread(value: float) -> str:
    """Format spread as percentage."""
    return f"{value*100:.2f}%"


def output_table(markets: List[Dict[str, Any]], metric: str):
    """Output markets as formatted table.

    Args:
        markets: List of ranked markets
        metric: Ranking metric used
    """
    if not markets:
        console.print("[yellow]No markets found matching criteria[/yellow]")
        return

    table = Table(title=f"Top Markets by {metric}", show_lines=False)

    # Add columns
    table.add_column("Rank", justify="right", style="cyan", width=5)
    table.add_column("Question", style="white", max_width=50)
    table.add_column("24h Vol", justify="right", style="green")
    table.add_column("Total Vol", justify="right", style="blue")
    table.add_column("Liquidity", justify="right", style="magenta")
    table.add_column("Spread", justify="right", style="yellow")
    table.add_column("Competitive", justify="right", style="red")

    if metric == "composite":
        table.add_column("Score", justify="right", style="bold green")

    # Add rows
    for i, market in enumerate(markets, 1):
        metrics = extract_market_metrics(market)
        question = market.get("question", "N/A")

        # Truncate long questions
        if len(question) > 47:
            question = question[:47] + "..."

        row = [
            str(i),
            question,
            format_number(metrics["volume24hr"]),
            format_number(metrics["volumeNum"]),
            format_number(metrics["liquidityNum"]),
            format_spread(metrics["spread"]),
            f"{metrics['competitive']:.2f}",
        ]

        if metric == "composite":
            score = market.get("_composite_score", 0.0)
            row.append(f"{score:.4f}")

        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]Showing {len(markets)} markets[/dim]")


def output_json(markets: List[Dict[str, Any]]):
    """Output markets as JSON array.

    Args:
        markets: List of ranked markets
    """
    # Remove internal fields
    cleaned = []
    for market in markets:
        market_copy = market.copy()
        market_copy.pop("_composite_score", None)
        cleaned.append(market_copy)

    print(json.dumps(cleaned, indent=2))


def output_csv(markets: List[Dict[str, Any]]):
    """Output markets as CSV to stdout.

    Args:
        markets: List of ranked markets
    """
    if not markets:
        return

    # Define columns
    columns = [
        "question",
        "volume24hr",
        "volumeNum",
        "liquidityNum",
        "spread",
        "competitive",
        "endDateIso",
    ]

    writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()

    for market in markets:
        # Extract and clean data
        row = {
            "question": market.get("question", ""),
            "volume24hr": extract_market_metrics(market)["volume24hr"],
            "volumeNum": extract_market_metrics(market)["volumeNum"],
            "liquidityNum": extract_market_metrics(market)["liquidityNum"],
            "spread": extract_market_metrics(market)["spread"],
            "competitive": extract_market_metrics(market)["competitive"],
            "endDateIso": market.get("endDateIso") or market.get("end_date_iso", ""),
        }
        writer.writerow(row)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rank Polymarket markets by volume, liquidity, and competitiveness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Top markets by 24h volume
  python -m poly_top --metric volume24hr --limit 20

  # Most liquid markets with thresholds
  python -m poly_top --metric liquidityNum --min-liquidity 10000 --limit 30

  # Tightest spreads among liquid markets
  python -m poly_top --metric tight_spread --min-liquidity 5000 --limit 15

  # Composite ranking
  python -m poly_top --metric composite --pages 3 --limit 50

  # JSON output
  python -m poly_top --metric volume24hr --format json --limit 10
        """,
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["volume24hr", "volumeNum", "liquidityNum", "competitive", "tight_spread", "composite"],
        default="volume24hr",
        help="Ranking metric (default: volume24hr)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of markets to display (default: 50)",
    )

    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=0.0,
        help="Minimum liquidity threshold (default: 0)",
    )

    parser.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="Minimum total volume threshold (default: 0)",
    )

    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.0,
        help="Minimum Yes probability (0-1, filters out near-certain No outcomes, default: 0)",
    )

    parser.add_argument(
        "--max-prob",
        type=float,
        default=1.0,
        help="Maximum Yes probability (0-1, filters out near-certain Yes outcomes, default: 1)",
    )

    parser.add_argument(
        "--active-only",
        action="store_true",
        default=True,
        help="Only include active markets (default: True)",
    )

    parser.add_argument(
        "--include-closed",
        action="store_true",
        default=False,
        help="Include closed markets (default: False)",
    )

    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of pages to fetch (default: 1)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="API request timeout in seconds (default: 30)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Main CLI entrypoint."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    try:
        # Determine API ordering
        # For tight_spread, we fetch by liquidity first, then sort locally
        # For composite, we fetch by volume and sort locally
        if args.metric == "tight_spread":
            api_order = "liquidityNum"
        elif args.metric == "composite":
            api_order = "volume24hr"
        elif args.metric == "competitive":
            api_order = None  # Competitive might not be supported for ordering
        else:
            api_order = args.metric

        # Fetch markets
        with GammaClient(timeout=args.timeout) as client:
            # Only show progress for table format (not JSON/CSV)
            if args.format == "table":
                console.print(f"[cyan]Fetching markets from Gamma API...[/cyan]")

            markets = client.fetch_markets_paginated(
                pages=args.pages,
                limit=100,  # Fetch 100 per page for better candidate set
                active=args.active_only if not args.include_closed else None,
                closed=args.include_closed if args.include_closed else False,
                order=api_order,
                ascending=False,
            )

            if not markets:
                if args.format == "table":
                    console.print("[yellow]No markets returned from API[/yellow]")
                return

            logger.info(f"Fetched {len(markets)} markets")

        # Rank markets
        ranked = rank_markets(
            markets,
            metric=args.metric,
            limit=args.limit,
            min_liquidity=args.min_liquidity,
            min_volume=args.min_volume,
            min_prob=args.min_prob,
            max_prob=args.max_prob,
        )

        # Output
        if args.format == "table":
            output_table(ranked, args.metric)
        elif args.format == "json":
            output_json(ranked)
        elif args.format == "csv":
            output_csv(ranked)

    except GammaAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
