"""CLI interface for Polymarket Edges v2."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from polymarket_edges.config import settings
from polymarket_edges.database import Database
from polymarket_edges.ingest import (
    ingest_markets,
    update_quotes as update_quotes_impl,
    parse_rules as parse_rules_impl,
)
from polymarket_edges.scoring_v1 import compute_scores
from polymarket_edges.selection import load_selected_markets
from polymarket_edges.workflows import (
    update_orderbooks_v2,
    compute_execution_metrics,
    detect_constraints,
    compute_features,
    compute_beliefs,
    build_reports,
    score_v2_outcomes,
)

app = typer.Typer(
    name="polymarket-edges",
    help="Production-ready execution-aware analytics for Polymarket data (v2)",
    add_completion=False,
)
console = Console()


@app.callback()
def main_callback():
    """Initialise logging and configuration."""
    settings.setup_logging()


@app.command()
def ingest(
    max_pages: int = typer.Option(
        10,
        "--max-pages",
        "-m",
        help="Maximum number of pages to fetch from Gamma API",
    ),
):
    """Fetch active markets from Gamma API and store locally."""
    console.print("[bold blue]Polymarket Edges v2 - Market Ingestion[/bold blue]\n")

    db = None
    try:
        db = Database()
        count = asyncio.run(ingest_markets(db, max_pages=max_pages))

        console.print(f"\n[green]Success![/green] Ingested [bold]{count}[/bold] markets.")
        console.print("\n[bold]Next steps (v2 pipeline):[/bold]")
        console.print("  1. [cyan]polymarket-edges update-orderbooks[/cyan] - Capture order book depth")
        console.print("  2. [cyan]polymarket-edges compute-execution[/cyan] - Simulate execution")
        console.print("  3. [cyan]polymarket-edges detect-constraints[/cyan] - Find arbitrage")
        console.print("  4. [cyan]polymarket-edges compute-features[/cyan] - Extract regime features")
        console.print("  5. [cyan]polymarket-edges compute-beliefs[/cyan] - Bayesian filtering")
        console.print("  6. [cyan]polymarket-edges parse-rules[/cyan] - LLM analysis")
        console.print("  7. [cyan]polymarket-edges build-reports[/cyan] - Generate reports")
        console.print("  8. [cyan]polymarket-edges score-v2[/cyan] - Multi-component scoring")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


# v2 Commands

@app.command(name="update-orderbooks")
def update_orderbooks(
    levels: int = typer.Option(
        None,
        "--levels",
        "-l",
        help="Number of order book levels to capture per side (default: 30)",
    ),
    max_concurrent: int = typer.Option(
        5,
        "--max-concurrent",
        "-c",
        help="Maximum concurrent requests to CLOB API",
    ),
    selected: str = typer.Option(
        None,
        "--selected",
        help="Path to JSON file with selected markets from poly_top",
    ),
):
    """Capture order book depth (v2).

    Fetches N levels of bids and asks for realistic execution simulation.
    """
    console.print("[bold blue]Polymarket Edges v2 - Order Book Depth Capture[/bold blue]\n")

    db = None
    try:
        db = Database()
        levels_to_use = levels or settings.orderbook_depth_levels

        # Load selected markets if provided
        selected_conditions = load_selected_markets(selected)
        if selected_conditions:
            console.print(f"[cyan]Filtering to {len(selected_conditions)} selected markets[/cyan]")

        console.print(f"Capturing {levels_to_use} levels per side...")

        count = asyncio.run(update_orderbooks_v2(db, levels=levels, max_concurrent=max_concurrent, selected_conditions=selected_conditions))

        console.print(f"\n[green]Success![/green] Updated [bold]{count}[/bold] order books with depth.")
        console.print("\n[bold]Next step:[/bold]")
        console.print("  [cyan]polymarket-edges compute-execution[/cyan] - Simulate execution across sizes")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command(name="compute-execution")
def compute_execution(
    sizes: str = typer.Option(
        None,
        "--sizes",
        "-s",
        help="Comma-separated trade sizes in USD (e.g., '25,100,250,1000')",
    ),
    selected: str = typer.Option(
        None,
        "--selected",
        help="Path to JSON file with selected markets from poly_top",
    ),
):
    """Compute execution metrics at multiple trade sizes (v2).

    Simulates VWAP, liquidity tax, slippage, and fill ratios.
    """
    console.print("[bold blue]Polymarket Edges v2 - Execution Metrics[/bold blue]\n")

    db = None
    try:
        db = Database()

        # Parse sizes
        size_buckets = None
        if sizes:
            size_buckets = [float(s.strip()) for s in sizes.split(",")]
        else:
            size_buckets = settings.trade_size_buckets

        # Load selected markets if provided
        selected_conditions = load_selected_markets(selected)
        if selected_conditions:
            console.print(f"[cyan]Filtering to {len(selected_conditions)} selected markets[/cyan]")

        console.print(f"Computing execution metrics for sizes: {size_buckets}")

        count = asyncio.run(compute_execution_metrics(db, sizes=size_buckets, selected_conditions=selected_conditions))

        console.print(f"\n[green]Success![/green] Computed [bold]{count}[/bold] execution metrics.")
        console.print("\n[bold]Next step:[/bold]")
        console.print("  [cyan]polymarket-edges detect-constraints[/cyan] - Find arbitrage opportunities")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command(name="detect-constraints")
def detect_constraints_cmd(
    size: float = typer.Option(
        None,
        "--size",
        "-s",
        help="Trade size for constraint detection (default: reference size)",
    ),
    selected: str = typer.Option(
        None,
        "--selected",
        help="Path to JSON file with selected markets from poly_top",
    ),
):
    """Detect constraint violations and arbitrage opportunities (v2).

    Checks complete set pricing in binary markets.
    """
    console.print("[bold blue]Polymarket Edges v2 - Constraint Detection[/bold blue]\n")

    db = None
    try:
        db = Database()
        size_to_use = size or settings.reference_size_bucket

        # Load selected markets if provided
        selected_conditions = load_selected_markets(selected)
        if selected_conditions:
            console.print(f"[cyan]Filtering to {len(selected_conditions)} selected markets[/cyan]")

        console.print(f"Detecting constraints at size: ${size_to_use}")

        count = asyncio.run(detect_constraints(db, size=size_to_use, selected_conditions=selected_conditions))

        console.print(f"\n[green]Success![/green] Detected [bold]{count}[/bold] constraint violations.")
        if count > 0:
            console.print("[yellow]⚠ Arbitrage opportunities found! Check dashboard for details.[/yellow]")
        console.print("\n[bold]Next step:[/bold]")
        console.print("  [cyan]polymarket-edges compute-features[/cyan] - Extract regime features")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command(name="compute-features")
def compute_features_cmd(
    window: str = typer.Option(
        "24h",
        "--window",
        "-w",
        help="Time window for features ('24h' or '7d')",
    ),
    selected: str = typer.Option(
        None,
        "--selected",
        help="Path to JSON file with selected markets from poly_top",
    ),
):
    """Compute regime and lifecycle features (v2).

    Extracts spread trends, volatility, market age, time-to-resolution.
    """
    console.print("[bold blue]Polymarket Edges v2 - Regime Features[/bold blue]\n")

    db = None
    try:
        db = Database()

        # Load selected markets if provided
        selected_conditions = load_selected_markets(selected)
        if selected_conditions:
            console.print(f"[cyan]Filtering to {len(selected_conditions)} selected markets[/cyan]")

        console.print(f"Computing features with window: {window}")

        count = asyncio.run(compute_features(db, window=window, selected_conditions=selected_conditions))

        console.print(f"\n[green]Success![/green] Computed [bold]{count}[/bold] regime features.")
        console.print("\n[bold]Next step:[/bold]")
        console.print("  [cyan]polymarket-edges compute-beliefs[/cyan] - Apply Bayesian filter")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command(name="compute-beliefs")
def compute_beliefs_cmd(
    selected: str = typer.Option(
        None,
        "--selected",
        help="Path to JSON file with selected markets from poly_top",
    ),
):
    """Compute Bayesian belief estimates (v2).

    Filters noisy market prices using a state-space model.
    """
    console.print("[bold blue]Polymarket Edges v2 - Bayesian Belief Filter[/bold blue]\n")

    db = None
    try:
        db = Database()

        # Load selected markets if provided
        selected_conditions = load_selected_markets(selected)
        if selected_conditions:
            console.print(f"[cyan]Filtering to {len(selected_conditions)} selected markets[/cyan]")

        count = asyncio.run(compute_beliefs(db, selected_conditions=selected_conditions))

        console.print(f"\n[green]Success![/green] Computed [bold]{count}[/bold] belief estimates.")
        console.print("\n[bold]Next step:[/bold]")
        console.print("  [cyan]polymarket-edges parse-rules[/cyan] - Extract structured rules")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command()
def parse_rules(
    provider: str = typer.Option(
        "local",
        "--provider",
        "-p",
        help="LLM provider to use: 'openai' or 'local'",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of markets to parse (None = all unparsed)",
    ),
    selected: str = typer.Option(
        None,
        "--selected",
        help="Path to JSON file with selected markets from poly_top",
    ),
):
    """Parse market rules using LLM (v2 with enhanced extraction).

    Extracts structured JSON with resolution criteria, edge cases, dispute risks.
    """
    console.print("[bold blue]Polymarket Edges v2 - Rules Parsing[/bold blue]\n")

    if provider == "openai" and not settings.openai_api_key:
        console.print(
            "[red]Error:[/red] OpenAI provider requires OPENAI_API_KEY environment variable.\n"
            "Set it in your .env file or use --provider local"
        )
        raise typer.Exit(1)

    db = None
    try:
        db = Database()

        # Load selected markets if provided
        selected_conditions = load_selected_markets(selected)
        if selected_conditions:
            console.print(f"[cyan]Filtering to {len(selected_conditions)} selected markets[/cyan]")

        count = asyncio.run(parse_rules_impl(db, provider_type=provider, limit=limit, selected_conditions=selected_conditions))

        console.print(f"\n[green]Success![/green] Parsed [bold]{count}[/bold] markets.")
        console.print("\n[bold]Next step:[/bold]")
        console.print("  [cyan]polymarket-edges build-reports[/cyan] - Generate markdown reports")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command(name="build-reports")
def build_reports_cmd(
    provider: str = typer.Option(
        "local",
        "--provider",
        "-p",
        help="LLM provider to use: 'openai' or 'local'",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of reports to generate",
    ),
    selected: str = typer.Option(
        None,
        "--selected",
        help="Path to JSON file with selected markets from poly_top",
    ),
):
    """Generate human-readable markdown reports (v2).

    Creates grounded narratives with execution analysis and risk assessment.
    """
    console.print("[bold blue]Polymarket Edges v2 - Report Generation[/bold blue]\n")

    if provider == "openai" and not settings.openai_api_key:
        console.print(
            "[red]Error:[/red] OpenAI provider requires OPENAI_API_KEY.\n"
            "Use --provider local for placeholder reports"
        )
        raise typer.Exit(1)

    db = None
    try:
        db = Database()

        # Load selected markets if provided
        selected_conditions = load_selected_markets(selected)
        if selected_conditions:
            console.print(f"[cyan]Filtering to {len(selected_conditions)} selected markets[/cyan]")

        console.print(f"Generating reports with provider: {provider}")

        count = asyncio.run(build_reports(db, provider_type=provider, limit=limit, selected_conditions=selected_conditions))

        console.print(f"\n[green]Success![/green] Generated [bold]{count}[/bold] reports.")
        console.print("\n[bold]Next step:[/bold]")
        console.print("  [cyan]polymarket-edges score-v2[/cyan] - Multi-component scoring")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command(name="score-v2")
def score_v2_cmd(
    selected: str = typer.Option(
        None,
        "--selected",
        help="Path to JSON file with selected markets from poly_top",
    ),
):
    """Compute v2 multi-component scores.

    Combines execution quality + rules clarity + constraint edge + regime opportunity.
    """
    console.print("[bold blue]Polymarket Edges v2 - Multi-Component Scoring[/bold blue]\n")

    db = None
    try:
        db = Database()

        # Load selected markets if provided
        selected_conditions = load_selected_markets(selected)
        if selected_conditions:
            console.print(f"[cyan]Filtering to {len(selected_conditions)} selected markets[/cyan]")

        console.print("Computing scores with weights:")
        console.print(f"  Execution: {settings.score_weight_execution:.2f}")
        console.print(f"  Rules: {settings.score_weight_rules:.2f}")
        console.print(f"  Constraint: {settings.score_weight_constraint:.2f}")
        console.print(f"  Regime: {settings.score_weight_regime:.2f}")

        count = asyncio.run(score_v2_outcomes(db, selected_conditions=selected_conditions))

        console.print(f"\n[green]Success![/green] Scored [bold]{count}[/bold] outcomes.")
        console.print("\n[bold]View results:[/bold]")
        console.print("  [cyan]polymarket-edges show-top-v2[/cyan] - Display top ranked markets")
        console.print("  [cyan]polymarket-edges serve[/cyan] - Launch dashboard")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command(name="show-top-v2")
def show_top_v2(
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Number of top markets to display",
    ),
):
    """Display top ranked markets (v2 scores).

    Shows execution quality, rules risk, constraint edge, and combined score.
    """
    import pandas as pd

    console.print("[bold blue]Polymarket Edges v2 - Top Ranked Markets[/bold blue]\n")

    db = None
    try:
        db = Database()
        scores = db.get_latest_v2_scores(limit=limit)

        if scores.empty:
            console.print("[yellow]No v2 scores found. Run the v2 pipeline first:[/yellow]")
            console.print("  1. polymarket-edges ingest")
            console.print("  2. polymarket-edges update-orderbooks")
            console.print("  3. polymarket-edges compute-execution")
            console.print("  4. polymarket-edges detect-constraints")
            console.print("  5. polymarket-edges compute-features")
            console.print("  6. polymarket-edges compute-beliefs")
            console.print("  7. polymarket-edges parse-rules")
            console.print("  8. polymarket-edges build-reports")
            console.print("  9. polymarket-edges score-v2")
            raise typer.Exit(0)

        # Create rich table
        table = Table(title=f"Top {len(scores)} Markets (v2)", show_lines=True)
        table.add_column("Rank", style="cyan", justify="right", width=5)
        table.add_column("Market", style="white", max_width=40)
        table.add_column("Outcome", style="yellow", width=6)
        table.add_column("Mid", style="green", justify="right", width=6)
        table.add_column("LiqTax", style="magenta", justify="right", width=7)
        table.add_column("ExecQ", style="blue", justify="right", width=5)
        table.add_column("RuleR", style="red", justify="right", width=5)
        table.add_column("ConE", style="cyan", justify="right", width=5)
        table.add_column("RegO", style="yellow", justify="right", width=5)
        table.add_column("Combined", style="bold green", justify="right", width=8)

        for idx, row in scores.iterrows():
            table.add_row(
                str(idx + 1),
                row["question"][:40] + "..." if len(row["question"]) > 40 else row["question"],
                row["outcome"],
                f"{row['mid_price']:.3f}" if pd.notna(row["mid_price"]) else "N/A",
                f"{row['liquidity_tax']:.4f}" if pd.notna(row["liquidity_tax"]) else "N/A",
                f"{row['execution_quality_score']:.1f}",
                f"{row['rules_risk_score']:.1f}",
                f"{row['constraint_edge_score']:.1f}",
                f"{row['regime_opportunity_score']:.1f}",
                f"{row['combined_score']:.1f}",
            )

        console.print(table)
        console.print("\n[dim]Legend: ExecQ=Execution Quality, RuleR=Rules Risk, ConE=Constraint Edge, RegO=Regime Opportunity[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


# Legacy v1 Commands (for compatibility)

@app.command(name="update-quotes")
def update_quotes(
    max_concurrent: int = typer.Option(
        5,
        "--max-concurrent",
        "-c",
        help="Maximum concurrent requests to CLOB API",
    ),
):
    """[v1] Fetch top-of-book quotes (legacy command)."""
    console.print("[bold blue]Polymarket Edges v1 - Quote Update[/bold blue]\n")
    console.print("[yellow]Note: This is the v1 command. For v2 with depth, use 'update-orderbooks'[/yellow]\n")

    db = None
    try:
        db = Database()
        count = asyncio.run(update_quotes_impl(db, max_concurrent=max_concurrent))

        console.print(f"\n[green]Success![/green] Updated [bold]{count}[/bold] quotes.")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command(name="score")
def score():
    """[v1] Compute v1 scores (legacy command)."""
    console.print("[bold blue]Polymarket Edges v1 - Scoring[/bold blue]\n")
    console.print("[yellow]Note: This is the v1 command. For v2 multi-component scoring, use 'score-v2'[/yellow]\n")

    db = None
    try:
        db = Database()
        count = compute_scores(db)

        console.print(f"\n[green]Success![/green] Computed [bold]{count}[/bold] v1 scores.")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command(name="show-top")
def show_top(
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Number of top markets to display",
    ),
):
    """[v1] Display top markets with v1 scores (legacy command)."""
    import pandas as pd

    console.print("[bold blue]Polymarket Edges v1 - Top Markets[/bold blue]\n")
    console.print("[yellow]Note: This shows v1 scores. For v2 scores, use 'show-top-v2'[/yellow]\n")

    db = None
    try:
        db = Database()
        scores = db.get_latest_scores(limit=limit)

        if scores.empty:
            console.print("[yellow]No v1 scores found.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=f"Top {len(scores)} Markets (v1)", show_lines=True)
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Market", style="white", max_width=50)
        table.add_column("Outcome", style="yellow")
        table.add_column("Mid", style="green", justify="right")
        table.add_column("Spread", style="magenta", justify="right")
        table.add_column("Trade", style="blue", justify="right")
        table.add_column("Risk", style="red", justify="right")
        table.add_column("Combined", style="bold green", justify="right")

        for idx, row in scores.iterrows():
            table.add_row(
                str(idx + 1),
                row["question"][:50] + "..." if len(row["question"]) > 50 else row["question"],
                row["outcome"],
                f"{row['mid_price']:.3f}" if pd.notna(row["mid_price"]) else "N/A",
                f"{row['spread']:.4f}" if pd.notna(row["spread"]) else "N/A",
                f"{row['tradability_score']:.1f}",
                f"{row['rules_risk_score']:.1f}",
                f"{row['combined_score']:.1f}",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


@app.command()
def serve(
    port: int = typer.Option(8501, "--port", "-p", help="Port to run Streamlit server"),
):
    """Launch the Streamlit dashboard."""
    console.print("[bold blue]Polymarket Edges v2 - Starting Dashboard[/bold blue]\n")

    # Find dashboard script
    app_path = Path(__file__).parent.parent.parent / "apps" / "dashboard.py"

    if not app_path.exists():
        console.print(f"[red]Error:[/red] Dashboard not found at {app_path}")
        raise typer.Exit(1)

    console.print(f"Starting Streamlit dashboard on port {port}...")
    console.print(f"Dashboard will open at [cyan]http://localhost:{port}[/cyan]\n")

    import subprocess

    try:
        subprocess.run(
            ["streamlit", "run", str(app_path), "--server.port", str(port)],
            check=True,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def pipeline(
    max_pages: int = typer.Option(10, "--max-pages", help="Maximum pages to ingest"),
    provider: str = typer.Option("local", "--provider", help="LLM provider"),
    limit: int = typer.Option(None, "--limit", help="Limit markets to parse"),
):
    """Run the complete v1 pipeline (legacy).

    For v2 pipeline, run commands individually.
    """
    console.print("[bold blue]Polymarket Edges v1 - Full Pipeline[/bold blue]\n")
    console.print("[yellow]Note: This runs the v1 pipeline. For v2, run commands individually.[/yellow]\n")

    db = None
    try:
        db = Database()

        console.print("[cyan]Step 1/4:[/cyan] Ingesting markets...")
        asyncio.run(ingest_markets(db, max_pages=max_pages))

        console.print("\n[cyan]Step 2/4:[/cyan] Updating quotes...")
        asyncio.run(update_quotes_impl(db))

        console.print("\n[cyan]Step 3/4:[/cyan] Parsing rules...")
        asyncio.run(parse_rules_impl(db, provider_type=provider, limit=limit))

        console.print("\n[cyan]Step 4/4:[/cyan] Computing scores...")
        compute_scores(db)

        console.print("\n[green]✓ v1 Pipeline completed successfully![/green]")
        console.print("\nRun [cyan]polymarket-edges serve[/cyan] to view the dashboard.")

    except Exception as e:
        console.print(f"\n[red]Pipeline failed:[/red] {e}")
        raise typer.Exit(1)
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    app()
