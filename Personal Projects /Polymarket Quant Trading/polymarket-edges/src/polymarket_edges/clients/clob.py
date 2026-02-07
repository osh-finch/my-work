"""Client for Polymarket CLOB API with depth capture (v2)."""

import asyncio
import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from polymarket_edges.config import settings
from polymarket_edges.models import CLOBOrderBookSummary

logger = logging.getLogger(__name__)


class CLOBClient:
    """Async client for Polymarket CLOB API with order book depth."""

    def __init__(
        self,
        base_url: str | None = None,
        rate_limit: float | None = None,
        timeout: float = 30.0,
        depth_levels: int | None = None,
    ):
        """Initialise CLOB API client.

        Args:
            base_url: API base URL (defaults to config)
            rate_limit: Requests per second (defaults to config)
            timeout: Request timeout in seconds
            depth_levels: Number of levels to capture per side (defaults to config)
        """
        self.base_url = (base_url or settings.clob_base_url).rstrip("/")
        self.rate_limit = rate_limit or settings.clob_rate_limit
        self.timeout = timeout
        self.depth_levels = depth_levels or settings.orderbook_depth_levels
        self._last_request_time = 0.0
        self._rate_limiter = asyncio.Semaphore(1)

    async def _rate_limit_wait(self) -> None:
        """Enforce rate limiting between requests."""
        async with self._rate_limiter:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self._last_request_time
            min_interval = 1.0 / self.rate_limit

            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)

            self._last_request_time = asyncio.get_event_loop().time()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make GET request with rate limiting and retries."""
        await self._rate_limit_wait()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.debug(f"GET {url} with params {params}")
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def get_order_book(self, token_id: str, with_depth: bool = True) -> CLOBOrderBookSummary | None:
        """Fetch order book for a token with depth.

        Args:
            token_id: Token ID (asset ID)
            with_depth: Whether to capture depth (default True)

        Returns:
            CLOBOrderBookSummary or None if not available
        """
        try:
            # CLOB API endpoint for order book
            data = await self._get("/book", params={"token_id": token_id})

            # The API returns the order book data
            if not data:
                logger.warning(f"Empty order book response for token {token_id}")
                return None

            # Extract bids and asks, limiting to depth_levels
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if with_depth:
                bids = bids[: self.depth_levels]
                asks = asks[: self.depth_levels]
            else:
                # Just top of book
                bids = bids[:1] if bids else []
                asks = asks[:1] if asks else []

            # Build summary from response
            summary = CLOBOrderBookSummary(
                market=data.get("market", ""),
                asset_id=token_id,
                timestamp=data.get("timestamp", 0),
                hash=data.get("hash"),
                bids=bids,
                asks=asks,
            )

            logger.debug(
                f"Token {token_id}: bid={summary.best_bid}, ask={summary.best_ask}, "
                f"depth={len(bids)} bids, {len(asks)} asks"
            )
            return summary

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Order book not found for token {token_id}")
                return None
            logger.error(
                f"HTTP error fetching order book for {token_id}: "
                f"{e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching order book for {token_id}: {e}")
            return None

    async def get_order_books_batch(
        self, token_ids: list[str], max_concurrent: int = 5, with_depth: bool = True
    ) -> dict[str, CLOBOrderBookSummary]:
        """Fetch order books for multiple tokens concurrently.

        Args:
            token_ids: List of token IDs
            max_concurrent: Maximum concurrent requests
            with_depth: Whether to capture depth

        Returns:
            Dictionary mapping token_id to CLOBOrderBookSummary
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(token_id: str) -> tuple[str, CLOBOrderBookSummary | None]:
            async with semaphore:
                summary = await self.get_order_book(token_id, with_depth=with_depth)
                return token_id, summary

        tasks = [fetch_with_semaphore(token_id) for token_id in token_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        order_books = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch fetch: {result}")
                continue

            token_id, summary = result
            if summary:
                order_books[token_id] = summary

        logger.info(
            f"Fetched {len(order_books)}/{len(token_ids)} order books successfully"
        )
        return order_books
