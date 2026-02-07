"""Client for Polymarket Gamma Markets API."""

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
from polymarket_edges.models import GammaMarket

logger = logging.getLogger(__name__)


class GammaClient:
    """Async client for Polymarket Gamma Markets API."""

    def __init__(
        self,
        base_url: str | None = None,
        rate_limit: float | None = None,
        timeout: float = 30.0,
    ):
        """Initialise Gamma API client.

        Args:
            base_url: API base URL (defaults to config)
            rate_limit: Requests per second (defaults to config)
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or settings.gamma_base_url).rstrip("/")
        self.rate_limit = rate_limit or settings.gamma_rate_limit
        self.timeout = timeout
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

    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GammaMarket]:
        """Fetch markets with pagination.

        Args:
            active: Include active markets
            closed: Include closed markets
            limit: Number of results per page
            offset: Pagination offset

        Returns:
            List of GammaMarket objects
        """
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }

        try:
            data = await self._get("/markets", params=params)

            # Handle both array response and paginated object response
            if isinstance(data, list):
                markets_data = data
            elif isinstance(data, dict) and "data" in data:
                markets_data = data["data"]
            else:
                markets_data = [data]

            markets = []
            for item in markets_data:
                try:
                    market = GammaMarket(**item)
                    markets.append(market)
                except Exception as e:
                    logger.warning(f"Failed to parse market {item.get('condition_id')}: {e}")
                    continue

            logger.info(f"Fetched {len(markets)} markets (offset={offset}, limit={limit})")
            return markets

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching markets: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            raise

    async def get_all_active_markets(self, max_pages: int = 10) -> list[GammaMarket]:
        """Fetch all active markets using pagination.

        Args:
            max_pages: Maximum number of pages to fetch (safety limit)

        Returns:
            List of all active GammaMarket objects
        """
        all_markets = []
        offset = 0
        limit = 100

        for page in range(max_pages):
            markets = await self.get_markets(active=True, closed=False, limit=limit, offset=offset)

            if not markets:
                logger.info(f"No more markets found at offset {offset}")
                break

            all_markets.extend(markets)
            offset += limit

            logger.info(f"Page {page + 1}: Total markets collected = {len(all_markets)}")

            # Stop if we got fewer results than the limit (last page)
            if len(markets) < limit:
                break

        logger.info(f"Collected {len(all_markets)} total active markets")
        return all_markets

    async def get_market_by_condition_id(self, condition_id: str) -> GammaMarket | None:
        """Fetch a specific market by condition ID.

        Args:
            condition_id: Market condition ID

        Returns:
            GammaMarket object or None if not found
        """
        try:
            # Try direct endpoint if available, otherwise search
            data = await self._get(f"/markets/{condition_id}")
            return GammaMarket(**data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Market not found: {condition_id}")
                return None
            raise
        except Exception as e:
            logger.error(f"Error fetching market {condition_id}: {e}")
            return None
