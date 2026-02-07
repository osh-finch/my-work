"""Polymarket Gamma Markets API client with retry logic."""

import logging
from typing import Any, Dict, List

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class GammaAPIError(Exception):
    """Exception raised for Gamma API errors."""

    pass


class GammaClient:
    """Client for Polymarket Gamma Markets API."""

    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, timeout: float = 30.0):
        """Initialize the Gamma API client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.client.close()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _request(
        self,
        endpoint: str,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Make HTTP request with retries.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            GammaAPIError: If API request fails
        """
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"

        try:
            logger.debug(f"GET {url} params={params}")
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            raise GammaAPIError(
                f"API request failed with status {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.TimeoutException as e:
            raise GammaAPIError(f"Request timed out after {self.timeout}s") from e
        except httpx.NetworkError as e:
            raise GammaAPIError(f"Network error: {e}") from e
        except Exception as e:
            raise GammaAPIError(f"Unexpected error: {e}") from e

    def fetch_markets(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        active: bool | None = None,
        closed: bool | None = None,
        order: str | None = None,
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch markets from Gamma API.

        Args:
            limit: Number of markets to fetch per page
            offset: Pagination offset
            active: Filter for active markets
            closed: Filter for closed markets
            order: Field to order by (e.g., 'volume24hr', 'liquidityNum')
            ascending: Sort ascending if True, descending if False

        Returns:
            List of market dicts

        Raises:
            GammaAPIError: If API request fails
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if order:
            params["order"] = order
            params["ascending"] = str(ascending).lower()

        response = self._request("/markets", params=params)
        logger.debug(f"Gamma API response: {response}")

        # Handle both list and dict responses
        if isinstance(response, list):
            return response
        elif isinstance(response, dict) and "data" in response:
            return response["data"]
        else:
            logger.warning(f"Unexpected response format: {type(response)}")
            return []

    def fetch_markets_paginated(
        self,
        *,
        pages: int = 1,
        limit: int = 100,
        active: bool | None = None,
        closed: bool | None = None,
        order: str | None = None,
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch multiple pages of markets and deduplicate.

        Args:
            pages: Number of pages to fetch
            limit: Markets per page
            active: Filter for active markets
            closed: Filter for closed markets
            order: Field to order by
            ascending: Sort direction

        Returns:
            Deduplicated list of markets

        Raises:
            GammaAPIError: If API request fails
        """
        all_markets = []
        seen_ids = set()

        for page in range(pages):
            offset = page * limit
            logger.info(f"Fetching page {page + 1}/{pages} (offset={offset})")

            markets = self.fetch_markets(
                limit=limit,
                offset=offset,
                active=active,
                closed=closed,
                order=order,
                ascending=ascending,
            )

            if not markets:
                logger.info(f"No more markets at page {page + 1}, stopping")
                break

            # Deduplicate by id or condition_id
            for market in markets:
                market_id = market.get("id") or market.get("condition_id") or market.get("conditionId")

                if market_id and market_id not in seen_ids:
                    seen_ids.add(market_id)
                    all_markets.append(market)

        logger.info(f"Fetched {len(all_markets)} unique markets across {pages} pages")
        return all_markets
