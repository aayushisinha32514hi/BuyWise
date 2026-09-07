"""
BuyWise — SerpAPI Google Shopping Provider (India)
====================================================
Concrete implementation of ShoppingSearchProvider using SerpAPI's
Google Shopping engine with gl=in (India), hl=en, currency=INR.

Free tier: 250 searches/month (recurring, no credit card).
Docs: https://serpapi.com/google-shopping-api
"""

import os
import logging
from datetime import datetime, timezone
from typing import List, Optional

import httpx

from backend.services.product_search.base import (
    ShoppingSearchProvider, ShoppingResult, ShoppingOffer
)

logger = logging.getLogger(__name__)


class SerpAPIProvider(ShoppingSearchProvider):
    """Google Shopping search via SerpAPI with India market defaults."""

    BASE_URL = "https://serpapi.com/search.json"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SERPAPI_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 10)

    def search(
        self,
        query: str,
        country: str = "in",
        language: str = "en",
        max_results: int = 30,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> List[ShoppingResult]:
        """
        Search Google Shopping via SerpAPI for Indian market results.
        Returns normalized ShoppingResult objects.
        """
        if not self.is_available():
            logger.warning("SerpAPI key not configured — cannot perform live search")
            return []

        params = {
            "engine": "google_shopping",
            "q": query,
            "gl": country,
            "hl": language,
            "location": "India",
            "google_domain": "google.co.in",
            "num": str(min(max_results, 40)),
            "api_key": self.api_key,
        }

        # SerpAPI supports tbs price filter parameter
        if min_price is not None and max_price is not None:
            params["tbs"] = f"mr:1,price:1,ppr_min:{int(min_price)},ppr_max:{int(max_price)}"
        elif max_price is not None:
            params["tbs"] = f"mr:1,price:1,ppr_max:{int(max_price)}"
        elif min_price is not None:
            params["tbs"] = f"mr:1,price:1,ppr_min:{int(min_price)}"

        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(self.BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("SerpAPI HTTP error: %s — %s", e.response.status_code, e.response.text[:300])
            return []
        except Exception as e:
            logger.error("SerpAPI request failed: %s", e)
            return []

        shopping_results = data.get("shopping_results", [])
        if not shopping_results:
            logger.info("SerpAPI returned 0 shopping results for query=%s", query)
            return []

        results: List[ShoppingResult] = []
        for item in shopping_results[:max_results]:
            result = self._parse_shopping_item(item, now_iso)
            if result and result.price > 0:
                results.append(result)

        logger.info("SerpAPI returned %d results for query=%s (country=%s)", len(results), query, country)
        return results

    def get_product_offers(self, product_id: str, page_token: str = "") -> List[ShoppingOffer]:
        """
        Fetch multi-retailer offers for a specific product using
        SerpAPI's google_immersive_product engine.
        """
        if not self.is_available() or not page_token:
            return []

        params = {
            "engine": "google_immersive_product",
            "page_token": page_token,
            "gl": "in",
            "hl": "en",
            "api_key": self.api_key,
        }

        try:
            with httpx.Client(timeout=12.0) as client:
                resp = client.get(self.BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error("SerpAPI product offers fetch failed: %s", e)
            return []

        offers: List[ShoppingOffer] = []

        # Parse online sellers from immersive product response
        sellers = data.get("sellers_results", {}).get("online_sellers", [])
        if not sellers:
            sellers = data.get("online_sellers", [])

        for seller in sellers:
            price = self._extract_price(seller.get("price", ""))
            if price <= 0:
                price = seller.get("extracted_price", 0.0) or 0.0
            if price <= 0:
                continue

            offers.append(ShoppingOffer(
                retailer=seller.get("name", seller.get("source", "Unknown")),
                price=float(price),
                currency="INR",
                url=seller.get("link", ""),
                delivery_info=seller.get("delivery", seller.get("additional_price", {}).get("delivery", "")),
                in_stock=True,
            ))

        # Mark best price
        if offers:
            best = min(offers, key=lambda o: o.price)
            best.is_best_price = True

        return offers

    def _parse_shopping_item(self, item: dict, fetched_at: str) -> Optional[ShoppingResult]:
        """Parse a single SerpAPI shopping_results item into a ShoppingResult."""
        title = item.get("title", "").strip()
        if not title:
            return None

        # Extract price — SerpAPI provides extracted_price as a float
        price = item.get("extracted_price", 0.0) or 0.0
        if price <= 0:
            price = self._extract_price(item.get("price", ""))
        if price <= 0:
            return None

        # Extract original price / old price
        old_price = item.get("extracted_old_price", 0.0) or 0.0
        if old_price <= 0:
            old_price_str = item.get("old_price", "")
            if old_price_str:
                old_price = self._extract_price(old_price_str)

        discount_pct = 0.0
        if old_price > price:
            discount_pct = round(((old_price - price) / old_price) * 100, 1)

        # Extract brand from title heuristic (first word or from extensions)
        brand = ""
        extensions = item.get("extensions", [])
        if extensions and isinstance(extensions, list):
            # Sometimes brand is in extensions
            brand = extensions[0] if extensions else ""

        # Rating and reviews
        rating = item.get("rating", None)
        if rating is not None:
            try:
                rating = float(rating)
            except (ValueError, TypeError):
                rating = None

        reviews = item.get("reviews", None)
        if reviews is not None:
            try:
                reviews = int(reviews)
            except (ValueError, TypeError):
                reviews = None

        return ShoppingResult(
            provider_product_id=str(item.get("product_id", "")),
            title=title,
            brand=brand,
            price=float(price),
            original_price=float(old_price) if old_price > 0 else float(price),
            currency="INR",
            discount_percentage=discount_pct,
            merchant=item.get("source", ""),
            merchant_url=item.get("link", ""),
            thumbnail=item.get("thumbnail", ""),
            rating=rating,
            review_count=reviews,
            offers=[],
            source_provider="SerpAPI Google Shopping",
            fetched_at=fetched_at,
            page_token=item.get("serpapi_product_api", item.get("immersive_product_page_token", "")),
        )

    @staticmethod
    def _extract_price(price_str: str) -> float:
        """Extract numeric price from formatted string like '₹49,999.00' or 'Rs. 1,299'."""
        if not price_str:
            return 0.0
        import re
        cleaned = re.sub(r'[^\d.]', '', price_str.replace(',', ''))
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
