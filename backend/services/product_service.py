"""
BuyWise — Product Service (Live Shopping Data, Multi-Retailer & AI Engine)
===========================================================================
Integrates:
1. Live Shopping Search via SerpAPI with 6-Hour SQLite Caching
2. Canonical Product Identity (p_<hash>) with Zero-Loss Retrieval
3. Real Technical Spec Extraction (RAM, CPU, GPU, Display, Camera, Battery, 5G, ANC)
4. AI Multi-Criteria Requirement & Preference Matching (Hybrid ML + MCDA)
5. Multi-Retailer Compare & Wishlist Store
6. Real Empirical Price Observation & History Store
"""

import os
import re
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union

from dotenv import load_dotenv
load_dotenv()

from backend.services.product_search.base import ShoppingResult, ShoppingOffer, generate_canonical_id
from backend.services.product_search.serpapi_provider import SerpAPIProvider
from backend.services.product_search.cache import (
    ensure_cache_tables, make_search_cache_key, get_cached_search,
    store_search_cache, get_cached_product_by_id,
    store_price_observation, get_price_history
)
from backend.services.product_search.normalizer import (
    normalize_result, classify_category, matches_category, compute_buywise_score
)
from backend.services.product_search.deduplicator import deduplicate_results
from backend.services.product_search.feature_extractor import extract_features
from backend.services.product_search.recommendation_engine import (
    AIRecommendationEngine, get_category_requirements
)
from backend.services.category_service import get_top_categories, get_search_query_for_category
from ml.sentiment.analyzer import ReviewSentimentAnalyzer

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), '../buywise.db')


class ProductService:
    def __init__(self):
        self.provider = SerpAPIProvider()
        self.recommender = AIRecommendationEngine()
        self.sentiment_analyzer = ReviewSentimentAnalyzer()
        ensure_cache_tables(DB_PATH)

    # ── Category & Requirement Metadata ──────────────────────

    def get_top_categories(self) -> List[Dict[str, Any]]:
        return get_top_categories()

    def get_category_requirements(self, category_id: str, subcategory: Optional[str] = None) -> List[Dict[str, Any]]:
        return get_category_requirements(category_id, subcategory=subcategory)

    # ── Search & Browse ──────────────────────────────────────

    def search_products(
        self,
        q: Optional[str] = None,
        category: Optional[str] = None,
        sub_category: Optional[str] = None,
        brand: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        sort_by: str = "score",
        page: int = 1,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Search products with natural query parsing, caching,
        spec extraction, deduplication, and strict category filtering.
        """
        search_query, extracted_max_price = self._build_search_query(q, category)

        if extracted_max_price and (max_price is None or max_price > extracted_max_price):
            max_price = extracted_max_price

        if not search_query:
            search_query = "best products India"

        if not self.provider.is_available():
            return {
                "total": 0,
                "page": page,
                "limit": limit,
                "total_pages": 0,
                "products": [],
                "source": "unavailable",
                "error": "Shopping data provider not configured. Set SERPAPI_KEY in .env file.",
            }

        cache_key = make_search_cache_key(search_query, "in", min_price, max_price)
        cached = get_cached_search(cache_key, DB_PATH)

        if cached is not None:
            products = cached
            source_label = "Google Shopping (cached)"
        else:
            raw_results = self.provider.search(
                query=search_query,
                country="in",
                language="en",
                max_results=40,
                min_price=min_price,
                max_price=max_price,
            )

            if not raw_results:
                return {
                    "total": 0,
                    "page": page,
                    "limit": limit,
                    "total_pages": 0,
                    "products": [],
                    "source": "Google Shopping",
                    "message": f"No results found for '{search_query}'. Try widening your search terms.",
                }

            normalized = [normalize_result(r) for r in raw_results]
            # Extract specs for each
            for r in normalized:
                r.specs = extract_features(r.title, r.buywise_category)

            deduped = deduplicate_results(normalized)
            store_search_cache(cache_key, search_query, deduped, db_path=DB_PATH)
            products = get_cached_search(cache_key, DB_PATH) or []
            source_label = "Google Shopping"

        # Record price observations for top products
        for p in products[:10]:
            cid = p.get("canonical_id") or p.get("id")
            if cid and p.get("price", 0) > 0:
                store_price_observation(
                    canonical_id=str(cid),
                    merchant=p.get("merchant", "Online Store"),
                    price=float(p["price"]),
                    currency=p.get("currency", "INR"),
                    product_url=p.get("link", ""),
                    db_path=DB_PATH
                )

        # Apply post-filters
        filtered = self._apply_filters(
            products,
            category=category,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
        )

        filtered = self._apply_sort(filtered, sort_by)

        total = len(filtered)
        total_pages = max((total + limit - 1) // limit, 1)
        start = (page - 1) * limit
        page_items = filtered[start:start + limit]

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "products": page_items,
            "source": source_label,
        }

    # ── Product Details, Multi-Retailer Offers & Compare ─────

    def get_product_by_id(self, product_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete product details from cache using canonical ID.
        Enriches with live multi-retailer offers, price spread stats, and sentiment insights.
        """
        product = get_cached_product_by_id(product_id, DB_PATH)
        if not product:
            return None

        cid = str(product.get("canonical_id") or product.get("id"))

        # Extract specs if missing
        if not product.get("specs"):
            product["specs"] = extract_features(product.get("title", ""), product.get("main_category", ""))

        # Record empirical price observation
        if product.get("price", 0) > 0:
            store_price_observation(
                canonical_id=cid,
                merchant=product.get("merchant", "Online Store"),
                price=float(product["price"]),
                currency=product.get("currency", "INR"),
                product_url=product.get("link", product.get("merchant_url", "")),
                db_path=DB_PATH
            )

        # Enrich with VADER sentiment insights
        product["sentiment_insights"] = self.sentiment_analyzer.extract_product_highlights(
            product.get("title", product.get("name", "")),
            product.get("rating")
        )

        # Multi-Retailer Offers Integration
        offers_list = product.get("offers") or []

        # If only 1 offer and page_token exists, fetch live immersive offers from Google Shopping
        page_tok = product.get("page_token") or ""
        prov_id = product.get("provider_product_id") or ""
        if len(offers_list) <= 1 and page_tok and self.provider.is_available():
            try:
                live_offers = self.provider.get_product_offers(prov_id, page_tok)
                if live_offers:
                    offers_list = [
                        {
                            "retailer": o.retailer,
                            "price": o.price,
                            "currency": o.currency,
                            "url": o.url,
                            "delivery_info": o.delivery_info,
                            "in_stock": o.in_stock,
                            "is_best_price": o.is_best_price,
                        }
                        for o in live_offers
                    ]
            except Exception as e:
                logger.debug("Live multi-retailer offer fetch skipped: %s", e)

        # Fallback primary offer if no offers list
        if not offers_list and product.get("merchant"):
            offers_list = [{
                "retailer": product["merchant"],
                "price": product.get("price", 0),
                "currency": "INR",
                "url": product.get("merchant_url", product.get("link", "")),
                "delivery_info": "Standard Online Delivery",
                "in_stock": True,
                "is_best_price": True,
            }]

        formatted_offers = self._format_offers_for_frontend(offers_list)
        product["retailer_offers"] = formatted_offers
        product["offers"] = formatted_offers

        # Calculate Price Spread Stats
        if formatted_offers:
            prices = [o["price"] for o in formatted_offers if o.get("price", 0) > 0]
            if prices:
                best_p = min(prices)
                high_p = max(prices)
                spread = round(high_p - best_p, 2)
                savings_pct = round(((high_p - best_p) / high_p) * 100, 1) if high_p > 0 else 0.0
                cheapest_off = min(formatted_offers, key=lambda x: x["price"])

                product["price_spread_info"] = {
                    "best_price": best_p,
                    "highest_price": high_p,
                    "price_spread": spread,
                    "savings_vs_highest": spread,
                    "savings_percentage": savings_pct,
                    "cheapest_retailer": cheapest_off.get("retailer", "Online Store"),
                    "retailer_count": len(formatted_offers),
                }

        product["similar_products"] = []
        return product

    def get_product_price_history(self, product_id: Union[str, int], days: int = 30) -> Dict[str, Any]:
        """Retrieve verified historical price observations for a canonical product."""
        return get_price_history(str(product_id).strip(), days=days, db_path=DB_PATH)

    def compare_products(self, product_ids: List[Union[str, int]]) -> Dict[str, Any]:
        """
        Compare 2 to 5 products with full multi-merchant pricing and AI winner badges.
        """
        products = []
        for pid in product_ids:
            p = self.get_product_by_id(pid)
            if p:
                products.append(p)

        if not products:
            return {
                "products": [],
                "comparison_summary": "No matching products found in shopping cache.",
            }

        # Calculate AI Winner Badges
        best_overall = max(products, key=lambda x: x.get("buywise_score", 0))
        best_overall_id = best_overall.get("id") or best_overall.get("canonical_id")

        best_value = max(products, key=lambda x: (x.get("discount_percentage") or 0.0))
        best_value_id = best_value.get("id") or best_value.get("canonical_id")

        best_rated = max(products, key=lambda x: (x.get("rating") or 0.0, x.get("review_count") or 0))
        best_rated_id = best_rated.get("id") or best_rated.get("canonical_id")

        summary = f"Compared {len(products)} products across verified Indian retailer feeds, specifications, ratings, and price efficiency."

        return {
            "products": products,
            "best_overall_id": best_overall_id,
            "best_value_id": best_value_id,
            "best_rated_id": best_rated_id,
            "comparison_summary": summary,
        }

    # ── AI Recommendation Engine ─────────────────────────────

    def recommend_products(
        self,
        category: str,
        sub_category: Optional[str] = None,
        requirements: Optional[Dict[str, int]] = None,
        preferred_brands: Optional[List[str]] = None,
        priority: str = "value",
        min_price: float = 0,
        max_price: float = 100000,
        min_rating: float = 0,
        limit: int = 8,
    ) -> Dict[str, Any]:
        """
        AI Match: Searches a large candidate pool (25–40 items) from live/cached data,
        extracts specs, scores with trained ML suitability model + preference vector weights,
        and returns explainable recommendations.
        """
        search_query = get_search_query_for_category(category, subcategory_id=sub_category)

        if not self.provider.is_available():
            return {
                "candidates_analyzed": 0,
                "total_matches": 0,
                "recommendations": [],
                "model_type": "Hybrid (Gradient Boosting Regressor + Multi-Criteria Preference Matching)"
            }

        # Collect Candidate Pool
        cache_key = make_search_cache_key(f"rec_pool_{search_query}", "in", min_price, max_price)
        cached = get_cached_search(cache_key, DB_PATH)

        if cached is not None:
            candidate_pool = cached
        else:
            raw_results = self.provider.search(
                query=f"best {search_query}",
                country="in",
                language="en",
                max_results=40,
                min_price=min_price if min_price > 0 else None,
                max_price=max_price if max_price > 0 else None,
            )

            normalized = [normalize_result(r) for r in raw_results]
            for r in normalized:
                r.specs = extract_features(r.title, r.buywise_category)
            deduped = deduplicate_results(normalized)

            store_search_cache(cache_key, search_query, deduped, db_path=DB_PATH)
            candidate_pool = get_cached_search(cache_key, DB_PATH) or []

        # Run Hybrid ML Multi-Criteria Ranking
        ranked_items, total_analyzed = self.recommender.rank_candidates(
            candidates=candidate_pool,
            category=category,
            subcategory=sub_category,
            user_requirements=requirements,
            preferred_brands=preferred_brands,
            priority=priority,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            limit=limit,
        )

        return {
            "candidates_analyzed": total_analyzed,
            "total_matches": len(ranked_items),
            "recommendations": ranked_items,
            "model_type": "Hybrid (Gradient Boosting Regressor + Multi-Criteria Preference Matching)"
        }

    # ── Internal Helpers ─────────────────────────────────────

    def _build_search_query(self, q: Optional[str], category: Optional[str]) -> Tuple[str, Optional[float]]:
        extracted_max_price = None
        parts = []

        if q and q.strip():
            cleaned, price = self._parse_natural_query(q)
            if cleaned:
                parts.append(cleaned)
            extracted_max_price = price

        if category and category.strip().lower() not in ("all", "any", ""):
            cat_query = get_search_query_for_category(category)
            if not parts:
                parts.append(cat_query)
            elif not any(kw in parts[0].lower() for kw in cat_query.lower().split()[:2]):
                parts.append(cat_query)

        return " ".join(parts), extracted_max_price

    def _parse_natural_query(self, query: str) -> Tuple[str, Optional[float]]:
        cleaned_q = query.strip()
        extracted_max_price = None

        under_match = re.search(
            r'(?:under|below|less than|within|upto|up to)\s*(?:rs\.?|inr|₹)?\s*([0-9]+k|[0-9]+)',
            cleaned_q, re.IGNORECASE
        )
        if under_match:
            val_str = under_match.group(1).lower()
            if 'k' in val_str:
                extracted_max_price = float(val_str.replace('k', '')) * 1000.0
            else:
                extracted_max_price = float(val_str)
            cleaned_q = re.sub(
                r'(?:under|below|less than|within|upto|up to)\s*(?:rs\.?|inr|₹)?\s*[0-9]+k?',
                '', cleaned_q, flags=re.IGNORECASE
            ).strip()

        return cleaned_q, extracted_max_price

    def _apply_filters(
        self,
        products: List[Dict],
        category: Optional[str] = None,
        brand: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
    ) -> List[Dict]:
        filtered = products

        if category and category.strip().lower() not in ("all", "any", ""):
            cat_id = category.strip().lower()
            filtered = [p for p in filtered if matches_category(p.get("title", ""), cat_id)]

        if brand and brand.strip():
            brand_lower = brand.strip().lower()
            filtered = [p for p in filtered if brand_lower in (p.get("brand", "") or "").lower()]

        if min_price and min_price > 0:
            filtered = [p for p in filtered if (p.get("price") or 0) >= min_price]

        if max_price and max_price > 0:
            filtered = [p for p in filtered if (p.get("price") or 0) <= max_price]

        if min_rating and min_rating > 0:
            filtered = [p for p in filtered if (p.get("rating") or 0) >= min_rating]

        return filtered

    def _apply_sort(self, products: List[Dict], sort_by: str) -> List[Dict]:
        if sort_by == "price_asc":
            return sorted(products, key=lambda p: p.get("price", 0))
        elif sort_by == "price_desc":
            return sorted(products, key=lambda p: p.get("price", 0), reverse=True)
        elif sort_by == "rating":
            return sorted(products, key=lambda p: (p.get("rating") or 0, p.get("review_count") or 0), reverse=True)
        elif sort_by == "reviews":
            return sorted(products, key=lambda p: p.get("review_count") or 0, reverse=True)
        else:
            return sorted(products, key=lambda p: p.get("buywise_score", 0), reverse=True)

    def _format_offers_for_frontend(self, offers: List[Dict]) -> List[Dict]:
        formatted = []
        for o in offers:
            formatted.append({
                "retailer": o.get("retailer", "Online Store"),
                "badge": o.get("badge", ""),
                "color": o.get("color", "#4F46E5"),
                "price": float(o.get("price", 0)),
                "original_price": float(o.get("original_price", o.get("price", 0))),
                "discount_percentage": float(o.get("discount_percentage", 0)),
                "in_stock": bool(o.get("in_stock", True)),
                "delivery_info": str(o.get("delivery_info", "Verified Delivery")),
                "url": str(o.get("url", "")),
                "is_best_price": bool(o.get("is_best_price", False)),
            })
        return formatted
