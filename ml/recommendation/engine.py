"""
BuyWise — Recommendation Engine Interface (ml/recommendation/engine.py)
========================================================================
Delegates to the AI Recommendation Engine with multi-criteria preference matching.
"""

from typing import List, Dict, Any, Optional
from backend.services.product_service import ProductService


class BuyWiseRecommender:
    def __init__(self):
        self.service = ProductService()

    def recommend_products(
        self,
        category: Optional[str] = "Laptops",
        requirements: Optional[Dict[str, int]] = None,
        min_price: Optional[float] = 0.0,
        max_price: Optional[float] = 100000.0,
        min_rating: Optional[float] = 0.0,
        preferred_brands: Optional[List[str]] = None,
        priority: str = "value",
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        result = self.service.recommend_products(
            category=category or "Laptops",
            requirements=requirements,
            preferred_brands=preferred_brands or [],
            priority=priority,
            min_price=min_price or 0.0,
            max_price=max_price or 100000.0,
            min_rating=min_rating or 0.0,
            limit=limit
        )
        return result.get("recommendations", [])
