"""
BuyWise — Abstract Shopping Search Provider & Canonical Data Models
=====================================================================
Defines the interface that all shopping data providers must implement
and canonical data structures for products, offers, and specifications.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import hashlib
import re


def generate_canonical_id(
    title: str,
    brand: str = "",
    merchant: str = "",
    url: str = "",
    provider_product_id: str = ""
) -> str:
    """
    Generate a deterministic, collision-resistant canonical product ID.
    If provider_product_id is provided, uses it as primary key.
    Otherwise, hashes normalized title + brand.
    Returns a clean string formatted as 'p_<16_hex_chars>'.
    """
    if provider_product_id and provider_product_id.strip():
        raw_key = f"gpid:{provider_product_id.strip()}"
    else:
        # Normalize title: lowercase, strip noise words & punctuation
        clean_title = re.sub(r'[^\w\s]', '', title.lower().strip())
        clean_title = re.sub(r'\s+', ' ', clean_title)
        clean_brand = brand.lower().strip()
        raw_key = f"{clean_title}|{clean_brand}"
    
    h = hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:16]
    return f"p_{h}"


@dataclass
class ShoppingOffer:
    """A single retailer offer for a product."""
    retailer: str
    price: float
    currency: str = "INR"
    url: str = ""
    delivery_info: str = ""
    in_stock: bool = True
    is_best_price: bool = False


@dataclass
class ShoppingResult:
    """A normalized product result with stable canonical identity."""
    # Canonical Identity
    canonical_id: str = ""                 # Deterministic ID: 'p_8f3a1c02b9d4e7f1'
    provider_product_id: str = ""          # Provider's unique product ID
    title: str = ""
    brand: str = ""

    # Pricing
    price: float = 0.0
    original_price: float = 0.0
    currency: str = "INR"
    discount_percentage: float = 0.0

    # Merchant / source
    merchant: str = ""
    merchant_url: str = ""

    # Media
    thumbnail: str = ""

    # Quality signals
    rating: Optional[float] = None
    review_count: Optional[int] = None

    # Multi-retailer offers (populated after deduplication or detail fetch)
    offers: List[ShoppingOffer] = field(default_factory=list)

    # Extracted Specifications & Features (RAM, Storage, GPU, Battery, etc.)
    specs: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    source_provider: str = ""              # e.g. "SerpAPI Google Shopping"
    fetched_at: str = ""                   # ISO timestamp
    page_token: str = ""                   # For fetching detailed multi-merchant offers

    # BuyWise computed fields (filled by normalizer / AI engine)
    buywise_category: str = ""
    buywise_score: float = 0.0
    match_score: float = 0.0
    recommendation_reason: str = ""

    def __post_init__(self):
        if not self.canonical_id and self.title:
            self.canonical_id = generate_canonical_id(
                title=self.title,
                brand=self.brand,
                merchant=self.merchant,
                url=self.merchant_url,
                provider_product_id=self.provider_product_id
            )


class ShoppingSearchProvider(ABC):
    """Abstract base class for shopping search data providers."""

    @abstractmethod
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
        Search for products matching the query.
        Returns a list of normalized ShoppingResult objects.
        """
        pass

    @abstractmethod
    def get_product_offers(self, product_id: str, page_token: str = "") -> List[ShoppingOffer]:
        """
        Fetch all retailer offers for a specific product.
        Uses the provider's product detail / immersive product endpoint.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and reachable."""
        pass
