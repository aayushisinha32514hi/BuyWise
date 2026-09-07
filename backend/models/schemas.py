"""
BuyWise — Pydantic API Schemas (Live Shopping & AI Decision Support Edition)
=============================================================================
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


class RetailerOffer(BaseModel):
    retailer: str
    badge: str = ""
    color: str = "#4F46E5"
    price: float
    original_price: float = 0
    discount_percentage: float = 0
    in_stock: bool = True
    delivery_info: str = ""
    url: str = ""
    is_best_price: bool = False


class PriceSpreadInfo(BaseModel):
    best_price: float
    highest_price: float
    price_spread: float
    savings_vs_highest: float
    savings_percentage: float
    cheapest_retailer: str
    retailer_count: int


class ProductBase(BaseModel):
    id: Union[str, int] = ""
    canonical_id: Optional[str] = ""
    name: str = ""
    title: str = ""
    brand: Optional[str] = ""
    main_category: str = ""
    sub_category: str = ""
    price: float = 0
    original_price: Optional[float] = 0
    discount_percentage: Optional[float] = 0
    rating: Optional[float] = None
    review_count: Optional[int] = None
    buywise_score: Optional[float] = 0
    match_score: Optional[float] = 0
    ml_suitability_score: Optional[float] = None
    recommendation_reason: Optional[str] = ""
    specs: Optional[Dict[str, Any]] = {}
    image: Optional[str] = ""
    thumbnail: Optional[str] = ""
    link: Optional[str] = ""
    merchant: Optional[str] = ""
    merchant_url: Optional[str] = ""
    source: Optional[str] = ""
    fetched_at: Optional[str] = ""

    class Config:
        extra = "allow"


class ProductDetail(ProductBase):
    retailer_offers: Optional[List[RetailerOffer]] = []
    sentiment_insights: Optional[Dict[str, Any]] = None
    similar_products: Optional[List[Dict[str, Any]]] = []
    offers: Optional[List[Dict[str, Any]]] = []
    price_spread_info: Optional[PriceSpreadInfo] = None


class CategoryRequirementItem(BaseModel):
    key: str
    label: str
    desc: str
    default: int = 3


class TopCategory(BaseModel):
    id: str
    name: str
    short_name: str
    icon: str
    search_queries: List[str] = []

    class Config:
        extra = "allow"


class PaginatedProductsResponse(BaseModel):
    total: int
    page: int
    limit: int
    total_pages: int
    products: List[Dict[str, Any]]
    source: Optional[str] = ""
    error: Optional[str] = None
    message: Optional[str] = None

    class Config:
        extra = "allow"


class RecommendationRequest(BaseModel):
    category: Optional[str] = "Laptops"
    sub_category: Optional[str] = None
    requirements: Optional[Dict[str, int]] = {}
    min_price: Optional[float] = 0
    max_price: Optional[float] = None
    min_rating: Optional[float] = 0
    preferred_brands: Optional[List[str]] = []
    priority: Optional[str] = "value"
    limit: Optional[int] = 8


class RecommendationItem(BaseModel):
    id: Union[str, int] = ""
    canonical_id: Optional[str] = ""
    name: str = ""
    title: str = ""
    brand: Optional[str] = ""
    main_category: str = ""
    sub_category: str = ""
    price: float = 0
    original_price: Optional[float] = 0
    discount_percentage: Optional[float] = 0
    rating: Optional[float] = None
    review_count: Optional[int] = None
    buywise_score: Optional[float] = 0
    match_score: float = 0
    ml_suitability_score: Optional[float] = None
    recommendation_reason: str = ""
    specs: Optional[Dict[str, Any]] = {}
    image: Optional[str] = ""
    thumbnail: Optional[str] = ""
    merchant: Optional[str] = ""

    class Config:
        extra = "allow"


class RecommendationResponse(BaseModel):
    candidates_analyzed: int = 0
    total_matches: int = 0
    query_params: Dict[str, Any] = {}
    recommendations: List[Dict[str, Any]]
    model_type: Optional[str] = "Hybrid (Gradient Boosting Regressor + Multi-Criteria Preference Matching)"

    class Config:
        extra = "allow"


class CompareRequest(BaseModel):
    product_ids: List[Union[str, int]] = Field(..., min_length=2, max_length=5)


class CompareResponse(BaseModel):
    products: List[Dict[str, Any]]
    best_overall_id: Optional[Union[str, int]] = None
    best_value_id: Optional[Union[str, int]] = None
    best_rated_id: Optional[Union[str, int]] = None
    comparison_summary: str = ""

    class Config:
        extra = "allow"


# ── NLP Requirements Extraction Schemas ──────────────────────
class NLRequirementParseRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=500)


class NLRequirementParseResponse(BaseModel):
    category: str
    sub_category: Optional[str] = None
    min_price: float
    max_price: float
    min_rating: float
    priority: str
    preferred_brands: List[str]
    requirements: Dict[str, int]
    use_cases: List[str]
    extracted_summary: str


# ── Price History & Observation Schemas ─────────────────────
class PriceObservationItem(BaseModel):
    merchant: str
    price: float
    currency: str = "INR"
    observed_at: str
    product_url: Optional[str] = ""


class PriceHistoryResponse(BaseModel):
    canonical_id: str
    observations: List[PriceObservationItem]
    current_price: Optional[float] = None
    lowest_observed_price: Optional[float] = None
    highest_observed_price: Optional[float] = None
    average_price: Optional[float] = None
    price_change_pct: float = 0.0
    trend: str = "building"  # 'rising', 'falling', 'stable', 'building'
    trend_label: str = "Price history building"
    has_history: bool = False
    days: int = 30


# ── User Authentication & Profile Schemas ───────────────────
class UserSignupRequest(BaseModel):
    email: str
    username: str
    password: str


class UserLoginRequest(BaseModel):
    email: str
    password: str


class UserProfileResponse(BaseModel):
    id: int
    email: str
    username: str
    created_at: Optional[str] = ""
    saved_count: int = 0
    history_count: int = 0


class AuthResponse(BaseModel):
    success: bool
    user: Optional[Dict[str, Any]] = None
    token: Optional[str] = None
    error: Optional[str] = None


class SavedToggleRequest(BaseModel):
    product_id: Union[str, int]
