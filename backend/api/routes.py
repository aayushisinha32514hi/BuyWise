from fastapi import APIRouter, HTTPException, Query, Header
from typing import Optional, List, Dict, Any, Union
from backend.models.schemas import (
    ProductBase, ProductDetail, TopCategory, CategoryRequirementItem,
    PaginatedProductsResponse, RecommendationRequest,
    RecommendationResponse, CompareRequest, CompareResponse,
    UserSignupRequest, UserLoginRequest, AuthResponse, UserProfileResponse,
    SavedToggleRequest, NLRequirementParseRequest, NLRequirementParseResponse,
    PriceHistoryResponse
)
from backend.services.product_service import ProductService
from backend.services.auth_service import AuthService, get_user_id_from_token
from backend.services.nlp_extractor import parse_natural_language_requirements

router = APIRouter()
service = ProductService()
auth_service = AuthService()

# ── Category & Requirement Endpoints ───────────────────────────
@router.get("/categories/top", response_model=List[TopCategory])
def list_top_categories():
    """Retrieve 10 curated high-level Indian shopping categories with icons."""
    return service.get_top_categories()

@router.get("/categories/{category_id}/requirements", response_model=List[CategoryRequirementItem])
def get_category_buying_requirements(category_id: str, subcategory: Optional[str] = Query(None)):
    """Retrieve dynamic purchasing requirements with default importance weights for a category and subcategory."""
    return service.get_category_requirements(category_id, subcategory=subcategory)

# ── Natural Language Requirement Understanding ────────────────
@router.post("/ai-match/parse-requirements", response_model=NLRequirementParseResponse)
def parse_user_prompt(req: NLRequirementParseRequest):
    """
    Parse an unstructured natural language shopping query into structured requirements,
    budget constraints, and category importance weights.
    """
    parsed = parse_natural_language_requirements(req.text)
    return parsed

# ── Product Search & Details ──────────────────────────────────
@router.get("/products", response_model=PaginatedProductsResponse)
def get_products(
    q: Optional[str] = Query(None, description="Search query or natural prompt (e.g. 'laptop under 80000')"),
    category: Optional[str] = Query(None, description="Filter by top category (e.g. 'laptops', 'mobiles')"),
    sub_category: Optional[str] = Query(None, description="Filter by sub category"),
    brand: Optional[str] = Query(None, description="Filter by brand"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price in INR"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price in INR"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="Minimum star rating"),
    sort_by: str = Query("score", description="Sort option: score, price_asc, price_desc, rating, reviews"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
):
    """Search and filter live shopping catalog with natural intent parsing & caching."""
    return service.search_products(
        q=q,
        category=category,
        sub_category=sub_category,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        sort_by=sort_by,
        page=page,
        limit=limit
    )

@router.get("/products/{product_id}", response_model=ProductDetail)
def get_product(product_id: str, authorization: Optional[str] = Header(None)):
    """Retrieve product details with Multi-Retailer comparison offers, price spread & VADER sentiment."""
    product = service.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found in current shopping cache")

    # Record browsing history if user is logged in
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        user_id = get_user_id_from_token(token)
        if user_id:
            auth_service.record_history(user_id, product_id)

    return product

@router.get("/products/{product_id}/price-history", response_model=PriceHistoryResponse)
def get_product_price_history(product_id: str, days: int = Query(30, ge=1, le=365)):
    """Retrieve real empirical price history and trend analytics for a product."""
    return service.get_product_price_history(product_id, days=days)

# ── ML Recommendation & Comparison ────────────────────────────
@router.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(req: RecommendationRequest):
    """
    Intelligent AI Decision-Support Recommendation Endpoint:
    Matches user weighted requirements against extracted product features using a trained
    Gradient Boosting ML suitability model blended with multi-criteria optimization.
    """
    res = service.recommend_products(
        category=req.category or "Laptops",
        sub_category=req.sub_category,
        requirements=req.requirements or {},
        preferred_brands=req.preferred_brands or [],
        priority=req.priority or "value",
        min_price=req.min_price or 0,
        max_price=req.max_price or 100000,
        min_rating=req.min_rating or 0,
        limit=req.limit or 8
    )
    return {
        "candidates_analyzed": res.get("candidates_analyzed", 0),
        "total_matches": res.get("total_matches", 0),
        "query_params": req.dict(),
        "recommendations": res.get("recommendations", []),
        "model_type": res.get("model_type", "Hybrid (Gradient Boosting Regressor + Multi-Criteria Preference Matching)")
    }

@router.post("/compare", response_model=CompareResponse)
def compare_products(req: CompareRequest):
    """Compare 2 to 5 products side-by-side with multi-retailer pricing and AI winner badges."""
    return service.compare_products(req.product_ids)

# ── User Account & Wishlist Endpoints ─────────────────────────
@router.post("/auth/signup", response_model=AuthResponse)
def signup(req: UserSignupRequest):
    res = auth_service.signup(req.email, req.username, req.password)
    return res

@router.post("/auth/login", response_model=AuthResponse)
def login(req: UserLoginRequest):
    res = auth_service.login(req.email, req.password)
    return res

@router.get("/auth/me", response_model=UserProfileResponse)
def get_current_user_profile(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    token = authorization.replace("Bearer ", "").strip()
    user_id = get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    profile = auth_service.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile

@router.post("/user/saved/toggle")
def toggle_saved(req: SavedToggleRequest, authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    token = authorization.replace("Bearer ", "").strip()
    user_id = get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return auth_service.toggle_saved_product(user_id, req.product_id)

@router.get("/user/saved")
def get_user_saved(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    token = authorization.replace("Bearer ", "").strip()
    user_id = get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return auth_service.get_saved_products(user_id)

@router.get("/user/history")
def get_user_history(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    token = authorization.replace("Bearer ", "").strip()
    user_id = get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return auth_service.get_history(user_id)
