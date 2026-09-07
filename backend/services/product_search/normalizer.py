"""
BuyWise — Shopping Result Normalizer
======================================
Normalizes raw SerpAPI responses into clean BuyWise product objects.
- Maps products to BuyWise's 10 macro categories using keyword matching
- Calculates BuyWise composite score from available fields
- NEVER invents missing data — marks unavailable fields as null
"""

import re
import math
from typing import Optional, Dict, Any

from backend.services.product_search.base import ShoppingResult


# Category keyword mapping for strict classification
CATEGORY_KEYWORDS = {
    "mobiles": [
        "smartphone", "mobile phone", "iphone", "samsung galaxy", "oneplus",
        "redmi", "realme", "vivo", "oppo", "motorola", "nokia phone",
        "pixel phone", "nothing phone", "poco", "iqoo", "phone case",
        "mobile cover", "screen protector", "charger", "power bank",
    ],
    "laptops": [
        "laptop", "notebook", "macbook", "chromebook", "thinkpad",
        "ideapad", "pavilion", "inspiron", "vivobook", "zenbook",
        "gaming laptop", "ultrabook", "2-in-1 laptop",
    ],
    "electronics": [
        "headphone", "earphone", "earbuds", "speaker", "bluetooth speaker",
        "soundbar", "amplifier", "microphone", "camera", "dslr", "mirrorless",
        "gopro", "drone", "tripod", "memory card", "hard drive", "ssd",
        "pendrive", "usb", "keyboard", "mouse", "monitor", "webcam",
        "router", "powerbank",
    ],
    "appliances": [
        "air conditioner", "split ac", "window ac", "refrigerator",
        "fridge", "washing machine", "microwave", "oven", "water purifier",
        "vacuum cleaner", "air purifier", "iron", "mixer grinder",
        "induction cooktop", "geyser", "water heater", "dishwasher",
        "chimney", "food processor", "juicer", "toaster",
    ],
    "tvs": [
        "television", "smart tv", "led tv", "oled tv", "qled tv",
        "4k tv", "android tv", "fire tv", "projector", "home theater",
        "streaming device", "set top box", "soundbar",
    ],
    "fashion": [
        "shirt", "t-shirt", "jeans", "trousers", "kurta", "saree",
        "lehenga", "dress", "jacket", "hoodie", "sweatshirt", "blazer",
        "suit", "ethnic wear", "western wear", "palazzo", "skirt",
        "shorts", "track pants", "innerwear", "lingerie",
    ],
    "footwear": [
        "shoes", "sneakers", "running shoe", "sports shoe", "sandal",
        "slipper", "flip flop", "boot", "loafer", "formal shoe",
        "casual shoe", "floater", "crocs", "heels", "flats", "wedges",
        "football studs", "training shoe",
    ],
    "beauty": [
        "moisturizer", "sunscreen", "face wash", "shampoo", "conditioner",
        "serum", "lipstick", "foundation", "mascara", "eyeliner",
        "nail polish", "perfume", "deodorant", "trimmer", "hair dryer",
        "straightener", "body lotion", "face cream", "makeup",
        "skincare", "grooming",
    ],
    "jewellery": [
        "watch", "wristwatch", "smartwatch", "fitness band", "bracelet",
        "necklace", "earring", "ring", "pendant", "chain", "bangle",
        "anklet", "mangalsutra", "sunglasses", "wallet", "handbag",
        "clutch",
    ],
    "home": [
        "bedsheet", "curtain", "pillow", "mattress", "sofa", "table",
        "chair", "bookshelf", "lamp", "led light", "wall decor",
        "painting", "rug", "carpet", "storage", "organizer",
        "kitchen", "cookware", "utensil", "dinner set", "bottle",
        "container", "lunchbox",
    ],
}


def classify_category(title: str) -> str:
    """
    Classify a product title into one of BuyWise's 10 macro categories.
    Returns the category ID (e.g., 'mobiles', 'laptops') or 'uncategorized'.
    """
    title_lower = title.lower()

    best_cat = "uncategorized"
    best_score = 0

    for cat_id, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in title_lower:
                # Longer keyword matches get higher weight
                score += len(kw)
        if score > best_score:
            best_score = score
            best_cat = cat_id

    return best_cat


def matches_category(title: str, target_category: str) -> bool:
    """
    Check if a product title strictly belongs to the target category.
    Used for strict post-filtering to prevent HDMI cables in Mobiles, etc.
    """
    if not target_category or target_category.lower() in ("all", "any", ""):
        return True

    classified = classify_category(title)
    return classified == target_category.lower()


def compute_buywise_score(
    price: float,
    original_price: float,
    rating: Optional[float],
    review_count: Optional[int],
    discount_pct: float,
) -> float:
    """
    Calculate BuyWise composite score (0-100) from available product signals.
    Uses Bayesian smoothing for rating and log-scaling for review volume.
    NEVER invents data — uses conservative defaults for missing fields.
    """
    # Rating component (0-1) with Bayesian smoothing
    if rating is not None and rating > 0:
        c, m = 10, 3.8  # Prior strength and prior mean
        rev = review_count or 0
        smoothed = (c * m + rating * rev) / (c + rev) if rev > 0 else rating
        norm_rating = min(smoothed / 5.0, 1.0)
    else:
        norm_rating = 0.5  # Neutral default when no rating available

    # Popularity component (0-1)
    if review_count is not None and review_count > 0:
        norm_popularity = min(math.log1p(review_count) / math.log1p(25000), 1.0)
    else:
        norm_popularity = 0.1  # Low default for unknown popularity

    # Discount/value component (0-1)
    norm_discount = min(max(discount_pct, 0.0) / 100.0, 1.0)

    # Composite score
    score = (
        0.35 * norm_rating +
        0.25 * norm_popularity +
        0.25 * norm_discount +
        0.15 * 0.5  # Neutral brand score (no preferred brands in general scoring)
    ) * 100.0

    return round(max(0.0, min(score, 100.0)), 1)


def normalize_result(result: ShoppingResult) -> ShoppingResult:
    """
    Enrich a ShoppingResult with BuyWise category classification and composite score.
    """
    result.buywise_category = classify_category(result.title)
    result.buywise_score = compute_buywise_score(
        price=result.price,
        original_price=result.original_price,
        rating=result.rating,
        review_count=result.review_count,
        discount_pct=result.discount_percentage,
    )

    # Clean up brand — extract from title if empty
    if not result.brand:
        result.brand = _extract_brand_from_title(result.title)

    return result


def _extract_brand_from_title(title: str) -> str:
    """Heuristic: extract the first capitalized word as brand name."""
    if not title:
        return ""
    words = title.split()
    if words:
        first = words[0].strip("()")
        if first and first[0].isupper() and len(first) >= 2:
            return first
    return ""
