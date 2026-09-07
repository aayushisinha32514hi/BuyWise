"""
BuyWise — Product Deduplication Engine
========================================
Groups multiple retailer offers under one canonical product.

Dedup strategy (in priority order):
1. provider_product_id — Google's internal product identifier (most reliable)
2. Normalized brand + model + title fuzzy matching (fallback)

Each canonical product holds a list of ShoppingOffer objects with
real merchant names, prices, and URLs from the shopping search provider.
"""

import re
from typing import List, Dict, Any
from difflib import SequenceMatcher

from backend.services.product_search.base import ShoppingResult, ShoppingOffer


def deduplicate_results(results: List[ShoppingResult]) -> List[ShoppingResult]:
    """
    Deduplicate a list of ShoppingResults into canonical products.
    Multiple offers for the same product are merged into offers[].
    """
    if not results:
        return []

    # Group by provider_product_id first
    groups: Dict[str, List[ShoppingResult]] = {}
    ungrouped: List[ShoppingResult] = []

    for r in results:
        if r.provider_product_id and r.provider_product_id.strip():
            key = r.provider_product_id.strip()
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        else:
            ungrouped.append(r)

    # Try to merge ungrouped items by title similarity
    for item in ungrouped:
        merged = False
        norm_title = _normalize_title(item.title)
        for key, group in groups.items():
            ref_title = _normalize_title(group[0].title)
            if _titles_match(norm_title, ref_title):
                group.append(item)
                merged = True
                break
        if not merged:
            # Create a new group with a synthetic key
            synth_key = f"_title_{_normalize_title(item.title)[:60]}"
            if synth_key in groups:
                groups[synth_key].append(item)
            else:
                groups[synth_key] = [item]

    # Merge each group into a single canonical product
    canonical: List[ShoppingResult] = []
    for group in groups.values():
        merged = _merge_group(group)
        if merged:
            canonical.append(merged)

    return canonical


def _merge_group(group: List[ShoppingResult]) -> ShoppingResult:
    """Merge a group of duplicate results into one canonical product."""
    # Use the result with the most information as the primary
    primary = max(group, key=lambda r: (
        (1 if r.rating else 0) +
        (1 if r.review_count else 0) +
        (1 if r.thumbnail else 0) +
        (1 if r.brand else 0)
    ))

    # Collect all unique offers
    seen_merchants = set()
    all_offers: List[ShoppingOffer] = []

    for r in group:
        merchant_key = r.merchant.strip().lower() if r.merchant else ""
        if merchant_key and merchant_key not in seen_merchants:
            seen_merchants.add(merchant_key)
            all_offers.append(ShoppingOffer(
                retailer=r.merchant,
                price=r.price,
                currency=r.currency,
                url=r.merchant_url,
                delivery_info="",
                in_stock=True,
                is_best_price=False,
            ))

    # Also add any pre-existing offers
    for r in group:
        for offer in (r.offers or []):
            mk = offer.retailer.strip().lower()
            if mk and mk not in seen_merchants:
                seen_merchants.add(mk)
                all_offers.append(offer)

    # If only one result and no existing offers, create the primary offer
    if not all_offers and primary.merchant:
        all_offers.append(ShoppingOffer(
            retailer=primary.merchant,
            price=primary.price,
            currency=primary.currency,
            url=primary.merchant_url,
            delivery_info="",
            in_stock=True,
            is_best_price=True,
        ))

    # Mark best price
    if all_offers:
        for o in all_offers:
            o.is_best_price = False
        best = min(all_offers, key=lambda o: o.price)
        best.is_best_price = True

        # Use the best price as the canonical price
        primary.price = best.price

    primary.offers = all_offers

    # Use best rating from the group
    ratings = [r.rating for r in group if r.rating is not None and r.rating > 0]
    if ratings:
        primary.rating = max(ratings)

    # Use highest review count from the group
    reviews = [r.review_count for r in group if r.review_count is not None and r.review_count > 0]
    if reviews:
        primary.review_count = max(reviews)

    # Use best thumbnail
    if not primary.thumbnail:
        for r in group:
            if r.thumbnail:
                primary.thumbnail = r.thumbnail
                break

    return primary


def _normalize_title(title: str) -> str:
    """Normalize a product title for comparison."""
    t = title.lower().strip()
    # Remove common noise words
    t = re.sub(r'\b(with|for|and|the|in|of|by|from|new|latest|pack|set|combo)\b', '', t)
    # Remove special characters
    t = re.sub(r'[^\w\s]', '', t)
    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _titles_match(title1: str, title2: str) -> bool:
    """Check if two normalized titles are similar enough to be the same product."""
    if title1 == title2:
        return True

    # Do not merge if storage capacities differ (e.g., 128GB vs 256GB vs 512GB)
    storage1 = set(re.findall(r'\b\d+\s*(?:gb|tb)\b', title1))
    storage2 = set(re.findall(r'\b\d+\s*(?:gb|tb)\b', title2))
    if storage1 and storage2 and storage1 != storage2:
        return False

    # Do not merge if model modifiers differ (e.g., Pro vs Plus vs Max vs Ultra vs Air)
    tokens1 = set(title1.split())
    tokens2 = set(title2.split())
    modifiers = {'plus', 'pro', 'max', 'mini', 'ultra', 'fe', 'lite', 'air', 'prime', 'neo'}
    mod1 = tokens1.intersection(modifiers)
    mod2 = tokens2.intersection(modifiers)
    if mod1 != mod2:
        return False

    ratio = SequenceMatcher(None, title1, title2).ratio()
    return ratio >= 0.88
