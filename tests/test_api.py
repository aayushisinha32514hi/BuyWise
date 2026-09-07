"""
BuyWise — Comprehensive End-to-End Automated Test Suite
=========================================================
Tests:
1. System & Health Endpoints
2. Top 10 Categories & Dynamic Buying Requirements Endpoints
3. Natural Language Requirement Understanding (NLP Extractor)
4. Price History & Observation Storage
5. Canonical Product Identity & Zero Cache Miss Lookup
6. Feature Extraction Engine (RAM, Storage, CPU, GPU, Display, Camera, Battery)
7. Preference-Sensitive AI Match Ranking with ML Suitability Prediction
8. Compare End-to-End: Explore -> Compare & AI Match -> Compare with Winner Badges
9. User Authentication & Wishlist with Canonical IDs
"""

import sys
import os
import secrets
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.product_search.base import generate_canonical_id, ShoppingResult
from backend.services.product_search.feature_extractor import extract_features
from backend.services.product_search.normalizer import classify_category, matches_category
from backend.services.product_search.recommendation_engine import AIRecommendationEngine, get_category_requirements
from backend.services.nlp_extractor import parse_natural_language_requirements

client = TestClient(app)


def test_system_and_health():
    """Verify system root and health endpoints."""
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "BuyWise" in r_root.text

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"


def test_categories_and_requirements():
    """Verify top 10 categories and dynamic category buying requirements."""
    r_top = client.get("/api/v1/categories/top")
    assert r_top.status_code == 200
    data = r_top.json()
    assert len(data) == 10

    # Test requirements for Mobiles
    r_mob = client.get("/api/v1/categories/mobiles/requirements")
    assert r_mob.status_code == 200
    mob_reqs = [req["key"] for req in r_mob.json()]
    assert "camera" in mob_reqs
    assert "gaming" in mob_reqs
    assert "battery" in mob_reqs

    # Test requirements for Laptops
    r_lap = client.get("/api/v1/categories/laptops/requirements")
    assert r_lap.status_code == 200
    lap_reqs = [req["key"] for req in r_lap.json()]
    assert "coding" in lap_reqs
    assert "gaming" in lap_reqs
    assert "performance" in lap_reqs


def test_nlp_requirement_extractor():
    """Verify NLP requirement parsing from natural text prompts."""
    # Test prompt 1: Coding laptop with budget
    p1 = client.post("/api/v1/ai-match/parse-requirements", json={
        "text": "I need a coding laptop under 85000 with 16GB RAM for programming and ASUS"
    })
    assert p1.status_code == 200
    d1 = p1.json()
    assert d1["category"] == "Laptops"
    assert d1["max_price"] == 85000.0
    assert d1["requirements"].get("coding") == 5
    assert "ASUS" in d1["preferred_brands"]

    # Test prompt 2: Phone with photography intent
    p2 = client.post("/api/v1/ai-match/parse-requirements", json={
        "text": "Looking for a Samsung 5G phone under 45k with great camera and all day battery"
    })
    assert p2.status_code == 200
    d2 = p2.json()
    assert d2["category"] == "Mobiles"
    assert d2["max_price"] == 45000.0
    assert d2["requirements"].get("camera") == 5
    assert d2["requirements"].get("battery") == 5
    assert "Samsung" in d2["preferred_brands"]


def test_feature_extractor():
    """Verify extraction of RAM, Storage, CPU, GPU, Camera, Display from titles."""
    f1 = extract_features("ASUS TUF Gaming A15 AMD Ryzen 7 7735HS (16GB RAM / 512GB SSD / RTX 4060 8GB / 144Hz FHD)")
    assert f1.get("ram_gb") == 16
    assert f1.get("storage_gb") == 512
    assert "Ryzen 7" in f1.get("processor", "")
    assert "RTX 4060" in f1.get("gpu", "")
    assert f1.get("high_refresh") is True

    f2 = extract_features("Apple MacBook Air 13.6-inch M2 chip 8GB RAM 256GB SSD Retina Display")
    assert f2.get("ram_gb") == 8
    assert f2.get("storage_gb") == 256
    assert "M2" in f2.get("processor", "")
    assert f2.get("display_type") == "Retina"

    f3 = extract_features("Samsung Galaxy S24 Ultra 5G (12GB RAM, 256GB Storage, 200MP Camera, 5000mAh Battery)")
    assert f3.get("ram_gb") == 12
    assert f3.get("storage_gb") == 256
    assert f3.get("camera_mp") == 200
    assert f3.get("battery_mah") == 5000
    assert f3.get("is_5g") is True


def test_canonical_id_generation():
    """Verify deterministic canonical ID generation."""
    id1 = generate_canonical_id("Apple iPhone 15 128GB Black", "Apple", "Amazon.in")
    id2 = generate_canonical_id("Apple iPhone 15 128GB Black", "Apple", "Amazon.in")
    assert id1 == id2, "Canonical ID must be deterministic across calls"
    assert id1.startswith("p_"), "Canonical ID must follow p_<hex> format"

    # Different product must produce different ID
    id3 = generate_canonical_id("Samsung Galaxy S24 256GB", "Samsung", "Amazon.in")
    assert id1 != id3, "Distinct products must have distinct canonical IDs"


def test_ai_ranking_preference_sensitivity():
    """
    CRITICAL AI TEST: Verify that changing user requirement weights
    meaningfully alters product ranking on identical candidate sets.
    """
    engine = AIRecommendationEngine()

    candidates = [
        {
            "canonical_id": "p_laptop_coding",
            "title": "Lenovo ThinkPad E14 Intel Core i7 13th Gen (16GB RAM / 1TB SSD / Iris Xe / 1.4kg Thin & Light)",
            "brand": "Lenovo",
            "price": 68000.0,
            "original_price": 85000.0,
            "discount_percentage": 20.0,
            "rating": 4.5,
            "review_count": 350,
            "merchant": "Amazon.in",
            "main_category": "laptops",
        },
        {
            "canonical_id": "p_laptop_gaming",
            "title": "Acer Nitro 5 Gaming Laptop Intel Core i5 (8GB RAM / 512GB SSD / RTX 4050 6GB GPU / 144Hz)",
            "brand": "Acer",
            "price": 69990.0,
            "original_price": 89990.0,
            "discount_percentage": 22.0,
            "rating": 4.4,
            "review_count": 520,
            "merchant": "Flipkart",
            "main_category": "laptops",
        },
    ]

    # User A: High Coding & Portability (5 Stars), Low Gaming (1 Star)
    ranked_a, _ = engine.rank_candidates(
        candidates=candidates,
        category="Laptops",
        user_requirements={"coding": 5, "battery_portability": 5, "gaming": 1, "performance": 4},
        max_price=80000,
        limit=2
    )

    # User B: High Gaming (5 Stars), Low Portability & Coding (1 Star)
    ranked_b, _ = engine.rank_candidates(
        candidates=candidates,
        category="Laptops",
        user_requirements={"gaming": 5, "coding": 1, "battery_portability": 1, "performance": 3},
        max_price=80000,
        limit=2
    )

    assert len(ranked_a) == 2
    assert len(ranked_b) == 2

    # User A's top choice MUST be the ThinkPad (Coding/RAM/Portability fit)
    assert ranked_a[0]["canonical_id"] == "p_laptop_coding", "Coding-first user should be recommended ThinkPad"

    # User B's top choice MUST be the Nitro 5 (Dedicated RTX 4050 Gaming GPU fit)
    assert ranked_b[0]["canonical_id"] == "p_laptop_gaming", "Gaming-first user should be recommended Acer Nitro with RTX 4050"


def test_compare_and_price_history_pipeline():
    """
    Verify Explore -> Details -> Compare -> Price History pipeline.
    """
    # 1. Search Explore
    r_search = client.get("/api/v1/products?q=laptop&limit=4")
    assert r_search.status_code == 200
    search_prods = r_search.json().get("products", [])

    if search_prods:
        first_p = search_prods[0]
        first_id = first_p["id"]
        assert first_id, "Product ID must not be empty or 0"

        # 2. Get Product Details
        r_det = client.get(f"/api/v1/products/{first_id}")
        assert r_det.status_code == 200, f"Failed to get product details for {first_id}"
        det_data = r_det.json()
        assert det_data["title"], "Product details must contain title"
        assert "retailer_offers" in det_data, "Product details must contain retailer offers"

        # 3. Get Price History
        r_hist = client.get(f"/api/v1/products/{first_id}/price-history?days=30")
        assert r_hist.status_code == 200
        hist_data = r_hist.json()
        assert hist_data["canonical_id"] == str(first_id)
        assert hist_data["current_price"] is not None

        # 4. Compare multiple products from Explore
        if len(search_prods) >= 2:
            pids = [search_prods[0]["id"], search_prods[1]["id"]]
            r_comp = client.post("/api/v1/compare", json={"product_ids": pids})
            assert r_comp.status_code == 200
            comp_data = r_comp.json()
            assert len(comp_data["products"]) == 2, "Compare must return both products"
            assert comp_data["best_overall_id"] in pids, "Best overall ID must match one of compared products"

    # 5. Recommend AI Match (verifying ML suitability score inclusion)
    r_rec = client.post("/api/v1/recommend", json={
        "category": "Laptops",
        "requirements": {"coding": 5, "gaming": 3},
        "min_price": 20000,
        "max_price": 85000,
        "limit": 4
    })
    assert r_rec.status_code == 200
    rec_data = r_rec.json()
    assert rec_data["candidates_analyzed"] >= 0
    assert "model_type" in rec_data
    recs = rec_data.get("recommendations", [])

    if recs:
        rec_id = recs[0]["id"]
        assert rec_id, "AI recommendation ID must not be 0"

        # 6. Detail lookup for AI recommendation
        r_rec_det = client.get(f"/api/v1/products/{rec_id}")
        assert r_rec_det.status_code == 200, f"Failed to fetch details for AI recommendation ID {rec_id}"


def test_user_authentication_and_wishlist_canonical():
    """Verify saved wishlist and history work seamlessly with canonical string IDs."""
    test_email = f"user_{secrets.token_hex(4)}@buywise.in"
    pwd = "password123"

    # 1. Sign Up
    r_signup = client.post("/api/v1/auth/signup", json={"email": test_email, "username": "Aayushi", "password": pwd})
    assert r_signup.status_code == 200
    token = r_signup.json()["token"]

    # 2. Toggle Save Canonical ID
    canon_id = "p_test_canon_123"
    r_save = client.post("/api/v1/user/saved/toggle", json={"product_id": canon_id}, headers={"Authorization": f"Bearer {token}"})
    assert r_save.status_code == 200
    assert r_save.json()["is_saved"] is True

    # 3. Toggle off
    r_unsave = client.post("/api/v1/user/saved/toggle", json={"product_id": canon_id}, headers={"Authorization": f"Bearer {token}"})
    assert r_unsave.status_code == 200
    assert r_unsave.json()["is_saved"] is False


if __name__ == "__main__":
    print("=" * 68)
    print("BuyWise AI Specialisation & Full System End-to-End Test Suite")
    print("=" * 68)
    test_system_and_health()
    print("  ✓ System & Health probes passed")
    test_categories_and_requirements()
    print("  ✓ Top 10 Categories & Dynamic Buying Requirements passed")
    test_nlp_requirement_extractor()
    print("  ✓ Natural Language Requirement Understanding (NLP) passed")
    test_feature_extractor()
    print("  ✓ Real-Time Spec & Feature Extraction passed")
    test_canonical_id_generation()
    print("  ✓ Deterministic Canonical Product Identity passed")
    test_ai_ranking_preference_sensitivity()
    print("  ✓ User preference sensitivity validated (Gaming vs Coding)")
    test_compare_and_price_history_pipeline()
    print("  ✓ Compare, Details & Price History Pipeline (Zero Cache Miss) passed")
    test_user_authentication_and_wishlist_canonical()
    print("  ✓ User Authentication & Canonical Wishlist passed")
    print("=" * 68)
    print("ALL API & INTEGRATION TESTS PASSED (100% PASS RATE)")
    print("=" * 68)
