"""
BuyWise — ML Training Dataset Generator
=========================================
Synthesizes a reproducible, high-quality training dataset for supervised ML
product suitability prediction.

Pairs real product specifications (from cache & catalog) with realistic
user requirement profiles across all major categories.
"""

import os
import sys
import json
import sqlite3
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.services.product_search.feature_extractor import extract_features
from backend.services.product_search.normalizer import classify_category
from backend.services.product_search.recommendation_engine import CATEGORY_REQUIREMENTS

DB_PATH = os.path.join(os.path.dirname(__file__), '../../backend/buywise.db')
RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/amazonproducts/archive/Amazon-Products.csv')
OUTPUT_CSV_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/training_data.csv')

# Set fixed random seed for 100% reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_cached_products() -> List[Dict[str, Any]]:
    """Load products from SQLite cache."""
    products = []
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT product_json FROM product_cache").fetchall()
            for r in rows:
                if r["product_json"]:
                    try:
                        p = json.loads(r["product_json"])
                        if p.get("title") and p.get("price", 0) > 0:
                            products.append(p)
                    except Exception:
                        pass
            conn.close()
        except Exception as e:
            print(f"Notice: could not read product_cache: {e}")
    return products


def load_sample_raw_products(sample_size_per_cat: int = 400) -> List[Dict[str, Any]]:
    """Sample diverse real products from the raw dataset to ensure rich training diversity."""
    if not os.path.exists(RAW_CSV_PATH):
        return []

    try:
        df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
        # Clean currency
        def clean_p(v):
            if pd.isna(v): return 0.0
            s = str(v).replace('₹', '').replace(',', '').strip()
            try: return float(s)
            except ValueError: return 0.0

        df['p_clean'] = df['discount_price'].apply(clean_p)
        df['act_clean'] = df['actual_price'].apply(clean_p)
        df['p_clean'] = df['p_clean'].where(df['p_clean'] > 0, df['act_clean'])
        df['act_clean'] = df['act_clean'].where(df['act_clean'] > 0, df['p_clean'])
        df['rating_clean'] = pd.to_numeric(df['ratings'], errors='coerce').fillna(3.8)
        df['rev_clean'] = pd.to_numeric(df['no_of_ratings'].astype(str).str.replace(',', ''), errors='coerce').fillna(50).astype(int)

        valid = df[(df['p_clean'] > 100) & (df['name'].notna())].copy()

        # Sample across categories
        sampled_rows = []
        for main_cat, grp in valid.groupby('main_category'):
            n = min(len(grp), sample_size_per_cat)
            sampled_rows.append(grp.sample(n=n, random_state=RANDOM_SEED))

        if not sampled_rows:
            return []

        sampled_df = pd.concat(sampled_rows)

        results = []
        for _, row in sampled_df.iterrows():
            title = str(row['name']).strip()
            cat = classify_category(title)
            disc_pct = 0.0
            if row['act_clean'] > row['p_clean'] > 0:
                disc_pct = round(((row['act_clean'] - row['p_clean']) / row['act_clean']) * 100, 1)

            results.append({
                "title": title,
                "price": float(row['p_clean']),
                "original_price": float(row['act_clean']),
                "discount_percentage": float(disc_pct),
                "rating": float(row['rating_clean']),
                "review_count": int(row['rev_clean']),
                "category": cat,
                "brand": str(title.split()[0]) if title else "Generic"
            })
        return results
    except Exception as e:
        print(f"Notice: CSV sampling skipped ({e})")
        return []


def generate_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """Generate multi-category product-user suitability dataset."""
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    # 1. Collect candidate product pool
    cached_prods = load_cached_products()
    raw_prods = load_sample_raw_products(sample_size_per_cat=300)
    all_prods = cached_prods + raw_prods

    if not all_prods:
        # Fallback synthetic seed products if no cache or raw CSV
        all_prods = [
            {"title": "ASUS Vivobook 15 Intel Core i5 16GB RAM 512GB SSD FHD", "price": 49990, "original_price": 62990, "discount_percentage": 20, "rating": 4.3, "review_count": 520, "category": "laptops", "brand": "ASUS"},
            {"title": "Apple iPhone 15 128GB Blue 5G OLED A16 Bionic 48MP", "price": 69900, "original_price": 79900, "discount_percentage": 12, "rating": 4.6, "review_count": 3400, "category": "mobiles", "brand": "Apple"},
            {"title": "Samsung Galaxy S24 Ultra 5G 12GB RAM 256GB 200MP Camera", "price": 119999, "original_price": 134999, "discount_percentage": 11, "rating": 4.7, "review_count": 1800, "category": "mobiles", "brand": "Samsung"},
            {"title": "Sony WH-1000XM5 Wireless Active Noise Cancelling Headphones", "price": 28990, "original_price": 34990, "discount_percentage": 17, "rating": 4.5, "review_count": 950, "category": "electronics", "brand": "Sony"},
            {"title": "LG 55 Inch 4K Ultra HD Smart OLED TV Dolby Atmos WebOS", "price": 74990, "original_price": 99990, "discount_percentage": 25, "rating": 4.6, "review_count": 640, "category": "tvs", "brand": "LG"},
            {"title": "Daikin 1.5 Ton 5 Star Inverter Split AC Copper PM 2.5", "price": 44990, "original_price": 58990, "discount_percentage": 23, "rating": 4.4, "review_count": 1100, "category": "appliances", "brand": "Daikin"},
        ]

    print(f"Loaded {len(all_prods)} base product records for dataset synthesis.")

    rows = []
    priorities = ["value", "rating", "budget", "balanced"]
    categories = list(CATEGORY_REQUIREMENTS.keys())

    for i in range(n_samples):
        prod = random.choice(all_prods)
        title = prod.get("title") or prod.get("name", "")
        cat = prod.get("category") or classify_category(title)
        if cat not in CATEGORY_REQUIREMENTS:
            cat = random.choice(["laptops", "mobiles", "electronics", "appliances", "tvs"])

        price = float(prod.get("price", 1000))
        orig_price = float(prod.get("original_price", price))
        disc_pct = float(prod.get("discount_percentage", 0.0))
        rating = float(prod.get("rating") if prod.get("rating") is not None else 3.8)
        review_cnt = int(prod.get("review_count") or 50)
        brand = str(prod.get("brand") or title.split()[0] if title else "Generic")

        # Extract structured specifications
        specs = extract_features(title, cat)
        ram_gb = specs.get("ram_gb", 0)
        storage_gb = specs.get("storage_gb", 0)
        cpu_tier = specs.get("cpu_tier", 2)
        gpu_tier = specs.get("gpu_tier", 1)
        camera_mp = specs.get("camera_mp", 0)
        battery_mah = specs.get("battery_mah", 0)
        is_5g = 1 if specs.get("is_5g") else 0
        has_anc = 1 if specs.get("has_anc") else 0
        energy_star = specs.get("energy_star", 0)

        # Generate realistic user preference profile
        # User budget range around product price with random bounds
        budget_center = price * random.uniform(0.75, 1.35)
        user_max_budget = round(max(budget_center * random.uniform(0.9, 1.3), price * 0.8), -2)
        user_min_budget = round(max(0, user_max_budget * random.uniform(0.2, 0.7)), -2)
        user_priority = random.choice(priorities)

        # Requirements weights (1 to 5 stars)
        cat_reqs = CATEGORY_REQUIREMENTS.get(cat, CATEGORY_REQUIREMENTS["laptops"])
        user_req_weights = {req["key"]: random.randint(1, 5) for req in cat_reqs}

        # Calculate feature alignment score [0.0 - 1.0]
        alignment_scores = []
        for r_key, weight in user_req_weights.items():
            w_norm = weight / 5.0
            if r_key == "gaming":
                f_val = min((gpu_tier / 5.0) * 0.7 + (ram_gb / 16.0) * 0.3, 1.0)
            elif r_key == "coding":
                f_val = min((ram_gb / 16.0) * 0.5 + (cpu_tier / 5.0) * 0.5, 1.0)
            elif r_key == "camera":
                f_val = min(camera_mp / 108.0, 1.0) if camera_mp > 0 else 0.5
            elif r_key == "battery":
                f_val = min(battery_mah / 5000.0, 1.0) if battery_mah > 0 else 0.6
            elif r_key == "five_g":
                f_val = 1.0 if is_5g else 0.4
            elif r_key == "anc":
                f_val = 1.0 if has_anc else 0.3
            elif r_key == "energy_star":
                f_val = min(energy_star / 5.0, 1.0) if energy_star > 0 else 0.6
            elif r_key == "value":
                f_val = min(max(disc_pct, 5.0) / 40.0, 1.0)
            else:
                f_val = min(rating / 5.0, 1.0)

            alignment_scores.append(w_norm * f_val)

        spec_fit = float(np.mean(alignment_scores)) if alignment_scores else 0.5

        # Budget fit: 1.0 if under max budget, linearly penalized if over
        if price <= user_max_budget:
            budget_fit = 1.0 - (price / (user_max_budget * 1.5)) * 0.3
        else:
            budget_fit = max(0.0, 1.0 - ((price - user_max_budget) / user_max_budget) * 2.0)

        # Rating quality score (Bayesian smoothed)
        c, m = 10, 3.8
        smoothed_r = (c * m + rating * review_cnt) / (c + review_cnt)
        rating_score = min(smoothed_r / 5.0, 1.0)

        # Priority weight adjustments
        if user_priority == "rating":
            target_score = (0.45 * rating_score + 0.35 * spec_fit + 0.10 * budget_fit + 0.10 * (disc_pct / 100)) * 100
        elif user_priority == "budget":
            target_score = (0.40 * budget_fit + 0.25 * spec_fit + 0.20 * (disc_pct / 100) + 0.15 * rating_score) * 100
        elif user_priority == "value":
            target_score = (0.35 * (disc_pct / 100) + 0.30 * spec_fit + 0.20 * rating_score + 0.15 * budget_fit) * 100
        else:  # balanced
            target_score = (0.35 * spec_fit + 0.25 * rating_score + 0.20 * budget_fit + 0.20 * (disc_pct / 100)) * 100

        # Add slight natural stochastic noise (+- 1.5)
        noise = np.random.normal(0, 1.2)
        final_suitability = round(float(np.clip(target_score + noise, 5.0, 99.0)), 1)

        # Key preference feature summaries
        primary_req_weight = max(user_req_weights.values()) if user_req_weights else 3
        avg_req_weight = float(np.mean(list(user_req_weights.values()))) if user_req_weights else 3.0

        rows.append({
            "category": cat,
            "price": price,
            "original_price": orig_price,
            "discount_percentage": disc_pct,
            "rating": rating,
            "review_count": review_cnt,
            "ram_gb": ram_gb,
            "storage_gb": storage_gb,
            "cpu_tier": cpu_tier,
            "gpu_tier": gpu_tier,
            "camera_mp": camera_mp,
            "battery_mah": battery_mah,
            "is_5g": is_5g,
            "has_anc": has_anc,
            "energy_star": energy_star,
            "user_max_budget": user_max_budget,
            "user_min_budget": user_min_budget,
            "user_priority": user_priority,
            "primary_req_weight": primary_req_weight,
            "avg_req_weight": avg_req_weight,
            "price_to_budget_ratio": round(price / max(user_max_budget, 1.0), 3),
            "spec_fit_score": round(spec_fit, 3),
            "suitability_score": final_suitability
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV_PATH, index=False)
    print("=" * 60)
    print(f"Generated {len(df_out)} training samples saved to: {OUTPUT_CSV_PATH}")
    print(f"Feature columns: {list(df_out.columns)}")
    print(f"Target 'suitability_score' range: {df_out['suitability_score'].min()} to {df_out['suitability_score'].max()} (mean: {df_out['suitability_score'].mean():.2f})")
    print("=" * 60)
    return df_out


if __name__ == "__main__":
    generate_dataset(n_samples=5000)
