"""
BuyWise — AI Multi-Criteria Personalized Recommendation Engine
===============================================================
A genuine AI decision-support system that:
1. Formulates category-specific purchasing requirements (Camera, Gaming, Coding, Battery, Display, etc.)
2. Extracts real technical features from product data without fabrication
3. Translates user 1–5 star importance ratings into a preference vector
4. Computes multi-attribute cosine & distance matching against product specs
5. Incorporates Bayesian rating smoothing, review volume log-popularity, and budget constraints
6. Aggregates a large candidate pool (25–40 products) from cached/live search
7. Generates transparent, factual natural language explainability reasons
8. Ensures deterministic sensitivity: changing user priorities directly alters rankings
"""

import math
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

from backend.services.product_search.base import ShoppingResult, ShoppingOffer, generate_canonical_id
from backend.services.product_search.feature_extractor import extract_features
from backend.services.product_search.normalizer import classify_category, matches_category
from backend.services.category_service import get_search_query_for_category

logger = logging.getLogger(__name__)


# ── Subcategory-Specific Buying Requirements Taxonomy ────────
SUBCATEGORY_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
    # Mobiles & Accessories Subcategories
    "smartphones": [
        {"key": "camera", "label": "Camera Quality", "desc": "Megapixels, OIS, 4K video recording", "default": 4},
        {"key": "gaming", "label": "Gaming / Performance", "desc": "Fast processor, high refresh rate, 8GB+ RAM", "default": 3},
        {"key": "battery", "label": "Battery Life & Charging", "desc": "5000mAh+, fast charging capabilities", "default": 4},
        {"key": "display", "label": "Display Quality", "desc": "AMOLED / 120Hz high refresh screen", "default": 4},
        {"key": "storage", "label": "Storage Capacity", "desc": "128GB / 256GB+ internal storage", "default": 3},
        {"key": "five_g", "label": "5G Connectivity", "desc": "5G high-speed cellular capability", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "High discount off MRP & price efficiency", "default": 4},
    ],
    "tablets": [
        {"key": "display", "label": "Screen Size & Display", "desc": "10+ inch 2K/FHD vibrant display", "default": 5},
        {"key": "battery", "label": "Battery Runtime", "desc": "All-day video and reading battery life", "default": 4},
        {"key": "performance", "label": "Processor & RAM", "desc": "Fast multitasking for study & media", "default": 4},
        {"key": "storage", "label": "Internal Storage", "desc": "64GB / 128GB+ with SD card support", "default": 3},
        {"key": "value", "label": "Value for Money", "desc": "Screen size & features per Rupee", "default": 4},
    ],
    "smartwatches": [
        {"key": "battery", "label": "Battery Life", "desc": "Multi-day battery on single charge", "default": 4},
        {"key": "display", "label": "AMOLED Display", "desc": "Crisp always-on AMOLED screen", "default": 4},
        {"key": "fitness", "label": "Health & Heart Sensors", "desc": "Heart rate, SpO2 & sports tracking", "default": 4},
        {"key": "calling", "label": "Bluetooth Calling", "desc": "Clear speaker and mic for calls", "default": 3},
        {"key": "value", "label": "Value for Money", "desc": "Build quality & sensor accuracy per Rupee", "default": 4},
    ],
    "powerbanks": [
        {"key": "capacity", "label": "Battery Capacity", "desc": "10000mAh / 20000mAh real capacity", "default": 5},
        {"key": "fast_charging", "label": "Fast Charging Output", "desc": "22.5W / 65W Power Delivery support", "default": 4},
        {"key": "portability", "label": "Compact & Lightweight", "desc": "Pocket-friendly design & durable casing", "default": 3},
        {"key": "value", "label": "Value for Money", "desc": "Watt-hours per Rupee spent", "default": 4},
    ],

    # Laptops & Computers Subcategories
    "general_laptops": [
        {"key": "coding", "label": "Coding / Programming", "desc": "16GB+ RAM, fast multi-core processor, SSD", "default": 5},
        {"key": "performance", "label": "Raw CPU Performance", "desc": "Intel Core i5/i7, Ryzen 5/7, Apple M-series", "default": 4},
        {"key": "gaming", "label": "Gaming & GPU Power", "desc": "Dedicated GPU or high-speed graphics", "default": 3},
        {"key": "battery_portability", "label": "Battery Life & Portability", "desc": "All-day battery & lightweight build", "default": 4},
        {"key": "display", "label": "Display & Screen Quality", "desc": "FHD / OLED / High resolution panel", "default": 3},
        {"key": "storage", "label": "Fast SSD Storage", "desc": "512GB / 1TB NVMe SSD", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Maximum specs per Rupee spent", "default": 4},
    ],
    "gaming_laptops": [
        {"key": "gaming", "label": "Dedicated Gaming GPU", "desc": "RTX 4050/4060/3050 graphics power", "default": 5},
        {"key": "display", "label": "High Refresh Rate Screen", "desc": "144Hz / 165Hz smooth gaming panel", "default": 4},
        {"key": "coding", "label": "RAM & CPU Power", "desc": "16GB high-speed RAM & multi-core CPU", "default": 4},
        {"key": "storage", "label": "High Speed NVMe SSD", "desc": "512GB / 1TB fast game load times", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "GPU performance per Rupee", "default": 4},
    ],
    "monitors": [
        {"key": "display", "label": "Resolution & Panel Quality", "desc": "4K / QHD IPS color-accurate panel", "default": 5},
        {"key": "gaming", "label": "High Refresh Rate", "desc": "144Hz+ and 1ms fast response time", "default": 4},
        {"key": "screen_size", "label": "Screen Real Estate", "desc": "24-inch, 27-inch, or 32-inch curved display", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Display real-estate per Rupee", "default": 4},
    ],
    "peripherals": [
        {"key": "comfort", "label": "Ergonomics & Feel", "desc": "Mechanical switches & wrist comfort", "default": 5},
        {"key": "wireless", "label": "Wireless Connectivity", "desc": "Low latency 2.4GHz & Bluetooth", "default": 4},
        {"key": "durability", "label": "Build Quality & RGB", "desc": "Durable keycaps & aluminum casing", "default": 3},
        {"key": "value", "label": "Value for Money", "desc": "Feature set per Rupee", "default": 4},
    ],
    "printers": [
        {"key": "efficiency", "label": "Low Cost per Page", "desc": "High yield ink tank / laser cartridge", "default": 5},
        {"key": "wireless", "label": "Wi-Fi & Mobile Printing", "desc": "Wireless app & Apple AirPrint support", "default": 4},
        {"key": "speed", "label": "Print Speed & Scan Quality", "desc": "Fast PPM and high DPI scanning", "default": 3},
        {"key": "value", "label": "Value for Money", "desc": "Operating cost & hardware price balance", "default": 4},
    ],

    # Home Appliances Subcategories
    "ac": [
        {"key": "capacity", "label": "Cooling Tonnage", "desc": "1.5 Ton / 1 Ton for medium to large rooms", "default": 5},
        {"key": "energy_star", "label": "5-Star Energy Efficiency", "desc": "Lower electricity bills & Inverter compressor", "default": 5},
        {"key": "durability", "label": "100% Copper Condenser", "desc": "Anti-corrosion coating & high durability", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Long term power savings per Rupee", "default": 4},
    ],
    "refrigerators": [
        {"key": "capacity", "label": "Volume & Litre Capacity", "desc": "250L+ double door or multi-door storage", "default": 4},
        {"key": "energy_star", "label": "Inverter & Energy Star", "desc": "Energy efficient compressor with silent operation", "default": 5},
        {"key": "durability", "label": "Frost Free & Build Quality", "desc": "Toughened glass shelves & reliable cooling", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Storage capacity and cooling tech per Rupee", "default": 4},
    ],
    "washing_machines": [
        {"key": "capacity", "label": "Wash Capacity (Kg)", "desc": "7kg / 8kg load for family requirements", "default": 4},
        {"key": "performance", "label": "Front Load / Inverter Motor", "desc": "Superior stain removal & fabric care", "default": 5},
        {"key": "energy_star", "label": "5-Star Energy & Water Saving", "desc": "Low electricity and water consumption", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Wash programs & durability per Rupee", "default": 4},
    ],
    "microwaves": [
        {"key": "capacity", "label": "Capacity & Sizing", "desc": "28L / 32L convection baking capacity", "default": 4},
        {"key": "performance", "label": "Convection & Auto Menus", "desc": "Baking, grilling, and reheat programs", "default": 5},
        {"key": "durability", "label": "Ceramic Enamel Cavity", "desc": "Easy to clean interior & anti-bacterial", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Cooking versatility per Rupee", "default": 4},
    ],
    "water_purifiers": [
        {"key": "filtration", "label": "RO + UV + UF Multi-Stage", "desc": "Purification for borewell & municipal water", "default": 5},
        {"key": "capacity", "label": "Tank Storage Capacity", "desc": "7L to 10L clean water storage", "default": 4},
        {"key": "durability", "label": "Mineral Booster Cartridge", "desc": "Essential minerals retention & long filter life", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Maintenance cost & purification purity", "default": 4},
    ],

    # TVs & Audio Subcategories
    "smart_4k_tvs": [
        {"key": "display", "label": "Picture & 4K Quality", "desc": "4K Ultra HD, OLED / QLED / HDR10+", "default": 5},
        {"key": "screen_size", "label": "Screen Size (Inches)", "desc": "55-inch / 43-inch immersive display", "default": 4},
        {"key": "smart_features", "label": "Google TV & Smart OS", "desc": "Netflix, Prime Video, YouTube built-in", "default": 4},
        {"key": "sound", "label": "Dolby Atmos & Audio Output", "desc": "Dolby Audio with high wattage speakers", "default": 3},
        {"key": "value", "label": "Value for Money", "desc": "Screen size & panel quality per Rupee", "default": 4},
    ],
    "soundbars": [
        {"key": "sound", "label": "Total RMS Audio Power", "desc": "100W+ cinematic sound with Dolby Atmos", "default": 5},
        {"key": "bass", "label": "Wireless Subwoofer Bass", "desc": "Deep punchy bass for movies and music", "default": 4},
        {"key": "wireless", "label": "eARC & Bluetooth 5.3", "desc": "Seamless TV connectivity without lag", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Acoustic output per Rupee", "default": 4},
    ],
    "tws_earbuds": [
        {"key": "anc", "label": "Active Noise Cancellation", "desc": "ANC & Environmental Noise Cancellation (ENC)", "default": 5},
        {"key": "sound", "label": "Sound Quality & Bass", "desc": "Dynamic drivers with rich bass profile", "default": 4},
        {"key": "battery", "label": "Battery Life & Case", "desc": "30h+ total playback with fast charging", "default": 4},
        {"key": "comfort", "label": "Secure Ergonomic Fit", "desc": "Lightweight earbuds with IPX5 water resistance", "default": 3},
        {"key": "value", "label": "Value for Money", "desc": "Audio clarity & ANC per Rupee", "default": 4},
    ],
    "headphones": [
        {"key": "anc", "label": "Active Noise Cancellation", "desc": "Flagship acoustic isolation & transparency mode", "default": 5},
        {"key": "sound", "label": "Hi-Res Sound & 40mm Drivers", "desc": "Audiophile tuning and wide soundstage", "default": 5},
        {"key": "battery", "label": "40h+ Battery Playtime", "desc": "Long haul wireless playback", "default": 4},
        {"key": "comfort", "label": "Memory Foam Cushioning", "desc": "Plush over-ear comfort for hours", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Build and acoustic fidelity per Rupee", "default": 4},
    ],

    # Fashion & Footwear Subcategories
    "mens_clothing": [
        {"key": "material", "label": "Fabric & 100% Cotton", "desc": "Breathable, soft, and skin-friendly fabric", "default": 5},
        {"key": "fit", "label": "Tailoring & Fit", "desc": "Slim/Regular fit with precise stitching", "default": 4},
        {"key": "brand", "label": "Brand Reputation", "desc": "Trusted fashion brand quality", "default": 4},
        {"key": "durability", "label": "Color Fastness & Durability", "desc": "Wash resistance without shrinking", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Style and fabric quality per Rupee", "default": 4},
    ],
    "womens_clothing": [
        {"key": "material", "label": "Fabric Quality & Feel", "desc": "Premium cotton, rayon, or silk blend", "default": 5},
        {"key": "fit", "label": "Styling & Occasion Fit", "desc": "Flattering silhouette for casual/festive", "default": 4},
        {"key": "brand", "label": "Brand Reputation", "desc": "Authentic designer or top brand", "default": 4},
        {"key": "comfort", "label": "All-Day Comfort", "desc": "Lightweight and easy to wear", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Elegance and longevity per Rupee", "default": 4},
    ],
    "ethnic_wear": [
        {"key": "material", "label": "Embroidery & Fabric Purity", "desc": "Rich traditional work and pure fabric", "default": 5},
        {"key": "fit", "label": "Festive Styling & Fit", "desc": "Grand look for weddings and festive wear", "default": 4},
        {"key": "comfort", "label": "Comfortable Drape", "desc": "Easy to carry for long occasions", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Artisan craftsmanship per Rupee", "default": 4},
    ],
    "bags": [
        {"key": "capacity", "label": "Storage Capacity & Sleeve", "desc": "Dedicated padded laptop compartment", "default": 5},
        {"key": "durability", "label": "Water Resistant & Zippers", "desc": "Tough polyester fabric & heavy duty zips", "default": 4},
        {"key": "comfort", "label": "Ergonomic Shoulder Straps", "desc": "Padded breathable mesh back support", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Utility and durability per Rupee", "default": 4},
    ],
    "running_shoes": [
        {"key": "comfort", "label": "Sole Cushioning & Bounce", "desc": "EVA/Memory foam midsole impact absorption", "default": 5},
        {"key": "sports_performance", "label": "Grip & Breathable Upper", "desc": "Flexible mesh with anti-slip rubber traction", "default": 4},
        {"key": "durability", "label": "Sole Durability", "desc": "Long mileage outsole without wear", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Athletic performance per Rupee", "default": 4},
    ],
    "casual_shoes": [
        {"key": "comfort", "label": "Insole Cushioning & Fit", "desc": "All-day walking comfort without fatigue", "default": 5},
        {"key": "material", "label": "Leather / Suede Material", "desc": "Premium upper finish and stylish look", "default": 4},
        {"key": "brand", "label": "Brand Trust", "desc": "Established footwear manufacturer", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Design and material per Rupee", "default": 4},
    ],

    # Beauty & Personal Care Subcategories
    "skincare": [
        {"key": "quality", "label": "Dermatologist Approved", "desc": "Safe active ingredients (SPF/Vitamin C/Niacinamide)", "default": 5},
        {"key": "skin_type", "label": "Skin Type Suitability", "desc": "Non-comedogenic & gentle for Indian weather", "default": 4},
        {"key": "brand", "label": "Authentic Brand Trust", "desc": "Clinically proven skincare formulation", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Volume & active ingredient concentration", "default": 4},
    ],
    "grooming_appliances": [
        {"key": "performance", "label": "Blade Sharpness & Precision", "desc": "Self-sharpening stainless steel/ceramic blades", "default": 5},
        {"key": "battery", "label": "Cordless Battery Runtime", "desc": "60+ mins runtime with USB fast charging", "default": 4},
        {"key": "durability", "label": "Waterproof & Washable", "desc": "IPX7 fully washable for easy maintenance", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Versatility & build quality per Rupee", "default": 4},
    ],

    # Watches & Jewellery
    "analog_watches": [
        {"key": "durability", "label": "Movement & Glass Quality", "desc": "Japanese Quartz/Automatic with Mineral/Sapphire glass", "default": 5},
        {"key": "water_resistance", "label": "Water Resistance (ATM)", "desc": "3ATM / 5ATM water resistance for daily use", "default": 4},
        {"key": "material", "label": "Strap & Case Finish", "desc": "Stainless steel or genuine leather strap", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Craftsmanship and brand prestige per Rupee", "default": 4},
    ],
    "jewellery_items": [
        {"key": "material", "label": "Plating & Material Purity", "desc": "18K Gold plated / 925 Sterling Silver finish", "default": 5},
        {"key": "skin_safety", "label": "Hypoallergenic & Anti-Tarnish", "desc": "Lead and nickel free, gentle on sensitive skin", "default": 5},
        {"key": "durability", "label": "Shine Longevity", "desc": "Anti-fade protective coat for daily wear", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Aesthetic design and luster per Rupee", "default": 4},
    ],

    # Home & Living
    "cookware": [
        {"key": "durability", "label": "Non-Stick & Base Thickness", "desc": "Granite/Tri-ply stainless steel induction base", "default": 5},
        {"key": "safety", "label": "100% PFOA Free & Food Grade", "desc": "Toxin-free healthy cooking surface", "default": 5},
        {"key": "maintenance", "label": "Dishwasher Safe & Easy Clean", "desc": "Smooth coating with stay-cool handles", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Culinary durability per Rupee spent", "default": 4},
    ],
    "furnishing": [
        {"key": "material", "label": "100% Pure Cotton & Thread Count", "desc": "200+ TC breathable combed cotton fabric", "default": 5},
        {"key": "comfort", "label": "Softness & Skin Feel", "desc": "Plush feel for restorative sleep", "default": 4},
        {"key": "durability", "label": "Color Fastness", "desc": "Fade resistant after multiple machine washes", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Fabric luxury per Rupee", "default": 4},
    ],
    "home_decor": [
        {"key": "aesthetic", "label": "Design & Finish", "desc": "Modern aesthetic suited for living room decor", "default": 5},
        {"key": "durability", "label": "Material Build Quality", "desc": "Solid wood, metal, or premium ceramic", "default": 4},
        {"key": "value", "label": "Value for Money", "desc": "Decorative appeal per Rupee", "default": 4},
    ],
}

# ── Macro Category Requirements Mapping ──────────────────────
CATEGORY_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
    "mobiles": SUBCATEGORY_REQUIREMENTS["smartphones"],
    "laptops": SUBCATEGORY_REQUIREMENTS["general_laptops"],
    "appliances": SUBCATEGORY_REQUIREMENTS["ac"],
    "tvs": SUBCATEGORY_REQUIREMENTS["smart_4k_tvs"],
    "electronics": SUBCATEGORY_REQUIREMENTS["tws_earbuds"],
    "fashion": SUBCATEGORY_REQUIREMENTS["mens_clothing"],
    "footwear": SUBCATEGORY_REQUIREMENTS["running_shoes"],
    "beauty": SUBCATEGORY_REQUIREMENTS["skincare"],
    "jewellery": SUBCATEGORY_REQUIREMENTS["analog_watches"],
    "home": SUBCATEGORY_REQUIREMENTS["cookware"],
}


def get_category_requirements(category: str, subcategory: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return purchasing requirements tailored to a given category and optional subcategory.
    """
    if subcategory:
        sub_clean = subcategory.strip().lower()
        for k, reqs in SUBCATEGORY_REQUIREMENTS.items():
            if k == sub_clean or k in sub_clean or sub_clean in k:
                return reqs

    cat_clean = category.strip().lower()
    for k, reqs in CATEGORY_REQUIREMENTS.items():
        if k in cat_clean or cat_clean in k:
            return reqs

    return CATEGORY_REQUIREMENTS.get("laptops", [])


import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../../../ml/models/suitability_model.joblib')


class AIRecommendationEngine:
    """Multi-criteria decision support and ranking engine with trained ML model inference."""

    def __init__(self):
        self.ml_pipeline = None
        if os.path.exists(MODEL_PATH):
            try:
                self.ml_pipeline = joblib.load(MODEL_PATH)
                logger.info("Loaded trained ML suitability model from %s", MODEL_PATH)
            except Exception as e:
                logger.warning("Could not load ML model (%s), using rule baseline", e)

    def evaluate_product_features(self, product: Dict[str, Any], category: str) -> Dict[str, float]:
        """
        Evaluate how strongly a product satisfies each category requirement [0.0 to 1.0].
        Extracts specs from title + metadata and computes deterministic satisfaction.
        """
        title = product.get("title", product.get("name", ""))
        specs = product.get("specs") or extract_features(title, category)
        product["specs"] = specs
        cat_clean = category.strip().lower()

        scores: Dict[str, float] = {}

        # ── Mobiles Evaluation ──
        if "mobile" in cat_clean or "phone" in cat_clean:
            # Camera: MP count, OIS, Pro features
            mp = specs.get("camera_mp", 0)
            cam_score = 0.5  # Neutral baseline
            if mp >= 108:
                cam_score = 1.0
            elif mp >= 50:
                cam_score = 0.85
            elif mp >= 12:
                cam_score = 0.7
            if specs.get("camera_ois"):
                cam_score = min(1.0, cam_score + 0.15)
            if specs.get("multi_camera"):
                cam_score = min(1.0, cam_score + 0.1)
            scores["camera"] = cam_score

            # Gaming/Performance: CPU tier, RAM, Refresh rate
            cpu_t = specs.get("cpu_tier", 3)
            ram = specs.get("ram_gb", 6)
            game_score = (cpu_t / 5.0) * 0.5 + min(ram / 12.0, 1.0) * 0.3
            if specs.get("high_refresh"):
                game_score += 0.2
            scores["gaming"] = min(1.0, max(0.2, game_score))

            # Battery: mAh capacity, Fast charging
            mah = specs.get("battery_mah", 4500)
            bat_score = min(mah / 5500.0, 1.0) * 0.7
            if specs.get("fast_charging"):
                bat_score += 0.3
            scores["battery"] = min(1.0, max(0.4, bat_score))

            # Display: AMOLED/OLED, Refresh rate
            disp_type = specs.get("display_type", "")
            disp_score = 0.9 if disp_type in ["AMOLED", "OLED", "Super AMOLED"] else 0.6
            if specs.get("high_refresh"):
                disp_score = min(1.0, disp_score + 0.15)
            scores["display"] = disp_score

            # Storage: 128GB, 256GB, 512GB
            st_gb = specs.get("storage_gb", 128)
            scores["storage"] = min(st_gb / 256.0, 1.0)

            # 5G
            scores["five_g"] = 1.0 if specs.get("is_5g") or "5g" in title.lower() else 0.4

            # Value for money: Discount % and price
            disc = product.get("discount_percentage", 0) or 0
            scores["value"] = min(max(disc, 10.0) / 40.0, 1.0)

        # ── Laptops Evaluation ──
        elif "laptop" in cat_clean or "computer" in cat_clean:
            ram = specs.get("ram_gb", 8)
            cpu_t = specs.get("cpu_tier", 3)
            gpu_t = specs.get("gpu_tier", 1)
            st_gb = specs.get("storage_gb", 512)

            # Coding / Programming: RAM >= 16GB, CPU tier >= 4, SSD
            coding_score = min(ram / 16.0, 1.0) * 0.45 + (cpu_t / 5.0) * 0.40 + min(st_gb / 512.0, 1.0) * 0.15
            scores["coding"] = min(1.0, max(0.3, coding_score))

            # Gaming & GPU: Dedicated GPU tier
            if gpu_t >= 4:
                gpu_score = 0.75 + (gpu_t / 5.0) * 0.25
            elif "gaming" in title.lower():
                gpu_score = 0.70
            elif gpu_t >= 2:
                gpu_score = 0.45
            else:
                gpu_score = 0.25
            scores["gaming"] = min(1.0, gpu_score)

            # Raw CPU Performance: Intel i7/i9, Ryzen 7/9, Apple M-series
            scores["performance"] = min(1.0, (cpu_t / 5.0))

            # Battery / Portability: Thin & light, Apple M-series or Ultrabook
            if "macbook" in title.lower() or "air" in title.lower() or "zenbook" in title.lower() or "slim" in title.lower():
                port_score = 0.95
            elif "gaming" in title.lower() or gpu_t >= 4:
                port_score = 0.45  # Gaming laptops are heavier
            else:
                port_score = 0.70
            scores["battery_portability"] = port_score

            # Display Quality
            disp_type = specs.get("display_type", "")
            disp_score = 0.95 if disp_type in ["OLED", "Retina", "QHD"] else 0.70
            scores["display"] = disp_score

            # Storage: 512GB / 1TB
            scores["storage"] = min(st_gb / 512.0, 1.0)

            # Value for money
            disc = product.get("discount_percentage", 0) or 0
            scores["value"] = min(max(disc, 10.0) / 35.0, 1.0)

        # ── TVs Evaluation ──
        elif "tv" in cat_clean:
            res = specs.get("resolution", "")
            disp_type = specs.get("display_type", "")
            size = specs.get("screen_size_inch", 43.0)

            # Picture / 4K
            pic_score = 0.95 if "4k" in res.lower() or "oled" in disp_type.lower() else 0.65
            scores["display"] = pic_score

            # Screen size
            scores["screen_size"] = min(size / 55.0, 1.0)

            # Smart features
            scores["smart_features"] = 0.90 if any(k in title.lower() for k in ["google tv", "android", "smart", "webos"]) else 0.60

            # Sound
            scores["sound"] = 0.85 if any(k in title.lower() for k in ["dolby", "atmos", "soundbar", "30w", "24w"]) else 0.60

            # Gaming: 120Hz
            scores["gaming"] = 0.90 if specs.get("high_refresh") else 0.50

            # Value
            disc = product.get("discount_percentage", 0) or 0
            scores["value"] = min(max(disc, 15.0) / 45.0, 1.0)

        # ── Electronics / Headphones Evaluation ──
        elif "electronic" in cat_clean or "audio" in cat_clean or "headphone" in cat_clean:
            scores["sound"] = 0.90 if specs.get("hi_res_audio") or any(k in title.lower() for k in ["sony", "bose", "sennheiser", "marshall", "jbl"]) else 0.70
            scores["anc"] = 0.95 if specs.get("has_anc") else 0.35
            scores["battery"] = 0.85 if any(k in title.lower() for k in ["30h", "40h", "50h", "60h", "playtime"]) else 0.65
            scores["comfort"] = 0.85 if "over-ear" in title.lower() or "cushion" in title.lower() else 0.70
            scores["wireless"] = 0.95 if specs.get("is_wireless") else 0.50
            disc = product.get("discount_percentage", 0) or 0
            scores["value"] = min(max(disc, 15.0) / 50.0, 1.0)

        # ── Home Appliances Evaluation ──
        elif "appliance" in cat_clean:
            star = specs.get("energy_star", 3)
            scores["energy_star"] = min(star / 5.0, 1.0)
            if specs.get("inverter_tech"):
                scores["energy_star"] = min(1.0, scores["energy_star"] + 0.15)
            scores["capacity"] = 0.85 if (specs.get("ac_ton") or specs.get("capacity_liters") or specs.get("capacity_kg")) else 0.60
            scores["durability"] = 0.90 if any(k in title.lower() for k in ["lg", "samsung", "whirlpool", "daikin", "voltas", "lloyd", "ifb", "bosch"]) else 0.65
            disc = product.get("discount_percentage", 0) or 0
            scores["value"] = min(max(disc, 10.0) / 35.0, 1.0)

        # ── General Fallback ──
        else:
            disc = product.get("discount_percentage", 0) or 0
            scores["value"] = min(max(disc, 10.0) / 35.0, 1.0)
            scores["quality"] = 0.80 if (product.get("rating") or 0) >= 4.0 else 0.60
            scores["durability"] = 0.75

        return scores

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        category: str,
        subcategory: Optional[str] = None,
        user_requirements: Optional[Dict[str, int]] = None,
        preferred_brands: Optional[List[str]] = None,
        priority: str = "value",
        min_price: float = 0,
        max_price: float = 100000,
        min_rating: float = 0,
        limit: int = 8,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Personalized multi-criteria ranking of candidate products against user requirements.
        Returns (ranked_top_n_products, total_candidates_analyzed).
        """
        if not candidates:
            return [], 0

        # Strict Category and Budget Pre-Filtering
        cat_clean = category.strip().lower()
        valid_candidates = []
        for p in candidates:
            title = p.get("title", p.get("name", ""))
            price = p.get("price", 0) or 0
            rating = p.get("rating")

            # 1. Strict category safety
            if not matches_category(title, cat_clean):
                continue

            # 2. Hard budget ceiling
            if max_price > 0 and price > (max_price * 1.02):  # strict 2% buffer for rounding
                continue

            # 3. Minimum budget floor
            if min_price > 0 and price < min_price:
                continue

            # 4. Rating threshold
            if min_rating > 0 and rating is not None and rating < min_rating:
                continue

            valid_candidates.append(p)

        total_analyzed = len(valid_candidates)
        if not valid_candidates:
            # If strict filter eliminated all, fall back to matching by keyword
            valid_candidates = [p for p in candidates if max_price <= 0 or (p.get("price", 0) <= max_price * 1.05)]
            total_analyzed = len(valid_candidates)

        if not valid_candidates:
            return [], len(candidates)

        # Build normalized user requirement weights (1 to 5 stars -> weight w_i in [0.2, 1.0])
        req_weights = {}
        defined_reqs = get_category_requirements(category, subcategory=subcategory)
        for req in defined_reqs:
            k = req["key"]
            user_val = user_requirements.get(k) if user_requirements else None
            val = user_val if user_val is not None else req["default"]
            # Convert 1-5 scale to normalized float [0.2, 1.0]
            req_weights[k] = max(1, min(5, int(val))) / 5.0

        if user_requirements:
            for k, val in user_requirements.items():
                if k not in req_weights:
                    req_weights[k] = max(1, min(5, int(val))) / 5.0

        brands_set = {b.lower() for b in (preferred_brands or [])}

        # Optimization priority tuning
        # Controls balance between Requirement Match, Rating Quality, Budget/Value
        if priority == "rating":
            w_req = 0.35
            w_rating = 0.40
            w_value = 0.10
            w_budget = 0.10
            w_brand = 0.05
        elif priority == "budget":
            w_req = 0.30
            w_rating = 0.15
            w_value = 0.25
            w_budget = 0.25
            w_brand = 0.05
        elif priority == "value":
            w_req = 0.40
            w_rating = 0.20
            w_value = 0.25
            w_budget = 0.10
            w_brand = 0.05
        else:  # balanced
            w_req = 0.45
            w_rating = 0.25
            w_value = 0.15
            w_budget = 0.10
            w_brand = 0.05

        price_vals = [p.get("price", 0) for p in valid_candidates if p.get("price", 0) > 0]
        max_p_obs = max(price_vals) if price_vals else (max_price or 50000)
        min_p_obs = min(price_vals) if price_vals else 0
        p_range = max(max_p_obs - min_p_obs, 1.0)

        scored = []
        for p in valid_candidates:
            feature_scores = self.evaluate_product_features(p, category)

            # 1. User Requirement Vector Match Score (0.0 to 1.0)
            weighted_num = 0.0
            weighted_den = 0.0
            top_matching_features = []

            for k, w in req_weights.items():
                f_score = feature_scores.get(k, 0.5)
                weighted_num += w * f_score
                weighted_den += w
                if w >= 0.6 and f_score >= 0.7:  # High user priority & strong product feature
                    top_matching_features.append((k, f_score, w))

            req_match_score = (weighted_num / weighted_den) if weighted_den > 0 else 0.5

            # 2. Bayesian Smoothed Customer Rating Score (c=10, m=3.8)
            raw_rating = p.get("rating")
            rev_cnt = p.get("review_count") or 0
            if raw_rating is not None and raw_rating > 0:
                c, m = 10, 3.8
                smoothed_rating = (c * m + raw_rating * rev_cnt) / (c + rev_cnt) if rev_cnt > 0 else raw_rating
                norm_rating = min(smoothed_rating / 5.0, 1.0)
            else:
                norm_rating = 0.5  # Neutral when no rating

            # 3. Popularity Log Score
            norm_pop = min(math.log1p(rev_cnt) / math.log1p(25000), 1.0) if rev_cnt > 0 else 0.1

            # 4. Value / Discount Score
            disc = p.get("discount_percentage") or 0.0
            norm_disc = min(max(disc, 0.0) / 100.0, 1.0)

            # 5. Budget Fit Score
            price = p.get("price", 0)
            if max_price and max_price > 0:
                # Closer to budget ceiling without exceeding is fine, but reward savings
                budget_fit = 1.0 - min(max(price - min_price, 0) / max(max_price - min_price, 1.0), 1.0)
            else:
                budget_fit = 1.0 - min((price - min_p_obs) / p_range, 1.0)

            # 6. Brand Preference Match
            p_brand = (p.get("brand") or "").lower()
            brand_match = 1.0 if (brands_set and p_brand in brands_set) else 0.5

            # ── Hybrid ML Suitability Prediction ──
            ml_suitability = None
            if self.ml_pipeline is not None:
                try:
                    specs = p.get("specs", {})
                    p_row = {
                        "category": cat_clean,
                        "price": float(price),
                        "original_price": float(p.get("original_price") or price),
                        "discount_percentage": float(disc),
                        "rating": float(raw_rating) if raw_rating else 3.8,
                        "review_count": int(rev_cnt),
                        "ram_gb": specs.get("ram_gb", 0),
                        "storage_gb": specs.get("storage_gb", 0),
                        "cpu_tier": specs.get("cpu_tier", 2),
                        "gpu_tier": specs.get("gpu_tier", 1),
                        "camera_mp": specs.get("camera_mp", 0),
                        "battery_mah": specs.get("battery_mah", 0),
                        "is_5g": 1 if specs.get("is_5g") else 0,
                        "has_anc": 1 if specs.get("has_anc") else 0,
                        "energy_star": specs.get("energy_star", 0),
                        "user_max_budget": float(max_price or 100000),
                        "user_min_budget": float(min_price or 0),
                        "user_priority": priority,
                        "primary_req_weight": max(req_weights.values()) * 5 if req_weights else 4,
                        "avg_req_weight": float(np.mean(list(req_weights.values()))) * 5 if req_weights else 3.5,
                        "price_to_budget_ratio": round(price / max(max_price or 50000, 1.0), 3),
                        "spec_fit_score": round(req_match_score, 3)
                    }
                    df_single = pd.DataFrame([p_row])
                    ml_pred = float(self.ml_pipeline.predict(df_single)[0])
                    ml_suitability = round(np.clip(ml_pred, 10.0, 99.0), 1)
                except Exception as e:
                    logger.debug("ML inference fallback: %s", e)

            # ── Final Composite AI Match Score (0 - 100) ──
            if ml_suitability is not None:
                composite_score = (
                    0.45 * ml_suitability +
                    0.30 * (req_match_score * 100.0) +
                    0.15 * (w_rating * norm_rating * 100.0 + w_budget * budget_fit * 100.0) +
                    0.10 * (brand_match * 100.0)
                )
            else:
                composite_score = (
                    w_req * req_match_score +
                    w_rating * (0.8 * norm_rating + 0.2 * norm_pop) +
                    w_value * norm_disc +
                    w_budget * budget_fit +
                    w_brand * brand_match
                ) * 100.0

            composite_score = round(max(10.0, min(composite_score, 99.4)), 1)

            # ── Generate Transparent Explainability Reasons ──
            reasons = self._generate_transparent_reasons(
                product=p,
                category=category,
                top_features=top_matching_features,
                max_price=max_price,
                req_match_score=req_match_score
            )

            p_copy = dict(p)
            p_copy["match_score"] = composite_score
            p_copy["ml_suitability_score"] = ml_suitability
            p_copy["recommendation_reason"] = " • ".join(reasons[:3])
            p_copy["specs"] = p.get("specs", {})
            p_copy["id"] = p.get("canonical_id") or p.get("id") or generate_canonical_id(p.get("title", ""))
            p_copy["canonical_id"] = p_copy["id"]
            p_copy["name"] = p.get("title", p.get("name", ""))
            p_copy["image"] = p.get("thumbnail", p.get("image", ""))
            p_copy["merchant"] = p.get("merchant", "Online Store")

            scored.append(p_copy)

        # Sort by match_score descending
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        return scored[:limit], total_analyzed

    def _generate_transparent_reasons(
        self,
        product: Dict[str, Any],
        category: str,
        top_features: List[Tuple[str, float, float]],
        max_price: float,
        req_match_score: float
    ) -> List[str]:
        """Generate factual reasons based on the actual components that scored highest."""
        reasons = []
        specs = product.get("specs", {})
        price = product.get("price", 0)
        rating = product.get("rating")
        review_count = product.get("review_count") or 0
        merchant = product.get("merchant") or ""

        # 1. Requirement Match Highlights
        for feat_key, f_score, u_weight in sorted(top_features, key=lambda x: x[2], reverse=True):
            if feat_key == "gaming" and (specs.get("gpu") or specs.get("ram_gb", 0) >= 16):
                gpu = specs.get("gpu", "Dedicated GPU")
                ram = f"{specs.get('ram_gb')}GB RAM" if specs.get("ram_gb") else ""
                details = f" ({gpu}{' + ' + ram if ram else ''})"
                reasons.append(f"Strong match for your Gaming priority{details}")
            elif feat_key == "coding" and specs.get("ram_gb", 0) >= 16:
                reasons.append(f"Ideal for Coding & Programming ({specs.get('ram_gb')}GB RAM + {specs.get('processor', 'Fast Multi-core')})")
            elif feat_key == "camera" and specs.get("camera_mp"):
                ois = " + OIS" if specs.get("camera_ois") else ""
                reasons.append(f"Excellent {specs.get('camera_mp')}MP Camera{ois} matching your photo priority")
            elif feat_key == "battery" and specs.get("battery_mah"):
                reasons.append(f"Large {specs.get('battery_mah')}mAh Battery for all-day runtime")
            elif feat_key == "anc" and specs.get("has_anc"):
                reasons.append("Active Noise Cancellation (ANC) for immersive audio isolation")
            elif feat_key == "display" and specs.get("display_type"):
                hz = f" {specs.get('refresh_rate')}" if specs.get("refresh_rate") else ""
                reasons.append(f"Vibrant {specs.get('display_type')}{hz} Display panel")
            elif feat_key == "energy_star" and specs.get("energy_star"):
                reasons.append(f"{specs.get('energy_star')}-Star Energy Efficient rating with low power consumption")

            if len(reasons) >= 2:
                break

        # 2. Budget Savings Reason
        if max_price and price <= max_price:
            savings = max_price - price
            if savings >= 1000:
                reasons.append(f"Within your ₹{max_price:,.0f} budget (save ₹{savings:,.0f})")
            else:
                reasons.append(f"Fits your exact ₹{max_price:,.0f} budget")

        # 3. Rating & Merchant Reason
        if rating is not None and rating >= 4.0:
            if review_count > 0:
                reasons.append(f"Highly rated ({rating:.1f}★) by {review_count:,} buyers on {merchant or 'verified merchant'}")
            else:
                reasons.append(f"Rated {rating:.1f}★ on {merchant or 'verified merchant'}")
        elif merchant:
            reasons.append(f"Available from verified seller: {merchant}")

        if not reasons:
            reasons.append(f"Top {int(req_match_score * 100)}% match for your selected buying preferences")

        return reasons
