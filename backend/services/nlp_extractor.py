"""
BuyWise — Natural Language Requirement Understanding (NLP Extractor)
=====================================================================
Parses unstructured natural language shopping prompts into structured
category requirements, budget bounds, brand preferences, and priority weights.
"""

import re
from typing import Dict, Any, List, Optional, Tuple

from backend.services.product_search.normalizer import classify_category
from backend.services.product_search.recommendation_engine import CATEGORY_REQUIREMENTS

BRAND_LIST = [
    "Apple", "Samsung", "OnePlus", "Xiaomi", "Redmi", "Realme", "Vivo", "Oppo",
    "Motorola", "Google", "Nothing", "iQOO", "Poco", "ASUS", "Lenovo", "HP",
    "Dell", "Acer", "MSI", "Sony", "Bose", "Sennheiser", "JBL", "Boat", "Noise",
    "LG", "Daikin", "Voltas", "Whirlpool", "Lloyd", "Carrier", "IFB", "Bosch",
    "Nike", "Adidas", "Puma", "Reebok", "Titan", "Fastrack", "Casio"
]

CATEGORY_DISPLAY_MAP = {
    "mobiles": "Mobiles",
    "laptops": "Laptops",
    "tvs": "TVs",
    "electronics": "Electronics",
    "appliances": "Appliances",
    "footwear": "Footwear",
    "fashion": "Fashion",
    "beauty": "Beauty",
    "jewellery": "Jewellery",
    "home": "Home"
}


def parse_natural_language_requirements(text: str) -> Dict[str, Any]:
    """
    Parse natural language text and return structured requirement payload
    ready for user review and AI Match ranking.
    """
    if not text or not str(text).strip():
        return {
            "category": "Laptops",
            "min_price": 0.0,
            "max_price": 60000.0,
            "min_rating": 4.0,
            "priority": "value",
            "preferred_brands": [],
            "requirements": {},
            "use_cases": [],
            "extracted_summary": "Default criteria"
        }

    raw = str(text).strip()
    raw_lower = raw.lower()

    # 1. Extract Category and Subcategory with boundary-aware priority regex
    detected_cat_id = None
    detected_subcat_id = None

    # Mobiles & Subcategories
    if re.search(r'\b(?:phone|phones|smartphone|smartphones|mobile|mobiles|iphone|android|oneplus|galaxy|redmi|realme|iqoo|pixel)\b', raw_lower) and not re.search(r'\b(?:headphone|earphone|microphone)\b', raw_lower):
        detected_cat_id = "mobiles"
        if re.search(r'\b(?:tablet|tablets|ipad|ipads|tab)\b', raw_lower):
            detected_subcat_id = "tablets"
        elif re.search(r'\b(?:smartwatch|smartwatches|watch|fitness band|tracker)\b', raw_lower):
            detected_subcat_id = "smartwatches"
        elif re.search(r'\b(?:power bank|powerbank|charger|power banks)\b', raw_lower):
            detected_subcat_id = "powerbanks"
        elif re.search(r'\b(?:case|cover|cable|charger|adapter)\b', raw_lower):
            detected_subcat_id = "mobile_accessories"
        else:
            detected_subcat_id = "smartphones"

    # Laptops & Subcategories
    elif re.search(r'\b(?:laptop|laptops|macbook|notebook|chromebook|thinkpad|ideapad|vivobook|legion|omen|tuf|predator|rog|zephyrus|alienware)\b', raw_lower):
        detected_cat_id = "laptops"
        if re.search(r'\b(?:game|gaming|gpu|rtx|graphics|fps|144hz|165hz|nvidia|radeon)\b', raw_lower):
            detected_subcat_id = "gaming_laptops"
        elif re.search(r'\b(?:touch|2-in-1|convertible|stylus|foldable|touchscreen)\b', raw_lower):
            detected_subcat_id = "touch_laptops"
        elif re.search(r'\b(?:monitor|monitors|display screen)\b', raw_lower):
            detected_subcat_id = "monitors"
        elif re.search(r'\b(?:keyboard|keyboards|mouse|mice)\b', raw_lower):
            detected_subcat_id = "peripherals"
        elif re.search(r'\b(?:printer|printers|scanner)\b', raw_lower):
            detected_subcat_id = "printers"
        else:
            detected_subcat_id = "general_laptops"

    # TVs & Audio
    elif re.search(r'\b(?:tv|tvs|television|televisions|smart tv|led tv|oled tv|qled tv|4k tv|55 inch|43 inch|65 inch|32 inch)\b', raw_lower):
        detected_cat_id = "tvs"
        if re.search(r'\b(?:soundbar|soundbars|home theater|subwoofer)\b', raw_lower):
            detected_subcat_id = "soundbars"
        elif re.search(r'\b(?:firetv|chromecast|streaming stick|tv box)\b', raw_lower):
            detected_subcat_id = "streaming_devices"
        else:
            detected_subcat_id = "smart_4k_tvs"

    # Audio & Electronics
    elif re.search(r'\b(?:headphone|headphones|earphone|earphones|earbuds|tws|airpods|speaker|speakers|soundbar|camera|dslr)\b', raw_lower):
        detected_cat_id = "electronics"
        if re.search(r'\b(?:tws|earbuds|airpods|wireless buds|anc earbuds)\b', raw_lower):
            detected_subcat_id = "tws_earbuds"
        elif re.search(r'\b(?:over-ear|headphone|headphones|headset)\b', raw_lower):
            detected_subcat_id = "headphones"
        elif re.search(r'\b(?:speaker|speakers|bluetooth speaker|soundbar)\b', raw_lower):
            detected_subcat_id = "bluetooth_speakers"
        elif re.search(r'\b(?:camera|dslr|mirrorless|vlog)\b', raw_lower):
            detected_subcat_id = "cameras"
        else:
            detected_subcat_id = "tws_earbuds"

    # Appliances
    elif re.search(r'\b(?:ac|split ac|air conditioner|refrigerator|fridge|washing machine|microwave|oven|cooler|purifier|geyser)\b', raw_lower):
        detected_cat_id = "appliances"
        if re.search(r'\b(?:ac|split ac|air conditioner|ton|inverter ac)\b', raw_lower):
            detected_subcat_id = "ac"
        elif re.search(r'\b(?:refrigerator|refrigerators|fridge|frost free|double door)\b', raw_lower):
            detected_subcat_id = "refrigerators"
        elif re.search(r'\b(?:washing machine|washing machines|washer|front load|top load)\b', raw_lower):
            detected_subcat_id = "washing_machines"
        elif re.search(r'\b(?:microwave|oven|convection)\b', raw_lower):
            detected_subcat_id = "microwaves"
        elif re.search(r'\b(?:purifier|ro|water purifier)\b', raw_lower):
            detected_subcat_id = "water_purifiers"
        else:
            detected_subcat_id = "ac"

    # Footwear
    elif re.search(r'\b(?:shoes|shoe|sneakers|sneaker|sandals|boots|crocs|loafers|running shoes)\b', raw_lower):
        detected_cat_id = "footwear"
        if re.search(r'\b(?:running|sports|gym|marathon|jogging|training)\b', raw_lower):
            detected_subcat_id = "running_shoes"
        elif re.search(r'\b(?:sandals|slides|slippers|flip flops)\b', raw_lower):
            detected_subcat_id = "sandals"
        else:
            detected_subcat_id = "casual_shoes"

    # Fashion
    elif re.search(r'\b(?:shirt|t-shirt|tshirt|jeans|dress|jacket|hoodie|kurta|saree|bag|backpack)\b', raw_lower):
        detected_cat_id = "fashion"
        if re.search(r'\b(?:bag|backpack|duffle|handbag|laptop bag)\b', raw_lower):
            detected_subcat_id = "bags"
        elif re.search(r'\b(?:kurta|saree|ethnic|lehenga|sherwani)\b', raw_lower):
            detected_subcat_id = "ethnic_wear"
        elif re.search(r'\b(?:women|dress|skirt|top|tops)\b', raw_lower):
            detected_subcat_id = "womens_clothing"
        else:
            detected_subcat_id = "mens_clothing"

    # Beauty
    elif re.search(r'\b(?:perfume|cream|lotion|shampoo|lipstick|skincare|moisturizer|sunscreen|trimmer|shaver)\b', raw_lower):
        detected_cat_id = "beauty"
        if re.search(r'\b(?:trimmer|shaver|hair dryer|straightener)\b', raw_lower):
            detected_subcat_id = "grooming_appliances"
        elif re.search(r'\b(?:perfume|deodorant|fragrance|body spray)\b', raw_lower):
            detected_subcat_id = "fragrances"
        elif re.search(r'\b(?:sunscreen|serum|face wash|moisturizer|skincare)\b', raw_lower):
            detected_subcat_id = "skincare"
        else:
            detected_subcat_id = "skincare"

    # Jewellery
    elif re.search(r'\b(?:watch|wristwatch|necklace|bracelet|ring|earring|jewellery)\b', raw_lower):
        detected_cat_id = "jewellery"
        if re.search(r'\b(?:watch|wristwatch|analog|chronograph)\b', raw_lower):
            detected_subcat_id = "analog_watches"
        else:
            detected_subcat_id = "jewellery_items"

    # Home
    elif re.search(r'\b(?:bedsheet|curtain|pillow|chair|table|cookware|bottle|mixer|grinder)\b', raw_lower):
        detected_cat_id = "home"
        if re.search(r'\b(?:cookware|pan|pot|mixer|grinder|bottle)\b', raw_lower):
            detected_subcat_id = "cookware"
        elif re.search(r'\b(?:bedsheet|curtain|pillow|bedding|blanket)\b', raw_lower):
            detected_subcat_id = "furnishing"
        else:
            detected_subcat_id = "home_decor"

    if not detected_cat_id:
        detected_cat_id = classify_category(raw)
        if detected_cat_id == "uncategorized":
            detected_cat_id = "laptops"
        detected_subcat_id = "general_laptops"

    category_display = CATEGORY_DISPLAY_MAP.get(detected_cat_id, "Laptops")

    # 2. Extract Budget Bounds (Min and Max Price)
    min_price = 0.0
    max_price = 50000.0

    # Max budget regex patterns: "under 60000", "below 80k", "within 1.5 lakh", "<= 45000"
    max_match = re.search(
        r'(?:under|below|less than|within|upto|up to|budget|max)\s*(?:rs\.?|inr|₹)?\s*([0-9]+(?:\.[0-9]+)?)\s*(k|lakh|lac)?',
        raw_lower
    )
    if max_match:
        val = float(max_match.group(1))
        unit = (max_match.group(2) or "").lower()
        if unit == 'k':
            max_price = val * 1000.0
        elif unit in ('lakh', 'lac'):
            max_price = val * 100000.0
        else:
            max_price = val

    # Range regex: "between 40k and 80k", "40000 to 70000"
    range_match = re.search(
        r'(?:between|from)\s*(?:rs\.?|inr|₹)?\s*([0-9]+)\s*(k)?\s*(?:and|to|-)\s*(?:rs\.?|inr|₹)?\s*([0-9]+)\s*(k|lakh|lac)?',
        raw_lower
    )
    if range_match:
        min_v = float(range_match.group(1)) * (1000.0 if range_match.group(2) == 'k' else 1.0)
        max_v = float(range_match.group(3))
        unit_max = (range_match.group(4) or "").lower()
        if unit_max == 'k': max_v *= 1000.0
        elif unit_max in ('lakh', 'lac'): max_v *= 100000.0
        min_price = min_v
        max_price = max_v

    # 3. Extract Preferred Brands
    detected_brands = []
    for b in BRAND_LIST:
        if re.search(r'\b' + re.escape(b.lower()) + r'\b', raw_lower):
            detected_brands.append(b)

    # 4. Extract Optimization Priority Goal
    priority = "value"
    if any(k in raw_lower for k in ["top quality", "best quality", "premium", "highest rated", "best rated", "flagship"]):
        priority = "rating"
    elif any(k in raw_lower for k in ["cheapest", "lowest price", "budget saver", "tight budget", "affordable"]):
        priority = "budget"
    elif any(k in raw_lower for k in ["balanced", "overall", "all rounder", "all-rounder"]):
        priority = "balanced"
    else:
        priority = "value"

    # 5. Extract Quality / Rating Threshold
    min_rating = 0.0
    if any(k in raw_lower for k in ["top rated", "highly rated", "4 star", "4.5", "best"]):
        min_rating = 4.0
    elif any(k in raw_lower for k in ["good rating", "3.5"]):
        min_rating = 3.5

    # 6. Extract Use Cases and Formulate Category/Subcategory-Specific Importance Weights
    use_cases = []
    from backend.services.product_search.recommendation_engine import get_category_requirements
    cat_req_definitions = get_category_requirements(detected_cat_id, detected_subcat_id)
    req_weights: Dict[str, int] = {r["key"]: r["default"] for r in cat_req_definitions}

    # Intent keyword matching
    if any(w in raw_lower for w in ["game", "gaming", "gpu", "gta", "fps", "rtx", "graphics", "val", "nvidia", "radeon"]):
        use_cases.append("Gaming")
        if "gaming" in req_weights: req_weights["gaming"] = 5
        if "gpu" in req_weights: req_weights["gpu"] = 5
        if "performance" in req_weights: req_weights["performance"] = 5

    if any(w in raw_lower for w in ["code", "coding", "programming", "developer", "software", "python", "java", "vscode", "btech", "college", "16gb", "32gb", "ram"]):
        use_cases.append("Coding & Work")
        if "coding" in req_weights: req_weights["coding"] = 5
        if "performance" in req_weights: req_weights["performance"] = 5
        if "storage" in req_weights: req_weights["storage"] = 4

    if any(w in raw_lower for w in ["photo", "photography", "camera", "video", "reels", "vlog", "selfie", "ois", "4k video"]):
        use_cases.append("Photography")
        if "camera" in req_weights: req_weights["camera"] = 5

    if any(w in raw_lower for w in ["battery", "travel", "all day", "long lasting", "portable", "lightweight", "thin"]):
        use_cases.append("Portability & Battery")
        if "battery" in req_weights: req_weights["battery"] = 5
        if "battery_portability" in req_weights: req_weights["battery_portability"] = 5

    if any(w in raw_lower for w in ["screen", "display", "movie", "oled", "amoled", "4k", "55 inch", "43 inch", "65 inch", "ultra hd"]):
        use_cases.append("Media & Display")
        if "display" in req_weights: req_weights["display"] = 5
        if "screen_size" in req_weights: req_weights["screen_size"] = 5

    if any(w in raw_lower for w in ["sound", "music", "bass", "anc", "noise cancelling", "calls", "mic", "dolby"]):
        use_cases.append("Audio & Calls")
        if "sound" in req_weights: req_weights["sound"] = 5
        if "anc" in req_weights: req_weights["anc"] = 5

    if any(w in raw_lower for w in ["5 star", "inverter", "power saving", "electricity", "energy"]):
        use_cases.append("Energy Efficiency")
        if "energy_star" in req_weights: req_weights["energy_star"] = 5

    if any(w in raw_lower for w in ["5g", "fast internet"]):
        use_cases.append("5G Network")
        if "five_g" in req_weights: req_weights["five_g"] = 5

    # Summary generator
    high_reqs = [r["label"] for r in cat_req_definitions if req_weights.get(r["key"], 0) == 5]
    summary_parts = [f"{category_display}"]
    if detected_subcat_id:
        summary_parts[0] = f"{category_display} ({detected_subcat_id.replace('_', ' ').title()})"
    if high_reqs:
        summary_parts.append(f"focused on {', '.join(high_reqs[:2])}")
    if max_price > 0:
        summary_parts.append(f"under ₹{max_price:,.0f}")
    if detected_brands:
        summary_parts.append(f"({', '.join(detected_brands)})")

    extracted_summary = " • ".join(summary_parts)

    return {
        "category": category_display,
        "sub_category": detected_subcat_id,
        "min_price": float(min_price),
        "max_price": float(max_price),
        "min_rating": float(min_rating),
        "priority": priority,
        "preferred_brands": detected_brands,
        "requirements": req_weights,
        "use_cases": use_cases,
        "extracted_summary": extracted_summary,
    }
