"""
BuyWise — Indian Shopping Category & Subcategory Taxonomy Service
==================================================================
Provides structured 10 Macro-Categories and 45+ Curated Subcategories
with search queries and category classification mappings.
"""

from typing import List, Dict, Any, Optional

# Top 10 Indian Shopping Categories with curated subcategories
TOP_CATEGORIES = [
    {
        "id": "mobiles",
        "name": "Mobiles & Accessories",
        "short_name": "Mobiles",
        "icon": "📱",
        "search_queries": ["smartphones", "mobile phones India"],
        "subcategories": [
            {"id": "smartphones", "name": "Smartphones", "search_query": "smartphones 5G mobile phones India"},
            {"id": "tablets", "name": "Tablets & iPads", "search_query": "tablets iPads Android tablets India"},
            {"id": "smartwatches", "name": "Smartwatches & Bands", "search_query": "smartwatches fitness trackers India"},
            {"id": "powerbanks", "name": "Power Banks", "search_query": "power banks fast charging India"},
            {"id": "mobile_accessories", "name": "Mobile Accessories", "search_query": "mobile phone chargers cases cables India"},
        ]
    },
    {
        "id": "laptops",
        "name": "Laptops & Computers",
        "short_name": "Laptops",
        "icon": "💻",
        "search_queries": ["laptops", "notebooks India"],
        "subcategories": [
            {"id": "general_laptops", "name": "Laptops & Ultrabooks", "search_query": "laptops ultrabooks SSD India"},
            {"id": "gaming_laptops", "name": "Gaming Laptops", "search_query": "gaming laptops RTX GPU 144Hz India"},
            {"id": "touch_laptops", "name": "2-in-1 Touch Laptops", "search_query": "2-in-1 touch laptops convertible India"},
            {"id": "monitors", "name": "Monitors & Displays", "search_query": "computer monitors 4K IPS gaming monitors India"},
            {"id": "peripherals", "name": "Keyboards & Mice", "search_query": "mechanical keyboards wireless mouse India"},
            {"id": "printers", "name": "Printers & Scanners", "search_query": "all-in-one ink tank laser printers India"},
        ]
    },
    {
        "id": "appliances",
        "name": "Home Appliances",
        "short_name": "Appliances",
        "icon": "❄️",
        "search_queries": ["home appliances India"],
        "subcategories": [
            {"id": "ac", "name": "Air Conditioners", "search_query": "split air conditioners 1.5 ton 5 star inverter AC India"},
            {"id": "refrigerators", "name": "Refrigerators", "search_query": "refrigerators double door frost free inverter fridge India"},
            {"id": "washing_machines", "name": "Washing Machines", "search_query": "front load top load washing machines inverter India"},
            {"id": "microwaves", "name": "Microwave Ovens", "search_query": "convection microwave ovens India"},
            {"id": "air_coolers", "name": "Air Coolers", "search_query": "desert personal air coolers honeycomb India"},
            {"id": "vacuum_cleaners", "name": "Vacuum Cleaners", "search_query": "robotic cordless vacuum cleaners India"},
            {"id": "water_purifiers", "name": "Water Purifiers", "search_query": "RO UV UF water purifiers TDS controller India"},
            {"id": "geysers", "name": "Geysers & Water Heaters", "search_query": "instant storage water geysers 5 star India"},
        ]
    },
    {
        "id": "tvs",
        "name": "TVs & Home Theater",
        "short_name": "TVs",
        "icon": "📺",
        "search_queries": ["smart TV television India"],
        "subcategories": [
            {"id": "smart_4k_tvs", "name": "Smart & 4K Ultra HD TVs", "search_query": "4K smart TVs OLED QLED 55 inch 43 inch India"},
            {"id": "soundbars", "name": "Soundbars & Home Theaters", "search_query": "soundbars Dolby Atmos home theater subwoofer India"},
            {"id": "streaming_devices", "name": "Streaming Sticks & Boxes", "search_query": "streaming sticks 4K media streamers India"},
        ]
    },
    {
        "id": "electronics",
        "name": "Audio & Electronics",
        "short_name": "Audio",
        "icon": "🎧",
        "search_queries": ["headphones earbuds speakers"],
        "subcategories": [
            {"id": "tws_earbuds", "name": "TWS Wireless Earbuds", "search_query": "TWS earbuds active noise cancellation ANC India"},
            {"id": "headphones", "name": "Over-Ear Headphones", "search_query": "over-ear wireless ANC headphones Hi-Res India"},
            {"id": "bluetooth_speakers", "name": "Bluetooth Speakers", "search_query": "portable Bluetooth speakers bass waterproof India"},
            {"id": "cameras", "name": "Cameras & Photography", "search_query": "mirrorless DSLR vlogging cameras India"},
        ]
    },
    {
        "id": "fashion",
        "name": "Fashion & Clothing",
        "short_name": "Fashion",
        "icon": "👕",
        "search_queries": ["clothing shirts t-shirts jeans"],
        "subcategories": [
            {"id": "mens_clothing", "name": "Men's Clothing", "search_query": "men shirts t-shirts jeans trousers India"},
            {"id": "womens_clothing", "name": "Women's Clothing", "search_query": "women dresses tops jeans western wear India"},
            {"id": "ethnic_wear", "name": "Ethnic & Festive Wear", "search_query": "ethnic wear kurta sets sarees lehengas India"},
            {"id": "western_wear", "name": "Western Wear", "search_query": "western wear jackets hoodies stylish outfits India"},
            {"id": "bags", "name": "Bags & Backpacks", "search_query": "laptop backpacks travel duffles handbags India"},
        ]
    },
    {
        "id": "footwear",
        "name": "Footwear & Shoes",
        "short_name": "Footwear",
        "icon": "👟",
        "search_queries": ["shoes sneakers sandals India"],
        "subcategories": [
            {"id": "running_shoes", "name": "Sports & Running Shoes", "search_query": "running shoes sports training sneakers India"},
            {"id": "casual_shoes", "name": "Casual & Formal Shoes", "search_query": "casual sneakers loafers leather formal shoes India"},
            {"id": "sandals", "name": "Sandals & Slippers", "search_query": "comfort sandals clogs slides slippers India"},
        ]
    },
    {
        "id": "beauty",
        "name": "Beauty & Personal Care",
        "short_name": "Beauty",
        "icon": "✨",
        "search_queries": ["beauty skincare makeup India"],
        "subcategories": [
            {"id": "skincare", "name": "Skincare & Sunscreens", "search_query": "sunscreen serums moisturizers face wash India"},
            {"id": "haircare", "name": "Haircare & Shampoos", "search_query": "shampoos conditioners hair serums hair oils India"},
            {"id": "makeup", "name": "Makeup & Cosmetics", "search_query": "lipstick foundation kajal mascara makeup India"},
            {"id": "grooming_appliances", "name": "Grooming & Hair Dryers", "search_query": "trimmers electric shavers hair dryers straighteners India"},
            {"id": "fragrances", "name": "Perfumes & Deodorants", "search_query": "perfumes body sprays deodorants India"},
        ]
    },
    {
        "id": "jewellery",
        "name": "Watches & Jewellery",
        "short_name": "Jewellery",
        "icon": "💍",
        "search_queries": ["watches jewellery India"],
        "subcategories": [
            {"id": "analog_watches", "name": "Analog & Luxury Watches", "search_query": "analog wristwatches chronograph watches India"},
            {"id": "jewellery_items", "name": "Fashion Jewellery", "search_query": "necklaces bracelets earrings rings India"},
        ]
    },
    {
        "id": "home",
        "name": "Home, Living & Kitchen",
        "short_name": "Home & Kitchen",
        "icon": "🏠",
        "search_queries": ["home decor kitchen furniture India"],
        "subcategories": [
            {"id": "cookware", "name": "Cookware & Kitchen Tools", "search_query": "non-stick cookware sets mixer grinders bottles India"},
            {"id": "furnishing", "name": "Home Furnishing & Bedding", "search_query": "bedsheets curtains pillows blankets India"},
            {"id": "home_decor", "name": "Home Decor & Lighting", "search_query": "wall decor lamps clocks organizers India"},
        ]
    },
]


def get_top_categories() -> List[Dict[str, Any]]:
    """Return the 10 curated Indian shopping categories with subcategories."""
    return TOP_CATEGORIES


def get_subcategories_for_category(category_id: str) -> List[Dict[str, Any]]:
    """Retrieve subcategories for a given category ID or name."""
    cat_clean = category_id.strip().lower()
    for cat in TOP_CATEGORIES:
        if cat["id"] == cat_clean or cat["short_name"].lower() == cat_clean or cat["name"].lower() == cat_clean:
            return cat.get("subcategories", [])
    return []


def get_search_query_for_category(category_id: str, subcategory_id: Optional[str] = None) -> str:
    """
    Map a BuyWise category and optional subcategory ID to a Google Shopping search query.
    """
    cat_clean = category_id.strip().lower()
    for cat in TOP_CATEGORIES:
        if cat["id"] == cat_clean or cat["short_name"].lower() == cat_clean or cat["name"].lower() == cat_clean:
            if subcategory_id:
                sub_clean = subcategory_id.strip().lower()
                for sub in cat.get("subcategories", []):
                    if sub["id"] == sub_clean or sub["name"].lower() == sub_clean:
                        return sub.get("search_query", sub["name"])
            return cat["search_queries"][0]

    # Fallback
    return f"{category_id} {subcategory_id or ''}".strip()
