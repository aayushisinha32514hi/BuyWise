"""
BuyWise — Real-Time Product Specification & Feature Extractor
==============================================================
Extracts structured technical specifications from product titles, extensions,
and snippet text without fabricating nonexistent details.

Features Extracted:
- Processors (Intel Core i3/i5/i7/i9, AMD Ryzen 3/5/7/9, Apple M-Series, Snapdragon, Dimensity, Bionic)
- RAM Capacity (4GB, 8GB, 16GB, 24GB, 32GB, 64GB)
- Storage Capacity (64GB, 128GB, 256GB, 512GB, 1TB, 2TB) + Storage Type (SSD, ROM, eMMC)
- GPU / Graphics (RTX 3050/4050/4060/4070/4080, GTX, Iris Xe, Radeon, Apple GPU)
- Display Quality (AMOLED, OLED, QLED, Retina, 120Hz, 144Hz, 165Hz, 4K, 2K, FHD)
- Camera Specs (Megapixels e.g. 50MP, 108MP, 200MP, OIS, 4K Video)
- Battery & Charging (4000mAh–6000mAh, Fast Charging, 65W/120W)
- Connectivity (5G, Wi-Fi 6, Bluetooth 5.3)
- Audio/ANC (Active Noise Cancellation, Transparency, Hi-Res Audio, 40mm)
- Appliance Capacity & Energy (1.5 Ton, 2 Ton, 5 Star, 3 Star, Inverter, 250L, 7kg)
"""

import re
from typing import Dict, Any, Optional


def extract_features(title: str, category: str = "", extra_text: str = "") -> Dict[str, Any]:
    """
    Extract structured specs and feature flags from a product title and optional text.
    """
    text = f"{title} {extra_text}".strip()
    t_lower = text.lower()

    specs: Dict[str, Any] = {}

    # 1. RAM Extraction
    ram_match = re.search(r'\b(4|6|8|12|16|24|32|64|128)\s*(?:gb\s*ram|gb|g\s*ram)\b', text, re.IGNORECASE)
    if ram_match:
        specs["ram_gb"] = int(ram_match.group(1))

    # 2. Storage Extraction
    storage_tb = re.search(r'\b([1-4])\s*(?:tb|t)\s*(?:ssd|storage|hdd|nvme)?\b', text, re.IGNORECASE)
    if storage_tb:
        specs["storage_gb"] = int(storage_tb.group(1)) * 1024
        specs["storage_display"] = f"{storage_tb.group(1)}TB"
    else:
        storage_gb = re.search(r'\b(64|128|256|512)\s*(?:gb|g)\s*(?:ssd|rom|storage|emmc|nvme|ufs)?\b', text, re.IGNORECASE)
        if storage_gb:
            specs["storage_gb"] = int(storage_gb.group(1))
            specs["storage_display"] = f"{storage_gb.group(1)}GB"

    # 3. Processor Extraction
    if "apple" in t_lower or "macbook" in t_lower or "ipad" in t_lower:
        m_match = re.search(r'\b(m[1-4](?:\s*(?:pro|max|ultra))?)\b', text, re.IGNORECASE)
        if m_match:
            specs["processor"] = f"Apple {m_match.group(1).upper()}"
            specs["cpu_tier"] = 5 if "max" in t_lower or "ultra" in t_lower or "m3" in t_lower or "m4" in t_lower else 4
        elif "a1" in t_lower or "bionic" in t_lower:
            specs["processor"] = "Apple Bionic"
            specs["cpu_tier"] = 4
    elif "intel" in t_lower or "core" in t_lower:
        i_match = re.search(r'\b(i[3579])(?:-|\s*)(\d{4,5}[a-z]*)?\b', text, re.IGNORECASE)
        if i_match:
            gen = f" {i_match.group(2)}" if i_match.group(2) else ""
            specs["processor"] = f"Intel Core {i_match.group(1).upper()}{gen}"
            tier = 3 if "i3" in i_match.group(1) else (4 if "i5" in i_match.group(1) else 5)
            specs["cpu_tier"] = tier
    elif "ryzen" in t_lower or "amd" in t_lower:
        r_match = re.search(r'\b(ryzen\s*[3579])(?:-|\s*)(\d{4}[a-z]*)?\b', text, re.IGNORECASE)
        if r_match:
            specs["processor"] = f"AMD {r_match.group(1).title()}"
            tier = 3 if "3" in r_match.group(1) else (4 if "5" in r_match.group(1) else 5)
            specs["cpu_tier"] = tier
    elif "snapdragon" in t_lower:
        specs["processor"] = "Qualcomm Snapdragon"
        specs["cpu_tier"] = 5 if "gen" in t_lower or "8" in t_lower else 4
    elif "dimensity" in t_lower:
        specs["processor"] = "MediaTek Dimensity"
        specs["cpu_tier"] = 5 if "9000" in t_lower or "8" in t_lower else 3

    # 4. GPU / Dedicated Graphics Extraction
    gpu_match = re.search(r'\b(rtx\s*(?:4090|4080|4070|4060|4050|3080|3070|3060|3050|2050)|gtx\s*\d{4}|radeon|iris\s*xe)\b', text, re.IGNORECASE)
    if gpu_match:
        gpu_name = gpu_match.group(1).upper()
        specs["gpu"] = gpu_name
        if "40" in gpu_name or "3080" in gpu_name or "3070" in gpu_name:
            specs["gpu_tier"] = 5  # High-end gaming
        elif "3050" in gpu_name or "3060" in gpu_name or "2050" in gpu_name:
            specs["gpu_tier"] = 4  # Mid gaming
        else:
            specs["gpu_tier"] = 2  # Integrated / entry

    # 5. Display Specs
    display_types = []
    if re.search(r'\b(amoled|super amoled)\b', t_lower):
        display_types.append("AMOLED")
    elif re.search(r'\b(oled)\b', t_lower):
        display_types.append("OLED")
    elif re.search(r'\b(qled)\b', t_lower):
        display_types.append("QLED")
    elif re.search(r'\b(retina)\b', t_lower):
        display_types.append("Retina")
    elif re.search(r'\b(ips)\b', t_lower):
        display_types.append("IPS")

    # Refresh rate
    hz_match = re.search(r'\b(90|120|144|165|240)\s*hz\b', t_lower)
    if hz_match:
        specs["refresh_rate"] = f"{hz_match.group(1)}Hz"
        specs["high_refresh"] = True

    # Resolution
    if re.search(r'\b(4k|ultra hd|uhd)\b', t_lower):
        specs["resolution"] = "4K UHD"
    elif re.search(r'\b(2k|qhd|wqhd)\b', t_lower):
        specs["resolution"] = "2K QHD"
    elif re.search(r'\b(fhd|full hd|1080p)\b', t_lower):
        specs["resolution"] = "Full HD"

    if display_types:
        specs["display_type"] = display_types[0]

    # Screen size (inches)
    size_match = re.search(r'\b(\d{1,2}(?:\.\d)?)\s*(?:inch|\"|\'|-inch)\b', t_lower)
    if size_match:
        specs["screen_size_inch"] = float(size_match.group(1))

    # 6. Camera Specs (Mobiles / Cameras)
    cam_mp = re.search(r'\b(\d{2,3})\s*mp\b', t_lower)
    if cam_mp:
        specs["camera_mp"] = int(cam_mp.group(1))
    if re.search(r'\b(ois|optical image stabilization)\b', t_lower):
        specs["camera_ois"] = True
    if re.search(r'\b(triple camera|quad camera|dual camera)\b', t_lower):
        specs["multi_camera"] = True

    # 7. Battery & 5G
    bat_match = re.search(r'\b(\d{4,5})\s*mah\b', t_lower)
    if bat_match:
        specs["battery_mah"] = int(bat_match.group(1))
    if re.search(r'\b(5g)\b', t_lower):
        specs["is_5g"] = True
    if re.search(r'\b(fast charg\w+|67w|120w|80w|100w|33w|45w)\b', t_lower):
        specs["fast_charging"] = True

    # 8. Audio / Headphones Specs
    if re.search(r'\b(anc|active noise cancell\w+|noise reduction)\b', t_lower):
        specs["has_anc"] = True
    if re.search(r'\b(wireless|bluetooth|tws|true wireless)\b', t_lower):
        specs["is_wireless"] = True
    if re.search(r'\b(hi-res|dolby atmos|spatial audio)\b', t_lower):
        specs["hi_res_audio"] = True

    # 9. Home Appliances Specs
    ton_match = re.search(r'\b(1|1\.5|2|2\.5)\s*ton\b', t_lower)
    if ton_match:
        specs["ac_ton"] = float(ton_match.group(1))
    star_match = re.search(r'\b([1-5])\s*star\b', t_lower)
    if star_match:
        specs["energy_star"] = int(star_match.group(1))
    if re.search(r'\b(inverter|dual inverter)\b', t_lower):
        specs["inverter_tech"] = True
    ltr_match = re.search(r'\b(\d{2,3})\s*(?:l|ltr|litre|liter)\b', t_lower)
    if ltr_match:
        specs["capacity_liters"] = int(ltr_match.group(1))
    kg_match = re.search(r'\b(\d(?:\.\d)?)\s*kg\b', t_lower)
    if kg_match:
        specs["capacity_kg"] = float(kg_match.group(1))

    return specs
