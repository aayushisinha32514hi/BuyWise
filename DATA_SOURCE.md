# DATA_SOURCE.md — BuyWise Dataset Documentation

> This document details the exact origin, structure, licensing, and processing of the real-world dataset used in BuyWise.

---

## 1. Primary Dataset: Amazon Products Dataset (India)

| Field | Description |
|---|---|
| **Platform** | Kaggle |
| **Dataset Title** | Amazon Products Sales Dataset |
| **Source Origin** | Scraped from Amazon India (`amazon.in`) |
| **Raw Records** | 551,585 product listings |
| **Cleaned Records in DB** | **448,237 unique products** |
| **Number of Sub-Categories** | 112 categories across 20 high-level domains |
| **Currency** | Indian Rupees (INR ₹) |
| **License** | Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) |
| **Usage** | Academic, non-commercial B.Tech student research & development |

---

## 2. Raw vs. Cleaned Data Pipeline

```
Raw CSV (551,585 rows)
       ↓  ml/scripts/preprocess.py
- Currency parsing (₹ and commas stripped to float)
- Rating count normalization (commas to integer)
- Brand name extraction from title tokens
- Discount % calculation: ((MRP - Price) / MRP) * 100
- Deduplication of identical product titles & categories
- Quality filtering (valid name, price > 0)
       ↓
Cleaned Dataset: data/processed/products_cleaned.csv (448,237 rows)
       ↓  backend/database/load_data.py
Indexed SQLite Database: backend/buywise.db (257.58 MB)
```

---

## 3. Categories Summary (Sample Top Categories)

1. **Air Conditioners** (720 products) — Lloyd, LG, Carrier, Voltas, Daikin
2. **Refrigerators** (2,160 products) — Samsung, LG, Whirlpool, Haier, Godrej
3. **Washing Machines** (1,440 products) — Bosch, LG, IFB, Samsung, Whirlpool
4. **Headphones** (9,600 products) — boAt, Sony, JBL, Boult, Noise, Sennheiser
5. **Cameras** (9,600 products) — Canon, Nikon, Sony, Panasonic, Fujifilm
6. **Watches** (17,515 products) — Titan, Fastrack, Casio, Fossil, Timex
7. **Sports Shoes** (16,537 products) — Puma, Nike, Adidas, Sparx, Campus
8. **Backpacks & Luggage** (19,152 products) — American Tourister, Safari, Skybags, Wildcraft
