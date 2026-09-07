# BuyWise — System Architecture & Data Flow

## 1. High-Level Architecture Diagram

```
+-------------------------------------------------------------------------+
|                              USER FRONTEND                              |
|   Responsive SPA (Mobile App-Shell / Tablet 2-Col / Desktop 4-Col Grid) |
|   Home • Explore • AI Match (NLP + Star Weights) • Compare • Saved      |
+------------------------------------+------------------------------------+
                                     |
                                     | HTTP REST API (JSON)
                                     v
+-------------------------------------------------------------------------+
|                             FASTAPI BACKEND                             |
|                                                                         |
|  [/api/v1/products]          [/api/v1/recommend]     [/api/v1/compare]  |
|  [/api/v1/ai-match/parse]    [/api/v1/price-history]  [/api/v1/auth]     |
+---------+--------------------------+---------------------+--------------+
          |                          |                     |
          v                          v                     v
+--------------------+   +---------------------+   +---------------------+
|  SEARCH & CACHE    |   |  ML RECOMMENDATION  |   |   PRICE & OFFERS    |
|                    |   |                     |   |                     |
| • SerpAPI Provider |   | • GBR Model ($R^2=0.97$) | • Multi-Retailer Table|
| • Normalizer       |   | • Feature Extractor | • Price Spread Calc |
| • Category Filter  |   | • Bayesian Ratings  | • Price Observations|
| • Deduplicator     |   | • Explainable XAI   | • Price History API |
+---------+----------+   +----------+----------+   +----------+----------+
          |                         |                         |
          +-------------------------+-------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                            SQLITE PERSISTENCE                           |
|                                                                         |
|  • search_cache        (6-Hour TTL query responses)                     |
|  • product_cache       (Canonical ID indexed product records)           |
|  • price_observations  (Continuous empirical price history tracking)    |
|  • users & user_saved  (Auth tokens, wishlists, browsing history)       |
+-------------------------------------------------------------------------+
```

---

## 2. Canonical Product Identity (`canonical_id`)

To eliminate cache misses between Explore, AI Match, Compare, Wishlist, and Product Details:
- Each product is assigned a deterministic 16-character hexadecimal identifier: `p_<hash>`.
- Generated from: `SHA-256(normalized_title + brand + merchant + url)`.
- All lookups query `product_cache(canonical_id)` with zero cache-miss rate.

---

## 3. Core REST API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Server & database liveness probe |
| `GET` | `/api/v1/categories/top` | 10 verified shopping macro-categories with metadata |
| `GET` | `/api/v1/categories/{id}/requirements` | Dynamic purchasing priorities for category |
| `POST` | `/api/v1/ai-match/parse-requirements` | NLP extraction from unstructured user prompt |
| `GET` | `/api/v1/products` | Paginated live shopping search with intent parsing |
| `GET` | `/api/v1/products/{id}` | Product details with multi-merchant offers & VADER sentiment |
| `GET` | `/api/v1/products/{id}/price-history` | Empirical price history observations and trend |
| `POST` | `/api/v1/recommend` | Hybrid ML suitability matching against user requirements |
| `POST` | `/api/v1/compare` | Multi-product side-by-side comparison with AI winner badges |
| `POST` | `/api/v1/auth/signup` | Register user account |
| `POST` | `/api/v1/auth/login` | Authenticate and obtain bearer token |
| `GET` | `/api/v1/user/saved` | Retrieve user saved wishlist |
| `POST` | `/api/v1/user/saved/toggle` | Save or remove product by canonical ID |
