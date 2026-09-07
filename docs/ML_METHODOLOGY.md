# BuyWise — Machine Learning & Recommendation Methodology

## 1. System Architecture Overview

BuyWise implements a **Hybrid AI Decision-Support Pipeline** combining:
1. **Supervised Machine Learning:** A trained Gradient Boosting Regressor predicting product-user suitability scores ($R^2 = 0.9708$).
2. **Multi-Criteria Decision Analysis (MCDA):** Category-specific preference vector weighting with 1–5 star importance scales.
3. **Empirical Feature Extraction:** Semantic regex specification parsing of real hardware components (RAM, CPU, GPU, Camera MP, Battery mAh, 5G, ANC, Energy Star).
4. **Bayesian Rating Smoothing:** m-estimate smoothing to prevent low-sample rating distortion.
5. **Natural Language Processing (NLP):** Entity and intent extraction from free-text user shopping queries.

---

## 2. Technical Component Taxonomy

To maintain strict engineering honesty and avoid misrepresenting rules as AI:

| Subsystem | Methodology | Underlying Technique |
|---|---|---|
| **Suitability Prediction** | Machine Learning (Supervised) | `GradientBoostingRegressor` (160 estimators, learning rate 0.08, max depth 5) |
| **Spec Extraction** | Deterministic / Semantic NLP | Regex boundary-matching and token normalization (`feature_extractor.py`) |
| **Intent Understanding** | Natural Language Processing | Intent classifier & boundary entity extraction (`nlp_extractor.py`) |
| **Customer Approval** | Lexicon NLP | VADER Sentiment Intensity Analyzer (`vaderSentiment`) |
| **Candidate Retrieval** | Live External Provider | Google Shopping India API via SerpAPI with 6h SQLite caching |
| **Hard Constraints** | Deterministic Filtering | Strict category boundary post-filtering & budget ceiling validation |

---

## 3. Mathematical Formulations

### A. Bayesian Smoothed Rating ($R_{smooth}$)
Raw averages can be distorted when a product has very few reviews (e.g. 5.0★ with 1 review):

$$R_{smooth} = \frac{C \cdot m + R \cdot N}{C + N}$$

Where:
- $R$: Raw average star rating ($0 \le R \le 5$)
- $N$: Total number of verified customer reviews
- $m = 3.8$: Prior global Indian market average rating
- $C = 10$: Confidence regularization weight

### B. Feature Alignment Vector Matching ($S_{spec}$)
For a product with extracted features $\mathbf{f}$ and user importance weights $\mathbf{w} \in [0.2, 1.0]^k$:

$$S_{spec} = \frac{\sum_{i=1}^k w_i \cdot \phi_i(\mathbf{f})}{\sum_{i=1}^k w_i}$$

Where $\phi_i(\mathbf{f}) \in [0.0, 1.0]$ represents the satisfaction function for category requirement $i$ (e.g. $\phi_{gaming}(\mathbf{f}) = 0.7 \cdot \frac{GPU_{tier}}{5} + 0.3 \cdot \frac{RAM}{16}$).

### C. Hybrid Decision Score ($S_{final}$)
The final ranking score balances model inference, hard preference alignment, rating quality, and budget elasticity:

$$S_{final} = 0.45 \cdot \hat{y}_{ML} + 0.30 \cdot (100 \cdot S_{spec}) + 0.15 \cdot (100 \cdot \text{Quality}) + 0.10 \cdot (100 \cdot \text{Brand})$$

Where $\hat{y}_{ML}$ is the Gradient Boosting suitability prediction.

---

## 4. Training Pipeline & Feature Engineering

The supervised training pipeline (`ml/scripts/train.py`) processes 22 structured features:
- **Product Features (14):** `price`, `original_price`, `discount_percentage`, `rating`, `review_count`, `ram_gb`, `storage_gb`, `cpu_tier`, `gpu_tier`, `camera_mp`, `battery_mah`, `is_5g`, `has_anc`, `energy_star`
- **Context Features (6):** `user_max_budget`, `user_min_budget`, `primary_req_weight`, `avg_req_weight`, `price_to_budget_ratio`, `spec_fit_score`
- **Categorical Features (2):** `category` (One-Hot Encoded), `user_priority` (One-Hot Encoded: Value, Rating, Budget, Balanced)

### Preprocessing Pipeline:
- Numerics scaled with `StandardScaler()`
- Categoricals transformed with `OneHotEncoder(handle_unknown='ignore')`
- Packaged as a unified `sklearn.pipeline.Pipeline` persisted via `joblib`.
