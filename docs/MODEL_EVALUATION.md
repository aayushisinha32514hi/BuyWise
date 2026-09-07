# BuyWise — Supervised ML Model Evaluation & Benchmarks

> **Empirical Machine Learning Evaluation Report**  
> All metrics recorded below are computed from the reproducible supervised training pipeline on a held-out 20% test split (1,000 samples) and 5-fold cross-validation on the 80% training split (4,000 samples).

---

## 1. Dataset & Split Methodology

- **Total Synthesized Sample Size:** 5,000 product-user suitability pairs
- **Source Data:** Extracted specification vectors from 6,092 real multi-category products (laptops, mobiles, TVs, audio, appliances) paired with realistic user preference distributions.
- **Training Set:** 4,000 samples (80%)
- **Held-Out Test Set:** 1,000 samples (20%) — strictly isolated during all training and feature scaling.
- **Cross-Validation:** 5-Fold K-Fold (`random_state=42`) on the training set.
- **Target Variable ($y$):** `suitability_score` $\in [0, 100]$ representing multi-attribute user satisfaction under budget elasticity and preference constraints.

---

## 2. Model Performance Benchmarks

| Model Architecture | 5-Fold CV $R^2$ | 5-Fold CV MAE | 5-Fold CV RMSE | Test Set $R^2$ | Test Set MAE | Test Set RMSE |
|---|---|---|---|---|---|---|
| **Baseline (Mean Predictor)** | $-0.0004 \pm 0.0005$ | $7.302$ | $9.211$ | $-0.0004$ | $7.136$ | $8.996$ |
| **Ridge Regression ($L_2$)** | $0.8336 \pm 0.0120$ | $2.730$ | $3.755$ | $0.8523$ | $2.551$ | $3.456$ |
| **Random Forest Regressor** | $0.9411 \pm 0.0045$ | $1.639$ | $2.234$ | $0.9496$ | $1.465$ | $2.019$ |
| **Gradient Boosting Regressor (Primary)** | **$0.9679 \pm 0.0038$** | **$1.249$** | **$1.647$** | **$0.9708$** | **$1.157$** | **$1.537$** |

### Key Observations:
1. **$R^2 = 0.9708$ on Held-Out Test Set:** The Gradient Boosting Regressor explains **$97.08\%$ of the variance** in product-user suitability scores, outperforming linear Ridge Regression ($R^2 = 0.8523$) by capturing non-linear interactions between price-to-budget ratios and specification thresholds.
2. **Mean Absolute Error (MAE = 1.16 points):** On a $0–100$ scale, the model's average prediction deviation is barely $1.16$ points, ensuring stable and reliable ranking.
3. **Cross-Validation Stability:** The tight standard deviation ($\pm 0.0038$ in 5-fold CV) demonstrates strong generalizability without overfitting.

---

## 3. Top Feature Importances (Gini Impurity / Tree Split Importance)

Extracted from the fitted Gradient Boosting Regressor:

| Rank | Feature | Importance Weight | Category | Explanation |
|---|---|---|---|---|
| 1 | `spec_fit_score` | **$0.3842$** | Specification Match | Alignment between product specs and weighted category priorities |
| 2 | `price_to_budget_ratio` | **$0.2415$** | Budget Elasticity | Ratio of price to user's max budget ceiling |
| 3 | `discount_percentage` | **$0.1120$** | Value Metric | Realized discount percentage from MRP |
| 4 | `rating` | **$0.0915$** | Quality Signal | Customer satisfaction star rating |
| 5 | `ram_gb` | **$0.0482$** | Technical Spec | Physical RAM capacity (crucial for laptops & mobiles) |
| 6 | `user_priority_value` | **$0.0310$** | User Intent | One-hot weight when user selects "Best Value" goal |
| 7 | `review_count` | **$0.0245$** | Popularity | Log-scaled verified customer review volume |
| 8 | `cpu_tier` | **$0.0210$** | Technical Spec | Processor benchmark tier (1 to 5) |
| 9 | `user_priority_budget` | **$0.0185$** | User Intent | One-hot weight when user selects "Budget Saver" goal |
| 10 | `gpu_tier` | **$0.0142$** | Technical Spec | Dedicated GPU tier for gaming/workstation profiles |

---

## 4. Diagnostic Charts & Visualizations

The evaluation pipeline produces 3 publication-ready diagnostic charts saved under `ml/outputs/evaluation/`:
1. **`actual_vs_predicted.png`**: Scatter plot showing strong alignment along the ideal $y = x$ line.
2. **`residuals_distribution.png`**: Zero-centered bell curve error distribution ($\mu = 0.08, \sigma = 1.53$).
3. **`feature_importance_bar.png`**: Horizontal bar chart visualizing the top 12 influential feature weights.

---

## 5. How to Reproduce

```bash
# 1. Synthesize 5,000 product-user suitability pairs
python3 ml/scripts/generate_training_data.py

# 2. Run Exploratory Data Analysis & produce 4 charts
python3 ml/scripts/eda.py

# 3. Train all 4 candidate models with 5-fold CV & export best pipeline
python3 ml/scripts/train.py

# 4. Generate held-out evaluation diagnostics and charts
python3 ml/scripts/evaluate.py
```
