"""
BuyWise — Comprehensive Model Evaluation & Diagnostic Visualizer
==================================================================
1. Evaluates saved Gradient Boosting Regressor against held-out test data
2. Generates Diagnostic Visualizations (Actual vs Predicted, Residuals, Feature Importance)
3. Outputs updated results table for documentation and benchmarking.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/training_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/suitability_model.joblib')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../outputs/evaluation')
EVAL_DOC_PATH = os.path.join(os.path.dirname(__file__), '../../docs/MODEL_EVALUATION.md')

NUMERIC_FEATURES = [
    'price', 'original_price', 'discount_percentage', 'rating', 'review_count',
    'ram_gb', 'storage_gb', 'cpu_tier', 'gpu_tier', 'camera_mp', 'battery_mah',
    'is_5g', 'has_anc', 'energy_star', 'user_max_budget', 'user_min_budget',
    'primary_req_weight', 'avg_req_weight', 'price_to_budget_ratio', 'spec_fit_score'
]

CATEGORICAL_FEATURES = ['category', 'user_priority']
TARGET = 'suitability_score'


def run_evaluation():
    print("=" * 68)
    print("BuyWise — Supervised Model Evaluation & Diagnostics")
    print("=" * 68)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print(f"Error: Model or Data not found. Run train.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values

    # Exactly match train.py 80/20 test split with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    y_pred_test = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)

    test_r2 = float(r2_score(y_test, y_pred_test))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    print(f"Test Set R² Score:  {test_r2:.4f} (Variance Explained)")
    print(f"Test Set MAE:       {test_mae:.3f} points")
    print(f"Test Set RMSE:      {test_rmse:.3f} points\n")

    # ── Diagnostic Chart 1: Actual vs Predicted ──
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred_test, alpha=0.45, color="#4F46E5", edgecolors='none', s=35)
    
    # Ideal 45-degree line
    lims = [min(y_test.min(), y_pred_test.min()) - 2, max(y_test.max(), y_pred_test.max()) + 2]
    ax.plot(lims, lims, color="#EF4444", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
    
    ax.set_title(f"Actual vs Predicted Suitability Score (Test R² = {test_r2:.3f})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Actual Suitability Score (Ground Truth)")
    ax.set_ylabel("Model Predicted Suitability Score")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc="upper left")
    plt.tight_layout()
    chart1 = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    fig.savefig(chart1, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Diagnostic Chart 1: {chart1}")

    # ── Diagnostic Chart 2: Residuals Distribution ──
    residuals = y_test - y_pred_test
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#7C3AED", bins=30, ax=ax)
    ax.axvline(0, color="#10B981", linestyle="--", linewidth=2, label="Zero Error Line")
    ax.set_title(f"Prediction Error (Residuals) Distribution (Mean = {residuals.mean():.2f}, Std = {residuals.std():.2f})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Error (Actual - Predicted Suitability)")
    ax.set_ylabel("Sample Count")
    ax.legend()
    plt.tight_layout()
    chart2 = os.path.join(OUTPUT_DIR, "residuals_distribution.png")
    fig.savefig(chart2, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Diagnostic Chart 2: {chart2}")

    # ── Diagnostic Chart 3: Feature Importances ──
    gb_model = pipeline.named_steps['regressor']
    fitted_prep = pipeline.named_steps['preprocessor']
    cat_names = fitted_prep.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_names = NUMERIC_FEATURES + cat_names
    importances = gb_model.feature_importances_

    feat_df = pd.DataFrame({'feature': all_names, 'importance': importances})
    feat_df = feat_df.sort_values(by='importance', ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_df, x='importance', y='feature', palette='mako', ax=ax)
    ax.set_title("Top 12 Most Influential Features (Gradient Boosting Gini Importance)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Normalized Feature Importance Score")
    ax.set_ylabel("Feature Name")
    plt.tight_layout()
    chart3 = os.path.join(OUTPUT_DIR, "feature_importance_bar.png")
    fig.savefig(chart3, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Diagnostic Chart 3: {chart3}")

    print("=" * 68)
    print("Evaluation Complete! All charts saved to ml/outputs/evaluation/")
    print("=" * 68)


if __name__ == '__main__':
    run_evaluation()
