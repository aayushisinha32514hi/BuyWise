"""
BuyWise — Exploratory Data Analysis (EDA) Pipeline
===================================================
Produces statistical summaries and publication-ready visual charts for:
1. Product price & discount distributions by category
2. Rating & review volume distributions
3. Category-specific suitability variations across priority goals
4. Feature correlation heatmap with target suitability score

Outputs saved to: ml/outputs/eda/
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for server execution
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/training_data.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../outputs/eda')


def run_eda():
    print("=" * 65)
    print("BuyWise — Exploratory Data Analysis (EDA)")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Training dataset not found at {DATA_PATH}. Run generate_training_data.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset successfully loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    # ── Summary Statistics ──
    print("--- Statistical Distribution of Key Numeric Variables ---")
    numeric_cols = ['price', 'discount_percentage', 'rating', 'review_count', 'spec_fit_score', 'suitability_score']
    print(df[numeric_cols].describe().round(2).to_string())
    print("\n")

    # Set aesthetic style
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

    # ── Chart 1: Price Distribution & Suitability ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Log-transformed price distribution
    sns.histplot(np.log10(df['price']), kde=True, ax=axes[0], color="#4F46E5", bins=30)
    axes[0].set_title("Product Price Distribution (Log10 Scale)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Log10(Price in INR)")
    axes[0].set_ylabel("Count")

    # Suitability Score Distribution
    sns.histplot(df['suitability_score'], kde=True, ax=axes[1], color="#10B981", bins=30)
    axes[1].axvline(df['suitability_score'].mean(), color='#EF4444', linestyle='--', label=f"Mean: {df['suitability_score'].mean():.1f}")
    axes[1].set_title("Target Suitability Score Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Suitability Score (0 - 100)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    chart1_path = os.path.join(OUTPUT_DIR, "price_and_suitability_distribution.png")
    fig.savefig(chart1_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Chart 1: {chart1_path}")

    # ── Chart 2: Category Suitability & Priority Goal Boxplots ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    top_cats = df['category'].value_counts().head(6).index
    df_top = df[df['category'].isin(top_cats)]

    sns.boxplot(data=df_top, x='category', y='suitability_score', hue='category', ax=axes[0], palette="Set2", legend=False)
    axes[0].set_title("Suitability Score Distribution by Category", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Product Category")
    axes[0].set_ylabel("Suitability Score")
    axes[0].tick_params(axis='x', rotation=25)

    sns.boxplot(data=df, x='user_priority', y='suitability_score', hue='user_priority', ax=axes[1], palette="pastel", legend=False)
    axes[1].set_title("Suitability Score by User Optimization Goal", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Optimization Priority (Value / Rating / Budget / Balanced)")
    axes[1].set_ylabel("Suitability Score")

    plt.tight_layout()
    chart2_path = os.path.join(OUTPUT_DIR, "category_and_priority_boxplots.png")
    fig.savefig(chart2_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Chart 2: {chart2_path}")

    # ── Chart 3: Correlation Heatmap ──
    corr_cols = [
        'suitability_score', 'spec_fit_score', 'price_to_budget_ratio',
        'rating', 'discount_percentage', 'review_count',
        'ram_gb', 'storage_gb', 'cpu_tier', 'gpu_tier',
        'camera_mp', 'battery_mah', 'is_5g', 'energy_star'
    ]
    corr_df = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix with Target Suitability Score", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    chart3_path = os.path.join(OUTPUT_DIR, "feature_correlation_heatmap.png")
    fig.savefig(chart3_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Chart 3: {chart3_path}")

    # ── Chart 4: Rating vs Suitability by Price Tier ──
    df['price_tier'] = pd.qcut(df['price'], q=3, labels=['Budget (< 33%)', 'Mid-Range (33-66%)', 'Premium (> 66%)'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df.sample(min(800, len(df)), random_state=42),
        x='spec_fit_score',
        y='suitability_score',
        hue='price_tier',
        size='rating',
        sizes=(20, 150),
        alpha=0.7,
        palette='viridis',
        ax=ax
    )
    ax.set_title("Spec Fit Score vs Suitability Score across Price Tiers", fontsize=12, fontweight='bold')
    ax.set_xlabel("Extracted Spec Fit Score (0.0 to 1.0)")
    ax.set_ylabel("Final Suitability Score (0 to 100)")
    plt.tight_layout()
    chart4_path = os.path.join(OUTPUT_DIR, "spec_fit_vs_suitability_scatter.png")
    fig.savefig(chart4_path, dpi=200)
    plt.close(fig)
    print(f"✓ Saved Chart 4: {chart4_path}")

    print("=" * 65)
    print("EDA Complete! All 4 analytical charts saved to ml/outputs/eda/")
    print("=" * 65)


if __name__ == '__main__':
    run_eda()
