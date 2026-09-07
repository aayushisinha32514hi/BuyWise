"""
BuyWise — Supervised ML Model Training Pipeline
=================================================
Trains, cross-validates, and exports:
1. Baseline Mean Predictor (DummyRegressor)
2. Ridge Regression (L2 Linear Comparison)
3. Random Forest Regressor
4. Gradient Boosting Regressor (Primary Production Model)

Includes full ColumnTransformer preprocessing (StandardScaler + OneHotEncoder),
80/20 train/test split (random_state=42), 5-fold cross-validation, and model persistence.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/training_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'suitability_model.joblib')
SUMMARY_SAVE_PATH = os.path.join(MODEL_DIR, 'training_summary.json')

NUMERIC_FEATURES = [
    'price', 'original_price', 'discount_percentage', 'rating', 'review_count',
    'ram_gb', 'storage_gb', 'cpu_tier', 'gpu_tier', 'camera_mp', 'battery_mah',
    'is_5g', 'has_anc', 'energy_star', 'user_max_budget', 'user_min_budget',
    'primary_req_weight', 'avg_req_weight', 'price_to_budget_ratio', 'spec_fit_score'
]

CATEGORICAL_FEATURES = ['category', 'user_priority']
TARGET = 'suitability_score'


def build_preprocessor() -> ColumnTransformer:
    """Construct sklearn ColumnTransformer for robust numeric scaling & categorical one-hot encoding."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def train_and_evaluate():
    print("=" * 68)
    print("BuyWise — Supervised Machine Learning Training Pipeline")
    print("=" * 68)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}. Run generate_training_data.py first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Loaded: {len(df):,} samples with {len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)} features.")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values

    # 80/20 Train/Test Split with fixed random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training Set: {len(X_train):,} samples | Held-out Test Set: {len(X_test):,} samples\n")

    preprocessor = build_preprocessor()

    # Candidate Models Dictionary
    models = {
        "Baseline (Mean Predictor)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=160, learning_rate=0.08, max_depth=5, min_samples_split=4, subsample=0.9, random_state=42
        )
    }

    results = {}
    fitted_pipelines = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("--- 5-Fold Cross-Validation Performance on Training Split ---")
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('regressor', model)
        ])

        # 5-fold cross-validation scoring
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
            return_train_score=True
        )

        cv_r2_mean = float(np.mean(cv_res['test_r2']))
        cv_r2_std = float(np.std(cv_res['test_r2']))
        cv_mae_mean = float(-np.mean(cv_res['test_neg_mean_absolute_error']))
        cv_rmse_mean = float(-np.mean(cv_res['test_neg_root_mean_squared_error']))

        print(f"Model: {name:<30} | 5-Fold CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f} | CV MAE: {cv_mae_mean:.3f} | CV RMSE: {cv_rmse_mean:.3f}")

        # Fit on full training split
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Evaluate on held-out 20% test set
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        train_r2 = float(r2_score(y_train, y_pred_train))
        test_r2 = float(r2_score(y_test, y_pred_test))
        train_mae = float(mean_absolute_error(y_train, y_pred_train))
        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

        results[name] = {
            "cv_5fold_r2_mean": round(cv_r2_mean, 4),
            "cv_5fold_r2_std": round(cv_r2_std, 4),
            "cv_5fold_mae": round(cv_mae_mean, 3),
            "cv_5fold_rmse": round(cv_rmse_mean, 3),
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "train_mae": round(train_mae, 3),
            "test_mae": round(test_mae, 3),
            "train_rmse": round(train_rmse, 3),
            "test_rmse": round(test_rmse, 3),
        }

    print("\n--- Held-out Test Set Evaluation (20% Split) ---")
    for name, m in results.items():
        print(f"Model: {name:<30} | Test R²: {m['test_r2']:.4f} | Test MAE: {m['test_mae']:.3f} | Test RMSE: {m['test_rmse']:.3f}")

    # Best model selection (Gradient Boosting Regressor)
    best_name = "Gradient Boosting Regressor"
    best_pipeline = fitted_pipelines[best_name]

    # Save trained model pipeline and metadata
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    print(f"\n✓ Saved production model pipeline to: {MODEL_SAVE_PATH}")

    # Extract feature importances
    gb_model = best_pipeline.named_steps['regressor']
    fitted_prep = best_pipeline.named_steps['preprocessor']
    cat_feature_names = fitted_prep.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERIC_FEATURES + cat_feature_names
    importances = gb_model.feature_importances_.tolist()

    feat_imp = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    results[best_name]["top_feature_importances"] = [
        {"feature": f, "importance": round(imp, 4)} for f, imp in feat_imp[:12]
    ]

    with open(SUMMARY_SAVE_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved training summary metadata to: {SUMMARY_SAVE_PATH}")

    print("=" * 68)
    print(f"SUCCESS: Trained and validated {best_name} (Test R²: {results[best_name]['test_r2']:.4f}, MAE: {results[best_name]['test_mae']:.2f})")
    print("=" * 68)


if __name__ == '__main__':
    train_and_evaluate()
