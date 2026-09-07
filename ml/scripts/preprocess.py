"""
BuyWise — Data Preprocessing & Cleaning Pipeline
=================================================
Processes raw Amazon India multi-category product catalog from:
data/raw/amazonproducts/archive/Amazon-Products.csv

Outputs:
data/processed/products_cleaned.csv
"""

import os
import re
import pandas as pd
import numpy as np

RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/amazonproducts/archive/Amazon-Products.csv')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DIR, 'products_cleaned.csv')

def clean_currency(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace('₹', '').replace(',', '').strip()
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def clean_integer(val):
    if pd.isna(val):
        return 0
    val_str = str(val).replace(',', '').strip()
    try:
        return int(float(val_str))
    except ValueError:
        return 0

def extract_brand(name):
    if pd.isna(name):
        return "Generic"
    # Take first 1 or 2 words before special characters
    tokens = str(name).strip().split()
    if not tokens:
        return "Generic"
    first_word = re.sub(r'[^a-zA-Z0-9]', '', tokens[0])
    if len(first_word) >= 2:
        return first_word.capitalize()
    return "Generic"

def run_pipeline():
    print("=" * 60)
    print("BuyWise Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"Reading raw data from: {RAW_CSV_PATH}")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    initial_count = len(df)
    print(f"Initial raw record count: {initial_count:,}")
    
    # Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    print("Cleaning numeric fields (prices, ratings, rating counts)...")
    df['discount_price_cleaned'] = df['discount_price'].apply(clean_currency)
    df['actual_price_cleaned'] = df['actual_price'].apply(clean_currency)
    
    # Fill actual price with discount price if actual is missing, or vice versa
    df['price'] = df['discount_price_cleaned'].fillna(df['actual_price_cleaned'])
    df['original_price'] = df['actual_price_cleaned'].fillna(df['price'])
    
    # Ratings cleaning
    df['ratings_cleaned'] = pd.to_numeric(df['ratings'], errors='coerce').fillna(0.0)
    df['no_of_ratings_cleaned'] = df['no_of_ratings'].apply(clean_integer)
    
    # Compute discount percentage
    df['discount_percentage'] = np.where(
        df['original_price'] > 0,
        np.clip(((df['original_price'] - df['price']) / df['original_price']) * 100, 0, 99),
        0.0
    )
    
    # Extract Brand
    print("Extracting brand names...")
    df['brand'] = df['name'].apply(extract_brand)
    
    # Clean text strings
    df['name'] = df['name'].astype(str).str.strip()
    df['main_category'] = df['main_category'].astype(str).str.strip().str.title()
    df['sub_category'] = df['sub_category'].astype(str).str.strip().str.title()
    df['image'] = df['image'].fillna('')
    df['link'] = df['link'].fillna('')
    
    # Filter valid products: must have valid name and positive price
    valid_df = df[(df['name'] != '') & (df['price'] > 0) & (df['name'] != 'nan')].copy()
    print(f"Valid products with positive price: {len(valid_df):,}")
    
    # Deduplicate exact same name and subcategory
    dedup_df = valid_df.drop_duplicates(subset=['name', 'sub_category']).copy()
    print(f"Products after deduplication: {len(dedup_df):,}")
    
    # Feature Engineering: BuyWise Base Score (0-100)
    # Composite of Rating (40%), Popularity/Log Review count (30%), Value for money/Discount (30%)
    rating_norm = np.clip(dedup_df['ratings_cleaned'] / 5.0, 0, 1.0)
    review_norm = np.clip(np.log1p(dedup_df['no_of_ratings_cleaned']) / np.log1p(50000), 0, 1.0)
    discount_norm = np.clip(dedup_df['discount_percentage'] / 100.0, 0, 1.0)
    
    dedup_df['buywise_score'] = np.round(
        (rating_norm * 45 + review_norm * 35 + discount_norm * 20), 1
    )
    
    # Keep final structured columns
    final_cols = [
        'name', 'brand', 'main_category', 'sub_category',
        'price', 'original_price', 'discount_percentage',
        'ratings_cleaned', 'no_of_ratings_cleaned',
        'buywise_score', 'image', 'link'
    ]
    
    final_df = dedup_df[final_cols].rename(columns={
        'ratings_cleaned': 'rating',
        'no_of_ratings_cleaned': 'review_count'
    })
    
    print(f"Saving cleaned dataset to: {PROCESSED_CSV_PATH}")
    final_df.to_csv(PROCESSED_CSV_PATH, index=False)
    
    print("=" * 60)
    print("Preprocessing Complete!")
    print(f"Total Cleaned Products: {len(final_df):,}")
    print(f"Categories Count: {final_df['sub_category'].nunique()} subcategories across {final_df['main_category'].nunique()} main categories")
    print(f"Sample price range: ₹{final_df['price'].min():,.0f} to ₹{final_df['price'].max():,.0f}")
    print(f"Average BuyWise Score: {final_df['buywise_score'].mean():.1f} / 100")
    print("=" * 60)

if __name__ == '__main__':
    run_pipeline()
