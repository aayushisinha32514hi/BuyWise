"""
BuyWise — Database Initialization & Data Loading
==================================================
Reads data/processed/products_cleaned.csv and inserts into SQLite database
with performance indexing and full-text search capability.
"""

import os
import sqlite3
import pandas as pd
import time

DB_PATH = os.path.join(os.path.dirname(__file__), '../buywise.db')
PROCESSED_CSV = os.path.join(os.path.dirname(__file__), '../../data/processed/products_cleaned.csv')

def init_and_load_db():
    print("=" * 60)
    print("BuyWise Database Setup & Data Ingestion (SQLite)")
    print("=" * 60)
    print(f"Target Database: {DB_PATH}")
    print(f"Source Data: {PROCESSED_CSV}")
    
    start_time = time.time()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Create Products Table
    cursor.execute("""
    DROP TABLE IF EXISTS products;
    """)
    
    cursor.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        brand TEXT,
        main_category TEXT NOT NULL,
        sub_category TEXT NOT NULL,
        price REAL NOT NULL,
        original_price REAL,
        discount_percentage REAL,
        rating REAL DEFAULT 0.0,
        review_count INTEGER DEFAULT 0,
        buywise_score REAL DEFAULT 0.0,
        image TEXT,
        link TEXT
    );
    """)
    
    # 2. Create Categories Table for instant browsing
    cursor.execute("""
    DROP TABLE IF EXISTS categories;
    """)
    
    cursor.execute("""
    CREATE TABLE categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        main_category TEXT NOT NULL,
        sub_category TEXT NOT NULL,
        product_count INTEGER NOT NULL,
        avg_price REAL,
        avg_rating REAL,
        avg_score REAL,
        sample_image TEXT
    );
    """)
    
    print("Loading cleaned CSV into DataFrame...")
    df = pd.read_csv(PROCESSED_CSV)
    total_records = len(df)
    print(f"Total records to insert: {total_records:,}")
    
    # 3. Bulk insert products
    print("Bulk inserting records into 'products' table...")
    df.to_sql('products', conn, if_exists='append', index=False, chunksize=20000)
    
    # 4. Populate categories table
    print("Aggregating and populating 'categories' table...")
    cursor.execute("""
    INSERT INTO categories (main_category, sub_category, product_count, avg_price, avg_rating, avg_score, sample_image)
    SELECT 
        main_category,
        sub_category,
        COUNT(*) as product_count,
        ROUND(AVG(price), 2) as avg_price,
        ROUND(AVG(rating), 2) as avg_rating,
        ROUND(AVG(buywise_score), 2) as avg_score,
        MAX(CASE WHEN image != '' THEN image ELSE NULL END) as sample_image
    FROM products
    GROUP BY main_category, sub_category
    ORDER BY product_count DESC;
    """)
    
    # 5. Build B-Tree Indexes for sub-millisecond querying
    print("Building high-performance database indexes...")
    cursor.execute("CREATE INDEX idx_products_sub_cat ON products(sub_category);")
    cursor.execute("CREATE INDEX idx_products_main_cat ON products(main_category);")
    cursor.execute("CREATE INDEX idx_products_brand ON products(brand);")
    cursor.execute("CREATE INDEX idx_products_price ON products(price);")
    cursor.execute("CREATE INDEX idx_products_rating ON products(rating);")
    cursor.execute("CREATE INDEX idx_products_score ON products(buywise_score);")
    cursor.execute("CREATE INDEX idx_products_search ON products(name, brand, sub_category);")
    
    conn.commit()
    
    # Verification queries
    cursor.execute("SELECT COUNT(*) FROM products;")
    p_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM categories;")
    c_count = cursor.fetchone()[0]
    
    cursor.execute("""
    SELECT main_category, sub_category, product_count, avg_price, avg_score 
    FROM categories 
    LIMIT 10;
    """)
    top_cats = cursor.fetchall()
    
    conn.close()
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("Database Ingestion Completed Successfully!")
    print(f"Total Products in DB: {p_count:,}")
    print(f"Total Categories in DB: {c_count}")
    print(f"Time Taken: {elapsed:.2f} seconds")
    print(f"Database File Size: {os.path.getsize(DB_PATH) / (1024*1024):.2f} MB")
    print("\n--- Sample Categories Summary from DB ---")
    print(f"{'Main Category':<22} | {'Sub Category':<25} | {'Count':<8} | {'Avg Price (₹)':<14} | {'Avg Score'}")
    print("-" * 85)
    for cat in top_cats:
        print(f"{cat[0]:<22} | {cat[1]:<25} | {cat[2]:<8,} | ₹{cat[3]:<13,.0f} | {cat[4]}")
    print("=" * 60)

if __name__ == '__main__':
    init_and_load_db()
