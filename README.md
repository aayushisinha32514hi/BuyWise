# BuyWise — AI-Powered Multi-Category Shopping Decision Support & Price Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **B.Tech CSE (AI Specialisation) Capstone Project**  
> BuyWise is an intelligent e-commerce decision-support system that combines **supervised machine learning**, **natural language requirement understanding**, **multi-criteria preference matching**, and **live multi-retailer price intelligence** for the Indian shopping market.

---

## 🚀 Key Features

### 1. 🧠 Hybrid ML Recommendation Engine
- **Supervised Suitability Model:** Gradient Boosting Regressor trained on 5,000 product-user suitability samples with **$R^2 = 0.9708$** and **MAE = 1.16 points** on held-out test data.
- **Category-Aware Buying Priorities:** Category taxonomies (Mobiles, Laptops, TVs, Audio, Appliances) with 1–5 star importance scales.
- **Preference Sensitivity:** Distinct user priorities (e.g. Gaming 5★ vs Coding 5★) dynamically alter product recommendations.
- **Transparent Explainability (XAI):** Natural language bullet points detailing exact reasons why each product was chosen based on model feature contributions.

### 2. 💬 NLP Shopping Intent Understanding
- **Unstructured Prompt Parsing:** Accepts freeform requests like *"Need a coding laptop under 85000 with 16GB RAM and fast SSD"* and auto-extracts target category, budget bounds, brand preferences, and priority weights.

### 3. 🛒 Live Multi-Retailer Price Intelligence
- **Real Shopping Data:** Real-time Indian shopping catalog indexed via SerpAPI (Google Shopping India, `gl=in`, INR).
- **Multi-Merchant Offers Table:** Side-by-side verified quotes across Amazon.in, Flipkart, Croma, Reliance Digital, etc., with direct purchase links.
- **Price Spread Analytics:** Highlights price differences across sellers (e.g. *"Save ₹4,000 — cheapest at Amazon"*).
- **Empirical Price History:** Records empirical observations on every search to build genuine price history trends without fabrication.

### 4. ⚖️ Multi-Product Comparison Matrix
- Compare 2 to 4 products side-by-side with full specification matrices, multi-merchant offers, and calculated **AI Winner Badges** (*Best Overall*, *Best Value*, *Top Rated*).

### 5. 📱 Responsive Modern Web Application
- Mobile-first single-page application with bottom navigation on phones, expanding seamlessly into multi-column desktop grids with top navigation on larger screens.

---

## 📊 Supervised Machine Learning Benchmarks

| Model Architecture | 5-Fold CV $R^2$ | 5-Fold CV MAE | Test Set $R^2$ | Test Set MAE | Test Set RMSE |
|---|---|---|---|---|---|
| **Baseline (Mean Predictor)** | $-0.0004$ | $7.30$ | $-0.0004$ | $7.14$ | $9.00$ |
| **Ridge Regression ($L_2$)** | $0.8336$ | $2.73$ | $0.8523$ | $2.55$ | $3.46$ |
| **Random Forest Regressor** | $0.9411$ | $1.64$ | $0.9496$ | $1.47$ | $2.02$ |
| **Gradient Boosting (Production)** | **$0.9679$** | **$1.25$** | **$0.9708$** | **$1.16$** | **$1.54$** |

*See full evaluation report and diagnostic charts in [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md).*

---

## 🛠️ System Architecture & Tech Stack

```
User Prompt / Requirements
        ↓
NLP Intent Extraction (`nlp_extractor.py`)
        ↓
Live Shopping Retrieval & 6-Hour Cache (`serpapi_provider.py` & `cache.py`)
        ↓
Semantic Spec Extraction (RAM, CPU, GPU, Camera, Battery, 5G, ANC)
        ↓
Supervised ML Suitability Prediction (`suitability_model.joblib`)
        ↓
Multi-Criteria Preference Vector Matching & Bayesian Smoothing
        ↓
Explainable Recommendations & Multi-Retailer Offers Table
```

- **Backend:** FastAPI, Uvicorn, SQLite, Pydantic, Python-dotenv
- **Machine Learning & NLP:** scikit-learn, joblib, pandas, numpy, vaderSentiment, matplotlib, seaborn
- **Frontend:** Vanilla JavaScript (ES6+ SPA), CSS Grid & Flexbox, HTML5

---

## 🏃 Quickstart & Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/aayushisinha32514hi/BuyWise.git
cd BuyWise

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```ini
SERPAPI_KEY=your_serpapi_key_here
ENVIRONMENT=development
```

### 3. Run the ML Pipeline (Optional - Pre-trained Model Included)
```bash
# Generate 5,000 synthetic training pairs
python3 ml/scripts/generate_training_data.py

# Run EDA & generate 4 analytical charts
python3 ml/scripts/eda.py

# Train candidate models & export Gradient Boosting pipeline
python3 ml/scripts/train.py

# Run evaluation & generate diagnostic charts
python3 ml/scripts/evaluate.py
```

### 4. Start the Application Server
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **http://127.0.0.1:8000** in your browser. Interactive API documentation is available at **http://127.0.0.1:8000/docs**.

---

## 🧪 Automated Testing

Run the test suite:
```bash
# Run ML unit tests
python3 tests/test_ml.py

# Run API & End-to-End integration tests
python3 tests/test_api.py
```

---

## 📁 Repository Structure

```
BuyWise/
├── backend/
│   ├── api/routes.py              # REST API route handlers
│   ├── database/                  # Database initialization & migrations
│   ├── models/schemas.py          # Pydantic data validation models
│   ├── services/
│   │   ├── auth_service.py        # User authentication & wishlist
│   │   ├── category_service.py    # Category taxonomies
│   │   ├── nlp_extractor.py       # Natural language requirement parsing
│   │   ├── product_service.py     # Product search, compare, & price history
│   │   └── product_search/
│   │       ├── base.py            # Canonical product identity
│   │       ├── cache.py           # SQLite cache & price observations
│   │       ├── deduplicator.py    # Cross-merchant deduplication
│   │       ├── feature_extractor.py # Regex spec extraction
│   │       ├── normalizer.py      # Category classification & BuyWise score
│   │       ├── recommendation_engine.py # Hybrid ML recommendation engine
│   │       └── serpapi_provider.py # SerpAPI Google Shopping client
│   └── main.py                    # FastAPI application entrypoint
├── data/
│   └── processed/training_data.csv# Synthesized ML training data
├── docs/
│   ├── ARCHITECTURE.md            # System architecture & data flow
│   ├── ML_METHODOLOGY.md          # Machine learning formulas & taxonomy
│   └── MODEL_EVALUATION.md        # Comprehensive benchmark reports
├── frontend/
│   ├── app.js                     # SPA frontend application controller
│   ├── index.html                 # Responsive single-page UI
│   └── styles.css                 # Responsive themes & layout styles
├── ml/
│   ├── models/                    # Serialized joblib production pipelines
│   ├── outputs/                   # Saved EDA & evaluation diagnostic charts
│   ├── scripts/                   # Training, EDA, and evaluation scripts
│   └── sentiment/analyzer.py      # VADER sentiment analysis
├── tests/
│   ├── test_api.py                # End-to-end API integration tests
│   └── test_ml.py                 # Supervised ML unit tests
├── Dockerfile                     # Production container specification
├── docker-compose.yml             # Container orchestration
└── requirements.txt               # Pinned project dependencies
```

---

## 📜 License
This project is open-source under the MIT License.
