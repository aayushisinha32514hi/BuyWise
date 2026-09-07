# PROJECT_PLAN.md — BuyWise Development Plan

> Solo project | B.Tech CSE Semester 3 | AI/ML + Full-Stack Mini-Project
> REVISED: 2026-08-29 — Project expanded to multi-category platform

---

## Project Definition (Revised)

**BuyWise** is an AI/ML-powered, mobile-first, multi-category product discovery,
comparison and recommendation platform using real-world product data.

It is NOT a laptop-only demo.
It is NOT a static product catalogue.
It is NOT a fake-AI showcase.

---

## Supported Product Categories (Target: 15–20)

1. Smartphones
2. Laptops
3. Tablets
4. Televisions / Smart TVs
5. Headphones
6. TWS / Earbuds
7. Smartwatches
8. Cameras
9. Gaming Consoles / Accessories
10. Computer Monitors
11. Printers
12. Wi-Fi Routers
13. Power Banks
14. Keyboards
15. Computer Mice
16. Refrigerators
17. Washing Machines
18. Air Conditioners
19. Air Fryers
20. Vacuum Cleaners

Final category list confirmed after data-source verification in Phase 1.

---

## Milestone Timeline

| Milestone | What | Date |
|---|---|---|
| M1 | Problem identification + source feasibility | 7–11 Sep 2026 |
| M2 | Data pipeline + EDA + ML model progress | 5–9 Oct 2026 |
| M3 | Full working application + evaluation + report | 23–27 Nov 2026 |

---

## Development Phases (Revised)

### PHASE 0 — Environment Setup ✅ DONE
- [x] Inspect OS, Python, Node.js, Git, database availability
- [x] Create project folder structure
- [x] Create .gitignore, README.md, DATA_SOURCE.md, ML_METHODOLOGY.md
- [x] Initialize git repository

### PHASE 1 — Architecture + Verified Real-Data Source Research ⬅️ IN PROGRESS
- [ ] Research real datasets for all 20 categories
- [ ] Identify which categories have verified real public datasets
- [ ] Identify any combined multi-category datasets (Amazon/Flipkart)
- [ ] Verify licensing/usage for each identified source
- [ ] Design multi-source data strategy
- [ ] Update DATA_SOURCE.md with verified sources
- [ ] Design category-aware database schema
- [ ] Finalize architecture (mobile-first, FastAPI, SQLite, ML pipeline)
- [ ] Write revised implementation plan
- [ ] WAIT FOR USER APPROVAL before Phase 2

### PHASE 2 — Real Data Acquisition
- [ ] Download verified real datasets per category
- [ ] Document exact download method, date, file hash
- [ ] Update DATA_SOURCE.md with all confirmed sources
- [ ] Initial inspection (shape, columns, dtypes, null counts)

### PHASE 3 — Data Cleaning + EDA
- [ ] Category-aware preprocessing pipeline
- [ ] Parse/standardize spec fields per category
- [ ] Handle missing values (documented strategy per field)
- [ ] EDA notebooks per category or combined
- [ ] Feature engineering (value_ratio, performance metrics, etc.)
- [ ] Save cleaned data to data/processed/

### PHASE 4 — ML Recommendation Pipeline
- [ ] Define target variable (per category or unified score)
- [ ] Feature selection per category
- [ ] Baseline model
- [ ] Candidate models (Ridge, Random Forest minimum)
- [ ] Train/test split (80/20, random_state=42)
- [ ] 5-fold cross-validation
- [ ] Model evaluation (MAE, RMSE, R²)
- [ ] Save best model(s) to ml/models/
- [ ] Update MODEL_EVALUATION.md with real results
- [ ] NLP review sentiment (if review data available)

### PHASE 5 — FastAPI Backend
- [ ] Project structure setup
- [ ] SQLite database + category-aware schema
- [ ] Data loading (CSV → DB)
- [ ] Product endpoints (list, search, filter, detail)
- [ ] Recommendation endpoint (ML inference)
- [ ] Compare endpoint
- [ ] Review/NLP endpoint (if reviews available)
- [ ] Input validation, error handling, logging
- [ ] Test backend with curl/httpie

### PHASE 6 — Mobile-First React Frontend
- [ ] Install Node.js (brew install node — confirm before running)
- [ ] Initialize React + Vite project
- [ ] Mobile-first component structure
- [ ] Bottom navigation / mobile nav
- [ ] Category browsing screen
- [ ] Search + filter screen (category-aware filters)
- [ ] Product cards (mobile-optimized)
- [ ] Product detail screen
- [ ] Compare screen (mobile-friendly horizontal scroll)
- [ ] Recommendations screen ("For You")
- [ ] Review insights screen (if NLP component exists)
- [ ] Purple/indigo theme, responsive cards, clean typography

### PHASE 7 — Frontend-Backend Integration
- [ ] Connect React to FastAPI via fetch/axios
- [ ] Handle loading/error/empty states
- [ ] Verify all screens work end-to-end

### PHASE 8 — Testing and Validation
- [ ] Backend unit tests (pytest)
- [ ] API endpoint tests
- [ ] ML pipeline smoke tests
- [ ] End-to-end integration test
- [ ] Fix bugs

### PHASE 9 — Polish + Documentation + Viva Preparation
- [ ] Complete README with screenshots
- [ ] Architecture diagram
- [ ] Setup instructions verified
- [ ] Viva Q&A document
- [ ] Milestone 2 progress report
- [ ] Final report

---

## Architecture Decision Log

| Decision | Choice | Reason |
|---|---|---|
| Backend | Python FastAPI | Same language as ML; clean REST API; auto-docs |
| Database | SQLite (initially) | Zero install; sufficient for prototype; easy schema migration |
| ML framework | scikit-learn | Standard, covers all needed algorithms; viva-explainable |
| Frontend | React + Vite | Mobile-first; component reuse; industry standard |
| UI priority | Mobile-first | Primary users are on phones; desktop secondary |
| NLP | NLTK + VADER | No labeled training data required; interpretable |
| Model | Random Forest (candidate) | Mixed data; feature importance; explainable |

---

## Scope Exclusions

- User authentication / login
- Payment / cart / order
- Real-time live price scraping
- Deep learning / neural networks
- Native mobile app (React Native / Flutter)
- Multiple languages / i18n

---

*Last updated: 2026-08-29 | Phase: 0 done, Phase 1 in progress — awaiting research results*
