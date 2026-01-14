# FMCG Jarvis - AI-Powered Business Analytics Assistant

A production-ready, end-to-end analytics system for FMCG sales and supply-chain decision support. Combines SQL-based descriptive analytics, machine learning predictions, and what-if simulations into an interactive Streamlit application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [ML Design and Feature Schema](#ml-design-and-feature-schema)
- [Project Structure](#project-structure)
- [Startup Health Checks](#startup-health-checks)
- [How to Run Locally](#how-to-run-locally)
- [Engineering Principles](#engineering-principles)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview

FMCG Jarvis is a business analytics assistant designed for Fast-Moving Consumer Goods (FMCG) decision-makers. It answers business questions across three analytics tiers:

| Tier | Type | Example Questions |
|------|------|-------------------|
| **Descriptive** | SQL-based | Which category sells the most? |
| **Predictive** | ML-based | What are the expected daily sales? |
| **Prescriptive** | Simulation | What if stock drops by 20%? |

The system is built for reliability: all dependencies are validated at startup, errors are handled gracefully with user-friendly messages, and feature parity between training and inference is strictly enforced.

---

## Key Capabilities

### Descriptive Analytics (SQL)

- Total units sold (year-wise filtering)
- Average daily sales
- Top-performing category, brand, region, channel
- Promotion vs non-promotion uplift analysis
- Stock availability impact on sales

### Predictive Analytics (ML)

- Expected daily sales under current conditions
- Expected sales under active promotion
- Expected sales without promotion (baseline)

### Prescriptive Analytics (What-If Simulations)

- **Stock scenarios**: Impact of +/-20% stock level changes
- **Delivery delay**: Effect of increased delivery times on demand
- **Promotion-stock interaction**: Should we promote when stock is low?
- **Promotion removal**: Revenue impact of turning off promotions
- **Feature importance**: Identify key demand drivers

---

## System Architecture

```
+-------------------------------------------------------------+
|                     STREAMLIT UI (app.py)                   |
|                   Question dropdown + results               |
+---------------------------+---------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|                  JARVIS DISPATCHER (jarvis.py)              |
|         Exact-match routing -> NLP fallback -> handlers     |
+--------+------------------+-----------------+---------------+
         |                  |                 |
         v                  v                 v
+---------------+  +----------------+  +-----------------+
|  jarvis_sql   |  |   jarvis_ml    |  |   jarvis_nlp    |
|               |  |                |  |                 |
| - sales_sum   |  | - expected     |  | - detect_intent |
| - top_cat     |  | - stock_sim    |  | - rule-based    |
| - promo_eff   |  | - promo_sim    |  | - conservative  |
+-------+-------+  +--------+-------+  +-----------------+
        |                   |
        v                   v
+---------------+  +----------------+
|   SQLite DB   |  |  XGBoost Model |
|  (fmcg_data)  |  |  (xgb_model)   |
+---------------+  +----------------+
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| app.py | Streamlit UI, health checks, question routing |
| jarvis.py | Central dispatcher, error formatting, handler functions |
| jarvis_sql.py | SQL query execution with structured error handling |
| jarvis_ml.py | ML predictions and simulations with feature alignment |
| jarvis_nlp.py | Rule-based intent detection for free-text questions |

---

## ML Design and Feature Schema

### Model Specification

| Property | Value |
|----------|-------|
| Algorithm | XGBoost Regressor |
| Target | log1p(units_sold) |
| Inverse Transform | expm1(prediction) |
| Training | Offline (notebook) |
| Inference | Loaded via pickle |

### Canonical Feature Schema

The model expects exactly these features in this order:

```python
FEATURES = [
    "product_id",           # Encoded product identifier
    "market_id",            # Encoded market identifier
    "year",                 # Calendar year
    "promotion_flag",       # Binary: 0 = no promo, 1 = promo
    "price",                # Unit price
    "discount_pct",         # Discount percentage (default: 0.0)
    "stock_available",      # Available inventory units
    "delivery_delay_days",  # Days to deliver
    "stock_ratio"           # stock_available / max(stock) per product-market
]
```

### Feature Alignment

All ML simulations use the align_features() function to ensure column order matches model.feature_names_in_. This prevents silent prediction errors from feature misalignment.

```python
def align_features(model, X):
    """Reorder X columns to match model.feature_names_in_ exactly."""
    if hasattr(model, "feature_names_in_"):
        required = list(model.feature_names_in_)
        missing = [f for f in required if f not in X.columns]
        if missing:
            raise MLSimulationError(f"Missing features: {missing}")
        return X[required]
    return X
```

---

## Project Structure

```
fmcg-jarvis/
|-- app.py                      # Streamlit application entry point
|-- fmcg_jarvis/
|   |-- __init__.py
|   |-- jarvis.py               # Main dispatcher + handlers
|   |-- jarvis_sql.py           # SQL query layer
|   |-- jarvis_ml.py            # ML prediction + simulation layer
|   +-- jarvis_nlp.py           # Intent detection (rule-based)
|-- tools/
|   |-- xgb_model.pkl           # Trained XGBoost model
|   +-- X_test.pkl              # Feature sample for simulations
|-- data/
|   +-- fmcg_data.db            # SQLite database
|-- notebooks/
|   +-- main.ipynb              # Data pipeline + model training
|-- requirements.txt
|-- .gitignore
+-- README.md
```

### Database Schema

```
+--------------+     +--------------+     +--------------+
|   products   |     |   markets    |     |   calendar   |
+--------------+     +--------------+     +--------------+
| product_id   |     | market_id    |     | date (PK)    |
| sku          |     | region       |     | year         |
| brand        |     | channel      |     | month        |
| segment      |     +--------------+     | quarter      |
| category     |                          | day_of_week  |
| pack_type    |                          +--------------+
+--------------+
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
                  +-------------------+
                  |       sales       |
                  +-------------------+
                  | sale_id (PK)      |
                  | date (FK)         |
                  | product_id (FK)   |
                  | market_id (FK)    |
                  | price_unit        |
                  | promotion_flag    |
                  | delivery_days     |
                  | stock_available   |
                  | delivered_qty     |
                  | units_sold        |
                  +-------------------+
```

---

## Startup Health Checks

The application validates all dependencies before accepting queries:

```
System Health Check
============================================
[OK] Database: Database is healthy
[OK] ML Model: Model is healthy
[OK] Features: Features loaded (X rows, Y columns)
```

| Check | Validation |
|-------|------------|
| Database | File exists, connection works, required tables present |
| ML Model | File exists, loads successfully, has predict() method |
| Features | File exists, loads as DataFrame, non-empty |

If any check fails, the app displays a clear error message and stops.

---

## How to Run Locally

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/officialpk956-wq/fmcg-jarvis-analytics
cd fmcg-jarvis

# Create virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
sqlalchemy>=2.0.0
xgboost>=3.0.0
scikit-learn>=1.3.0
```

### Run the Application

```bash
streamlit run app.py
```

The app will open at http://localhost:8501

### Verify Installation

```bash
# Test XGBoost installation
python -c "from xgboost import XGBRegressor; print('OK')"

# Test database connection
python -c "import sqlite3; sqlite3.connect('data/fmcg_data.db').execute('SELECT 1')"
```

---

## Engineering Principles

### 1. Strict Feature Parity

The same feature schema is used in training (notebook) and inference (app). The align_features() function enforces column order at runtime.

### 2. Fail-Fast with Clear Errors

All errors are caught and transformed into user-friendly messages. Internal details (SQL queries, stack traces) are logged but never exposed to users.

```python
class SQLExecutionError(Exception):
    """Wraps database errors with safe context for UI display."""
    def __init__(self, context: SQLErrorContext, original: Exception):
        self.context = context  # Safe to show
        self.original = original  # Logged only
```

### 3. Dropdown-First Routing

Predefined questions are matched exactly before NLP fallback. This ensures deterministic behavior for known questions.

```python
if question in DROPDOWN_HANDLERS:
    return DROPDOWN_HANDLERS[question]()  # Direct call, no NLP
```

### 4. Conservative NLP

The intent detector uses explicit keyword matching, not ML. Unknown intents return a helpful "I don't understand" message rather than guessing.

### 5. Connection Lifecycle Management

SQLAlchemy Engine (not Connection) is passed throughout the app. Each query opens a fresh connection from the pool, preventing stale connection issues on Streamlit Cloud.

---

## Future Improvements

| Area | Improvement |
|------|-------------|
| **NLP** | Replace rule-based intent detection with a lightweight classifier or LLM integration |
| **Caching** | Add @st.cache_data for expensive SQL queries |
| **Time Series** | Add Prophet or ARIMA for seasonal demand forecasting |
| **Explainability** | Integrate SHAP values for per-prediction explanations |
| **Testing** | Add pytest suite for SQL and ML simulation functions |
| **CI/CD** | GitHub Actions for automated testing and deployment |

---

## Author

Developed as a portfolio project demonstrating end-to-end ML system design, from data engineering to production deployment.

**Skills demonstrated:**
- SQL analytics and star schema design
- XGBoost regression with proper train/inference separation
- Streamlit application architecture
- Error handling and production hardening
- Clean code organization and documentation

---

## License

This project is available for educational and portfolio purposes.
