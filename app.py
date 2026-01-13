# =========================
# FMCG JARVIS – STREAMLIT APP
# =========================

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from fmcg_jarvis.jarvis import ask_jarvis

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="FMCG Jarvis",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 FMCG Jarvis")
st.caption("AI-powered Business Analytics Assistant")

# -------------------------
# PATH SETUP (MATCHES YOUR DIRECTORY)
# -------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "tools" / "xgb_model.pkl"
XTEST_PATH = BASE_DIR / "tools" / "X_test.pkl"
DB_PATH = BASE_DIR / "data" / "fmcg_data.db"

# Required tables for the app to function
REQUIRED_TABLES = ["sales"]


# -------------------------
# HEALTH CHECK FUNCTIONS
# -------------------------
def check_database_health() -> tuple[bool, Optional[Engine], str]:
    """
    Validate database at startup:
    1. File exists
    2. SQLAlchemy engine connects
    3. Required tables exist
    
    Returns:
        (success, engine_or_none, message)
        
    Note: Returns an Engine (not Connection) because:
    - Engine manages connection pool efficiently
    - Connections should be short-lived within `with` blocks
    - Engine survives Streamlit reruns without stale issues
    """
    db_path_str = str(DB_PATH)
    
    # Check 1: File exists
    if not DB_PATH.exists():
        return False, None, f"Database file not found: `{db_path_str}`"
    
    # Check 2: Create SQLAlchemy engine and test connection
    try:
        # SQLite URL format: sqlite:///path/to/file.db
        engine = create_engine(
            f"sqlite:///{db_path_str}",
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,  # Verify connections before use
        )
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as e:
        return False, None, f"Cannot connect to database: {e}"
    
    # Check 3: Required tables exist
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            existing_tables = {row[0] for row in result.fetchall()}
            missing_tables = set(REQUIRED_TABLES) - existing_tables
            
            if missing_tables:
                engine.dispose()
                return False, None, f"Missing required tables: `{', '.join(missing_tables)}`"
    except SQLAlchemyError as e:
        engine.dispose()
        return False, None, f"Cannot verify tables: {e}"
    
    return True, engine, "Database is healthy"


def check_model_health():
    """
    Validate ML model at startup:
    1. Model file exists
    2. Model loads successfully
    3. Model has predict method
    Returns: (success: bool, model_or_none, message: str)
    """
    # Check 1: File exists
    if not MODEL_PATH.exists():
        return False, None, f"Model file not found: `{MODEL_PATH.name}`"
    
    # Check 2: Model loads
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return False, None, f"Cannot load model: {e}"
    
    # Check 3: Model has predict method
    if not hasattr(model, "predict"):
        return False, None, "Model is invalid (no predict method)"
    
    return True, model, "Model is healthy"


def check_features_health():
    """
    Validate feature data at startup:
    1. File exists
    2. Loads as DataFrame
    3. Not empty
    Returns: (success: bool, X_or_none, message: str)
    """
    # Check 1: File exists
    if not XTEST_PATH.exists():
        return False, None, f"Features file not found: `{XTEST_PATH.name}`"
    
    # Check 2: Loads successfully
    try:
        X = pd.read_pickle(XTEST_PATH)
    except Exception as e:
        return False, None, f"Cannot load features: {e}"
    
    # Check 3: Not empty
    if X is None or (hasattr(X, 'empty') and X.empty):
        return False, None, "Features file is empty"
    
    return True, X, f"Features loaded ({len(X):,} rows, {len(X.columns)} columns)"


# -------------------------
# STARTUP VALIDATION
# -------------------------
def run_startup_checks() -> tuple[Optional[Engine], Any, Any]:
    """
    Run all health checks and display status.
    Returns: (engine, model, X) or stops the app on failure.
    """
    st.subheader("🔍 System Health Check")
    
    all_passed = True
    engine, model, X = None, None, None
    
    # Database check
    db_ok, engine, db_msg = check_database_health()
    if db_ok:
        st.success(f"✅ **Database:** {db_msg}")
    else:
        st.error(f"❌ **Database:** {db_msg}")
        all_passed = False
    
    # Model check
    model_ok, model, model_msg = check_model_health()
    if model_ok:
        st.success(f"✅ **ML Model:** {model_msg}")
    else:
        st.error(f"❌ **ML Model:** {model_msg}")
        all_passed = False
    
    # Features check
    features_ok, X, features_msg = check_features_health()
    if features_ok:
        st.success(f"✅ **Features:** {features_msg}")
    else:
        st.error(f"❌ **Features:** {features_msg}")
        all_passed = False
    
    st.divider()
    
    if not all_passed:
        st.error(
            "🚫 **Jarvis cannot start due to missing dependencies.**\n\n"
            "Please verify that all required files are deployed correctly."
        )
        st.stop()
    
    return engine, model, X


# Run health checks at startup
engine, model, X_test = run_startup_checks()


# -------------------------
# UNIFIED QUESTIONS LIST
# -------------------------
QUESTIONS = [
    "— Descriptive Analytics —",
    "Total units sold in 2024",
    "Average daily sales in 2024",
    "Which category sells the most?",
    "Which region performs best?",
    "Which brand performs best?",
    "Which channel performs best?",
    "Do promotions increase sales?",
    "— Predictive Analytics —",
    "What are the expected daily sales?",
    "Expected sales under promotion",
    "Expected sales without promotion",
    "— Prescriptive Analytics (What-If) —",
    "What if stock drops by 20%?",
    "What if stock increases by 20%?",
    "What if delivery delay increases?",
    "Should we promote when stock is low?",
    "What if we turn off promotions?",
    "What affects sales the most?",
]


# -------------------------
# BUSINESS QUESTION SET
# -------------------------
st.subheader("📊 Ask Jarvis a business question")

question = st.selectbox(
    "Choose a question:",
    QUESTIONS,
    index=1  # Default to first actual question
)

# -------------------------
# ANSWER LOGIC
# -------------------------

def is_section_header(q):
    """Check if the selected item is a section header."""
    return q.startswith("—") and q.endswith("—")

# -------------------------
# RESPONSE DISPLAY
# -------------------------
st.divider()

if is_section_header(question):
    st.info("👆 Please select a question from the dropdown above.")
else:
    with st.spinner("🔍 Analyzing..."):
        answer = ask_jarvis(question, engine, model, X_test)
    st.markdown(answer)

# -------------------------
# FOOTER
# -------------------------
st.divider()

# -------------------------
# HOW JARVIS WORKS
# -------------------------
with st.expander("ℹ️ How Jarvis Works"):
    st.markdown("""
**FMCG Jarvis** is an AI-powered assistant designed to support data-driven decision making across your supply chain and sales operations.

### 📊 Descriptive Analytics
Jarvis answers questions about historical performance—total sales, top-performing categories, regions, brands, and channels—by querying your business database in real time.

### 📈 Predictive Analytics
Using a trained machine learning model, Jarvis forecasts expected sales under different conditions, such as with or without promotions, helping you anticipate demand.

### 🔮 What-If Simulations
Jarvis enables scenario planning by simulating the impact of changes—like stock adjustments or delivery delays—on sales performance, so you can make informed decisions before acting.

### 📉 Visual Exploration (Power BI)
For interactive dashboards and deeper visual analysis, Jarvis is complemented by Power BI reports that allow stakeholders to explore trends, drill down by dimension, and share insights across teams.

---
*Jarvis combines structured queries, predictive models, and simulation logic to deliver actionable insights—no guesswork required.*
    """)

st.caption("Built with ❤️ for FMCG analytics | Jarvis v1.0")
