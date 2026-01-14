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
# PATH SETUP
# -------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "tools" / "xgb_model.pkl"
XTEST_PATH = BASE_DIR / "tools" / "X_test.pkl"
DB_PATH = BASE_DIR / "data" / "fmcg_data.db"

REQUIRED_TABLES = ["sales"]

# -------------------------
# HEALTH CHECK FUNCTIONS
# -------------------------
def check_database_health() -> tuple[bool, Optional[Engine], str]:
    """
    Validate database at startup:
    1. File exists
    2. Engine connects
    3. Required tables exist
    """
    engine: Optional[Engine] = None
    db_path_str = str(DB_PATH)

    # 1. File exists
    if not DB_PATH.exists():
        return False, None, f"Database file not found: `{db_path_str}`"

    # 2. Create engine + test connection
    try:
        engine = create_engine(
            f"sqlite:///{db_path_str}",
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as e:
        if engine is not None:
            engine.dispose()
        return False, None, f"Cannot connect to database: {e}"

    # 3. Verify required tables
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
    if not MODEL_PATH.exists():
        return False, None, f"Model file not found: `{MODEL_PATH.name}`"

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return False, None, f"Cannot load model: {e}"

    if not hasattr(model, "predict"):
        return False, None, "Model is invalid (no predict method)"

    return True, model, "Model is healthy"


def check_features_health():
    if not XTEST_PATH.exists():
        return False, None, f"Features file not found: `{XTEST_PATH.name}`"

    try:
        X = pd.read_pickle(XTEST_PATH)
    except Exception as e:
        return False, None, f"Cannot load features: {e}"

    if X is None or X.empty:
        return False, None, "Features file is empty"

    return True, X, f"Features loaded ({len(X):,} rows, {len(X.columns)} columns)"


# -------------------------
# STARTUP VALIDATION
# -------------------------
def run_startup_checks() -> tuple[Optional[Engine], Any, Any]:
    st.subheader("🔍 System Health Check")

    all_passed = True
    engine, model, X = None, None, None

    db_ok, engine, db_msg = check_database_health()
    if db_ok:
        st.success(f"✅ **Database:** {db_msg}")
    else:
        st.error(f"❌ **Database:** {db_msg}")
        all_passed = False

    model_ok, model, model_msg = check_model_health()
    if model_ok:
        st.success(f"✅ **ML Model:** {model_msg}")
    else:
        st.error(f"❌ **ML Model:** {model_msg}")
        all_passed = False

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


engine, model, X_test = run_startup_checks()

# -------------------------
# QUESTIONS
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

st.subheader("📊 Ask Jarvis a business question")

question = st.selectbox("Choose a question:", QUESTIONS, index=1)

def is_section_header(q: str) -> bool:
    return q.startswith("—") and q.endswith("—")

st.divider()

if is_section_header(question):
    st.info("👆 Please select a question from the dropdown above.")
else:
    with st.spinner("🔍 Analyzing..."):
        answer = ask_jarvis(question, engine, model, X_test)
    st.markdown(answer)

st.divider()

with st.expander("ℹ️ How Jarvis Works"):
    st.markdown("""
**FMCG Jarvis** is an AI-powered assistant designed to support data-driven decision making.

- 📊 Descriptive analytics via SQL
- 📈 Predictive analytics via ML
- 🔮 What-if simulations for decisions
    """)

st.caption("Built with ❤️ for FMCG analytics | Jarvis v1.0")
