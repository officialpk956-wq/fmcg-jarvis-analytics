# =========================
# FMCG JARVIS – STREAMLIT APP (UI v2 – PARTIAL PREDEFINED QUESTIONS)
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

# -------------------------
# HEADER
# -------------------------
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
    engine: Optional[Engine] = None

    if not DB_PATH.exists():
        return False, None, "Database file not found"

    try:
        engine = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = [row[0] for row in result.fetchall()]
            missing = set(REQUIRED_TABLES) - set(tables)

            if missing:
                return False, None, f"Missing tables: {', '.join(missing)}"

    except SQLAlchemyError as e:
        return False, None, str(e)

    return True, engine, "Healthy"


def check_model_health():
    if not MODEL_PATH.exists():
        return False, None, "Model file missing"

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return True, model, "Healthy"


def check_features_health():
    if not XTEST_PATH.exists():
        return False, None, "Feature file missing"

    X = pd.read_pickle(XTEST_PATH)
    return True, X, f"{len(X):,} rows, {len(X.columns)} columns"


# -------------------------
# STARTUP CHECKS
# -------------------------
def run_startup_checks():
    db_ok, engine, _ = check_database_health()
    model_ok, model, _ = check_model_health()
    feat_ok, X, _ = check_features_health()

    if not all([db_ok, model_ok, feat_ok]):
        st.error("🚫 System failed health checks")
        st.stop()

    return engine, model, X


engine, model, X_test = run_startup_checks()

# -------------------------
# TOP SUMMARY BAR
# -------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(X_test):,}")
col2.metric("Features", len(X_test.columns))
col3.metric("Model", "Healthy")

st.divider()

# -------------------------
# HERO SECTION – ASK JARVIS
# -------------------------
st.markdown("## 💬 Ask Jarvis")
st.caption(
    "Ask business questions in plain English. "
    "Jarvis analyzes the data and responds with insights and recommendations."
)

question = st.text_input(
    "Ask a business question",
    placeholder="e.g. What happens if sales drop by 20%?"
)

st.markdown("**Try asking:**")
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Average daily sales in 2024"):
        question = "Average daily sales in 2024"

with c2:
    if st.button("Do promotions increase sales?"):
        question = "Do promotions increase sales?"

with c3:
    if st.button("What if stock drops by 20%?"):
        question = "What if stock drops by 20%?"

st.divider()

# -------------------------
# PREDEFINED QUESTIONS (FILTERED)
# -------------------------
PREDEFINED_QUESTIONS = [
    "Total units sold in 2024",
    "Average daily sales in 2024",
    "Do promotions increase sales?",
    "What are the expected daily sales?",
    "Expected sales under promotion",
    "Expected sales without promotion",
    "What if stock drops by 20%?",
    "What if stock increases by 20%?",
    "What if delivery delay increases?",
    "Should we promote when stock is low?",
    "What if we turn off promotions?",
    "What affects sales the most?",
]

selected = st.selectbox(
    "Or choose a predefined question:",
    ["— Select —"] + PREDEFINED_QUESTIONS
)

if selected != "— Select —":
    question = selected

# -------------------------
# ANSWER OUTPUT
# -------------------------
if question:
    with st.spinner("🔍 Analyzing..."):
        response = ask_jarvis(question, engine, model, X_test)

    st.markdown("### 📊 Insight")
    st.info(response)

    st.markdown("### 💡 Recommendation")
    st.success("Use this insight to guide data-driven decisions.")

st.divider()

# -------------------------
# SYSTEM HEALTH (LOW WEIGHT)
# -------------------------
with st.expander("🩺 System Health"):
    st.success("Database: Healthy")
    st.success("ML Model: Loaded")
    st.success("Features: Ready")

st.caption("Built with ❤️ for FMCG analytics | Jarvis v2.0")
