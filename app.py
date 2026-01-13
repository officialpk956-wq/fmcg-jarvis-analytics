# =========================
# FMCG JARVIS – STREAMLIT APP
# =========================

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import sqlite3

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
# LOADERS (CACHED)
# -------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("❌ Model file not found in deployment.")
        st.stop()

    if not XTEST_PATH.exists():
        st.error("❌ X_test file not found in deployment.")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_pickle(XTEST_PATH)

    return model, X_test


@st.cache_data
def load_features():
    with open(XTEST_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# -------------------------
# LOAD REQUIRED RESOURCES
# -------------------------
try:
    model, X_test = load_model()
    X_test = load_features()
    conn = load_db()
    st.success("✅ Model & features loaded successfully")
except Exception as e:
    st.error("❌ Failed to load required resources")
    st.exception(e)
    st.stop()

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
        answer = ask_jarvis(question, conn, model, X_test)
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
