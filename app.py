# =========================
# FMCG JARVIS – STREAMLIT APP
# =========================

import streamlit as st
import pickle
import sqlite3
from pathlib import Path

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
    [
        "Total units sold in 2024",
        "Do promotions increase sales?",
        "What if stock drops by 20%?",
        "Should we promote when stock is low?",
        "What affects sales the most?"
    ]
)

# -------------------------
# ANSWER LOGIC
# -------------------------
def total_units_2024(conn):
    q = """
    SELECT SUM(units_sold) 
    FROM sales_fact sf
    JOIN calendar c ON sf.date = c.date
    WHERE c.year = 2024
    """
    return conn.execute(q).fetchone()[0]

def promo_effect(conn):
    q = """
    SELECT promotion_flag, AVG(units_sold)
    FROM sales_fact
    GROUP BY promotion_flag
    """
    rows = conn.execute(q).fetchall()
    no_promo, promo = rows[0][1], rows[1][1]
    uplift = ((promo - no_promo) / no_promo) * 100
    return no_promo, promo, uplift

def simulate_stock_drop(model, X, drop_pct=0.2):
    X_sim = X.copy()
    if "stock_available" not in X_sim.columns:
        return None

    base_pred = model.predict(X_sim)
    X_sim["stock_available"] *= (1 - drop_pct)
    sim_pred = model.predict(X_sim)

    base_units = base_pred.mean()
    sim_units = sim_pred.mean()

    return {
        "avg_base_units": base_units,
        "avg_sim_units": sim_units,
        "avg_delta_units": sim_units - base_units,
        "pct_change": ((sim_units - base_units) / base_units) * 100
    }

def key_drivers(model, X):
    importances = model.feature_importances_
    features = X.columns
    top = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:3]
    return top

# -------------------------
# RESPONSE DISPLAY
# -------------------------
st.divider()

if question == "Total units sold in 2024":
    total = total_units_2024(conn)
    st.metric("📦 Total Units Sold (2024)", f"{int(total):,}")

elif question == "Do promotions increase sales?":
    no_promo, promo, uplift = promo_effect(conn)
    st.write(f"📉 Avg sales without promotion: **{no_promo:.2f} units**")
    st.write(f"📈 Avg sales with promotion: **{promo:.2f} units**")
    st.success(f"🚀 Promotions increase sales by **{uplift:.1f}%**")

elif question == "What if stock drops by 20%?":
    result = simulate_stock_drop(model, X_test)
    if result is None:
        st.error("Stock feature not available in model input.")
    else:
        st.warning(
            f"""
📉 **Stock Reduction Impact Analysis**

• Avg daily sales (baseline): **{result['avg_base_units']:.2f}**  
• Avg daily sales (after 20% drop): **{result['avg_sim_units']:.2f}**  
• Avg unit loss per day: **{abs(result['avg_delta_units']):.2f}**  
• Expected demand change: **{result['pct_change']:.2f}%**
"""
        )

elif question == "Should we promote when stock is low?":
    st.info(
        """
⚠️ **Promotion & Stock Interaction**

Promotions increase demand, but when inventory is constrained,
their effectiveness drops sharply.

**Recommendation:**  
✔️ Fix stock availability before running promotions.
"""
    )

elif question == "What affects sales the most?":
    drivers = key_drivers(model, X_test)
    st.write("📊 **Top Sales Drivers (Model-Based)**")
    for f, v in drivers:
        st.write(f"• **{f}** → {v:.2%}")

# -------------------------
# FOOTER
# -------------------------
st.divider()
st.caption("Built with ❤️ for FMCG analytics | Jarvis v1.0")
