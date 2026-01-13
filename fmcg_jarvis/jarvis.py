import pandas as pd
import numpy as np

from .jarvis_nlp import detect_intent
from .jarvis_sql import (
    sales_summary,
    top_category,
    region_performance,
    promo_effect,
)
from .jarvis_ml import (
    expected_sales,
    simulate_stock_drop,
    simulate_promo_stock_interaction,
    simulate_promo_off,
)


def ask_jarvis(question, conn, model, X):
    q = question.lower()
    intent = detect_intent(question)

    # ========== NEW DROPDOWN QUESTIONS (substring matching) ==========

    # ---------- DESCRIPTIVE (SQL) ----------

    # 1. "Average daily sales in 2024"
    if "average" in q and "daily" in q and "sales" in q and "2024" in q:
        query = """
            SELECT AVG(units_sold) AS avg_daily_sales
            FROM sales
            WHERE strftime('%Y', date) = '2024'
        """
        avg_sales = pd.read_sql(query, conn)["avg_daily_sales"].iloc[0]
        return (
            f"📊 Average daily sales in 2024: **{avg_sales:.2f} units per day**.\n\n"
            f"📌 Recommendation: Use this baseline to set realistic daily targets."
        )

    # 2. "Which brand performs best?"
    if "brand" in q and ("best" in q or "performs" in q):
        query = """
            SELECT brand, SUM(units_sold) AS total_units
            FROM sales
            GROUP BY brand
            ORDER BY total_units DESC
            LIMIT 1
        """
        res = pd.read_sql(query, conn).iloc[0]
        return (
            f"🏆 The best performing brand is **{res['brand']}**, "
            f"with **{int(res['total_units']):,} units sold**.\n\n"
            f"📌 Recommendation: Prioritize inventory and promotions for this brand."
        )

    # 3. "Which channel performs best?"
    if "channel" in q and ("best" in q or "performs" in q):
        query = """
            SELECT channel, SUM(units_sold) AS total_units
            FROM sales
            GROUP BY channel
            ORDER BY total_units DESC
            LIMIT 1
        """
        res = pd.read_sql(query, conn).iloc[0]
        return (
            f"🏆 The best performing channel is **{res['channel']}**, "
            f"with **{int(res['total_units']):,} units sold**.\n\n"
            f"📌 Recommendation: Focus distribution and marketing efforts on this channel."
        )

    # ---------- PREDICTIVE (ML) ----------

    # 4. "Expected sales under promotion"
    if "expected" in q and "sales" in q and ("under" in q or "with" in q) and "promotion" in q:
        try:
            X_promo = X.copy()
            X_promo["promotion_flag"] = 1
            preds = np.expm1(model.predict(X_promo))
            avg_sales = float(preds.mean())
            return (
                f"📈 Expected sales **under promotion**: **{avg_sales:.2f} units per SKU per day**.\n\n"
                f"📌 Recommendation: Plan inventory to meet increased demand during promotions."
            )
        except Exception:
            return (
                "⚠️ Promotion scenario could not be simulated due to data constraints.\n\n"
                "📌 Recommendation: Review promotion flag availability in the dataset."
            )

    # 5. "Expected sales without promotion"
    if "expected" in q and "sales" in q and "without" in q and "promotion" in q:
        try:
            X_no_promo = X.copy()
            X_no_promo["promotion_flag"] = 0
            preds = np.expm1(model.predict(X_no_promo))
            avg_sales = float(preds.mean())
            return (
                f"📉 Expected sales **without promotion**: **{avg_sales:.2f} units per SKU per day**.\n\n"
                f"📌 Recommendation: Use this as a baseline to measure promotional uplift."
            )
        except Exception:
            return (
                "⚠️ Baseline scenario could not be simulated due to data constraints.\n\n"
                "📌 Recommendation: Review promotion flag availability in the dataset."
            )

    # ---------- PRESCRIPTIVE (WHAT-IF) ----------

    # 6. "What if stock increases by 20%?"
    if "what if" in q and "stock" in q and "increase" in q:
        try:
            X_sim = X.copy()
            X_sim["stock_available"] *= 1.2
            if "stock_ratio" in X_sim.columns:
                X_sim["stock_ratio"] *= 1.2
            base = np.expm1(model.predict(X))
            sim = np.expm1(model.predict(X_sim))
            pct_change = (sim.mean() / base.mean() - 1) * 100
            return (
                f"📦 A **20% increase in stock** is expected to change average daily sales "
                f"by **{pct_change:+.1f}%** (from {base.mean():.2f} to {sim.mean():.2f} units).\n\n"
                f"📌 Recommendation: {'Increase stock levels to capture additional demand.' if pct_change > 0 else 'Stock increase may have diminishing returns; focus on other levers.'}"
            )
        except Exception:
            return (
                "⚠️ Stock increase scenario could not be simulated due to data constraints.\n\n"
                "📌 Recommendation: Review inventory and demand signals."
            )

    # 7. "What if delivery delay increases?"
    if "what if" in q and "delivery" in q and ("delay" in q or "increase" in q):
        try:
            X_sim = X.copy()
            if "delivery_delay_days" in X_sim.columns:
                X_sim["delivery_delay_days"] += 2  # Simulate 2-day increase
            elif "delivery_days" in X_sim.columns:
                X_sim["delivery_days"] += 2
            base = np.expm1(model.predict(X))
            sim = np.expm1(model.predict(X_sim))
            pct_change = (sim.mean() / base.mean() - 1) * 100
            return (
                f"🚚 A **2-day increase in delivery delay** is expected to change sales "
                f"by **{pct_change:+.1f}%** (from {base.mean():.2f} to {sim.mean():.2f} units).\n\n"
                f"📌 Recommendation: {'Minimize delivery delays to protect sales.' if pct_change < 0 else 'Delivery delay has limited impact; focus on other factors.'}"
            )
        except Exception:
            return (
                "⚠️ Delivery delay scenario could not be simulated due to data constraints.\n\n"
                "📌 Recommendation: Review delivery metrics availability in the dataset."
            )

    # ========== END NEW DROPDOWN QUESTIONS ==========

    # ---------- DESCRIPTIVE ----------
    if intent == "sales_summary":
        total = sales_summary(conn, year=2024)
        return f"📦 Total units sold in 2024: **{int(total):,} units**."

    if intent == "top_category":
        res = top_category(conn)
        return (
            f"🏆 The best performing category is **{res['category']}**, "
            f"with **{int(res['total_units']):,} units sold**."
        )

    if intent == "region_performance":
        res = region_performance(conn)
        return (
            f"🌍 The top performing region is **{res['region']}**, "
            f"contributing **{int(res['total_units']):,} units**."
        )

    if intent == "promo_effect":
        res = promo_effect(conn)
        uplift = (res["promo"] / res["no_promo"] - 1) * 100
        return (
            f"📊 Promotions increase average sales by **~{uplift:.1f}%** "
            f"compared to non-promotional periods."
        )

    # ---------- PREDICTIVE ----------
    if intent == "expected_sales":
        try:
            res = expected_sales(model, X)
            return (
                f"📈 Expected daily sales are approximately "
                f"**{res['expected_daily_sales']:.2f} units per SKU** "
                f"under current conditions."
            )
        except Exception:
            return (
                "⚠️ Sales prediction could not be generated due to data constraints.\n\n"
                "📌 Recommendation: Review model inputs and feature availability."
            )

    # ---------- PRESCRIPTIVE ----------
    if intent == "stock_drop":
        try:
            res = simulate_stock_drop(model, X)
            return (
                f"📉 A 20% reduction in stock is expected to reduce average daily sales "
                f"by **{abs(res['pct_change']):.1f}%** "
                f"(from {res['base']:.2f} to {res['sim']:.2f} units). "
                f"This indicates strong inventory dependency."
            )
        except Exception:
            return (
                "⚠️ Stock drop scenario could not be simulated due to data constraints.\n\n"
                "📌 Recommendation: Review inventory and demand signals."
            )

    if intent == "promo_stock_interaction":
        try:
            r = simulate_promo_stock_interaction(model, X)
            normal = (r["promo_normal"] / r["no_promo_normal"] - 1) * 100
            low = (r["promo_low_stock"] / r["no_promo_low_stock"] - 1) * 100

            return (
                f"⚠️ Promotions increase demand by **~{normal:.1f}%** under normal stock, "
                f"but only **~{low:.1f}%** when inventory is constrained.\n\n"
                f"📌 Recommendation: fix stock availability before running promotions."
            )
        except Exception:
            return (
                "⚠️ Promotion-stock interaction could not be simulated due to data constraints.\n\n"
                "📌 Recommendation: Review inventory and promotion data availability."
            )

    if intent == "promo_off":
        try:
            res = simulate_promo_off(model, X)
            return (
                f"📉 Turning off promotions is expected to reduce sales by "
                f"**{abs(res['pct_change']):.1f}%**, indicating promotions are a "
                f"significant demand driver."
            )
        except Exception:
            return (
                "⚠️ Promotion-off scenario could not be simulated due to data constraints.\n\n"
                "📌 Recommendation: Review promotion flag availability in the dataset."
            )

    # ---------- FALLBACK ----------
    return "🤖 I understood the question, but I cannot answer it yet."
