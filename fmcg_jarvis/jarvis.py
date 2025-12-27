from fmcg_jarvis.jarvis_nlp import detect_intent
from fmcg_jarvis.jarvis_sql import (
    sales_summary,
    top_category,
    region_performance,
    promo_effect,
)
from fmcg_jarvis.jarvis_ml import (
    expected_sales,
    simulate_stock_drop,
    simulate_promo_stock_interaction,
    simulate_promo_off,
)


def ask_jarvis(question, conn, model, X):
    intent = detect_intent(question)

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
        res = expected_sales(model, X)
        return (
            f"📈 Expected daily sales are approximately "
            f"**{res['expected_daily_sales']:.2f} units per SKU** "
            f"under current conditions."
        )

    # ---------- PRESCRIPTIVE ----------
    if intent == "stock_drop":
        res = simulate_stock_drop(model, X)
        return (
            f"📉 A 20% reduction in stock is expected to reduce average daily sales "
            f"by **{abs(res['pct_change']):.1f}%** "
            f"(from {res['base']:.2f} to {res['sim']:.2f} units). "
            f"This indicates strong inventory dependency."
        )

    if intent == "promo_stock_interaction":
        r = simulate_promo_stock_interaction(model, X)
        normal = (r["promo_normal"] / r["no_promo_normal"] - 1) * 100
        low = (r["promo_low_stock"] / r["no_promo_low_stock"] - 1) * 100

        return (
            f"⚠️ Promotions increase demand by **~{normal:.1f}%** under normal stock, "
            f"but only **~{low:.1f}%** when inventory is constrained.\n\n"
            f"📌 Recommendation: fix stock availability before running promotions."
        )

    if intent == "promo_off":
        res = simulate_promo_off(model, X)
        return (
            f"📉 Turning off promotions is expected to reduce sales by "
            f"**{abs(res['pct_change']):.1f}%**, indicating promotions are a "
            f"significant demand driver."
        )

    # ---------- FALLBACK ----------
    return "🤖 I understood the question, but I cannot answer it yet."
