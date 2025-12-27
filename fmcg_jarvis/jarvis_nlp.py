import re

def detect_intent(text: str) -> str:
    text = text.lower()

    # ---------- DESCRIPTIVE (SQL) ----------
    if "total" in text and "sold" in text:
        return "sales_summary"

    if "category" in text and ("most" in text or "top" in text):
        return "top_category"

    if "region" in text and ("best" in text or "top" in text):
        return "region_performance"

    if "promotion" in text and "increase" in text:
        return "promo_effect"

    # ---------- PREDICTIVE (ML) ----------
    if "expected" in text and "sales" in text:
        return "expected_sales"

    if "predict" in text or "forecast" in text:
        return "forecast_sales"

    # ---------- PRESCRIPTIVE (WHAT-IF) ----------
    if "promotion" in text and "stock" in text:
        return "promo_stock_interaction"

    if "stock" in text and any(
        kw in text for kw in ["drop", "drops", "reduce", "reduction", "%"]
    ):
        return "stock_drop"

    if "promotion" in text and ("stop" in text or "off" in text):
        return "promo_off"

    if "affect" in text or "driver" in text:
        return "key_drivers"

    return "unknown"
