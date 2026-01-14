import pandas as pd
import numpy as np
import logging
from typing import Any, Optional
from .jarvis_ml import align_features

from sqlalchemy.engine import Engine

from .jarvis_nlp import detect_intent
from .jarvis_sql import (
    sales_summary,
    top_category,
    region_performance,
    promo_effect,
    safe_get_scalar,
    safe_get_row,
    execute_sql,
    SQLExecutionError,
)
from .jarvis_ml import (
    expected_sales,
    simulate_stock_drop,
    simulate_promo_stock_interaction,
    simulate_promo_off,
    MLSimulationError,
)

# Module logger
logger = logging.getLogger(__name__)


def _format_sql_error_response(exc: SQLExecutionError) -> str:
    """
    Format a SQLExecutionError into a user-friendly response string.
    Includes error ID for support correlation without exposing sensitive data.
    """
    ctx = exc.context
    return (
        f"⚠️ **{ctx.user_message}**\n\n"
        f"📌 {ctx.technical_hint}\n\n"
        f"🔍 Error ID: `{ctx.error_id}` (reference this if contacting support)"
    )


def _format_generic_error_response(operation: str, exc: Exception) -> str:
    """
    Format a generic exception into a user-friendly response.
    Does NOT expose exception details to prevent information leakage.
    """
    # Log the full error for debugging
    logger.error(f"Error during {operation}: {type(exc).__name__}: {exc}")
    
    return (
        f"⚠️ **An error occurred while {operation}.**\n\n"
        f"📌 Recommendation: Please try again. If the problem persists, "
        f"contact support with details about what you were trying to do."
    )


def _format_ml_error_response(exc: MLSimulationError, operation: str) -> str:
    """
    Format an MLSimulationError into a user-friendly response.
    Provides specific guidance for missing features.
    """
    logger.error(f"ML error during {operation}: {exc.message}")
    
    if exc.missing_features:
        feature_list = ", ".join(f"`{f}`" for f in exc.missing_features)
        return (
            f"⚠️ **Cannot run {operation}.**\n\n"
            f"Missing required data columns: {feature_list}\n\n"
            f"📌 Recommendation: Ensure your dataset includes these fields, "
            f"or contact support if the data should be available."
        )
    
    return (
        f"⚠️ **{operation.capitalize()} could not be completed.**\n\n"
        f"📌 Recommendation: Review model inputs and feature availability. "
        f"If the problem persists, contact support."
    )


def _safe_pct_change(sim_mean: float, base_mean: float) -> float:
    """
    Calculate percentage change safely.
    Returns NaN when baseline is zero (percentage change is undefined).
    """
    if base_mean == 0:
        logger.warning(f"Division by zero in pct_change - returning NaN: {sim_mean}/{base_mean}")
        return float('nan')
    return (sim_mean / base_mean - 1) * 100


def _format_pct(value: float, absolute: bool = False) -> str:
    """
    Format a percentage value for display, handling NaN gracefully.
    
    Args:
        value: Percentage value (may be NaN if baseline was zero)
        absolute: If True, display absolute value with no sign
        
    Returns:
        Formatted string like "+5.2%", "5.2%", or "N/A (no baseline)"
    """
    import math
    if math.isnan(value):
        return "N/A (no baseline data)"
    if absolute:
        return f"{abs(value):.1f}%"
    return f"{value:+.1f}%"


def ask_jarvis(
    question: str,
    engine: Engine,
    model: Any,
    X: pd.DataFrame
) -> str:
    """
    Main entry point for Jarvis Q&A.
    Returns a user-friendly answer or a clear error explanation.
    
    Args:
        question: User's question string
        engine: SQLAlchemy Engine for database access (NOT a Connection)
        model: Trained ML model with .predict() method
        X: Feature DataFrame for ML predictions
    
    Returns:
        Formatted markdown response string
    
    Control flow:
    1. Input validation (engine, model, X)
    2. Dropdown questions via exact substring matching (exit immediately on match)
    3. NLP intent-based routing as fallback for free-text questions
    """
    # ---------- INPUT VALIDATION ----------
    if engine is None:
        return (
            "❌ **Database connection unavailable.**\n\n"
            "📌 Recommendation: Check that the database file exists and the app has restarted correctly."
        )
    
    if model is None:
        return (
            "❌ **Prediction model not loaded.**\n\n"
            "📌 Recommendation: Verify that the model file (xgb_model.pkl) exists in the tools folder."
        )
    
    if X is None or (hasattr(X, 'empty') and X.empty):
        return (
            "❌ **Feature data (X_test) not available.**\n\n"
            "📌 Recommendation: Verify that X_test.pkl exists and contains valid data."
        )

    # ==========================================================================
    # PHASE 1: DROPDOWN QUESTIONS (exact text matching)
    # These are predefined questions from the UI dropdown.
    # Each handler returns immediately on match - no fallthrough to NLP.
    # ==========================================================================

    # ---------- DROPDOWN HANDLER FUNCTIONS ----------

    def _handle_total_units_2024() -> str:
        try:
            total = sales_summary(engine, year=2024)
            if total == 0:
                return (
                    "⚠️ No sales data available for 2024.\n\n"
                    "📌 Recommendation: Verify that 2024 data has been loaded."
                )
            return f"📦 Total units sold in 2024: **{int(total):,} units**."
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving sales summary", e)

    def _handle_average_daily_sales_2024() -> str:
        try:
            query = """
                SELECT COALESCE(AVG(units_sold), 0) AS avg_daily_sales
                FROM sales
                WHERE SUBSTR(date, 1, 4) = '2024'
                   OR date LIKE '2024-%'
                   OR date LIKE '2024/%'
            """
            df = execute_sql(query, engine, operation="average daily sales lookup")
            avg_sales = safe_get_scalar(df, "avg_daily_sales", default=0)
            if avg_sales == 0:
                return (
                    "⚠️ No sales data found for 2024.\n\n"
                    "📌 Recommendation: Verify that 2024 data has been loaded."
                )
            return (
                f"📊 Average daily sales in 2024: **{avg_sales:.2f} units per day**.\n\n"
                f"📌 Recommendation: Use this baseline to set realistic daily targets."
            )
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving average daily sales", e)

    def _handle_category_best() -> str:
        try:
            res = top_category(engine)
            if res.get("_empty"):
                return (
                    "⚠️ No category data available for the selected period.\n\n"
                    "📌 Recommendation: Verify that category information exists in the dataset."
                )
            return (
                f"🏆 The best performing category is **{res['category']}**, "
                f"with **{int(res['total_units']):,} units sold**."
            )
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving category performance", e)

    def _handle_region_best() -> str:
        try:
            res = region_performance(engine)
            if res.get("_empty"):
                return (
                    "⚠️ No region data available for the selected period.\n\n"
                    "📌 Recommendation: Verify that region information exists in the dataset."
                )
            return (
                f"🌍 The top performing region is **{res['region']}**, "
                f"contributing **{int(res['total_units']):,} units**."
            )
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving region performance", e)

    def _handle_brand_best() -> str:
        try:
            query = """
                SELECT 
                    COALESCE(brand, 'Unknown') AS brand, 
                    COALESCE(SUM(units_sold), 0) AS total_units
                FROM sales
                WHERE brand IS NOT NULL
                GROUP BY brand
                ORDER BY total_units DESC
                LIMIT 1
            """
            df = execute_sql(query, engine, operation="brand performance lookup")
            res = safe_get_row(df, as_dict=True)
            if res is None:
                return (
                    "⚠️ No brand data available for the selected period.\n\n"
                    "📌 Recommendation: Verify that brand information exists in the dataset."
                )
            return (
                f"🏆 The best performing brand is **{res['brand']}**, "
                f"with **{int(res['total_units']):,} units sold**.\n\n"
                f"📌 Recommendation: Prioritize inventory and promotions for this brand."
            )
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving brand performance", e)

    def _handle_channel_best() -> str:
        try:
            query = """
                SELECT 
                    COALESCE(channel, 'Unknown') AS channel, 
                    COALESCE(SUM(units_sold), 0) AS total_units
                FROM sales
                WHERE channel IS NOT NULL
                GROUP BY channel
                ORDER BY total_units DESC
                LIMIT 1
            """
            df = execute_sql(query, engine, operation="channel performance lookup")
            res = safe_get_row(df, as_dict=True)
            if res is None:
                return (
                    "⚠️ No channel data available for the selected period.\n\n"
                    "📌 Recommendation: Verify that channel information exists in the dataset."
                )
            return (
                f"🏆 The best performing channel is **{res['channel']}**, "
                f"with **{int(res['total_units']):,} units sold**.\n\n"
                f"📌 Recommendation: Focus distribution and marketing efforts on this channel."
            )
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving channel performance", e)

    def _handle_promo_effect() -> str:
        try:
            res = promo_effect(engine)
            if res.get("_empty"):
                return (
                    "⚠️ No promotion data available for the selected period.\n\n"
                    "📌 Recommendation: Verify that the dataset includes promotion information."
                )
            if res["no_promo"] == 0:
                return (
                    "⚠️ No baseline (non-promotional) sales data found.\n\n"
                    "📌 Recommendation: Verify that the dataset includes non-promotional periods."
                )
            uplift = (res["promo"] / res["no_promo"] - 1) * 100
            return (
                f"📊 Promotions increase average sales by **~{uplift:.1f}%** "
                f"compared to non-promotional periods."
            )
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving promotion effect", e)

    def _handle_expected_sales() -> str:
        try:
            res = expected_sales(model, X)
            return (
                f"📈 Expected daily sales are approximately "
                f"**{res['expected_daily_sales']:.2f} units per SKU** "
                f"under current conditions."
            )
        except MLSimulationError as e:
            return _format_ml_error_response(e, "sales prediction")
        except Exception as e:
            return _format_generic_error_response("generating sales prediction", e)

    def _handle_expected_under_promo() -> str:
        try:
            if "promotion_flag" not in X.columns:
                return (
                    "⚠️ **Cannot simulate promotion scenario.**\n\n"
                    "Missing required column: `promotion_flag`\n\n"
                    "📌 Recommendation: Ensure promotion data is included in the dataset."
                )
            X_promo = X.copy()
            X_promo["promotion_flag"] = 1
            preds = np.expm1(model.predict(align_features(model, X_promo)))
            avg_sales = float(preds.mean())
            return (
                f"📈 Expected sales **under promotion**: **{avg_sales:.2f} units per SKU per day**.\n\n"
                f"📌 Recommendation: Plan inventory to meet increased demand during promotions."
            )
        except Exception as e:
            return _format_generic_error_response("simulating promotion scenario", e)

    def _handle_expected_without_promo() -> str:
        try:
            if "promotion_flag" not in X.columns:
                return (
                    "⚠️ **Cannot simulate baseline scenario.**\n\n"
                    "Missing required column: `promotion_flag`\n\n"
                    "📌 Recommendation: Ensure promotion data is included in the dataset."
                )
            X_no_promo = X.copy()
            X_no_promo["promotion_flag"] = 0
            preds = np.expm1(model.predict(align_features(model, X_no_promo)))
            avg_sales = float(preds.mean())
            return (
                f"📉 Expected sales **without promotion**: **{avg_sales:.2f} units per SKU per day**.\n\n"
                f"📌 Recommendation: Use this as a baseline to measure promotional uplift."
            )
        except Exception as e:
            return _format_generic_error_response("simulating baseline scenario", e)

    def _handle_stock_drop() -> str:
        try:
            res = simulate_stock_drop(model, X)
            pct_str = _format_pct(res['pct_change'], absolute=True)
            return (
                f"📉 A 20% reduction in stock is expected to reduce average daily sales "
                f"by **{pct_str}** "
                f"(from {res['base']:.2f} to {res['sim']:.2f} units). "
                f"This indicates strong inventory dependency."
            )
        except MLSimulationError as e:
            return _format_ml_error_response(e, "stock drop simulation")
        except Exception as e:
            return _format_generic_error_response("simulating stock drop", e)

    def _handle_stock_increase() -> str:
        try:
            if "stock_available" not in X.columns:
                return (
                    "⚠️ **Cannot simulate stock increase.**\n\n"
                    "Missing required column: `stock_available`\n\n"
                    "📌 Recommendation: Ensure inventory data is included in the dataset."
                )
            X_sim = X.copy()
            X_sim["stock_available"] *= 1.2
            if "stock_ratio" in X_sim.columns:
                X_sim["stock_ratio"] *= 1.2
            base = np.expm1(model.predict(align_features(model, X)))
            sim = np.expm1(model.predict(align_features(model, X_sim)))
            base_mean = float(base.mean())
            sim_mean = float(sim.mean())
            pct_change = _safe_pct_change(sim_mean, base_mean)
            import math
            pct_str = _format_pct(pct_change)
            if math.isnan(pct_change):
                recommendation = "Unable to calculate impact due to missing baseline data."
            elif pct_change > 0:
                recommendation = "Increase stock levels to capture additional demand."
            else:
                recommendation = "Stock increase may have diminishing returns; focus on other levers."
            return (
                f"📦 A **20% increase in stock** is expected to change average daily sales "
                f"by **{pct_str}** (from {base_mean:.2f} to {sim_mean:.2f} units).\n\n"
                f"📌 Recommendation: {recommendation}"
            )
        except Exception as e:
            return _format_generic_error_response("simulating stock increase", e)

    def _handle_delivery_delay() -> str:
        try:
            X_sim = X.copy()
            delay_col = None
            if "delivery_delay_days" in X_sim.columns:
                delay_col = "delivery_delay_days"
            elif "delivery_days" in X_sim.columns:
                delay_col = "delivery_days"
            
            if delay_col is None:
                return (
                    "⚠️ **Cannot simulate delivery delay.**\n\n"
                    "Missing required column: `delivery_delay_days` or `delivery_days`\n\n"
                    "📌 Recommendation: Ensure delivery metrics are included in the dataset."
                )
            
            X_sim[delay_col] += 2
            base = np.expm1(model.predict(align_features(model, X)))
            sim = np.expm1(model.predict(align_features(model, X_sim)))
            base_mean = float(base.mean())
            sim_mean = float(sim.mean())
            pct_change = _safe_pct_change(sim_mean, base_mean)
            import math
            pct_str = _format_pct(pct_change)
            if math.isnan(pct_change):
                recommendation = "Unable to calculate impact due to missing baseline data."
            elif pct_change < 0:
                recommendation = "Minimize delivery delays to protect sales."
            else:
                recommendation = "Delivery delay has limited impact; focus on other factors."
            return (
                f"🚚 A **2-day increase in delivery delay** is expected to change sales "
                f"by **{pct_str}** (from {base_mean:.2f} to {sim_mean:.2f} units).\n\n"
                f"📌 Recommendation: {recommendation}"
            )
        except Exception as e:
            return _format_generic_error_response("simulating delivery delay", e)

    def _handle_promo_stock_low() -> str:
        try:
            r = simulate_promo_stock_interaction(model, X)
            if r["no_promo_normal"] == 0 or r["no_promo_low_stock"] == 0:
                return (
                    "⚠️ Insufficient baseline data to calculate promotion-stock interaction.\n\n"
                    "📌 Recommendation: Verify that the dataset includes non-promotional periods."
                )
            normal = _safe_pct_change(r["promo_normal"], r["no_promo_normal"])
            low = _safe_pct_change(r["promo_low_stock"], r["no_promo_low_stock"])
            normal_str = _format_pct(normal, absolute=True)
            low_str = _format_pct(low, absolute=True)
            return (
                f"⚠️ Promotions increase demand by **~{normal_str}** under normal stock, "
                f"but only **~{low_str}** when inventory is constrained.\n\n"
                f"📌 Recommendation: fix stock availability before running promotions."
            )
        except MLSimulationError as e:
            return _format_ml_error_response(e, "promotion-stock interaction")
        except Exception as e:
            return _format_generic_error_response("simulating promotion-stock interaction", e)

    def _handle_promo_off() -> str:
        try:
            res = simulate_promo_off(model, X)
            pct_str = _format_pct(res['pct_change'], absolute=True)
            return (
                f"📉 Turning off promotions is expected to reduce sales by "
                f"**{pct_str}**, indicating promotions are a "
                f"significant demand driver."
            )
        except MLSimulationError as e:
            return _format_ml_error_response(e, "promotion-off simulation")
        except Exception as e:
            return _format_generic_error_response("simulating promotion turnoff", e)

    def _handle_feature_importance() -> str:
        try:
            if not hasattr(model, "feature_importances_"):
                return (
                    "⚠️ **Feature importance not available.**\n\n"
                    "📌 Recommendation: The current model does not expose feature importance. "
                    "Contact support if this feature is expected."
                )
            importances = model.feature_importances_
            feature_names = X.columns.tolist()
            sorted_idx = np.argsort(importances)[::-1][:5]
            top_features = [(feature_names[i], importances[i]) for i in sorted_idx]
            lines = [f"**{name}**: {imp:.3f}" for name, imp in top_features]
            return (
                f"📊 **Top factors affecting sales:**\n\n" +
                "\n".join(f"  • {line}" for line in lines) +
                "\n\n📌 Recommendation: Focus optimization efforts on these top drivers."
            )
        except Exception as e:
            return _format_generic_error_response("retrieving feature importance", e)

    # ---------- DROPDOWN DISPATCH TABLE (exact text → handler) ----------
    # Keys are EXACT dropdown text from app.py QUESTIONS list.
    # Each question maps to exactly one handler function.

    DROPDOWN_HANDLERS = {
        # Descriptive Analytics
        "Total units sold in 2024": _handle_total_units_2024,
        "Average daily sales in 2024": _handle_average_daily_sales_2024,
        "Which category sells the most?": _handle_category_best,
        "Which region performs best?": _handle_region_best,
        "Which brand performs best?": _handle_brand_best,
        "Which channel performs best?": _handle_channel_best,
        "Do promotions increase sales?": _handle_promo_effect,
        # Predictive Analytics
        "What are the expected daily sales?": _handle_expected_sales,
        "Expected sales under promotion": _handle_expected_under_promo,
        "Expected sales without promotion": _handle_expected_without_promo,
        # Prescriptive Analytics (What-If)
        "What if stock drops by 20%?": _handle_stock_drop,
        "What if stock increases by 20%?": _handle_stock_increase,
        "What if delivery delay increases?": _handle_delivery_delay,
        "Should we promote when stock is low?": _handle_promo_stock_low,
        "What if we turn off promotions?": _handle_promo_off,
        "What affects sales the most?": _handle_feature_importance,
    }

    # ---------- EXACT DROPDOWN MATCHING ----------
    # Exit immediately on match - no fallthrough to NLP.
    if question in DROPDOWN_HANDLERS:
        return DROPDOWN_HANDLERS[question]()

    # ==========================================================================
    # PHASE 2: NLP INTENT-BASED ROUTING (fallback for free-text questions)
    # Only reached if no dropdown pattern matched above.
    # ==========================================================================

    intent = detect_intent(question)

    # ---------- DESCRIPTIVE ----------
    if intent == "sales_summary":
        try:
            total = sales_summary(engine, year=2024)
            if total == 0:
                return (
                    "⚠️ No sales data available for 2024.\n\n"
                    "📌 Recommendation: Verify that 2024 data has been loaded."
                )
            return f"📦 Total units sold in 2024: **{int(total):,} units**."
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving sales summary", e)

    if intent == "top_category":
        try:
            res = top_category(engine)
            if res.get("_empty"):
                return (
                    "⚠️ No category data available for the selected period.\n\n"
                    "📌 Recommendation: Verify that category information exists in the dataset."
                )
            return (
                f"🏆 The best performing category is **{res['category']}**, "
                f"with **{int(res['total_units']):,} units sold**."
            )
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving category performance", e)

    if intent == "region_performance":
        try:
            res = region_performance(engine)
            if res.get("_empty"):
                return (
                    "⚠️ No region data available for the selected period.\n\n"
                    "📌 Recommendation: Verify that region information exists in the dataset."
                )
            return (
                f"🌍 The top performing region is **{res['region']}**, "
                f"contributing **{int(res['total_units']):,} units**."
            )
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving region performance", e)

    if intent == "promo_effect":
        try:
            res = promo_effect(engine)
            if res.get("_empty"):
                return (
                    "⚠️ No promotion data available for the selected period.\n\n"
                    "📌 Recommendation: Verify that the dataset includes promotion information."
                )
            if res["no_promo"] == 0:
                return (
                    "⚠️ No baseline (non-promotional) sales data found.\n\n"
                    "📌 Recommendation: Verify that the dataset includes non-promotional periods."
                )
            uplift = (res["promo"] / res["no_promo"] - 1) * 100
            return (
                f"📊 Promotions increase average sales by **~{uplift:.1f}%** "
                f"compared to non-promotional periods."
            )
        except SQLExecutionError as e:
            return _format_sql_error_response(e)
        except Exception as e:
            return _format_generic_error_response("retrieving promotion effect", e)

    # ---------- PREDICTIVE ----------
    if intent == "expected_sales":
        try:
            res = expected_sales(model, X)
            return (
                f"📈 Expected daily sales are approximately "
                f"**{res['expected_daily_sales']:.2f} units per SKU** "
                f"under current conditions."
            )
        except MLSimulationError as e:
            return _format_ml_error_response(e, "sales prediction")
        except Exception as e:
            return _format_generic_error_response("generating sales prediction", e)

    # ---------- PRESCRIPTIVE ----------
    if intent == "stock_drop":
        try:
            res = simulate_stock_drop(model, X)
            pct_str = _format_pct(res['pct_change'], absolute=True)
            return (
                f"📉 A 20% reduction in stock is expected to reduce average daily sales "
                f"by **{pct_str}** "
                f"(from {res['base']:.2f} to {res['sim']:.2f} units). "
                f"This indicates strong inventory dependency."
            )
        except MLSimulationError as e:
            return _format_ml_error_response(e, "stock drop simulation")
        except Exception as e:
            return _format_generic_error_response("simulating stock drop", e)

    if intent == "promo_stock_interaction":
        try:
            r = simulate_promo_stock_interaction(model, X)
            if r["no_promo_normal"] == 0 or r["no_promo_low_stock"] == 0:
                return (
                    "⚠️ Insufficient baseline data to calculate promotion-stock interaction.\n\n"
                    "📌 Recommendation: Verify that the dataset includes non-promotional periods."
                )
            normal = _safe_pct_change(r["promo_normal"], r["no_promo_normal"])
            low = _safe_pct_change(r["promo_low_stock"], r["no_promo_low_stock"])
            normal_str = _format_pct(normal, absolute=True)
            low_str = _format_pct(low, absolute=True)
            return (
                f"⚠️ Promotions increase demand by **~{normal_str}** under normal stock, "
                f"but only **~{low_str}** when inventory is constrained.\n\n"
                f"📌 Recommendation: fix stock availability before running promotions."
            )
        except MLSimulationError as e:
            return _format_ml_error_response(e, "promotion-stock interaction")
        except Exception as e:
            return _format_generic_error_response("simulating promotion-stock interaction", e)

    if intent == "promo_off":
        try:
            res = simulate_promo_off(model, X)
            pct_str = _format_pct(res['pct_change'], absolute=True)
            return (
                f"📉 Turning off promotions is expected to reduce sales by "
                f"**{pct_str}**, indicating promotions are a "
                f"significant demand driver."
            )
        except MLSimulationError as e:
            return _format_ml_error_response(e, "promotion-off simulation")
        except Exception as e:
            return _format_generic_error_response("simulating promotion turnoff", e)

    # ---------- FALLBACK: UNKNOWN INTENT ----------
    known_intents = [
        "sales_summary", "top_category", "region_performance", "promo_effect",
        "expected_sales", "stock_drop", "promo_stock_interaction", "promo_off",
        "forecast_sales", "key_drivers"
    ]
    
    if intent == "unknown":
        return (
            f"❓ **I couldn't understand this question.**\n\n"
            f"Your question: *\"{question}\"*\n\n"
            f"📌 Recommendation: Try selecting a predefined question from the dropdown, "
            f"or rephrase using keywords like 'sales', 'promotion', 'stock', or 'region'."
        )
    
    if intent not in known_intents:
        return (
            f"🚧 **This question type is recognized but not yet implemented.**\n\n"
            f"Detected intent: `{intent}`\n\n"
            f"📌 Recommendation: This feature may be under development. "
            f"Please try a different question or contact support."
        )
    
    # Catch-all: should never reach here if all intents are handled
    return (
        f"⚠️ **Unexpected error processing your question.**\n\n"
        f"Detected intent: `{intent}`\n\n"
        f"📌 Recommendation: Please try again or select a different question from the dropdown."
    )
