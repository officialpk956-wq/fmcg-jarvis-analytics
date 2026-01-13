import re
from typing import Set


def _has_word(text: str, word: str) -> bool:
    """
    Check if a word exists as a complete word (not substring).
    Uses word boundaries to prevent 'off' matching 'office'.
    """
    pattern = rf'\b{re.escape(word)}\b'
    return bool(re.search(pattern, text))


def _has_any_word(text: str, words: Set[str]) -> bool:
    """Check if any of the words exist as complete words."""
    return any(_has_word(text, w) for w in words)


def _has_all_words(text: str, words: Set[str]) -> bool:
    """Check if all of the words exist as complete words."""
    return all(_has_word(text, w) for w in words)


def detect_intent(text: str) -> str:
    """
    Rule-based intent detection for fallback NLP routing.
    
    Design principles:
    1. Conservative: prefer "unknown" over misclassification
    2. Word boundaries: prevent partial matches (e.g., "off" in "office")
    3. Only implemented intents: no dead code paths
    4. Order matters: more specific patterns first
    
    Implemented intents (downstream handlers exist):
    - sales_summary, top_category, region_performance, promo_effect
    - expected_sales
    - stock_drop, promo_stock_interaction, promo_off
    
    NOT implemented (removed):
    - forecast_sales, key_drivers
    """
    text = text.lower().strip()
    
    # Early exit for empty/trivial input
    if len(text) < 5:
        return "unknown"

    # ==========================================================================
    # PRESCRIPTIVE (most specific patterns first to avoid false positives)
    # ==========================================================================
    
    # "promotion" + "stock" → promo_stock_interaction
    # Must have BOTH words to avoid false positives
    if _has_word(text, "promotion") and _has_word(text, "stock"):
        # Disambiguation: if asking about turning off, that's promo_off
        if _has_any_word(text, {"off", "stop", "disable", "without"}):
            return "promo_off"
        return "promo_stock_interaction"
    
    # "turn off promotions" / "stop promotions" → promo_off
    # Requires explicit promo-related word + explicit off-action word
    if _has_word(text, "promotion") or _has_word(text, "promo"):
        off_actions = {"off", "stop", "disable", "cancel", "end"}
        if _has_any_word(text, off_actions):
            return "promo_off"
    
    # "stock drop" / "stock reduce" → stock_drop
    # REMOVED: dangerous '%' heuristic (matches any percentage question)
    # REMOVED: substring matching for "drop" (could match "dropdown")
    if _has_word(text, "stock"):
        stock_actions = {"drop", "drops", "reduce", "reduction", "decrease", "fall", "decline"}
        if _has_any_word(text, stock_actions):
            return "stock_drop"

    # ==========================================================================
    # PREDICTIVE
    # ==========================================================================
    
    # "expected sales" → expected_sales
    if _has_word(text, "expected") and _has_word(text, "sales"):
        return "expected_sales"
    
    # REMOVED: "forecast_sales" intent - not implemented downstream
    # REMOVED: "predict" / "forecast" patterns - too vague, causes false positives

    # ==========================================================================
    # DESCRIPTIVE (SQL queries)
    # ==========================================================================
    
    # "total sold" / "total units sold" → sales_summary
    if _has_word(text, "total") and _has_any_word(text, {"sold", "sales", "units"}):
        return "sales_summary"
    
    # "category" + ranking word → top_category
    if _has_word(text, "category"):
        ranking_words = {"most", "top", "best", "highest", "leading"}
        if _has_any_word(text, ranking_words):
            return "top_category"
    
    # "region" + ranking word → region_performance
    if _has_word(text, "region"):
        ranking_words = {"best", "top", "highest", "leading", "performs"}
        if _has_any_word(text, ranking_words):
            return "region_performance"
    
    # "promotion" + "increase" / "effect" / "impact" → promo_effect
    # This must come AFTER promo_off and promo_stock_interaction checks
    if _has_word(text, "promotion") or _has_word(text, "promo"):
        effect_words = {"increase", "effect", "impact", "affect", "boost", "help"}
        if _has_any_word(text, effect_words):
            return "promo_effect"

    # ==========================================================================
    # FALLBACK: Conservative default
    # ==========================================================================
    
    # REMOVED: "key_drivers" intent - not implemented downstream
    # REMOVED: vague "affect" / "driver" patterns - too many false positives
    
    return "unknown"
