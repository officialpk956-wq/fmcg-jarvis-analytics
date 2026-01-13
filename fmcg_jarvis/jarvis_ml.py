import numpy as np
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MLSimulationError(Exception):
    """Raised when ML simulation fails due to data or model issues."""
    def __init__(self, message: str, missing_features: Optional[List[str]] = None):
        self.message = message
        self.missing_features = missing_features or []
        super().__init__(message)


# ---------- HELPERS ----------
def _safe_divide(numerator: float, denominator: float) -> float:
    """
    Safe division that returns NaN when denominator is zero.
    
    Returns NaN (not 0.0) because percentage change is mathematically
    undefined when baseline is zero. Callers should check with math.isnan()
    and display an appropriate message.
    """
    if denominator == 0:
        logger.warning(f"Division by zero - returning NaN: {numerator}/{denominator}")
        return float('nan')
    return numerator / denominator


def _validate_required_features(X, required: List[str]) -> List[str]:
    """Check if required features exist in DataFrame. Returns list of missing."""
    return [f for f in required if f not in X.columns]


def align_features(model, X):
    """
    Reorder X columns to match model.feature_names_in_ exactly.
    Prevents XGBoost feature name mismatch errors.
    
    Args:
        model: Trained model with optional feature_names_in_ attribute
        X: Input DataFrame
        
    Returns:
        DataFrame with columns reordered to match model expectations
        
    Raises:
        MLSimulationError: If model requires features not present in X,
            with missing_features populated for debugging
    """
    X_aligned = X.copy()
    if hasattr(model, "feature_names_in_"):
        required_features = list(model.feature_names_in_)
        available_features = set(X_aligned.columns)
        missing = [f for f in required_features if f not in available_features]
        
        if missing:
            logger.error(
                f"Model requires {len(required_features)} features, "
                f"but {len(missing)} are missing: {missing}"
            )
            raise MLSimulationError(
                f"Model requires features not present in data: {missing}",
                missing_features=missing
            )
        
        X_aligned = X_aligned[required_features]
    return X_aligned


# ---------- PREDICTIVE ----------
def expected_sales(model, X) -> Dict[str, float]:
    """Predict expected daily sales under current conditions."""
    try:
        X_aligned = align_features(model, X)
        preds = np.expm1(model.predict(X_aligned))
        return {
            "expected_daily_sales": float(preds.mean())
        }
    except MLSimulationError:
        raise
    except Exception as e:
        logger.error(f"expected_sales failed: {type(e).__name__}: {e}")
        raise MLSimulationError(f"Prediction failed: {type(e).__name__}")


# ---------- PRESCRIPTIVE ----------
def simulate_stock_drop(model, X, drop_pct: float = 0.2) -> Dict[str, float]:
    """Simulate impact of stock reduction on sales."""
    # Validate required feature
    missing = _validate_required_features(X, ["stock_available"])
    if missing:
        raise MLSimulationError(
            f"Cannot simulate stock drop: missing columns {missing}",
            missing_features=missing
        )
    
    try:
        X_base = X.copy()
        X_sim = X.copy()
        X_sim["stock_available"] *= (1 - drop_pct)
        if "stock_ratio" in X_sim.columns:
            X_sim["stock_ratio"] *= (1 - drop_pct)

        X_base_aligned = align_features(model, X_base)
        X_sim_aligned = align_features(model, X_sim)

        base = np.expm1(model.predict(X_base_aligned))
        sim = np.expm1(model.predict(X_sim_aligned))
        
        base_mean = float(base.mean())
        sim_mean = float(sim.mean())

        return {
            "base": base_mean,
            "sim": sim_mean,
            "pct_change": (_safe_divide(sim_mean, base_mean) - 1) * 100,
        }
    except MLSimulationError:
        raise
    except Exception as e:
        logger.error(f"simulate_stock_drop failed: {type(e).__name__}: {e}")
        raise MLSimulationError(f"Stock simulation failed: {type(e).__name__}")


def simulate_promo_stock_interaction(model, X, stock_drop_pct: float = 0.2) -> Dict[str, float]:
    """Simulate how promotions interact with stock levels."""
    # Validate required features
    missing = _validate_required_features(X, ["promotion_flag", "stock_available"])
    if missing:
        raise MLSimulationError(
            f"Cannot simulate promo-stock interaction: missing columns {missing}",
            missing_features=missing
        )
    
    try:
        def run(promo, stock_drop=False):
            X_tmp = X.copy()
            X_tmp["promotion_flag"] = promo
            if stock_drop:
                X_tmp["stock_available"] *= (1 - stock_drop_pct)
                if "stock_ratio" in X_tmp.columns:
                    X_tmp["stock_ratio"] *= (1 - stock_drop_pct)
            X_aligned = align_features(model, X_tmp)
            return float(np.expm1(model.predict(X_aligned)).mean())

        no_promo_normal = run(0, False)
        promo_normal = run(1, False)
        no_promo_low_stock = run(0, True)
        promo_low_stock = run(1, True)

        return {
            "base": no_promo_normal,
            "sim": promo_normal,
            "pct_change": (_safe_divide(promo_normal, no_promo_normal) - 1) * 100,
            "no_promo_normal": no_promo_normal,
            "promo_normal": promo_normal,
            "no_promo_low_stock": no_promo_low_stock,
            "promo_low_stock": promo_low_stock,
        }
    except MLSimulationError:
        raise
    except Exception as e:
        logger.error(f"simulate_promo_stock_interaction failed: {type(e).__name__}: {e}")
        raise MLSimulationError(f"Promo-stock simulation failed: {type(e).__name__}")


def simulate_promo_off(model, X) -> Dict[str, float]:
    """Simulate impact of turning off promotions."""
    # Validate required feature
    missing = _validate_required_features(X, ["promotion_flag"])
    if missing:
        raise MLSimulationError(
            f"Cannot simulate promo-off: missing columns {missing}",
            missing_features=missing
        )
    
    try:
        X_base = X.copy()
        X_sim = X.copy()
        X_sim["promotion_flag"] = 0

        X_base_aligned = align_features(model, X_base)
        X_sim_aligned = align_features(model, X_sim)

        base = np.expm1(model.predict(X_base_aligned))
        sim = np.expm1(model.predict(X_sim_aligned))
        
        base_mean = float(base.mean())
        sim_mean = float(sim.mean())

        return {
            "base": base_mean,
            "sim": sim_mean,
            "pct_change": (_safe_divide(sim_mean, base_mean) - 1) * 100,
        }
    except MLSimulationError:
        raise
    except Exception as e:
        logger.error(f"simulate_promo_off failed: {type(e).__name__}: {e}")
        raise MLSimulationError(f"Promo-off simulation failed: {type(e).__name__}")

