import numpy as np


# ---------- HELPER ----------
def align_features(model, X):
    """
    Reorder X columns to match model.feature_names_in_ exactly.
    Prevents XGBoost feature name mismatch errors.
    """
    X_aligned = X.copy()
    if hasattr(model, "feature_names_in_"):
        X_aligned = X_aligned[model.feature_names_in_]
    return X_aligned


# ---------- PREDICTIVE ----------
def expected_sales(model, X):
    X_aligned = align_features(model, X)
    preds = np.expm1(model.predict(X_aligned))
    return {
        "expected_daily_sales": float(preds.mean())
    }


# ---------- PRESCRIPTIVE ----------
def simulate_stock_drop(model, X, drop_pct=0.2):
    X_base = X.copy()
    X_sim = X.copy()
    X_sim["stock_available"] *= (1 - drop_pct)
    if "stock_ratio" in X_sim.columns:
        X_sim["stock_ratio"] *= (1 - drop_pct)

    X_base_aligned = align_features(model, X_base)
    X_sim_aligned = align_features(model, X_sim)

    base = np.expm1(model.predict(X_base_aligned))
    sim = np.expm1(model.predict(X_sim_aligned))

    return {
        "base": float(base.mean()),
        "sim": float(sim.mean()),
        "pct_change": float((sim.mean() / base.mean() - 1) * 100),
    }


def simulate_promo_stock_interaction(model, X, stock_drop_pct=0.2):
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
        "pct_change": float((promo_normal / no_promo_normal - 1) * 100),
        "no_promo_normal": no_promo_normal,
        "promo_normal": promo_normal,
        "no_promo_low_stock": no_promo_low_stock,
        "promo_low_stock": promo_low_stock,
    }


def simulate_promo_off(model, X):
    X_base = X.copy()
    X_sim = X.copy()
    X_sim["promotion_flag"] = 0

    X_base_aligned = align_features(model, X_base)
    X_sim_aligned = align_features(model, X_sim)

    base = np.expm1(model.predict(X_base_aligned))
    sim = np.expm1(model.predict(X_sim_aligned))

    return {
        "base": float(base.mean()),
        "sim": float(sim.mean()),
        "pct_change": float((sim.mean() / base.mean() - 1) * 100),
    }

