import numpy as np

# ---------- PREDICTIVE ----------
def expected_sales(model, X):
    preds = np.expm1(model.predict(X))
    return {
        "expected_daily_sales": float(preds.mean())
    }


# ---------- PRESCRIPTIVE ----------
def simulate_stock_drop(model, X, drop_pct=0.2):
    X_sim = X.copy()
    X_sim["stock_available"] *= (1 - drop_pct)
    X_sim["stock_ratio"] *= (1 - drop_pct)

    base = np.expm1(model.predict(X))
    sim = np.expm1(model.predict(X_sim))

    return {
        "base": float(base.mean()),
        "sim": float(sim.mean()),
        "pct_change": float((sim.mean() / base.mean() - 1) * 100),
    }


def simulate_promo_stock_interaction(model, X, stock_drop_pct=0.2):
    X_base = X.copy()

    def run(promo, stock_drop=False):
        X_tmp = X_base.copy()
        X_tmp["promotion_flag"] = promo
        if stock_drop:
            X_tmp["stock_available"] *= (1 - stock_drop_pct)
            X_tmp["stock_ratio"] *= (1 - stock_drop_pct)
        return np.expm1(model.predict(X_tmp)).mean()

    return {
        "no_promo_normal": run(0, False),
        "promo_normal": run(1, False),
        "no_promo_low_stock": run(0, True),
        "promo_low_stock": run(1, True),
    }


def simulate_promo_off(model, X):
    X_tmp = X.copy()
    X_tmp["promotion_flag"] = 0

    base = np.expm1(model.predict(X))
    sim = np.expm1(model.predict(X_tmp))

    return {
        "base": float(base.mean()),
        "sim": float(sim.mean()),
        "pct_change": float((sim.mean() / base.mean() - 1) * 100),
    }
