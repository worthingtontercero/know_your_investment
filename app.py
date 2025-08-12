from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import numpy as np

app = Flask(__name__)

# =============== Helpers ===============

def _monthly_series(ticker: str, lookback_years: int):
    """
    Return:
      r  : pandas Series of monthly simple returns
      lr : numpy array of monthly log returns
    """
    end = pd.Timestamp.today(tz="UTC").normalize()
    start = end - pd.DateOffset(years=lookback_years, months=1)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False, threads=False)
    if df is None or df.empty:
        return None, None
    monthly = df["Close"].resample("M").last().dropna()
    r = monthly.pct_change().dropna()
    if len(r) < 12:
        return None, None
    lr = np.log1p(r.values)
    return r, lr

def _estimate_beta(stock_r: pd.Series, bench_r: pd.Series):
    """OLS beta using monthly simple returns."""
    df = pd.concat([stock_r, bench_r], axis=1, join="inner").dropna()
    if df.shape[0] < 12:
        return None
    s = df.iloc[:, 0].values
    b = df.iloc[:, 1].values
    var_b = np.var(b, ddof=1)
    if var_b == 0:
        return None
    cov = np.cov(s, b, ddof=1)[0, 1]
    return float(cov / var_b)

def compute_stock_summary(
    ticker: str,
    horizon_months: int,
    amount: float,
    lookback_years: int = 3,
    projection: bool = True,
):
    """
    Build monthly returns from adjusted closes and compute summary stats.
    Optionally project FV using geometric mean and ±1σ variants.
    """
    end = pd.Timestamp.today(tz="UTC").normalize()
    start = end - pd.DateOffset(years=lookback_years, months=1)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False, threads=False)
    if df is None or df.empty:
        return None

    monthly = df["Close"].resample("M").last().dropna()
    rets = monthly.pct_change().dropna()
    if len(rets) < 12:
        return None

    gmean = (1.0 + rets).prod() ** (1.0 / len(rets)) - 1.0   # monthly geometric mean
    amean = float(rets.mean())                                # monthly arithmetic mean
    std = float(rets.std())                                   # monthly std

    result = {
        "ticker": ticker,
        "count": int(len(rets)),
        "gmean": float(gmean),
        "amean": float(amean),
        "std": std,
        "bench": None,
    }

    if projection:
        n = max(0, int(horizon_months))
        base_fv = amount * (1 + gmean) ** n
        low_fv  = amount * (1 + (gmean - std)) ** n
        high_fv = amount * (1 + (gmean + std)) ** n
        result["proj_base"] = {"fv": float(base_fv), "gain": float(base_fv - amount)}
        result["proj_low"]  = {"fv": float(low_fv),  "gain": float(low_fv - amount)}
        result["proj_high"] = {"fv": float(high_fv), "gain": float(high_fv - amount)}

    return result

def compute_stock_forecast(
    ticker: str,
    amount: float,
    horizon_months: int,
    lookback_years: int = 3,
    benchmark: str = "SPY",
    rf_annual: float = 0.03,
    erp_annual: float = 0.05,
    capm_weight: float = 0.5,
    sims: int = 5000,
):
    """
    Monte Carlo forecast with drift blended between historical log-mean and CAPM,
    and volatility = historical monthly log σ.
    """
    r_stock, lr_stock = _monthly_series(ticker, lookback_years)
    if r_stock is None:
        return None

    # Historical stats (log)
    m_log_hist = float(np.mean(lr_stock))
    s_log_hist = float(np.std(lr_stock, ddof=1))

    # Beta vs benchmark
    beta_val = None
    if benchmark:
        r_bench, _ = _monthly_series(benchmark, lookback_years)
        if r_bench is not None:
            beta_val = _estimate_beta(r_stock, r_bench)

    # CAPM monthly arithmetic mean
    rf_m  = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    erp_m = (1.0 + erp_annual) ** (1.0 / 12.0) - 1.0
    e_capm_m = rf_m + (beta_val if beta_val is not None else 1.0) * erp_m
    # convert to log mean
    m_log_capm = float(np.log1p(e_capm_m))

    # Blend drift in log space
    w = max(0.0, min(1.0, capm_weight))
    m_log_blend = w * m_log_capm + (1.0 - w) * m_log_hist
    s_log = s_log_hist

    # Simulate
    n = max(1, int(horizon_months))
    sims = max(100, int(sims))
    rng = np.random.default_rng()
    lr_paths = rng.normal(loc=m_log_blend, scale=s_log, size=(sims, n))
    growth_factors = np.exp(lr_paths.sum(axis=1))
    fv = amount * growth_factors

    p05, p25, p50, p75, p95 = np.percentile(fv, [5, 25, 50, 75, 95])
    annualized_med = (p50 / amount) ** (12.0 / n) - 1.0 if amount > 0 else 0.0

    return {
        "ticker": ticker,
        "lookback": lookback_years,
        "count": int(len(r_stock)),
        "hist": {"m_log": m_log_hist, "s_log": s_log_hist},
        "beta": {"bench": benchmark if benchmark else None, "value": beta_val},
        "capm": {"e_m_arith": float(e_capm_m), "m_log": m_log_capm, "rf_m": rf_m, "erp_m": erp_m},
        "blend": int(round(w * 100)),
        "sims": sims,
        "forecast": {
            "fv_p05": float(p05), "fv_p25": float(p25), "fv_p50": float(p50),
            "fv_p75": float(p75), "fv_p95": float(p95),
            "annualized_med": float(annualized_med),
        },
    }

def compute_dca_forecast(
    ticker: str,
    initial: float,
    monthly: float,
    months: int,
    lookback_years: int = 3,
    benchmark: str = "SPY",
    rf_annual: float = 0.03,
    erp_annual: float = 0.05,
    capm_weight: float = 0.5,
    sims: int = 5000,
    timing: str = "EOM",
    goal: float | None = None,
    target_conf: float = 0.80,
):
    # Historical monthly returns (simple + log) for drift/vol
    r_stock, lr_stock = _monthly_series(ticker, lookback_years)
    if r_stock is None:
        return None

    m_log_hist = float(np.mean(lr_stock))
    s_log_hist = float(np.std(lr_stock, ddof=1))

    beta_val = None
    if benchmark:
        r_bench, _ = _monthly_series(benchmark, lookback_years)
        if r_bench is not None:
            beta_val = _estimate_beta(r_stock, r_bench)

    # Monthly CAPM expectation → log mean
    rf_m  = (1.0 + rf_annual) ** (1.0/12.0) - 1.0
    erp_m = (1.0 + erp_annual) ** (1.0/12.0) - 1.0
    e_capm_m = rf_m + (beta_val if beta_val is not None else 1.0) * erp_m
    m_log_capm = float(np.log1p(e_capm_m))

    # Blend drift; keep historical volatility
    w = float(np.clip(capm_weight, 0.0, 1.0))
    m_log = w * m_log_capm + (1.0 - w) * m_log_hist
    s_log = s_log_hist

    n = max(1, int(months))
    sims = max(100, int(sims))
    rng = np.random.default_rng()
    lr_paths = rng.normal(loc=m_log, scale=s_log, size=(sims, n))
    gf = np.exp(lr_paths)                  # growth factors per month
    total = gf.prod(axis=1)                # growth of $1 over n months
    cumprod_fwd = gf.cumprod(axis=1)

    if timing.upper() == "BOM":
        # contribution grows from month t (including that month)
        denom = np.concatenate([np.ones((sims, 1)), cumprod_fwd[:, :-1]], axis=1)
        S = (total[:, None] / denom).sum(axis=1)
        mode = "BOM"
    else:
        # contribution added after month t growth (end-of-month)
        S = (total[:, None] / cumprod_fwd).sum(axis=1)
        mode = "EOM"

    fv = initial * total + monthly * S
    p05, p25, p50, p75, p95 = np.percentile(fv, [5, 25, 50, 75, 95])

    prob_goal = None
    required_monthly = None
    if goal is not None and goal > 0:
        prob_goal = float(np.mean(fv >= goal))
        # For each path, the monthly needed to hit goal: (goal - initial*total) / S
        req = (goal - initial * total) / S
        req = np.maximum(0.0, req)  # no negative requirement
        tc = float(np.clip(target_conf, 0.0, 0.999))
        required_monthly = float(np.quantile(req, tc))

    invested = float(initial + monthly * n)

    return {
        "ticker": ticker,
        "lookback": lookback_years,
        "count": int(len(r_stock)),
        "timing": mode,
        "beta": {"bench": benchmark if benchmark else None, "value": beta_val},
        "forecast": {
            "fv_p05": float(p05), "fv_p25": float(p25), "fv_p50": float(p50),
            "fv_p75": float(p75), "fv_p95": float(p95),
            "prob_goal": prob_goal,
            "required_monthly": required_monthly,
        },
        "invested": invested,
        "conf": int(round(target_conf * 100)),
    }

# =============== Routes ===============

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/bonds")
def bonds():
    return render_template("bonds.html")

@app.route("/stocks")
def stocks_home():
    # Hub page with buttons to the sub-tools
    return render_template("stocks_home.html")

@app.route("/stocks/correlation", methods=["GET", "POST"])
def stocks_correlation():
    ctx = {"form": {"lookback": 3, "freq": "M"}, "result": None, "error": None}

    if request.method == "POST":
        raw = (request.form.get("tickers") or "").upper()
        tickers = [t for t in {tok.strip(", ").strip() for tok in raw.replace(",", " ").split()} if t]
        lookback = int(request.form.get("lookback") or 3)
        freq = (request.form.get("freq") or "M").upper()

        ctx["form"] = {"tickers": ", ".join(tickers), "lookback": lookback, "freq": freq}

        if len(tickers) < 2:
            ctx["error"] = "Please enter at least two tickers."
            return render_template("stocks_correlation.html", **ctx)

        try:
            res = compute_correlation_matrix(tickers, lookback_years=lookback, freq=freq)
            if res is None:
                ctx["error"] = "Could not compute correlations (insufficient overlapping data)."
            else:
                ctx["result"] = res
        except Exception as e:
            ctx["error"] = f"Error: {e}"

    return render_template("stocks_correlation.html", **ctx)


def compute_correlation_matrix(tickers, lookback_years=3, freq="M"):
    """
    Returns dict with tickers, correlation matrix (list of lists),
    simple stats, and a CSS class heatmap for each cell.
    """
    end = pd.Timestamp.today(tz="UTC").normalize()
    start = end - pd.DateOffset(years=lookback_years, months=1)

    frames = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            continue
        close = df["Close"].copy()
        if freq.upper() == "M":
            close = close.resample("M").last()
        # simple returns
        r = close.pct_change().dropna()
        r.name = t
        frames.append(r)

    if not frames:
        return None

    rets = pd.concat(frames, axis=1, join="inner").dropna()
    if rets.shape[1] < 2 or rets.shape[0] < 6:
        return None

    corr = rets.corr()

    # Stats over off-diagonal values
    n = corr.shape[0]
    mask_offdiag = ~np.eye(n, dtype=bool)
    vals = corr.where(mask_offdiag).stack()
    avg_pair = float(vals.mean()) if len(vals) else None
    min_pair = float(vals.min()) if len(vals) else None
    max_pair = float(vals.max()) if len(vals) else None

    # Discrete class buckets (negative -> red, near 0 -> white, positive -> green)
    def bucket_class(c):
        c = float(max(-1.0, min(1.0, c)))
        if c <= -0.75:    return "cneg3"
        elif c <= -0.40:  return "cneg2"
        elif c <= -0.10:  return "cneg1"
        elif c <  0.10:   return "czero"
        elif c <  0.40:   return "cpos1"
        elif c <  0.75:   return "cpos2"
        else:             return "cpos3"

    matrix_vals = corr.values.tolist()
    class_matrix = [[bucket_class(v) for v in row] for row in corr.values]

    title = f"{'Daily' if freq.upper()=='D' else 'Monthly'} return correlations — {lookback_years}y lookback"

    return {
        "title": title,
        "tickers": list(corr.columns),
        "matrix": matrix_vals,
        "class_matrix": class_matrix,
        "avg_pairwise": avg_pair,
        "min_pair": min_pair,
        "max_pair": max_pair,
        "periods": int(rets.shape[0]),
    }


@app.route("/stocks/trailing", methods=["GET", "POST"])
def stocks_trailing():
    ctx = {"form": {}, "result": None, "error": None}

    if request.method == "POST":
        ticker   = (request.form.get("ticker") or "").upper().strip()
        amount   = float(request.form.get("amount") or 0)
        horizon  = int(float(request.form.get("horizon") or 0))
        bench    = (request.form.get("benchmark") or "").upper().strip() or None
        lookback = int(request.form.get("lookback") or 3)

        ctx["form"] = {
            "ticker": ticker, "amount": amount, "horizon": horizon,
            "lookback": lookback, "benchmark": bench or ""
        }

        try:
            res = compute_stock_summary(ticker, horizon, amount, lookback_years=lookback)
            if res is None:
                ctx["error"] = f"No data found for ticker '{ticker}'."
            else:
                if bench:
                    bres = compute_stock_summary(bench, 12, 0,
                                                 lookback_years=lookback, projection=False)
                    if bres:
                        res["bench"] = {"symbol": bench, "gmean": bres["gmean"]}
                ctx["result"] = res
        except Exception as e:
            ctx["error"] = f"Error: {e}"
    else:
        ctx["form"] = {"amount": 1000, "horizon": 12, "lookback": 3, "benchmark": ""}

    # This still uses your existing template file:
    return render_template("stocks.html", **ctx)

@app.route("/stocks/dca", methods=["GET", "POST"])
def stocks_dca():
    ctx = {"form": {"months": 24, "lookback": 3, "timing": "EOM",
                    "rf": 3.0, "erp": 5.0, "blend": 50, "sims": 5000,
                    "benchmark": "SPY"},
           "result": None, "error": None}

    if request.method == "POST":
        ticker   = (request.form.get("ticker") or "").upper().strip()
        initial  = float(request.form.get("initial") or 0)
        monthly  = float(request.form.get("monthly") or 0)
        months   = int(float(request.form.get("months") or 0))
        lookback = int(request.form.get("lookback") or 3)
        timing   = (request.form.get("timing") or "EOM").upper()
        bench    = (request.form.get("benchmark") or "SPY").upper().strip() or "SPY"
        rf_ann   = float(request.form.get("rf") or 3.0) / 100.0
        erp_ann  = float(request.form.get("erp") or 5.0) / 100.0
        blend    = float(request.form.get("blend") or 50) / 100.0
        sims     = int(float(request.form.get("sims") or 5000))
        goal     = request.form.get("goal")
        conf     = float(request.form.get("conf") or 80) / 100.0
        goal_val = float(goal) if goal not in (None, "",) else None

        ctx["form"] = {"ticker": ticker, "initial": initial, "monthly": monthly,
                       "months": months, "lookback": lookback, "timing": timing,
                       "benchmark": bench, "rf": rf_ann*100, "erp": erp_ann*100,
                       "blend": int(blend*100), "sims": sims, "goal": goal_val, "conf": int(conf*100)}

        try:
            res = compute_dca_forecast(
                ticker=ticker, initial=initial, monthly=monthly, months=months,
                lookback_years=lookback, benchmark=bench,
                rf_annual=rf_ann, erp_annual=erp_ann, capm_weight=blend,
                sims=sims, timing=timing, goal=goal_val, target_conf=conf
            )
            if res is None:
                ctx["error"] = f"Could not run forecast for '{ticker}'."
            else:
                ctx["result"] = res
        except Exception as e:
            ctx["error"] = f"Error: {e}"

    return render_template("stocks_dca.html", **ctx)


@app.route("/stocks/forecast", methods=["GET", "POST"])
def stocks_forecast():
    ctx = {"form": {}, "result": None, "error": None}

    if request.method == "POST":
        ticker   = (request.form.get("ticker") or "").upper().strip()
        amount   = float(request.form.get("amount") or 0)
        horizon  = int(float(request.form.get("horizon") or 0))
        lookback = int(request.form.get("lookback") or 3)
        bench    = (request.form.get("benchmark") or "SPY").upper().strip() or "SPY"
        rf_ann   = float(request.form.get("rf") or 3.0)
        erp_ann  = float(request.form.get("erp") or 5.0)
        blend    = int(float(request.form.get("blend") or 50))
        sims     = int(float(request.form.get("sims") or 5000))

        ctx["form"] = {
            "ticker": ticker, "amount": amount, "horizon": horizon, "lookback": lookback,
            "benchmark": bench, "rf": rf_ann, "erp": erp_ann, "blend": blend, "sims": sims
        }

        try:
            res = compute_stock_forecast(
                ticker=ticker, amount=amount, horizon_months=horizon,
                lookback_years=lookback, benchmark=bench,
                rf_annual=rf_ann/100.0, erp_annual=erp_ann/100.0,
                capm_weight=blend/100.0, sims=sims
            )
            if res is None:
                ctx["error"] = f"Could not build forecast for '{ticker}'."
            else:
                ctx["result"] = res
        except Exception as e:
            ctx["error"] = f"Error: {e}"
    else:
        ctx["form"] = {"amount": 1000, "horizon": 12, "lookback": 3,
                       "benchmark": "SPY", "rf": 3.0, "erp": 5.0, "blend": 50, "sims": 5000}

    return render_template("stocks_forecast.html", **ctx)

# =============== Main ===============

if __name__ == "__main__":
    app.run(debug=True)
