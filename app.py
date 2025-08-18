from __future__ import annotations
from flask import render_template  # ensure this import exists
from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
import numpy as np
import json, os
# from dataclasses import dataclass  # not used; safe to remove

# Import Portfolio Lab helper
from portfolio_utils import analyze_portfolio

app = Flask(__name__)

# --- Sector mapping to SPDR ETFs (fallback = SPY) ---
SECTOR_ETFS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# =============== Helpers (your existing ones) ===============

def _load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _save_json(path: str, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except Exception:
        pass

def get_sector_for_ticker(ticker: str) -> str | None:
    """
    Try cached sector; else ask Yahoo. Cache result to avoid repeated calls.
    """
    cache_path = os.path.join(DATA_DIR, "sector_cache.json")
    cache = _load_json(cache_path) or {}
    t = ticker.upper().strip()
    if t in cache and cache[t]:
        return cache[t]

    try:
        info = yf.Ticker(t).get_info()  # may be slow the first time
        sector = info.get("sector") or info.get("Sector")
        if sector:
            cache[t] = sector
            _save_json(cache_path, cache)
            return sector
    except Exception:
        pass
    return None

def sector_proxy_symbol(sector: str | None) -> str:
    if not sector:
        return "SPY"
    return SECTOR_ETFS.get(sector, "SPY")

def compute_sector_sentiment(etf: str, lookback_years: int = 10, window_months: int = 3):
    """
    Very simple 'sentiment' proxy: momentum Z of the sector ETF.
    """
    end = pd.Timestamp.today(tz="UTC").normalize()
    start = end - pd.DateOffset(years=lookback_years, months=1)

    df = yf.download(etf, start=start, end=end, auto_adjust=True,
                     progress=False, threads=False)
    if df is None or df.empty:
        return {"z": 0.0, "quality_ok": False, "count_months": 0}

    # Ensure 'Close' is a 1-D Series
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze("columns")

    m = close.resample("M").last().dropna()
    r = m.pct_change().dropna()
    if isinstance(r, pd.DataFrame):
        r = r.squeeze("columns")

    if len(r) < (window_months + 24):
        return {"z": 0.0, "quality_ok": False, "count_months": int(len(r))}

    lr = np.log1p(r.to_numpy().reshape(-1))  # 1-D
    w = int(max(1, window_months))

    roll = pd.Series(lr).rolling(window=w).sum().dropna().to_numpy().reshape(-1)
    if len(roll) < 12:
        return {"z": 0.0, "quality_ok": False, "count_months": int(len(r))}

    latest = float(roll[-1])
    base = roll[:-1]
    mu, sd = float(np.mean(base)), float(np.std(base, ddof=1))
    z = 0.0 if sd <= 1e-12 else float(np.clip((latest - mu) / sd, -3.0, 3.0))

    return {"z": z, "quality_ok": True, "count_months": int(len(r))}

def apply_sector_sentiment(m_log_base: float, s_log_hist: float, z: float, blend: float):
    """
    Simpler adjustment: only 'level' via sector Z; tiny vol tilt.
    We also return the raw log-μ shift so we can apply a time decay later.
    """
    b = float(np.clip(blend, 0.0, 1.0))

    # Magnitude knobs (conservative defaults)
    gamma_mu = 0.15 * s_log_hist   # how strongly Z moves μ (in log space)
    gamma_sig = 0.08               # how strongly Z tilts σ (bullish -> slightly lower σ)

    adj_mu_log = b * gamma_mu * z
    cap = 0.5 * s_log_hist
    adj_mu_log = float(np.clip(adj_mu_log, -cap, cap))  # safety cap

    s_mult = 1.0 + b * (-gamma_sig * z)                 # z>0 => reduce σ a touch
    s_log_final = float(np.clip(s_log_hist * s_mult, 0.7 * s_log_hist, 1.5 * s_log_hist))

    # 'final' if you applied the full shift every month (we'll decay later)
    m_log_final = float(m_log_base + adj_mu_log)

    # Report arithmetic μ uplift (annualized) just for the UI
    mu_m0 = np.expm1(m_log_base)
    mu_m1 = np.expm1(m_log_final)
    uplift_annual = (1.0 + mu_m1) ** 12 - (1.0 + mu_m0) ** 12

    effects = {
        "mu_monthly_add": float(mu_m1 - mu_m0),
        "annual_uplift_pct": float(uplift_annual * 100.0),
        "adj_mu_log": adj_mu_log,  # for decayed use later
    }
    return m_log_final, s_log_final, effects

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

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze("columns")

    monthly = close.resample("M").last().dropna()
    r = monthly.pct_change().dropna()
    if isinstance(r, pd.DataFrame):
        r = r.squeeze("columns")

    if len(r) < 12:
        return None, None

    lr = np.log1p(r.to_numpy().reshape(-1))  # 1-D
    return r, lr

def _estimate_beta(stock_r: pd.Series, bench_r: pd.Series):
    """OLS beta using monthly simple returns."""
    df = pd.concat([stock_r, bench_r], axis=1, join="inner").dropna()
    if df.shape[0] < 12:
        return None
    s = np.asarray(df.iloc[:, 0], dtype=float).reshape(-1)
    b = np.asarray(df.iloc[:, 1], dtype=float).reshape(-1)
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

    # CAPM monthly arithmetic mean -> log mean
    rf_m  = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    erp_m = (1.0 + erp_annual) ** (1.0 / 12.0) - 1.0
    e_capm_m = rf_m + (beta_val if beta_val is not None else 1.0) * erp_m
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
        req = (goal - initial * total) / S
        req = np.maximum(0.0, req)
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
@app.route('/options-derivatives')
def options_derivatives_home():
    return render_template('options_derivatives.html')

@app.route('/options/black-scholes')
def options_black_scholes():
    return render_template('options_black_scholes.html')

@app.route('/options/payoff')
def options_payoff():
    return render_template('options_payoff.html')

@app.route('/options/iv')
def options_iv():
    return render_template('options_iv.html')

@app.route('/options/futures')
def options_futures():
    return render_template('futures_pricing.html')
    

@app.route('/options/greeks')
def options_greeks():
    return render_template('options_greeks.html')


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/bonds")
def bonds():
    return render_template("bonds.html")

@app.route("/stocks")
def stocks_home():
    return render_template("stocks_home.html")

@app.route("/stocks/old")
def stocks_home_old():
    return render_template("stocks.html")

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

    # Class buckets for heatmap
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

# NEW: sentiment-aware MC API (unchanged)
@app.route("/api/sentiment_forecast", methods=["POST"])
def api_sentiment_forecast():
    try:
        data = request.get_json(force=True) or {}
        ticker   = (data.get("ticker") or "").upper().strip()
        amount   = float(data.get("amount") or 0)
        monthly  = float(data.get("monthly") or 0)
        horizon  = int(data.get("horizon") or 12)
        lookback = int(data.get("lookback") or 3)
        simsN    = max(500, int(data.get("sims") or 10000))
        rf_ann   = float(data.get("rf") or 3.0) / 100.0
        erp_ann  = float(data.get("erp") or 5.0) / 100.0
        capm_w   = float(data.get("capm_weight") or 50.0) / 100.0

        # Sector sentiment knobs
        sent_b   = float(data.get("sentiment_blend") or 30.0) / 100.0
        sent_win = int(data.get("sent_window") or 3)
        manual   = bool(data.get("use_manual") or data.get("manual") or False)
        manZ     = float(data.get("manZ") or 0.0)

        if not ticker or amount < 0 or horizon <= 0:
            return jsonify({"error": "Ticker, amount ≥ 0, and horizon > 0 required."}), 400

        r_stock, lr_stock = _monthly_series(ticker, lookback)
        if r_stock is None:
            return jsonify({"error": f"No data for '{ticker}'."}), 400

        m_log_hist = float(np.mean(lr_stock))
        s_log_hist = float(np.std(lr_stock, ddof=1))

        # CAPM monthly expectation (arith) → log mean (using SPY for beta)
        r_bench, _ = _monthly_series("SPY", lookback)
        beta_val = _estimate_beta(r_stock, r_bench) if r_bench is not None else None
        rf_m  = (1.0 + rf_ann) ** (1.0/12.0) - 1.0
        erp_m = (1.0 + erp_ann) ** (1.0/12.0) - 1.0
        e_capm_m = rf_m + (beta_val if beta_val is not None else 1.0) * erp_m
        m_log_capm = float(np.log1p(e_capm_m))

        # Base drift blend
        w = float(np.clip(capm_w, 0.0, 1.0))
        m_log_base = float((1.0 - w) * m_log_hist + w * m_log_capm)

        # --- Sector detection & sentiment ---
        sector = get_sector_for_ticker(ticker)
        etf = sector_proxy_symbol(sector)

        if manual:
            z = float(np.clip(manZ, -3.0, 3.0))
            quality_ok = True
        else:
            ss = compute_sector_sentiment(etf, lookback_years=10, window_months=sent_win)
            z = float(ss["z"])
            quality_ok = bool(ss["quality_ok"])

        # If we don't trust the signal, kill the weight
        sent_blend_eff = sent_b if quality_ok else 0.0

        # Sector adjustment (we'll apply time-decay when simulating)
        _, s_log_final, effects = apply_sector_sentiment(
            m_log_base, s_log_hist, z, sent_blend_eff
        )
        adj_mu_log = float(effects["adj_mu_log"])

        # --- Simulation with half-life decay of the sector effect ---
        n = max(1, int(horizon))
        simsN = max(500, int(simsN))
        half_life = 6.0  # months; tweakable
        decay = np.power(0.5, np.arange(n) / half_life)
        means = m_log_base + adj_mu_log * decay

        rng = np.random.default_rng()
        lr_paths = rng.normal(loc=means, scale=s_log_final, size=(simsN, n))
        gf = np.exp(lr_paths)
        total = gf.prod(axis=1)

        if monthly > 0:
            cumprod_fwd = gf.cumprod(axis=1)
            S = (total[:, None] / cumprod_fwd).sum(axis=1)  # EOM contributions
            fv = amount * total + monthly * S
        else:
            fv = amount * total

        p05, p10, p25, p50, p75, p90, p95 = np.percentile(fv, [5, 10, 25, 50, 75, 90, 95])
        base_invested = float(amount + monthly * n)
        base_div = amount if amount > 0 else max(1e-9, base_invested)
        annualized_med = (p50 / base_div) ** (12.0 / n) - 1.0

        lo, hi = float(np.min(fv)), float(np.max(fv))
        bins = 60 if simsN >= 100_000 else 40
        if hi <= lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, bins + 1)
        hist, _ = np.histogram(fv, bins=edges)
        mids = 0.5 * (edges[:-1] + edges[1:])
        labels = [f"${int(x):,}" for x in mids]

        return jsonify({
            "ticker": ticker,
            "forecast": {
                "fv_p05": float(p05), "fv_p10": float(p10), "fv_p25": float(p25),
                "fv_p50": float(p50), "fv_p75": float(p75), "fv_p90": float(p90),
                "fv_p95": float(p95), "annualized_med": float(annualized_med)
            },
            "chart": {"labels": labels, "values": [int(v) for v in hist.tolist()]},
            "sector": {
                "name": sector or "Market",
                "etf": etf,
                "z": float(z),
                "window_months": int(sent_win),
                "weight": float(sent_blend_eff),
                "quality_ok": bool(quality_ok),
            },
            "effects": effects,
        })
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

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

    return render_template("stocks.html", **ctx)

@app.route("/stocks/sentiment")
def stocks_sentiment():
    return render_template("stocks_sentiment.html")

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

# ---------- Portfolio Lab ----------
@app.route("/stocks/portfolio")
def stocks_portfolio():
    return render_template("stocks_portfolio.html")

@app.route("/api/portfolio/analyze", methods=["POST"])
def api_portfolio_analyze():
    try:
        data = request.get_json(force=True)
        result = analyze_portfolio(
            tickers=data.get("tickers",""),
            weights_pct=data.get("weights",""),
            start=data.get("start","2019-01-01"),
            rf_annual=float(data.get("rf", 0.0)),
            market_symbol=data.get("market","^GSPC"),
            sims=int(data.get("sims", 2000)),
            horizon_days=int(data.get("horizon_days", 252)),
            conf=float(data.get("conf", 0.95)),
            drift_mode=data.get("drift","capm"),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
# ---------- END Portfolio Lab ----------

# =============== Main ===============
if __name__ == "__main__":
     
    app.run(debug=True)
