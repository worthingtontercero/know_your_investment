from __future__ import annotations

import os
import json
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

import pandas as pd
import yfinance as yf
import numpy as np

# Import Portfolio Lab helper
from portfolio_utils import analyze_portfolio

# Load environment variables from .env (if present)
load_dotenv()

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "change-me"),
)

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

    # Monthly CAPM expectation → log mean (using SPY for beta)
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

# ---------------------- Pro Forma API (NEW) ----------------------

def _latest_annual(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance returns statements with columns as period end dates (Timestamp).
    Keep the most recent 4 annual columns (rightmost) and ensure numeric.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    cols_sorted = sorted(df.columns, key=lambda x: pd.to_datetime(x))
    df = df[cols_sorted]
    df = df.iloc[:, -4:]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def _col_years(df: pd.DataFrame):
    """Return list of calendar years from the statement cols."""
    if df is None or df.empty:
        return []
    years = []
    for c in df.columns:
        try:
            y = pd.to_datetime(c).year
        except Exception:
            y = int(str(c)[:4])
        years.append(y)
    return years

def _avg_ratio(numer_series: pd.Series, denom_series: pd.Series, min_obs=1, default=np.nan):
    """
    Average ratio across overlapping nonzero observations.
    """
    try:
        df = pd.concat([numer_series, denom_series], axis=1)
        df.columns = ["num", "den"]
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df[df["den"] != 0]
        if df.shape[0] >= min_obs:
            r = (df["num"] / df["den"]).mean()
            if np.isfinite(r):
                return float(r)
        return float(default)
    except Exception:
        return float(default)

def _year_labels(last_year: int, h: int):
    return [str(y) for y in range(last_year, last_year + h + 1)]  # includes Y0 last_year

def _is_finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def _safe_num(x, default=None):
    """Return float(x) if finite; else default (or None)."""
    try:
        fx = float(x)
        return fx if np.isfinite(fx) else (default if default is not None else None)
    except Exception:
        return default if default is not None else None

def _sanitize_json(obj):
    """
    Recursively replace NaN/Inf with None so JSON is valid for browsers.
    Cast numpy types to native Python types.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


@app.get("/api/proforma")
def api_proforma():
    """
    Query:
      - ticker (str)  e.g. ?ticker=MSFT
      - h (int, horizon, default 5)
      - wacc (float %, optional override)
      - payout (float %, optional override)
      - debtSalesPct (float %, optional: tie debt to % of sales)
    """
    ticker = (request.args.get("ticker") or "").upper().strip()
    h = int(request.args.get("h", 5))
    wacc_pct = request.args.get("wacc", None)
    payout_pct = request.args.get("payout", None)
    debt_sales_pct = request.args.get("debtSalesPct", None)

    if not ticker:
        return jsonify({"error": "ticker is required"}), 400

    tkr = yf.Ticker(ticker)

    # Financial statements (annual)
    try:
        inc = _latest_annual(tkr.financials)              # Income Statement
        bs  = _latest_annual(tkr.balance_sheet)           # Balance Sheet
        cf  = _latest_annual(tkr.cashflow)                # Cash Flow
    except Exception as e:
        return jsonify({"error": f"Failed to fetch financials for {ticker}: {e}"}), 500

    if inc.empty or bs.empty or cf.empty:
        return jsonify({"error": f"No sufficient annual data for {ticker}."}), 404

    # Column years (last available fiscal year is Y0 base)
    years = _col_years(inc) or _col_years(bs) or _col_years(cf)
    base_year = int(years[-1])  # last reported year
    labels = _year_labels(base_year, h)  # e.g., ["2024","2025",...,"2029"]

    # Picker with flexible keys
    def pick(df: pd.DataFrame, candidates):
        for k in candidates:
            if k in df.index:
                return df.loc[k]
        lk = [i for i in df.index if any(k.lower() in i.lower() for k in candidates)]
        if lk:
            return df.loc[lk[0]]
        return pd.Series([np.nan]*df.shape[1], index=df.columns)

    # Income lines
    sales = pick(inc, ["Total Revenue", "Revenue", "Sales"])
    cogs  = pick(inc, ["Cost Of Revenue", "Cost of Revenue", "Cost Of Goods Sold", "Cost of goods sold"])
    sga   = pick(inc, ["Selling General Administrative", "Selling General & Administrative", "SG&A Expense"])
    rnd   = pick(inc, ["Research Development", "Research & Development", "R&D"])
    depis = pick(inc, ["Depreciation Amortization", "Depreciation & Amortization"])

    # CF lines
    capex = pick(cf, ["Capital Expenditures", "Capital Expenditure"]).abs()
    chg_wc = pick(cf, ["Change In Working Capital", "Change in Working Capital"])

    # Tax lines
    tax_exp = pick(inc, ["Income Tax Expense"])
    pretax  = pick(inc, ["Income Before Tax", "Pretax Income"])

    # Interest / debt
    int_exp = pick(inc, ["Interest Expense", "Interest Expense Non Operating"])
    st_debt = pick(bs, ["Short Long Term Debt", "Short Term Debt", "Short/Current Portion Of Long Term Debt"])
    lt_debt = pick(bs, ["Long Term Debt", "Long Term Debt Noncurrent", "Long-term Debt"])
    total_debt = st_debt.add(lt_debt, fill_value=0)

    # Balance seeds
    cash  = pick(bs, ["Cash And Cash Equivalents", "Cash And Short Term Investments", "Cash"])
    ppe   = pick(bs, ["Property Plant Equipment", "Property, Plant & Equipment Net", "Net PPE"])
    curr_assets = pick(bs, ["Total Current Assets"])
    curr_liab   = pick(bs, ["Total Current Liabilities"])
    equity_line = pick(bs, ["Total Stockholder Equity", "Stockholders Equity"])

    # Shares
    try:
        shares_out = float(tkr.info.get("sharesOutstanding") or tkr.fast_info.get("shares_outstanding") or np.nan)
    except Exception:
        shares_out = np.nan
    if not _is_finite(shares_out) or shares_out <= 0:
        shares_out = 1.0
    # ---------- derive drivers (3Y avg when possible) ----------
    cogs_pct   = _avg_ratio(cogs,  sales, min_obs=1, default=np.nan)
    sga_pct    = _avg_ratio(sga,   sales, min_obs=1, default=np.nan)
    rnd_pct    = _avg_ratio(rnd,   sales, min_obs=1, default=np.nan)
    dep_pct    = _avg_ratio(depis, sales, min_obs=1, default=np.nan)
    capex_pct  = _avg_ratio(capex, sales, min_obs=1, default=np.nan)


    # ΔNWC as % of ΔSales: use last diffs if available
    try:
        dsales = sales.diff().dropna()
        dnwc  = chg_wc.dropna() * -1  # CF sign -> +build
        nwc_pct = _avg_ratio(dnwc.reindex(dsales.index), dsales, min_obs=1, default=np.nan)
    except Exception:
        nwc_pct = np.nan

    # Effective tax rate
    tax_rate = _avg_ratio(tax_exp, pretax.replace(0, np.nan), min_obs=1, default=np.nan)

    # ---- FALLBACKS if any are NaN/Inf ----
    cogs_pct  = cogs_pct  if _is_finite(cogs_pct)  else 0.55
    sga_pct   = sga_pct   if _is_finite(sga_pct)   else 0.25
    rnd_pct   = rnd_pct   if _is_finite(rnd_pct)   else 0.01
    dep_pct   = dep_pct   if _is_finite(dep_pct)   else 0.03
    capex_pct = capex_pct if _is_finite(capex_pct) else 0.04
    nwc_pct   = nwc_pct   if _is_finite(nwc_pct)   else 0.08
    tax_rate  = min(max(tax_rate if _is_finite(tax_rate) else 0.21, 0.0), 0.35)

    # Interest rate proxy
    debt_avg = total_debt.rolling(2).mean().iloc[-1] if total_debt.shape[0] >= 2 else total_debt.iloc[-1]
    interest_rate = _safe_num(abs(int_exp.iloc[-1]) / debt_avg, 0.05) if (pd.notna(debt_avg) and debt_avg > 1) else 0.05

    # Debt as % of sales (optional tie)
    debt_sales_ratio = _safe_num(total_debt.iloc[-1] / sales.iloc[-1], 0.0) if (pd.notna(total_debt.iloc[-1]) and sales.iloc[-1] != 0) else 0.0

    # Payout ratio
    dividends = pick(cf, ["Cash Dividends Paid"])
    ni = pick(inc, ["Net Income", "Net Income Common Stockholders"])
    payout_ratio = _avg_ratio(dividends.abs(), ni.where(ni > 0), min_obs=1, default=0.35)
    payout_ratio = min(max(payout_ratio, 0.0), 0.8)

    # Base seeds (Y0)
    sales0 = float(sales.iloc[-1])
    cash0  = float(cash.iloc[-1]) if pd.notna(cash.iloc[-1]) else 0.0
    ppe0   = float(ppe.iloc[-1]) if pd.notna(ppe.iloc[-1]) else 0.0
    nwc0   = float(curr_assets.iloc[-1] - curr_liab.iloc[-1]) if (pd.notna(curr_assets.iloc[-1]) and pd.notna(curr_liab.iloc[-1])) else 0.0
    eq0    = float(equity_line.iloc[-1]) if pd.notna(equity_line.iloc[-1]) else max(0.0, (sales0*0.2))
    debt0  = float(total_debt.iloc[-1]) if pd.notna(total_debt.iloc[-1]) else 0.0

    # Optional overrides from query
    if wacc_pct is not None:
        try: wacc = float(wacc_pct)/100.0
        except: wacc = 0.08
    else:
        wacc = 0.08

    if payout_pct is not None:
        try: payout_ratio = min(max(float(payout_pct)/100.0, 0.0), 0.9)
        except: pass

    if debt_sales_pct is not None:
        try: debt_sales_ratio = max(float(debt_sales_pct)/100.0, 0.0)
        except: pass

    # Growth: last YoY if available, else 5%
    try:
        recent_growth = float((sales.iloc[-1] - sales.iloc[-2]) / sales.iloc[-2])
        g = recent_growth if np.isfinite(recent_growth) else 0.05
        g = float(np.clip(g, -0.10, 0.25))
    except Exception:
        g = 0.05

    # Shares fallback
    if not np.isfinite(shares_out) or shares_out <= 0:
        shares_out = 1.0  # avoid div by zero

    # ---------- Build forecast ----------
    H = int(h)
    Sales=[sales0]; COGS=[]; SGA=[]; RND=[]; Dep=[]; EBITDA=[]; EBIT=[]
    Interest=[]; EBT=[]; Taxes=[]; NI_=[]; Div_=[]; RE=[]
    CapEx=[]; dNWC=[]
    NWC=[nwc0]; PPE=[ppe0]; Debt=[debt0]; Cash=[cash0]; Equity=[eq0]
    CFO=[]; CFI=[]; CFF=[]; FCFu=[]

    # Y0 IS (display)
    COGS.append(sales0*cogs_pct)
    SGA.append(sales0*sga_pct)
    RND.append(sales0*rnd_pct)
    Dep.append(sales0*dep_pct)
    EBITDA.append(sales0 - COGS[0] - SGA[0] - RND[0])
    EBIT.append(EBITDA[0] - Dep[0])
    Interest.append(Debt[0]*interest_rate)
    EBT.append(EBIT[0] - Interest[0])
    Taxes.append(max(0.0, EBT[0]) * tax_rate)
    NI_.append(EBT[0] - Taxes[0])
    Div_.append(max(0.0, NI_[0]*payout_ratio))
    RE.append(NI_[0] - Div_[0])
    Equity[0] = Equity[0] + RE[0]

    for t in range(1, H+1):
        Sales.append(Sales[t-1]*(1+g))

        c = Sales[t]*cogs_pct
        s = Sales[t]*sga_pct
        r = Sales[t]*rnd_pct
        d = Sales[t]*dep_pct
        COGS.append(c); SGA.append(s); RND.append(r); Dep.append(d)
        ebitda = Sales[t] - c - s - r
        EBITDA.append(ebitda)
        ebit = ebitda - d
        EBIT.append(ebit)

        delta_sales = Sales[t] - Sales[t-1]
        dnw = delta_sales * nwc_pct
        dNWC.append(dnw)
        NWC.append(NWC[t-1] + dnw)

        cap = Sales[t] * capex_pct
        CapEx.append(cap)
        PPE.append(PPE[t-1] + cap - d)

        if debt_sales_ratio > 0:
            Debt.append(Sales[t]*debt_sales_ratio)
        else:
            Debt.append(Debt[t-1])
        interest = Debt[t]*interest_rate
        Interest.append(interest)

        ebt = ebit - interest
        EBT.append(ebt)
        tax_amt = max(0.0, ebt) * tax_rate
        Taxes.append(tax_amt)
        ni = ebt - tax_amt
        NI_.append(ni)

        div = max(0.0, ni*payout_ratio)
        Div_.append(div)
        re = ni - div
        RE.append(re)
        Equity.append(Equity[t-1] + re)

        fcfu = ebit*(1 - tax_rate) + d - cap - dnw
        FCFu.append(fcfu)

        non_cash_assets = NWC[t] + PPE[t]
        Cash.append(Debt[t] + Equity[t] - non_cash_assets)

        CFO.append(ni + d - dnw)
        CFI.append(-cap)
        CFF.append((Debt[t] - Debt[t-1]) - div)

    def _fmt(arr, d=0): 
        out=[]
        for a in arr:
            if a is None or (isinstance(a,float) and not np.isfinite(a)):
                out.append(None)
            else:
                out.append(round(float(a), d))
        return out

    payload = {
        "ticker": ticker,
        "base_year": base_year,
        "years": [str(y) for y in range(base_year, base_year + H + 1)],
        "shares_out": shares_out,
        "drivers": {
            "growth": float(g),
            "cogs_pct": float(cogs_pct),
            "sga_pct": float(sga_pct),
            "rnd_pct": float(rnd_pct),
            "dep_pct": float(dep_pct),
            "capex_pct": float(capex_pct),
            "nwc_pct": float(nwc_pct),
            "tax_rate": float(tax_rate),
            "interest_rate": float(interest_rate),
            "payout_ratio": float(payout_ratio),
            "debt_sales_ratio": float(debt_sales_ratio),
            "wacc": float(wacc),
        },
        "is": {
            "sales": _fmt(Sales, 0),
            "cogs": _fmt(COGS, 0),
            "sga": _fmt(SGA, 0),
            "rnd": _fmt(RND, 0),
            "dep": _fmt(Dep, 0),
            "ebitda": _fmt(EBITDA, 0),
            "ebit": _fmt(EBIT, 0),
            "interest": _fmt(Interest, 0),
            "taxes": _fmt(Taxes, 0),
            "net_income": _fmt(NI_, 0),
            "eps": _fmt([ni/shares_out if shares_out>0 else np.nan for ni in NI_], 4),
        },
        "bs": {
            "cash": _fmt(Cash, 0),
            "nwc": _fmt(NWC, 0),
            "ppe": _fmt(PPE, 0),
            "assets": _fmt([Cash[i]+NWC[i]+PPE[i] for i in range(0, H+1)], 0),
            "debt": _fmt(Debt, 0),
            "equity": _fmt(Equity, 0),
            "liab_equity": _fmt([Debt[i]+Equity[i] for i in range(0, H+1)], 0),
        },
        "cf": {
            "cfo": _fmt([0]+CFO, 0),
            "cfi": _fmt([0]+CFI, 0),
            "cff": _fmt([0]+CFF, 0),
            "delta_cash": _fmt([ ( ([0]+CFO)[i] + ([0]+CFI)[i] + ([0]+CFF)[i] ) for i in range(0, H+1) ], 0),
            "fcf_unlevered": _fmt([0]+FCFu, 0)
        }
    }
    return jsonify(_sanitize_json(payload))


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

# --- Pro Forma page (frontend template) ---
@app.route("/proforma")
def proforma():
    return render_template("proforma.html")

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

@app.route("/health")
def health():
    return "OK", 200

# =============== Main ===============
if __name__ == "__main__":
    app.run(debug=True)
