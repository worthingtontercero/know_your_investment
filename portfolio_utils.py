from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from typing import List, Tuple

# -------- Data ----------
def fetch_prices(tickers: List[str], start: str, end: str | None = None) -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        raise ValueError("Please provide at least one ticker.")

    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if data is None or data.empty:
        raise ValueError("No price data returned. Check tickers or start date.")

    # Select adjusted close
    adj = data["Adj Close"]
    # Robust column naming for single vs multi ticker
    if isinstance(adj, pd.Series):
        adj = adj.to_frame(name=tickers[0])   # single ticker: name the column as the ticker
    else:
        # multi: columns should already be tickers; ensure no 'Adj Close' residue
        adj.columns.name = None

    adj = adj.sort_index().dropna(how="all")
    adj = adj.ffill().dropna(how="any")  # require all series to be aligned
    if adj.empty:
        raise ValueError("Adjusted price matrix is empty after cleaning.")
    return adj

def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # log-returns (stable for aggregation)
    return np.log(prices / prices.shift(1)).dropna()

# -------- CAPM ----------
def capm_expected_returns(asset_rets: pd.DataFrame, market_rets: pd.Series, rf_annual: float):
    rf = rf_annual / 100.0
    td = 252
    mu_mkt_daily = market_rets.mean()
    mu_mkt_annual = (1 + mu_mkt_daily) ** td - 1

    betas = []
    for c in asset_rets.columns:
        cov = np.cov(asset_rets[c], market_rets)[0, 1]
        var_m = np.var(market_rets)
        betas.append(0.0 if var_m == 0 else cov / var_m)
    betas = pd.Series(betas, index=asset_rets.columns)

    exp = rf + betas * (mu_mkt_annual - rf)
    return exp, betas

# -------- Portfolio metrics ----------
def portfolio_annualized_stats(weights: np.ndarray, mu_daily: np.ndarray, cov_daily: np.ndarray, rf_annual: float):
    w = np.array(weights, dtype=float).reshape(-1)
    td = 252
    exp_daily = float(mu_daily @ w)
    exp_annual = (1 + exp_daily) ** td - 1
    vol_annual = float(np.sqrt(w @ (cov_daily * td) @ w))
    rf = rf_annual / 100.0
    sharpe = 0.0 if vol_annual == 0 else (exp_annual - rf) / vol_annual
    return exp_annual, vol_annual, sharpe

def portfolio_beta(weights: np.ndarray, betas: pd.Series) -> float:
    return float(np.dot(weights, betas.values)) if len(betas) else float("nan")

def max_drawdown(values: np.ndarray) -> float:
    peaks = np.maximum.accumulate(values)
    drawdowns = (values - peaks) / peaks
    return float(drawdowns.min())

def var_cvar(returns_daily: pd.Series, conf: float = 0.95) -> Tuple[float, float]:
    m = returns_daily.mean()
    s = returns_daily.std()
    from scipy.stats import norm
    z = norm.ppf(1 - conf)
    var_1d = -(m + z*s)  # positive loss
    cvar_1d = -(m + (norm.pdf(z)/(1-conf))*s)
    return float(var_1d * np.sqrt(252)), float(cvar_1d * np.sqrt(252))

def diversification_score(corr: pd.DataFrame) -> float:
    n = corr.shape[0]
    if n <= 1:
        return 50.0
    mask = ~np.eye(n, dtype=bool)
    avg_off = float(corr.values[mask].mean())
    return float(max(0.0, min(100.0, 100*(1-avg_off)/2)))

# -------- Optimization ----------
def _clean_weights(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)

def optimize_max_sharpe(mu_daily: pd.Series, cov_daily: pd.DataFrame, rf_annual: float) -> np.ndarray:
    n = len(mu_daily.values)
    def neg_sharpe(x):
        w = _clean_weights(np.array(x))
        exp_a, vol_a, sh = portfolio_annualized_stats(w, mu_daily.values, cov_daily.values, rf_annual)
        return -sh
    x0 = np.ones(n)/n
    cons = ({'type':'eq', 'fun': lambda x: np.sum(np.clip(x, 0, None)) - 1.0},)
    bnds = [(0.0, 1.0)] * n
    res = minimize(neg_sharpe, x0, bounds=bnds, constraints=cons, method='SLSQP', options={'maxiter': 500})
    return _clean_weights(res.x)

def efficient_frontier(mu_daily: pd.Series, cov_daily: pd.DataFrame, points: int = 30, rf_annual: float = 0.0):
    n = len(mu_daily.values)
    td = 252
    mu_ann_vec = (1 + mu_daily.values) ** td - 1
    t_min, t_max = float(mu_ann_vec.min()), float(mu_ann_vec.max())
    targets = np.linspace(t_min, t_max, points)

    vols, rets = [], []
    for tr in targets:
        def obj(x):
            w = _clean_weights(np.array(x))
            return float(w @ (cov_daily.values * td) @ w)
        cons = (
            {'type':'eq', 'fun': lambda x: np.sum(np.clip(x,0,None)) - 1.0},
            {'type':'eq', 'fun': lambda x: (1 + (mu_daily.values @ _clean_weights(np.array(x))))**td - 1 - tr}
        )
        bnds = [(0.0,1.0)]*n
        x0 = np.ones(n)/n
        try:
            res = minimize(obj, x0, bounds=bnds, constraints=cons, method='SLSQP', options={'maxiter':500})
            w = _clean_weights(res.x)
            exp_a, vol_a, _ = portfolio_annualized_stats(w, mu_daily.values, cov_daily.values, rf_annual)
            vols.append(vol_a); rets.append(exp_a)
        except Exception:
            pass
    return vols, rets

# -------- Monte Carlo ----------
def monte_carlo_portfolio(prices: pd.DataFrame, weights: np.ndarray, horizon_days=252, sims=2000,
                          drift_mode="capm", rf_annual=0.0, market_col=None):
    rets = prices_to_returns(prices)
    td = 252

    # choose drift/cov universe
    if drift_mode == "capm" and market_col and market_col in rets.columns:
        asset_rets = rets.drop(columns=[market_col])
        market_rets = rets[market_col]
        exp_capm_ann, _betas = capm_expected_returns(asset_rets, market_rets, rf_annual)
        mu = (1 + exp_capm_ann.values) ** (1/td) - 1
        Sigma = asset_rets.cov().values
        n = asset_rets.shape[1]
    else:
        mu = rets.mean().values
        Sigma = rets.cov().values
        n = rets.shape[1]

    # factorization
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals[eigvals < 1e-12] = 1e-12
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    # align weights
    w = np.array(weights, dtype=float).reshape(-1)
    if len(w) != n:
        w = w[:n] if len(w) > n else np.pad(w, (0, n-len(w)), constant_values=0)
        w = _clean_weights(w)

    sims_mat = np.zeros((sims, horizon_days+1))
    sims_mat[:, 0] = 1.0
    dt = 1/td

    for s in range(sims):
        z = np.random.normal(size=(horizon_days, n))
        shocks = z @ L.T
        port = 1.0
        for t in range(1, horizon_days+1):
            drift = (mu - 0.5*np.diag(Sigma)) * dt
            diffusion = shocks[t-1, :] * np.sqrt(dt)
            step = np.exp(drift + diffusion)
            port *= float(step @ w)
            sims_mat[s, t] = port
    return sims_mat

# -------- Orchestration ----------
def analyze_portfolio(tickers: str, weights_pct: str, start: str, rf_annual: float,
                      market_symbol: str = "^GSPC", sims: int = 2000, horizon_days: int = 252,
                      conf: float = 0.95, drift_mode: str = "capm"):
    tick_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not tick_list:
        raise ValueError("Enter at least one ticker (comma separated).")

    if weights_pct:
        raw = [float(x.strip()) for x in weights_pct.split(",") if x.strip()]
        if len(raw) != len(tick_list):
            raise ValueError(f"Number of weights ({len(raw)}) must equal number of tickers ({len(tick_list)}).")
        w = np.array(raw, dtype=float)
        s = w.sum()
        if s <= 0:
            raise ValueError("Weights must sum to a positive number.")
        w = w / s
    else:
        w = np.ones(len(tick_list)) / len(tick_list)

    # Include market for CAPM if needed
    all_tickers = tick_list + ([market_symbol] if (market_symbol and market_symbol not in tick_list) else [])
    prices = fetch_prices(all_tickers, start=start)
    # Keep columns in the same order as tick_list, plus market (if present)
    cols = [c for c in prices.columns if c in tick_list] + ([market_symbol] if market_symbol in prices.columns else [])
    prices = prices[cols]

    rets = prices_to_returns(prices)
    if rets.empty:
        raise ValueError("No returns after cleaning. Try an earlier start date.")

    td = 252
    if market_symbol in rets.columns:
        asset = rets[[c for c in rets.columns if c != market_symbol]]
        market = rets[market_symbol]
        exp_capm_ann, betas = capm_expected_returns(asset, market, rf_annual)
        mu_daily_vec = asset.mean().values
        cov_daily = asset.cov().values
        beta_pf = portfolio_beta(w, betas)
        corr = asset.corr()
        mu_series = pd.Series(mu_daily_vec, index=asset.columns)
        cov_df = asset.cov()
    else:
        asset = rets
        mu_daily_vec = asset.mean().values
        cov_daily = asset.cov().values
        beta_pf = float("nan")
        corr = asset.corr()
        mu_series = pd.Series(mu_daily_vec, index=asset.columns)
        cov_df = asset.cov()

    exp_annual, vol_annual, sharpe = portfolio_annualized_stats(w, mu_daily_vec, cov_daily, rf_annual)

    # Path stats
    port_ret_daily = (asset.values @ w)
    port_val = np.cumprod(1 + port_ret_daily)
    mdd = max_drawdown(port_val)

    # VaR/CVaR
    var_ann, cvar_ann = var_cvar(pd.Series(port_ret_daily), conf=conf)

    # Optimization + EF
    w_maxs = optimize_max_sharpe(mu_series, cov_df, rf_annual)
    exp_o, vol_o, sharpe_o = portfolio_annualized_stats(w_maxs, mu_series.values, cov_df.values, rf_annual)
    ef_vol, ef_ret = efficient_frontier(mu_series, cov_df, points=30, rf_annual=rf_annual)

    # Monte Carlo
    sims_mat = monte_carlo_portfolio(prices, w, horizon_days=horizon_days, sims=sims,
                                     drift_mode=drift_mode, rf_annual=rf_annual,
                                     market_col=market_symbol if market_symbol in prices.columns else None)
    show = int(min(100, sims_mat.shape[0]))
    traces = []
    for i in range(show):
        traces.append({"x": list(range(sims_mat.shape[1])), "y": sims_mat[i, :].tolist(),
                       "mode": "lines", "line": {"width": 1}, "opacity": 0.35})

    return {
        "note": f"Data points: {asset.shape[0]} days. Diversification score: {diversification_score(corr):.1f}/100.",
        "current_weights": [{"ticker": t, "weight": float(w[i])} for i, t in enumerate(asset.columns)],
        "opt_weights": [{"ticker": t, "weight": float(w_maxs[i])} for i, t in enumerate(asset.columns)],
        "metrics": {
            "exp_return": float(exp_annual),
            "vol": float(vol_annual),
            "sharpe": float(sharpe),
            "beta": float(beta_pf),
            "max_drawdown": float(mdd),
            "var": float(var_ann),
            "cvar": float(cvar_ann),
        },
        "opt_metrics": {
            "exp_return": float(exp_o),
            "vol": float(vol_o),
            "sharpe": float(sharpe_o),
        },
        "corr": { "labels": list(asset.columns), "z": corr.values.tolist() },
        "ef": { "vol": [float(x) for x in ef_vol], "ret": [float(y) for y in ef_ret] },
        "mc_traces": traces
    }
