from flask import Flask, render_template, request
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/bonds")
def bonds():
    return render_template("bonds.html")

# -------- Stocks --------
@app.route("/stocks", methods=["GET", "POST"])
def stocks():
    ctx = {"form": {}, "result": None, "error": None}

    if request.method == "POST":
        ticker = (request.form.get("ticker") or "").upper().strip()
        amount = float(request.form.get("amount") or 0)
        horizon = int(float(request.form.get("horizon") or 0))
        bench = (request.form.get("benchmark") or "").upper().strip() or None

        ctx["form"] = {"ticker": ticker, "amount": amount, "horizon": horizon}

        try:
            res = compute_stock_summary(ticker, horizon, amount)
            if res is None:
                ctx["error"] = f"No data found for ticker '{ticker}'."
            else:
                if bench:
                    bres = compute_stock_summary(bench, 12, 0, projection=False)
                    if bres:
                        res["bench"] = {"symbol": bench, "gmean": bres["gmean"]}
                ctx["result"] = res
        except Exception as e:
            ctx["error"] = f"Error: {e}"

    return render_template("stocks.html", **ctx)

def compute_stock_summary(ticker: str, horizon_months: int, amount: float, projection: bool=True):
    end = pd.Timestamp.today(tz="UTC").normalize()
    start = end - pd.DateOffset(years=3, months=1)  # a bit of buffer
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        return None

    # Monthly series from adjusted closes (use 'Close' when auto_adjust=True)
    monthly = df["Close"].resample("M").last().dropna()
    rets = monthly.pct_change().dropna()
    if len(rets) < 12:
        return None  # too little history

    # Means & std
    gmean = (1.0 + rets).prod() ** (1.0 / len(rets)) - 1.0
    amean = float(rets.mean())
    std = float(rets.std())

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

if __name__ == "__main__":
    app.run(debug=True)
