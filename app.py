import os
import pandas as pd
from pandas.tseries.offsets import BDay

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from mytslaapp.data import fetch_data
from mytslaapp.model import load_data, fit_auto_arima, forecast, simulate_paths
from mytslaapp.report import (
    plot_residuals,
    plot_forecast,
    plot_simulations,
    summary_stats,
)

app = FastAPI()

# 1) serve our output images
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# 2) template for index.html
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict")
async def predict():
    # ---- parameters ----
    TICKER    = "TSLA"
    START     = "2020-01-01"
    END       = "today"
    CSV_FILE  = "tesla.csv"
    M         = 8
    N_STEPS   = 30
    N_SIMS    = 1000
    OUTDIR    = "outputs"

    # 1) fetch & save
    fetch_data(TICKER, START, CSV_FILE)

    # 2) load & prep
    series = load_data(CSV_FILE, START, END)

    # 3) fit
    model = fit_auto_arima(series, m=M)

    # 4) residuals plot
    resid = pd.Series(model.resid()).dropna()
    plot_residuals(resid, OUTDIR)

    # 5) forecast plot
    fc_vals, fc_ci = forecast(model, n_periods=N_STEPS)
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + BDay(1), periods=N_STEPS, freq="B")
    plot_forecast(series, future_dates, fc_vals, fc_ci, OUTDIR)

    # 6) simulations plot
    sims = simulate_paths(series, model, n_sims=N_SIMS, n_steps=N_STEPS)
    plot_simulations(series, future_dates, sims, OUTDIR)

    # 7) summary
    final_vals = sims[:, -1]
    stats = summary_stats(final_vals, series.iloc[-1])

    return JSONResponse({
        "summary": stats,
        "images": {
            "residuals": "/outputs/residuals.png",
            "forecast": "/outputs/forecast.png",
            "simulations": "/outputs/simulations.png",
        }
    })
