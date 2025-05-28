import os
import json
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import BDay

from fastapi import FastAPI, Request, Query
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

# ensure outputs directory exists
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# serve our output images
app.mount("/outputs", StaticFiles(directory=OUTDIR), name="outputs")

# set up templates
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    # pass today's date into the template for default end-date
    today = datetime.now().strftime("%Y-%m-%d")
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "today": today}
    )


@app.get("/predict")
async def predict(
    ticker:    str = Query("TSLA"),
    start:     str = Query("2020-01-01"),
    end:       str = Query(None),
    m:         int = Query(8, description="seasonal period"),
    n_steps:   int = Query(30, description="forecast horizon"),
    n_sims:    int = Query(1000, description="number of simulation paths"),
):
    # resolve end date default
    if not end or end.lower() == "today":
        end = datetime.now().strftime("%Y-%m-%d")

    # base filename for this run
    base = f"{ticker}_{start}_{end}"
    resid_file = f"{base}_residuals.png"
    fcst_file  = f"{base}_forecast.png"
    sim_file   = f"{base}_simulations.png"
    summary_file = f"{base}_summary.json"

    # if all outputs exist, skip rerunning
    all_exist = all(os.path.exists(os.path.join(OUTDIR, fname))
                    for fname in (resid_file, fcst_file, sim_file, summary_file))
    if all_exist:
        stats = json.load(open(os.path.join(OUTDIR, summary_file)))
    else:
        # 1) fetch & save raw CSV
        csv_path = f"{base}.csv"
        fetch_data(ticker, start, csv_path)

        # 2) load & prep
        series = load_data(csv_path, start, end)

        # 3) fit ARIMA
        model = fit_auto_arima(series, m=m)

        # 4) residuals plot
        resid = pd.Series(model.resid()).dropna()
        plot_residuals(resid, OUTDIR)
        os.replace(os.path.join(OUTDIR, "residuals.png"),
                   os.path.join(OUTDIR, resid_file))

        # 5) forecast plot
        fc_vals, fc_ci = forecast(model, n_periods=n_steps)
        last_date = series.index[-1]
        future_dates = pd.date_range(
            start=last_date + BDay(1),
            periods=n_steps,
            freq="B"
        )
        plot_forecast(series, future_dates, fc_vals, fc_ci, OUTDIR)
        os.replace(os.path.join(OUTDIR, "forecast.png"),
                   os.path.join(OUTDIR, fcst_file))

        # 6) simulations plot
        sims = simulate_paths(series, model, n_sims=n_sims, n_steps=n_steps)
        plot_simulations(series, future_dates, sims, OUTDIR)
        os.replace(os.path.join(OUTDIR, "simulations.png"),
                   os.path.join(OUTDIR, sim_file))

        # 7) summary stats
        final_vals = sims[:, -1]
        stats = summary_stats(final_vals, series.iloc[-1])
        with open(os.path.join(OUTDIR, summary_file), "w") as f:
            json.dump(stats, f)

    # record last run parameters
    with open(os.path.join(OUTDIR, "last_run.json"), "w") as f:
        json.dump({"ticker": ticker, "start": start, "end": end}, f)

    return JSONResponse({
        "summary": stats,
        "images": {
            "residuals":   f"/outputs/{resid_file}",
            "forecast":    f"/outputs/{fcst_file}",
            "simulations": f"/outputs/{sim_file}",
        }
    })


@app.get("/latest")
async def latest():
    """Fetch the results of the most recent run, if any."""
    last_json = os.path.join(OUTDIR, "last_run.json")
    if not os.path.exists(last_json):
        return JSONResponse({}, status_code=204)

    last = json.load(open(last_json))
    base = f"{last['ticker']}_{last['start']}_{last['end']}"
    summary_file = os.path.join(OUTDIR, f"{base}_summary.json")

    if not os.path.exists(summary_file):
        return JSONResponse({}, status_code=204)

    stats = json.load(open(summary_file))
    return JSONResponse({
        "summary": stats,
        "images": {
            "residuals":   f"/outputs/{base}_residuals.png",
            "forecast":    f"/outputs/{base}_forecast.png",
            "simulations": f"/outputs/{base}_simulations.png",
        }
    })
