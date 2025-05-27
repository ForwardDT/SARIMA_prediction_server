import argparse
import os
import pandas as pd
from pandas.tseries.offsets import BDay

from mytslaapp.data import fetch_data
from mytslaapp.model import load_data, fit_auto_arima, forecast, simulate_paths
from mytslaapp.report import (
    plot_residuals,
    plot_forecast,
    plot_simulations,
    summary_stats,
    save_summary
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   default="TSLA")
    parser.add_argument("--start",    default="2020-01-01")
    parser.add_argument("--end",      default="today")
    parser.add_argument("--csv",      default="tesla.csv")
    parser.add_argument("--output",   default="outputs")
    parser.add_argument("--m",        type=int, default=8)
    parser.add_argument("--n_steps",  type=int, default=30)
    parser.add_argument("--n_sims",   type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Fetch & save
    df = fetch_data(args.ticker, args.start, args.csv)

    # 2. Load & prep
    series = load_data(args.csv, args.start, args.end)

    # 3. Fit
    model = fit_auto_arima(series, args.m)
    print(model.summary())

    # 4. Residuals
    resid = pd.Series(model.resid()).dropna()
    plot_residuals(resid, args.output)

    # 5. Forecast
    fc_vals, fc_ci = forecast(model, args.n_steps)
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + BDay(1),
                                 periods=args.n_steps,
                                 freq="B")
    plot_forecast(series, future_dates, fc_vals, fc_ci, args.output)

    # 6. Simulate
    sims = simulate_paths(series, model, args.n_sims, args.n_steps)
    plot_simulations(series, future_dates, sims, args.output)

    # 7. Summary
    final_vals = sims[:, -1]
    stats = summary_stats(final_vals, series.iloc[-1])
    save_summary(stats, args.output)

    print(f"All outputs saved in '{args.output}/'")

if __name__ == "__main__":
    main()
