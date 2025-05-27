import os
import numpy as np
import matplotlib.pyplot as plt

def plot_residuals(resid, outdir):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(resid)
    ax.set_title("Residuals")
    fig.savefig(os.path.join(outdir, "residuals.png"))
    plt.close(fig)

def plot_forecast(series, future_dates, fc_vals, fc_ci, outdir):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(series.index, series.values, label='Observed')
    ax.plot(future_dates, fc_vals, 'r--', label='Forecast')
    ax.fill_between(future_dates, fc_ci[:,0], fc_ci[:,1], alpha=0.5)
    ax.legend()
    ax.set_title("Forecast")
    fig.savefig(os.path.join(outdir, "forecast.png"))
    plt.close(fig)

def plot_simulations(series, future_dates, sims, outdir, max_plot=200):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(series.index, series.values, lw=1.5, label='Observed')
    for i in range(min(sims.shape[0], max_plot)):
        ax.plot(future_dates, sims[i], color='blue', alpha=0.05)
    ax.set_title("Simulated Paths")
    fig.savefig(os.path.join(outdir, "simulations.png"))
    plt.close(fig)

def summary_stats(final_vals, current_price):
    prop_above = np.mean(final_vals > current_price)
    prop_below = np.mean(final_vals < current_price)
    ratio     = prop_above / (prop_below + 1e-10)

    return {
        "Proportion ending above model end price":     round(prop_above, 2),
        "Proportion ending below model end price":     round(prop_below, 2),
        "Ratio up/down relative to model end price":   round(ratio, 2),
        "Mean simulated final price $":                  round(np.mean(final_vals), 2),
        "Median simulated final price $":                round(np.median(final_vals), 2),
        "Lower Quartile (25th percentile) $":            round(np.percentile(final_vals, 25), 2),
        "Upper Quartile (75th percentile) $":            round(np.percentile(final_vals, 75), 2),
    }

def save_summary(stats: dict, outdir: str):
    path = os.path.join(outdir, "summary.txt")
    with open(path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
