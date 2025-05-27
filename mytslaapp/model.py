import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import BDay
import statsmodels.api as sm
import pmdarima as pm

def load_data(csv_file: str, start_date: str, end_date: str) -> pd.Series:
    # resolve “today”
    if isinstance(end_date, str) and end_date.lower() == "today":
        end_date = datetime.now().strftime("%Y-%m-%d")
    df = (
        pd.read_csv(csv_file, skiprows=[1,2])
          .assign(Date=lambda d: pd.to_datetime(d['Price'], errors='coerce'),
                  Close=lambda d: pd.to_numeric(d['Close'], errors='coerce'))
          .dropna(subset=['Date','Close'])
          .set_index('Date')
          .sort_index()
    )
    df = df.asfreq('B').ffill().loc[start_date:end_date]
    return df['Close']

def fit_auto_arima(series: pd.Series, m: int):
    """
    Fits a seasonal ARIMA with d=1, D=1, seasonal_period=m.
    """
    model = pm.auto_arima(
        series,
        d=1, D=1, m=m,
        start_p=0, max_p=2,
        start_q=0, max_q=2,
        start_P=0, max_P=1,
        start_Q=0, max_Q=1,
        seasonal=True,
        trend='c',
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        information_criterion='aic'
    )
    return model

def forecast(model, n_periods: int):
    """
    Returns (forecast_values, conf_int) for next n_periods.
    """
    return model.predict(
        n_periods=n_periods,
        return_conf_int=True,
        alpha=0.05
    )

def simulate_paths(series: pd.Series, model, n_sims: int, n_steps: int) -> np.ndarray:
    """
    Simulates n_sims paths of length n_steps under the fitted SARIMAX.
    """
    # rebuild in statsmodels
    sm_mod = sm.tsa.SARIMAX(
        series,
        order=model.order,
        seasonal_order=model.seasonal_order,
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sm_res = sm_mod.fit(disp=False)
    sims = np.zeros((n_sims, n_steps))
    for i in range(n_sims):
        sims[i, :] = sm_res.simulate(nsimulations=n_steps, anchor='end')
    return sims
