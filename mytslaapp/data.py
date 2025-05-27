import yfinance as yf
import pandas as pd
from curl_cffi import requests

def fetch_data(ticker: str, start_date: str, csv_filename: str) -> pd.DataFrame:
    """
    Downloads historical data for `ticker` since `start_date` and saves to CSV.
    Returns the downloaded DataFrame.
    """
    session = requests.Session(impersonate="chrome")
    print(f"Downloading {ticker} from {start_date} …")
    df = yf.download(ticker, start=start_date, session=session)
    if df.empty:
        print(f"No data for {ticker}.")
    else:
        df.to_csv(csv_filename)
        print(f"Saved to {csv_filename}.")
    return df
