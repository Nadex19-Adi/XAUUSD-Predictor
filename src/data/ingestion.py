import yfinance as yf
import pandas as pd
from src.core.config import settings

def fetch_historical_data(symbol: str = settings.SYMBOL, period: str = settings.HISTORY_PERIOD, interval: str = settings.TIMEFRAME) -> pd.DataFrame:
    """
    Fetches historical OHLCV data from Yahoo Finance.
    """
    df = yf.download(symbol, period=period, interval=interval)
    df.columns = [col.lower() for col in df.columns]
    df.dropna(inplace=True)
    return df
