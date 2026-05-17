import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds core technical indicators to the OHLCV dataframe.
    """
    df = df.copy()
    
    # Momentum
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    
    # Trend
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema10'] = EMAIndicator(df['close'], window=10).ema_indicator()
    df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    
    # Volatility
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    
    # Price Action
    df['returns'] = df['close'].pct_change()
    
    # Target (next 5m direction)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df.dropna(inplace=True)
    return df
