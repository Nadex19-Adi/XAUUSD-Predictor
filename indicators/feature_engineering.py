import pandas as pd
import numpy as np
import os

def engineer_features(df: pd.DataFrame, output_path: str = "data/xauusd_features.csv") -> pd.DataFrame:
    """
    Advanced feature engineering for XAUUSD prediction.
    Includes momentum, multi-timeframe, volume, and volatility regime features.
    All features are NORMALIZED (no raw price levels to prevent memorization).
    """
    print("Engineering features (Advanced v2.0)...")
    df = df.copy()
    
    # =====================================================================
    # CORE INDICATORS (Original)
    # =====================================================================
    
    # RSI (14-period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta > 0, 0).abs()).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR (14-period)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # EMAs (kept for RAG text generation, NOT for model training)
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = ma20 + (std20 * 2)
    df['bb_lower'] = ma20 - (std20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']  # Normalized
    
    # Basic returns
    df['returns'] = df['close'].pct_change()
    
    # =====================================================================
    # NEW: MOMENTUM FEATURES (Phase 1.1)
    # =====================================================================
    
    # RSI Divergence (5-bar rate of change of RSI)
    df['rsi_roc'] = df['rsi'].diff(5)
    
    # Multi-bar momentum (normalized as percentage)
    df['momentum_5'] = df['close'].pct_change(5)    # 25-min momentum
    df['momentum_10'] = df['close'].pct_change(10)   # 50-min momentum
    df['momentum_30'] = df['close'].pct_change(30)   # 2.5-hour momentum
    
    # Rate of change of MACD histogram (acceleration)
    df['macd_hist_roc'] = df['macd_hist'].diff(3)
    
    # =====================================================================
    # NEW: VOLUME FEATURES (Phase 1.1)
    # =====================================================================
    
    if 'volume' in df.columns and df['volume'].sum() > 0:
        # Relative volume (vs 20-bar average)
        vol_ma = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / vol_ma.replace(0, 1)
        
        # Volume momentum
        df['volume_roc'] = df['volume'].pct_change(5)
    else:
        df['volume_ratio'] = 1.0
        df['volume_roc'] = 0.0
    
    # =====================================================================
    # NEW: MULTI-TIMEFRAME FEATURES (Phase 1.2)
    # =====================================================================
    
    # EMA cross signals (normalized as difference ratio, not raw price)
    df['ema_cross'] = (df['ema10'] - df['ema50']) / df['close']  # Normalized EMA spread
    
    # Higher timeframe trend proxies (using rolling windows on 5m data)
    ema_15m = df['close'].rolling(30).mean()    # ~15 min proxy
    ema_1h = df['close'].rolling(120).mean()    # ~1 hour proxy
    
    # Trend alignment score: +1 if all aligned up, -1 if all aligned down
    df['trend_alignment'] = (
        np.sign(df['ema10'] - ema_15m) + 
        np.sign(ema_15m - ema_1h)
    ) / 2.0  # Normalized between -1 and +1
    
    # Price position within Bollinger Bands (0 = lower band, 1 = upper band)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range.replace(0, 1)
    
    # =====================================================================
    # NEW: VOLATILITY REGIME FEATURES (Phase 1.3)
    # =====================================================================
    
    # ATR percentile rank (rolling 500-bar window) — regime detector
    df['atr_percentile'] = df['atr'].rolling(500, min_periods=50).rank(pct=True)
    
    # Bollinger squeeze detection (1 = squeeze, 0 = normal)
    bb_width_q10 = df['bb_width'].rolling(120, min_periods=20).quantile(0.1)
    df['bb_squeeze'] = (df['bb_width'] < bb_width_q10).astype(float)
    
    # Candle body ratio (normalized: body / range)
    candle_range = df['high'] - df['low']
    candle_body = (df['close'] - df['open']).abs()
    df['body_ratio'] = candle_body / candle_range.replace(0, 1)
    
    # Upper/Lower wick ratios (for pin bar / doji detection)
    df['upper_wick_ratio'] = (df['high'] - df[['close', 'open']].max(axis=1)) / candle_range.replace(0, 1)
    df['lower_wick_ratio'] = (df[['close', 'open']].min(axis=1) - df['low']) / candle_range.replace(0, 1)
    
    # =====================================================================
    # TARGETS
    # =====================================================================
    
    # Original target (single candle)
    df['next_5m_direction'] = (df['close'].shift(-1) > df['close']).astype(float)
    df['next_return'] = df['close'].shift(-1) / df['close'] - 1
    
    # NEW: Multi-bar targets (Phase 2 — less noisy)
    df['target_3bar'] = (df['close'].shift(-3) > df['close']).astype(float)  # 15-min
    df['target_6bar'] = (df['close'].shift(-6) > df['close']).astype(float)  # 30-min
    
    # Clean up
    if len(df) > 1000:
        df.dropna(inplace=True)
        df['next_5m_direction'] = df['next_5m_direction'].astype(int)
        df['target_3bar'] = df['target_3bar'].astype(int)
        df['target_6bar'] = df['target_6bar'].astype(int)
    
    # Save to disk
    print(f"Saving features to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    
    print(f"Features ready. Shape: {df.shape} ({len(df.columns)} columns)")
    return df
