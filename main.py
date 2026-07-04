import os
import sys
import subprocess
import time
import pandas as pd


def ensure_dirs():
    dirs = ['data', 'data/faiss', 'models', 'rag', 'indicators',
            'api', 'dashboard', 'training', 'notebooks', 'utils']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        init = os.path.join(d, '__init__.py')
        if not os.path.exists(init):
            with open(init, 'a'):
                pass


def resample_5m_to_15m(master_5m: str, master_15m: str) -> pd.DataFrame:
    """Resample 5-minute OHLCV candles to 15-minute candles."""
    print("Loading 5-min master data...")
    df = pd.read_csv(master_5m, parse_dates=["date"], index_col="date")
    print(f"  Loaded {len(df):,} rows (5-min candles)")

    print("Resampling to 15-min candles...")
    df_15m = df.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna(subset=["open"])

    df_15m.to_csv(master_15m)
    print(f"  Resampled: {len(df):,} -> {len(df_15m):,} rows (saved to {master_15m})")
    return df_15m


def main():
    print("=== XAUUSD Predictor Pipeline (15-Min Candles) ===")

    # 1. Ensure directory structure
    ensure_dirs()

    # 2. Data Ingestion: 5m -> 15m resampling
    master_5m = "data/xauusd_master_5m.csv"
    master_15m = "data/xauusd_master_15m.csv"

    if not os.path.exists(master_5m):
        from utils.merge_datasets import merge_xauusd_datasets
        merge_xauusd_datasets()

    if not os.path.exists(master_15m):
        df = resample_5m_to_15m(master_5m, master_15m)
    else:
        from data.data_loader import load_data
        df = load_data(csv_path=master_15m)

    # 3. Feature Engineering (on 15-min data)
    from indicators.feature_engineering import engineer_features
    df_features = engineer_features(df)

    # 4. Build FAISS Vector Index
    from rag.build_vector_db import build_db
    build_db("data/xauusd_features.csv")

    # 5. Train XGBoost Model
    from training.train import train_pipeline
    train_pipeline(df_features, "models/xgb_model.json")

    print("\nPipeline complete. Starting services...")

    # 6. Launch API
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", "8000"]
    )

    time.sleep(3)

    # 7. Launch Dashboard
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
        )
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        api_process.terminate()


if __name__ == "__main__":
    main()
