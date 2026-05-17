import pandas as pd
import os

def merge_xauusd_datasets():
    datasets_dir = "datasets"
    master_path = "data/xauusd_master_5m.csv"
    os.makedirs("data", exist_ok=True)

    all_dfs = []

    # 1. Load XAUUSD_2010-2023.csv (5m)
    p1 = os.path.join(datasets_dir, "XAUUSD_2010-2023.csv")
    if os.path.exists(p1):
        print(f"Loading {p1}...")
        df1 = pd.read_csv(p1)
        df1.columns = [c.lower() for c in df1.columns]
        ts_col = 'time' if 'time' in df1.columns else 'date'
        all_dfs.append(pd.DataFrame({
            'date': pd.to_datetime(df1[ts_col]),
            'open': df1['open'],
            'high': df1['high'],
            'low': df1['low'],
            'close': df1['close']
        }))

    # 2. Load Gold-Spot-XAUUSD-5-Minute-OHLC-Candles.csv (5m)
    p2 = os.path.join(datasets_dir, "Gold-Spot-XAUUSD-5-Minute-OHLC-Candles.csv")
    if os.path.exists(p2):
        print(f"Loading {p2}...")
        df2 = pd.read_csv(p2)
        df2.columns = [c.lower() for c in df2.columns]
        
        # Determine the best timestamp source
        ts_col = 'time' if 'time' in df2.columns else 'date'
        final_ts = pd.to_datetime(df2[ts_col])
        
        # Create a fresh DF with only the columns we need
        temp_df2 = pd.DataFrame({
            'date': final_ts,
            'open': df2['open'],
            'high': df2['high'],
            'low': df2['low'],
            'close': df2['close']
        })
        all_dfs.append(temp_df2)

    # 3. Load XAU_1m_data.csv (1m) and resample to 5m
    p3 = os.path.join(datasets_dir, "XAU_1m_data.csv")
    if os.path.exists(p3):
        print(f"Loading {p3} (1m data, resampling to 5m)...")
        df3 = pd.read_csv(p3)
        df3.columns = [c.lower() for c in df3.columns]
        df3['date'] = pd.to_datetime(df3['date'])
        df3.set_index('date', inplace=True)
        
        # Resample to 5m
        df3_5m = df3.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        df3_5m.reset_index(inplace=True)
        all_dfs.append(df3_5m[['date', 'open', 'high', 'low', 'close']])

    if not all_dfs:
        print("No datasets found in datasets/ folder.")
        return

    # Merge all
    print("Merging datasets and removing duplicates...")
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Strip timezones to avoid comparison errors
    master_df['date'] = pd.to_datetime(master_df['date'], utc=True).dt.tz_localize(None)
    
    master_df.drop_duplicates(subset=['date'], inplace=True)
    master_df.sort_values('date', inplace=True)
    
    # Fill missing volume with 1000
    if 'volume' not in master_df.columns:
        master_df['volume'] = 1000

    master_df.to_csv(master_path, index=False)
    print(f"Master dataset created at {master_path} with {len(master_df)} rows.")
    return master_path

if __name__ == "__main__":
    merge_xauusd_datasets()
