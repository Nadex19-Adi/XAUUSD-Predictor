import os
import sys
import subprocess
import time

def ensure_dirs():
    dirs = ['data', 'models', 'rag', 'indicators', 'api', 'dashboard', 'training', 'notebooks', 'utils']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        # Touch __init__.py
        with open(os.path.join(d, '__init__.py'), 'a'):
            pass

def main():
    print("=== XAUUSD Predictor Setup & Run ===")
    
    # 1. Ensure structure
    ensure_dirs()
    
    # 2. Data Ingestion (Using the new 2.4M row Master Dataset)
    from data.data_loader import load_data
    csv_path = "data/xauusd_master_5m.csv"
    if not os.path.exists(csv_path):
        from utils.merge_datasets import merge_xauusd_datasets
        csv_path = merge_xauusd_datasets()
        
    df = load_data(csv_path=csv_path)
    
    # 3. Feature Engineering
    from indicators.feature_engineering import engineer_features
    df_features = engineer_features(df)
    
    # 4. Build Vector DB (Chunked for Memory Safety)
    from rag.build_vector_db import build_db
    build_db("data/xauusd_features.csv")
    
    # 5. Train Model
    from training.train import train_pipeline
    train_pipeline(df_features, "models/xgb_model.json")
    
    print("\nPipeline execution complete. Starting services...")
    
    # 6. Launch API in background
    api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])
    
    # Wait a moment for API to start
    time.sleep(3)
    
    # 7. Launch Dashboard
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        api_process.terminate()

if __name__ == "__main__":
    main()
