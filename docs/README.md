# XAUUSD RAG Predictor (Perfect Case v2.0)

A hybrid **RAG + XGBoost** quantitative trading system for Gold (XAUUSD) using 13+ years of 5-minute historical data.

## Setup

1. Create a virtual environment: `python -m venv venv`
2. Activate environment:
    - Windows: `venv\Scripts\activate`
    - Unix: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Place Kaggle datasets in the `datasets/` folder:
    - `XAUUSD_2010-2023.csv`
    - `Gold-Spot-XAUUSD-5-Minute-OHLC-Candles.csv`
    - `XAU_1m_data.csv`

## Running

Execute the entire pipeline via `main.py`:
```bash
python main.py
```

This will:
1. Auto-merge all Kaggle CSVs into a 1.7M-row master dataset (`data/xauusd_features.csv`).
2. Engineer 14 technical features (RSI, MACD, BB Width, etc.).
3. **Build Sharded ChromaDB**: Use the sharding script to handle memory constraints:
   ```bash
   # Build all shards (legacy, mid, recent)
   python -m rag.build_vector_db --shard all
   ```
4. Train the XGBoost model with RAG augmentation and Walk-Forward Validation.
5. Launch the FastAPI server on port 8000.
6. Launch the Streamlit dashboard on port 8501.

## Timezone Auto-Fix
The API accepts timestamps in any timezone (IST, EST, Broker Time) and automatically normalizes them to UTC for accurate historical pattern matching.

## Project Structure
```text
├── api/           FastAPI backend with timezone normalization
├── dashboard/     Streamlit UI with system health monitor
├── data/          Master CSV + ChromaDB persistent storage
├── datasets/      Raw Kaggle CSV files (user-provided)
├── indicators/    Feature engineering (14 indicators)
├── models/        Trained XGBoost artifacts
├── rag/           RAG engine + vector DB builder
├── src/core/      Central Pydantic configuration
├── training/      Deep training pipeline (100k rows, 500 estimators)
├── utils/         Dataset merge & timezone utilities
└── main.py        Root orchestration script
```
