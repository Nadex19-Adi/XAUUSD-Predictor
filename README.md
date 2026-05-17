# XAUUSD Predictor

A **machine‑learning pipeline** that predicts the direction of the XAU/USD (Gold) spot price on a 5‑minute timeframe.

## Features
- **Feature engineering** with over 30 technical indicators (RSI, MACD, Bollinger Bands, ATR, volume ratios, multi‑time‑frame EMA crosses, etc.)
- **Sharded ChromaDB vector store** for fast similarity look‑ups (RAG‑style market memory)
- **XGBoost classifier** with regularisation and confidence‑filtering to surface high‑conviction trades
- **Time‑series aware train/validation split** (80/20 chronological split) and optional walk‑forward validation
- **Automated report generation** (`generate_report.py`) that produces a polished Word document with charts, tables, and a mind‑map.

## Quick Start
```bash
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Generate features (if not already present)
python data/feature_engineering.py   # create data/xauusd_features.csv

# 3️⃣ Train the master model
python train_final.py

# 4️⃣ Produce the project report
python generate_report.py
```

## Project Structure
```
XAUUSD‑Predictor/
├─ api/                # FastAPI endpoints (optional)
├─ data/               # Raw CSVs & engineered feature CSV
├─ datasets/           # External OHLC datasets (large files, git‑ignored)
├─ indicators/         # Feature‑engineering helper functions
├─ models/             # Trained XGBoost model + metadata JSON
├─ rag/                # Vector DB implementation (ChromaDB)
├─ training/           # Training utilities
├─ generate_report.py  # Builds the Word report
├─ todo_boost_accuracy.md  # Accuracy‑boost checklist (tracked)
├─ README.md           # ⬅ this file
└─ .gitignore          # ⬅ ignore large data files
```

## Contributing
1. Fork the repo.
2. Create a feature branch.
3. Ensure you **do not commit** the `datasets/` or `data/*.csv` files – they are listed in `.gitignore`.
4. Open a pull request.

## License
MIT – feel free to adapt for your own research or trading projects.
