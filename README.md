# XAUUSD Predictor v3.0

A hybrid AI trading system for Gold (XAUUSD) that combines **Market Memory (RAG)** with **XGBoost** classification on **15-minute candle data**.

## Architecture
- **Data Pipeline**: 5-min OHLCV -> 15-min resample -> 44 engineered features
- **Vector Search**: FAISS IVF-PQ index (~150 MB) for sub-10ms historical regime retrieval
- **Classifier**: XGBoost (hist-tree, 500 estimators, purged walk-forward CV)
- **API**: FastAPI with timezone-aware Pydantic models
- **UI**: Streamlit + Plotly / React + Vite

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (resample -> features -> FAISS index -> train -> serve)
python main.py
```

### Individual Steps
```bash
# Build FAISS vector index only
python -m rag.build_vector_db

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start Streamlit dashboard
streamlit run dashboard/app.py
```

### Docker
```bash
docker-compose up --build
```

## Project Structure
```
├── api/           FastAPI /predict endpoint
├── dashboard/     Streamlit + Plotly UI
├── data/          Master CSVs, FAISS index, features
├── docs/          TRD, PRD, Audit Report, Roadmap
├── frontend/      React + Vite web dashboard
├── indicators/    44-feature engineering pipeline
├── models/        Trained XGBoost model
├── rag/           FAISS-backed RAG market memory
├── src/core/      Pydantic central configuration
├── training/      XGBoost training with walk-forward CV
├── utils/         Dataset merge utilities
└── main.py        Root orchestration
```

## Documentation
See [docs/](docs/) for detailed documentation:
- [TRD](docs/TRD.md) - Technical Requirements
- [PRD](docs/PRD.md) - Product Requirements
- [Audit Report](docs/AUDIT_REPORT.md) - System Audit
- [Accuracy Roadmap](docs/ACCURACY_ROADMAP.md) - Improvement Plan
