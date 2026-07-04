# Project Documentation Overview

## Document Inventory

| Document | Status | Path |
|----------|--------|------|
| README (Root) | Updated | `README.md` |
| README (Docs) | Updated | `docs/README.md` |
| TRD | Updated | `docs/TRD.md` |
| PRD | Updated | `docs/PRD.md` |
| Audit Report | Updated | `docs/AUDIT_REPORT.md` |
| Accuracy Roadmap | Updated | `docs/ACCURACY_ROADMAP.md` |
| Project Monitor | Active | `docs/PROJECT_MONITOR.md` |

## System Architecture (v3.0)

```
5-Min CSV (2.4M rows)
    |
    v
15-Min Resample (800K rows)  <-- pd.resample("15min")
    |
    v
Feature Engineering (44 cols) <-- indicators/feature_engineering.py
    |
    +---> FAISS Index Build    <-- rag/build_vector_db.py
    |         |
    |         v
    |     FAISS IVF-PQ (~150MB) <-- data/faiss/
    |
    +---> XGBoost Training     <-- training/train.py
              |
              v
          xgb_model.json       <-- models/
              |
              v
          FastAPI /predict     <-- api/main.py
              |
              v
          Streamlit Dashboard  <-- dashboard/app.py
```

## Project Structure

```
XAUUSD predictor/
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI app with /predict endpoint
├── dashboard/
│   ├── __init__.py
│   └── app.py                  # Streamlit UI
├── data/
│   ├── __init__.py
│   ├── data_loader.py          # CSV loader utility
│   ├── xauusd_master_5m.csv    # Source 5-min data (2.4M rows)
│   ├── xauusd_master_15m.csv   # Resampled 15-min data (800K rows)
│   ├── xauusd_features.csv     # Engineered features (44 columns)
│   └── faiss/                  # FAISS index + metadata
│       ├── market_memory.index
│       └── metadata.parquet
├── docs/                       # All documentation
├── frontend/                   # React + Vite web dashboard
├── indicators/
│   ├── __init__.py
│   └── feature_engineering.py  # 44-feature engineering pipeline
├── models/
│   ├── __init__.py
│   └── xgb_model.json          # Trained XGBoost model
├── rag/
│   ├── __init__.py
│   ├── rag_engine.py           # FAISS-backed RAG engine
│   └── build_vector_db.py      # FAISS index builder
├── src/
│   └── core/
│       ├── __init__.py
│       └── config.py           # Pydantic Settings (central config)
├── training/
│   ├── __init__.py
│   └── train.py                # XGBoost training pipeline
├── utils/
│   ├── __init__.py
│   └── merge_datasets.py       # Dataset merge utility
├── main.py                     # Root orchestration (5m->15m->features->train->serve)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── generate_report.py          # DOCX report generator
├── .env.example
└── .gitignore
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Orchestrator | `main.py` | Full pipeline: resample, features, index, train, serve |
| Config | `src/core/config.py` | Pydantic Settings with all paths and hyperparams |
| Features | `indicators/feature_engineering.py` | 44-feature engineering from OHLCV |
| RAG Engine | `rag/rag_engine.py` | FAISS-backed market memory with thermal safety |
| Index Builder | `rag/build_vector_db.py` | Single unified IVF-PQ index builder |
| Trainer | `training/train.py` | XGBoost with walk-forward CV |
| API | `api/main.py` | FastAPI /predict with timezone normalization |
| Dashboard | `dashboard/app.py` | Streamlit + Plotly live visualization |
| Report Gen | `generate_report.py` | DOCX report generator for submissions |
