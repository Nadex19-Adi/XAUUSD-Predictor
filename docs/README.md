# XAUUSD Predictor - Documentation

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (resample -> features -> FAISS index -> train -> serve)
python main.py

# Or run individual steps:

# 1. Build FAISS vector index only
python -m rag.build_vector_db

# 2. Start API server only
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 3. Start Streamlit dashboard only
streamlit run dashboard/app.py
```

## Architecture
- **Data Flow**: 5-min CSV -> 15-min resample -> Feature Engineering (44 features) -> FAISS Index + XGBoost Training
- **RAG Backend**: FAISS IVF-PQ (single unified index, ~150 MB)
- **Classifier**: XGBoost (hist-tree, 500 estimators, walk-forward CV)
- **API**: FastAPI with timezone-aware Pydantic models
- **UI**: Streamlit + Plotly / React + Vite

## Project Structure
```
├── api/           FastAPI application
├── dashboard/     Streamlit UI
├── data/          Master CSVs, FAISS index, features
├── docs/          Project documentation
├── frontend/      React + Vite web dashboard
├── indicators/    Feature engineering (44 features)
├── models/        Trained XGBoost model
├── rag/           RAG engine + FAISS index builder
├── src/core/      Pydantic central configuration
├── training/      XGBoost training pipeline
├── utils/         Dataset merge utilities
└── main.py        Root orchestration script
```

## Key Documents
- [TRD.md](TRD.md) - Technical Requirements Document
- [PRD.md](PRD.md) - Product Requirements Document
- [AUDIT_REPORT.md](AUDIT_REPORT.md) - System Audit Report
- [ACCURACY_ROADMAP.md](ACCURACY_ROADMAP.md) - Accuracy Improvement Roadmap
