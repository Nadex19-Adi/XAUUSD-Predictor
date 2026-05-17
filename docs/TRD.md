# Technical Requirements Document (TRD): XAUUSD Predictor (Perfect Case v2.1)

## 1. System Architecture
The system follows a high-performance modular architecture combining a massive vector-retrieval engine (RAG) with a Gradient Boosted Decision Tree (XGBoost) classifier.

### 1.1 Technology Stack
- **Language:** Python 3.11 (Recommended for stability and pre-built wheels)
- **Data Fetching:** `yfinance`, Local Kaggle CSVs (XAUUSD 5m)
- **Technical Indicators:** `ta` (Technical Analysis library)
- **Vector DB:** `chromadb` (Optimized for 1.7M validated / 2.4M raw capacity)
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers`
- **Acceleration:** NVIDIA CUDA (RTX 3050 6GB) / AMD DirectML Fallback
- **Thermal Safety:** Custom `torch` thread limiting + 2s batch delays
- **Classifier:** `XGBoost` (Hist-based tree method for large scale)
- **API:** `FastAPI` (with timezone-aware Pydantic models)
- **UI:** `Streamlit` + `Plotly`

## 2. Directory Structure
```text
/
├── api/                # FastAPI application & Timezone Normalization
├── dashboard/          # Streamlit UI with system health monitor
├── data/               # Master CSV (2.4M rows), ChromaDB persistent storage
├── datasets/           # Raw Kaggle CSV files (user-provided)
├── indicators/         # Custom feature engineering (14 primary features)
├── models/             # Trained XGBoost artifacts (.json)
├── rag/                # Core RAG engine & Batch Indexing logic
├── src/                # Internal core modules
│   └── core/           # Pydantic-based central configuration (config.py)
├── training/           # Deep training pipeline (train.py)
├── utils/              # Merge & Timezone utility scripts
├── notebooks/          # Jupyter notebooks for exploration
├── main.py             # Root orchestration script
└── requirements.txt    # Dependency list
```

## 3. Core Logic & Data Flow

### 3.1 Data Ingestion (Master Merge)
1.  **Merging**: Combined three Kaggle datasets (2010-2023) into `data/xauusd_master_5m.csv`.
2.  **Normalization**: Standardized timestamps to UTC-Naive (Stripped timezones).
3.  **Resampling**: Resampled 1-minute historical data to a 5-minute OHLC format.

### 3.2 Sharded Perfect Memory (RAG) Engine
- **Indexing**: Optimized batch embedding (batch_size=5000) for 1.7M+ rows.
- **3-Shard Architecture**:
    - `gold_legacy`: 0 - 560,000 rows.
    - `gold_mid`: 560,000 - 1,120,000 rows.
    - `gold_recent`: 1,120,000 - End.
- **Memory Management**: Manual garbage collection (`gc.collect()`) after each shard build to maintain <16GB RAM usage.
- **Inference Normalization**: Converts incoming regional timestamps (IST, etc.) to UTC-Naive before lookup.
- **Analogy Query**: Top-5 historical twins recalled via cosine similarity across active shards.
- **Thermal Filter**: Strict `$lt` timestamp filter ensures zero data leakage (Past lookups only).
- **Thermal Safety Implementation**: 
    - `torch.set_num_threads(cores // 2)`: Prevents CPU from hitting 100% saturation.
    - `time.sleep(2)`: Forced pause between batches to allow GPU/CPU cooling.

### 3.3 Deep XGBoost Modeling
- **Training Set**: 100,000 row high-fidelity training window.
- **Model Config**: 
    - `n_estimators`: 500
    - `max_depth`: 6
    - `tree_method`: 'hist' (optimized for large datasets)
- **Cross-Validation**: 5-fold Purged Walk-Forward Cross-Validation.

## 4. API & Integration
- **Timezone Auto-Fix**: `normalize_to_utc_naive` function handles global input strings (IST/EST).
- **Live Inference Pipeline**: Feature engineering logic modified to preserve the tailing candle (NaN target) for real-time inference while ensuring clean training data.
- **Central Configuration**: Pydantic-based `Settings` model in `src/core/config.py` manages all paths and hyper-parameters.
- **Endpoint**: `/predict` accepts OHLC indicators + optional regional timestamp.
- **Serving**: Production-ready FastAPI implementation for low-latency signal generation.

## 5. Success Metrics (v2.1)
```yaml
success_metrics:
  directional_accuracy:
    target: ">57%"
    evaluation_window: "rolling 1000 trades"
    confidence_interval: "95%"
  
  inference_latency:
    target: "<300ms"
    percentiles: [p50, p95, p99]
    measurement: "from request to response"
  
  system_uptime:
    target: "99.9%"
    calculation: "monthly rolling"
  
  rag_retrieval_time:
    target: "<100ms"
    shard_parallel: true
```

## 6. Production Readiness Gaps & Mitigation
| Area | Status | Mitigation Plan |
|------|--------|------------------|
| API Authentication | ❌ Missing | Implement API Key/OAuth in FastAPI middleware |
| Rate Limiting | ❌ Missing | Implement `slowapi` for /predict endpoint |
| Monitoring | ❌ Missing | Add Prometheus metrics and /health endpoint |
| Model Versioning | ❌ Missing | Artifacts saved with timestamp + accuracy metadata |
| Input Validation | ⚠️ Partial | Enforce OHLC ranges and price > 0 in Pydantic models |

