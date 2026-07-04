# Technical Requirements Document (TRD): XAUUSD Predictor v3.0

## 1. System Architecture
The system follows a high-performance modular architecture combining a vector-retrieval engine (RAG) with a Gradient Boosted Decision Tree (XGBoost) classifier, operating on **15-minute resampled candle data** for reduced noise and improved signal quality.

### 1.1 Technology Stack
- **Language:** Python 3.11+
- **Data Source:** Local Kaggle CSVs (XAUUSD 5m), resampled to 15-minute OHLCV
- **Technical Indicators:** Custom feature engineering (44 features including momentum, volume, multi-timeframe, volatility regime)
- **Vector Index:** `faiss-cpu` (IVF-PQ compressed index, ~150 MB RAM)
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers`
- **Acceleration:** NVIDIA CUDA (RTX 3050 6GB) / AMD DirectML / CPU Fallback
- **Thermal Safety:** Custom `torch` thread limiting + batch cooldown delays
- **Classifier:** `XGBoost` (Hist-based tree method)
- **API:** `FastAPI` (with timezone-aware Pydantic models)
- **UI:** `Streamlit` + `Plotly` (Python), React + Vite (Web)

### 1.2 Why 15-Minute Candles?
- 5-minute gold data has excessive noise (random wicks, spread spikes, market maker activity)
- 15-minute candles smooth out noise while retaining intraday granularity
- Reduces dataset from ~2.4M to ~800K rows (3x faster training)
- Produces cleaner technical indicator signals (RSI, MACD, Bollinger Bands)
- Better pattern recognition for candle formations (hammer, doji, engulfing)

## 2. Directory Structure
```text
/
├── api/                # FastAPI application & Timezone Normalization
├── dashboard/          # Streamlit UI with system health monitor
├── data/               # Master CSVs, FAISS index, engineered features
│   ├── xauusd_master_5m.csv    # Source of truth (2.4M rows, 5-min)
│   ├── xauusd_master_15m.csv   # Resampled (800K rows, 15-min)
│   ├── xauusd_features.csv     # Engineered features (44 columns)
│   └── faiss/                  # FAISS IVF-PQ index + metadata
├── frontend/           # React + Vite web dashboard
├── indicators/         # Custom feature engineering (44 features)
├── models/             # Trained XGBoost artifacts (.json)
├── rag/                # Core RAG engine & FAISS index builder
├── src/core/           # Pydantic-based central configuration
├── training/           # XGBoost training pipeline
├── utils/              # Dataset merge utilities
├── docs/               # Project documentation
├── main.py             # Root orchestration (5m -> 15m -> features -> train)
└── requirements.txt    # Dependency list
```

## 3. Core Logic & Data Flow

### 3.1 Data Ingestion Pipeline
1. **Source Data**: Three Kaggle datasets merged into `data/xauusd_master_5m.csv` (2.4M rows, 2003-2026).
2. **Timestamp Normalization**: Standardized to UTC-Naive format.
3. **15-Minute Resampling**: `pd.resample("15min")` aggregation:
   - Open: first, High: max, Low: min, Close: last, Volume: sum
4. **Output**: `data/xauusd_master_15m.csv` (~800K clean 15-min candles).

### 3.2 Feature Engineering (44 Features)
- **Core Indicators**: RSI(14), MACD(12,26,9), ATR(14), EMA(10,50), Bollinger Bands(20,2)
- **Momentum**: RSI ROC(5), momentum(5,10,30), MACD histogram acceleration
- **Volume**: Relative volume ratio(20), volume ROC(5)
- **Multi-Timeframe**: EMA cross spread, trend alignment score, BB position
- **Volatility Regime**: ATR percentile(500), BB squeeze detection, body/wick ratios
- **Lagged Features**: Returns lag(1,2,4), RSI/MACD/returns rolling mean & std(10)
- **Targets**: next_15m_direction, next_return, target_3bar (45-min), target_6bar (90-min)

### 3.3 FAISS Market Memory (RAG) Engine
- **Backend**: FAISS IVF-PQ (replaces old 3-shard ChromaDB architecture)
- **Index Spec**: 1024 clusters, 48 sub-quantizers, 8-bit codes
- **Memory**: ~150 MB RAM (vs 4-10 GB with ChromaDB)
- **Disk**: ~150 MB (vs 9.7 GB with ChromaDB)
- **Search**: Cosine similarity via L2-normalized inner product, nprobe=32
- **Metadata**: Companion Parquet file with timestamps, regimes, targets
- **Inference**: Converts regional timestamps (IST, etc.) to UTC-Naive before lookup
- **Analogy Query**: Top-5 historical twins via FAISS + NumPy post-filter
- **Safety**: Zero data leakage (past-only lookups), thermal throttling

### 3.4 XGBoost Classifier
- **Training Set**: High-fidelity training window from 15-min features
- **Model Config**: n_estimators=500, max_depth=6, tree_method='hist'
- **Validation**: 5-fold Purged Walk-Forward Cross-Validation
- **Top Features**: body_ratio, returns_roll_std, wick ratios, momentum, ATR percentile

## 4. API & Integration
- **Timezone Auto-Fix**: `normalize_to_utc_naive` handles global input strings (IST/EST)
- **Live Inference**: Feature engineering preserves tailing candle (NaN target) for real-time use
- **Configuration**: Pydantic-based `Settings` model in `src/core/config.py`
- **Endpoint**: `/predict` accepts OHLC indicators + optional regional timestamp
- **Serving**: FastAPI with low-latency signal generation

## 5. Success Metrics (v3.0)
```yaml
success_metrics:
  directional_accuracy:
    target: ">57%"
    evaluation_window: "rolling 1000 trades"
    confidence_interval: "95%"
  
  inference_latency:
    target: "<300ms"
    percentiles: [p50, p95, p99]
  
  system_uptime:
    target: "99.9%"
    calculation: "monthly rolling"
  
  rag_retrieval_time:
    target: "<10ms"
    backend: "FAISS IVF-PQ"
```

## 6. Production Readiness
| Area | Status | Plan |
|------|--------|------|
| API Authentication | Missing | Implement API Key/OAuth in FastAPI middleware |
| Rate Limiting | Missing | Implement `slowapi` for /predict endpoint |
| Monitoring | Missing | Add Prometheus metrics and /health endpoint |
| Model Versioning | Missing | Timestamp + accuracy metadata on save |
| Input Validation | Partial | Enforce OHLC ranges and price > 0 in Pydantic |
