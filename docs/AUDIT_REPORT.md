# Audit Report: XAUUSD Predictor v3.0

> **Scope** - This audit covers the production pipeline as it stands today: FAISS vector index, RAG + XGBoost model, FastAPI + Streamlit UI, operating on 15-minute resampled candle data.

## Executive Summary

| Area | Finding | Risk Level |
|------|---------|------------|
| **Data pipeline** | Clean 5m -> 15m resampling with proper OHLCV aggregation | Low |
| **Vector index** | FAISS IVF-PQ: ~150 MB disk, ~150 MB RAM (vs 9.7 GB / 4-10 GB with old ChromaDB) | Low |
| **Feature engineering** | 44 normalized features covering momentum, volume, multi-timeframe, volatility | Low |
| **Model training** | XGBoost with purged walk-forward CV, top-20 feature selection via RFE | Medium |
| **Security posture** | No hardcoded secrets; config via Pydantic `Settings`; all I/O sandboxed in `data/` | Low |
| **Thermal safety** | Thread limiting + batch cooldowns prevent hardware damage | Low |

> **Bottom line:** The pipeline is **production-ready** for a mid-size quant operation on a single VM. The migration from 5-min to 15-min candles significantly reduces noise and improves signal quality.

## Architecture Diagram
```
+-------------------+     +-------------------+     +-------------------+
|  5-Min Master CSV |     |  15-Min Resample  |     |  Feature Engine   |
|  (2.4M rows)      | --> |  (800K rows)      | --> |  (44 features)    |
+-------------------+     +-------------------+     +-------------------+
                                                            |
                          +-------------------+     +-------------------+
                          |  FAISS IVF-PQ     | <-- |  Sentence         |
                          |  (~150 MB)        |     |  Embeddings       |
                          +-------------------+     +-------------------+
                                  |
                          +-------------------+     +-------------------+
                          |  XGBoost Model    |     |  FastAPI + UI     |
                          |  (Hist-Tree)      | --> |  /predict         |
                          +-------------------+     +-------------------+
```

## Component Analysis

### Data Pipeline
- **Source**: Three Kaggle datasets merged into `xauusd_master_5m.csv` (2.4M rows, 2003-2026)
- **Resampling**: 5m -> 15m via `pd.resample("15min")` with proper OHLCV aggregation
- **Feature Engineering**: 44 normalized features (no raw price levels to prevent memorization)
- **Technology**: pandas, numpy

### FAISS Vector Index
| Aspect | Specification |
|--------|--------------|
| **Index type** | IVF-PQ (1024 clusters, 48 sub-quantizers, 8-bit) |
| **Vectors** | ~800K (15-min candles) |
| **Disk** | ~150 MB |
| **RAM** | ~150 MB |
| **Search** | Cosine similarity, nprobe=32 |
| **Metadata** | Companion Parquet file |

### Risk Matrix
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Data quality** | Low | High | Automated NaN detection + dropna in pipeline |
| **Model overfit** | Medium | High | Purged walk-forward CV, feature importance filtering |
| **Memory overflow** | Low | High | FAISS uses 150 MB vs ChromaDB's 4-10 GB |
| **Hardware failure** | Low | High | Persistent FAISS files on disk; simple 2-file backup |
| **Stale predictions** | Medium | Medium | Retrain regularly with fresh data |

## Recommendations
1. **Containerize** - Dockerize the full stack (FastAPI, Streamlit, FAISS index)
2. **Add API Auth** - Implement API key or OAuth middleware
3. **Add Rate Limiting** - Use `slowapi` for the /predict endpoint
4. **Model Versioning** - Save models with timestamp + accuracy metadata
5. **Monitoring** - Add Prometheus metrics and /health endpoint
