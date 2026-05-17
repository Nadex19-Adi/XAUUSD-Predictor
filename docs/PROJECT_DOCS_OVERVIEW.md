# 📚 XAUUSD RAG Predictor — Quick Docs Overview

> **Last Updated**: 2026-05-07 04:12 PM IST | **All Shards**: ✅ Complete | **Total Records**: 1,687,267

---

## 🗺️ Doc Index
| File | What's Inside |
|------|---------------|
| [README.md](./README.md) | Setup, install, run commands |
| [PRD.md](./PRD.md) | Business goals & product requirements |
| [TRD.md](./TRD.md) | Architecture, sharding, model config |
| [AUDIT_REPORT.md](./AUDIT_REPORT.md) | Real-world sustainability & risk audit |
| [PROJECT_MONITOR.md](./PROJECT_MONITOR.md) | Live ops dashboard (shard status) |

---

## ⚡ Pipeline Health Check (Verified 04:12 PM)

### ✅ Data Layer
| Component | Status | Details |
|-----------|--------|---------|
| Master CSV | ✅ | `data/xauusd_master_5m.csv` (135 MB) |
| Feature CSV | ✅ | `data/xauusd_features.csv` (478 MB, 1.7M rows, 14 indicators) |
| ChromaDB | ✅ | `data/chromadb/` (9.7 GB, 3 shards, 1.68M vectors) |

### ✅ Model Layer
| Component | Status | Details |
|-----------|--------|---------|
| XGBoost (json) | ✅ | `models/xgb_model.json` (49 KB) — Walk-Forward trained |
| XGBoost (bst) | ✅ | `models/xauusd_model.bst` (4 MB) — 1000-estimator master model |

### ✅ Application Layer
| Component | Status | Details |
|-----------|--------|---------|
| FastAPI | ✅ | `api/main.py` — `/predict` endpoint with timezone auto-fix |
| Streamlit | ✅ | `dashboard/app.py` — Live chart + RAG analogues |
| Config | ✅ | `src/core/config.py` — Pydantic-based central settings |

### ✅ Intelligence Layer
| Component | Status | Details |
|-----------|--------|---------|
| Feature Engine | ✅ | `indicators/feature_engineering.py` — RSI, MACD, ATR, BB, EMA |
| RAG Engine | ✅ | `rag/rag_engine.py` — Cosine similarity, top-5 recall |
| Shard Builder | ✅ | `rag/build_vector_db.py` — 3-shard with GC + batch=5000 |
| Training | ✅ | `training/train.py` + `train_final.py` — Walk-Forward CV |

---

## ⚠️ Critical Findings

### 🔴 metrics.txt shows 100% accuracy — THIS IS A RED FLAG!
```
Walk-Forward Validation Results (6 splits):
Average Accuracy:  1.0000
Average Precision: 1.0000
Average Recall:    1.0000
```
**Why it's a problem**: Perfect metrics = **data leakage** or **target leak** in the training pipeline. No model achieves 100% on financial data. This needs immediate investigation.

**Likely cause**: `next_5m_direction` is computed from `close.shift(-1)` which may be leaking into feature columns during walk-forward splits.

### 🟡 config.py still references old collection name
`COLLECTION_NAME: str = "gold_patterns"` — but actual shards are `gold_legacy`, `gold_mid`, `gold_recent`. Config needs updating for shard-aware queries.

### 🟡 No train/test split in `train_final.py`
`model.fit(X, y)` trains on the **entire** 1.7M dataset with no holdout set. No way to measure real generalization performance.

---

## 🏗️ Project Structure (Verified)
```
XAUUSD predictor/
├── api/main.py              # FastAPI backend
├── dashboard/app.py         # Streamlit UI
├── data/
│   ├── xauusd_master_5m.csv # 135 MB raw OHLC
│   ├── xauusd_features.csv  # 478 MB engineered features
│   ├── data_loader.py       # yfinance + CSV loader
│   └── chromadb/            # 9.7 GB vector store (3 shards)
├── indicators/
│   └── feature_engineering.py  # 14 technical indicators
├── models/
│   ├── xgb_model.json       # 49 KB (walk-forward)
│   └── xauusd_model.bst     # 4 MB (1000-tree master)
├── rag/
│   ├── rag_engine.py        # MarketRAG class
│   └── build_vector_db.py   # Sharded builder
├── src/core/config.py       # Pydantic settings
├── training/train.py        # Walk-forward training
├── train_final.py           # Full-dataset master training
├── main.py                  # Orchestration entry point
├── Dockerfile               # Container support
├── docker-compose.yml       # Multi-service orchestration
├── requirements.txt         # Dependencies
└── docs/                    # You are here! 📍
```

---

## 🎯 Next Steps (Priority Order)
1. 🔴 **FIX DATA LEAKAGE** — Investigate 100% accuracy in `metrics.txt`
2. 🟡 **Update config.py** — Add shard-aware collection names
3. 🟡 **Add holdout test** in `train_final.py` — at minimum 80/20 split
4. 📅 **Retrain XGBoost** on clean sharded features
5. 📅 **Launch API + Dashboard** for live signals
