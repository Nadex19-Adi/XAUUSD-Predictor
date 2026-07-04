# Product Requirements Document (PRD): XAUUSD Predictor v3.0

## 1. Goal
To build a highly accurate, production-ready quantitative AI trading system for Gold (XAUUSD). The system leverages a **"Market Memory" Retrieval-Augmented Generation (RAG)** engine to recall similar past market regimes from a 23-year historical dataset, coupled with a deep **XGBoost classifier** for short-term price direction prediction.

**Key Objectives:**
- Provide reliable, high-confidence trading signals on 15-minute candles.
- **Noise Reduction**: Resample raw 5-min data to 15-min candles for cleaner signals and better pattern recognition.
- **Efficient Vector Search**: FAISS IVF-PQ index (~150 MB) for sub-10ms retrieval across 800K+ historical regimes.
- **Thermal Resilience**: Built-in hardware protection (thread limiting + batch cooldowns) to prevent system thermal shutdowns.
- **Timezone Resilience**: Automatic normalization of global timestamps (IST, EST, Broker Time) to UTC.
- Explain trade logic via historical regime similarity (XAI).

## 2. Target Audience
- **Quantitative / Algorithmic Traders**: Seeking a hybrid approach (NLP Embeddings + Gradient Boosting).
- **Intraday Traders**: Using 15-min intervals for cleaner execution signals with less noise than 5-min.
- **Portfolio Managers**: Needing an explainable AI system that provides historical context for every prediction.

## 3. Core Features & Capabilities

### 3.1 Market Memory (RAG Engine)
- **FAISS Backend**: Single unified IVF-PQ index replaces old 3-shard ChromaDB architecture.
- **Deep Indexing**: Queries 23 years of OHLCV data (2003-2026, ~800K 15-min candles).
- **Batch Optimization**: High-speed embedding and indexing with thermal safety.
- **Regime Recall**: Top-5 historical "twins" of the current market state.
- **Temporal Integrity**: Strict timestamp filtering ensures zero data leakage during backtesting.

### 3.2 Advanced Predictive Modeling (XGBoost)
- **Directional Classification**: Predicts if the next 15-min candle will be UP or DOWN.
- **44-Feature Set**: Momentum, volume, multi-timeframe, volatility regime, candle morphology, and lagged features.
- **Multi-Target**: Primary (next 15-min), 3-bar (45-min), 6-bar (90-min) prediction horizons.
- **Purged Walk-Forward CV**: Validates model stability across 5 temporal splits.

### 3.3 Reliability & Global Compatibility
- **Timezone Auto-Fix**: Handles IST and other regional inputs by normalizing to UTC-naive format.
- **Resampling Pipeline**: Automated 5m -> 15m conversion in `main.py`.

### 3.4 Monitoring & UI
- **Streamlit Dashboard**: Real-time visualization with Plotly candlestick charts.
- **React Web Dashboard**: Modern frontend built with Vite + React + TypeScript.
- **System Health**: Dataset status, index capacity, and RAG connectivity indicators.
- **Analogical Insight**: Visualizes closest historical matches for current price action.

## 4. Success Metrics (v3.0)
- **Directional Accuracy**: Target > 57% (rolling 1000 trades, 95% CI).
- **Inference Latency**: Target < 300ms (p95) for RAG + Prediction loop.
- **System Uptime**: Target 99.9% (monthly rolling).
- **RAG Retrieval Time**: Target < 10ms (FAISS IVF-PQ).
- **Thermal Stability**: 0 System Shutdowns (GPU < 85C, CPU < 90C).
