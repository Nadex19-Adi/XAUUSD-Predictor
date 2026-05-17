# Product Requirements Document (PRD): XAUUSD Predictor (Perfect Case v2.1)

## 1. Goal to Achieve
To build a highly accurate, production-ready quantitative AI trading system for Gold (XAUUSD). The system leverages a **"Market Memory" Retrieval-Augmented Generation (RAG)** engine to recall similar past market regimes from a massive 13-year historical dataset, coupled with a deep **XGBoost classifier** for short-term price direction prediction. 

**Key Objectives:**
- Provide reliable, high-confidence trading signals for the 5m timeframe.
- **Sharded Historical Recall**: Utilize a master dataset of 1.7M validated (2.4M raw capacity) rows (2010-2026) indexed across a 3-shard architecture (Legacy, Mid, Recent) to optimize RAM footprint (<16GB).
- **Memory-Efficient RAG**: Use sharding to maintain high-speed retrieval without OOM (Out of Memory) errors.
- **Thermal Resilience**: Built-in hardware protection logic (Thread limiting + Batch cooldowns) to prevent system thermal shutdowns during high-load embedding tasks.
- **Timezone Resilience**: Automatic normalization of global timestamps (IST, EST, Broker Time) to UTC.
- Explain trade logic via historical regime similarity (XAI).

## 2. Target Audience
*   **Quantitative Traders / Algorithmic Traders:** Seeking a hybrid approach (NLP Embeddings + Gradient Boosting).
*   **Day Traders**: Using 5m intervals for quick execution signals.
*   **Portfolio Managers:** Needing an explainable AI system that provides historical context for every prediction.

## 3. Core Features & Capabilities

### 3.1 Perfect Market Memory (RAG Engine)
- **Deep Indexing**: Queries ChromaDB across 13 years of OHLC action.
- **Batch Optimization**: High-speed embedding and indexing of millions of records.
- **Regime Recall**: Queries for the top-k historical "twins" of the current market state.
- **Temporal Integrity**: Strict filtering (`$lt`) ensures the RAG engine never "remembers" the future during backtesting.

### 3.2 Advanced Predictive Modeling (XGBoost)
- **Directional Classification**: Predicts if the next 5m candle will be UP or DOWN.
- **Expanded Feature Set**: Uses 14 input features including MACD Histograms, Bollinger Widths, and RAG-calculated similarity metrics.
- **Purged Walk-Forward CV**: Validates model stability across 5 temporal splits.

### 3.3 Reliability & Global Compatibility
- **Timezone Auto-Fix**: Seamlessly handles IST and other regional inputs by normalizing to UTC-naive format.
- **Synthetic Data Fallback**: Maintains operation even if external APIs are unavailable.

### 3.4 Monitoring & UI
- **Real-time Dashboard**: Built with Streamlit and Plotly for live visualization.
- **System Health Monitor**: Sidebar indicator showing dataset status, indexing capacity (2.4M), and RAG connectivity.
- **Inference Stability**: Optimized data pipeline preserves the most recent candle for zero-latency live signals.
- **Analogical Insight**: Visualizes the closest historical matches for the current price action.

## 4. Success Metrics (v2.1)
*   **Directional Accuracy**: Target > 57% (rolling 1000 trades, 95% CI).
*   **Inference Latency**: Target < 300ms (p95) for RAG + Prediction loop.
*   **System Uptime**: Target 99.9% (monthly rolling).
*   **RAG Retrieval Time**: Target < 100ms (shard-parallel enabled).
*   **Thermal Stability**: 0 System Shutdowns (GPU < 85°C, CPU < 90°C).
