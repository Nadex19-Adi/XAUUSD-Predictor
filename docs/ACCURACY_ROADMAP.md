# XAUUSD Predictor — Accuracy Improvement Roadmap

> **Current Baseline**: 52.28% Test Accuracy (honest, no leakage, 5m data)
> **New Baseline (15m)**: To be measured after retraining on 15-min candles
> **Realistic Target**: 58-65% (15-min candles have less noise than 5m)
> **Why not 90%?**: Financial markets have inherent noise. Even Renaissance Technologies
> (the most profitable quant fund in history) operates at ~55-60% hit rate on short timeframes.
> 90%+ accuracy on raw price prediction = data leakage, guaranteed.

---

## Phase 1: Feature Engineering (Target: 54-56%)
**Effort: 1-2 days | Impact: HIGH**

### 1.1 Add Momentum & Volume Features
```python
# indicators/feature_engineering.py — ADD these:
df['rsi_divergence'] = df['rsi'].diff(5)           # RSI momentum
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()  # Relative volume
df['close_vs_vwap'] = df['close'] / (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
df['momentum_10'] = df['close'] / df['close'].shift(10) - 1  # 10-bar momentum
df['momentum_30'] = df['close'] / df['close'].shift(30) - 1  # 30-bar momentum
```

### 1.2 Add Multi-Timeframe Features
```python
# Resample 5m to 15m and 1h, then merge back:
df['ema10_15m'] = df['close'].rolling(30).mean()   # 15m EMA proxy (30 x 5m = 150m)
df['ema10_1h'] = df['close'].rolling(120).mean()   # 1h EMA proxy
df['trend_alignment'] = ((df['ema10'] > df['ema10_15m']) & 
                          (df['ema10_15m'] > df['ema10_1h'])).astype(int)
```

### 1.3 Add Volatility Regime Detection
```python
df['atr_percentile'] = df['atr'].rolling(500).rank(pct=True)  # ATR percentile (regime)
df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(120).quantile(0.1)).astype(int)
```

---

## Phase 2: Smarter Target Engineering (Target: 56-58%)
**Effort: 1 day | Impact: MEDIUM-HIGH**

### 2.1 Use Multi-Bar Targets Instead of Next Candle
```python
# With 15-min base candles, multi-bar targets become even more powerful:
df['target_3bar'] = (df['close'].shift(-3) > df['close']).astype(int)  # 45-min direction
df['target_6bar'] = (df['close'].shift(-6) > df['close']).astype(int)  # 90-min direction
```
**Why?** Even 15-min single candles have noise. 45-90 minute trends are highly predictable.

### 2.2 Use Threshold-Based Targets
```python
# Only predict SIGNIFICANT moves (filter out noise):
threshold = df['atr'] * 0.3  # 30% of ATR
df['target_strong_up'] = (df['close'].shift(-3) - df['close'] > threshold).astype(int)
df['target_strong_down'] = (df['close'] - df['close'].shift(-3) > threshold).astype(int)
```

---

## Phase 3: Model Architecture (Target: 58-60%)
**Effort: 2-3 days | Impact: MEDIUM**

### 3.1 Ensemble of Specialists
```python
# Train separate models for different market regimes:
model_trending = XGBClassifier(...)   # Trained on trending periods only
model_ranging = XGBClassifier(...)    # Trained on ranging periods only

# At inference time, detect regime first, then use the right model
regime = detect_regime(current_data)
if regime == 'trending':
    prediction = model_trending.predict(X)
else:
    prediction = model_ranging.predict(X)
```

### 3.2 LightGBM + CatBoost Stacking
```python
# Stack 3 different gradient boosting models:
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

models = [
    XGBClassifier(...),
    LGBMClassifier(...),
    CatBoostClassifier(...)
]
# Use their predictions as inputs to a meta-learner (Logistic Regression)
```

### 3.3 LSTM/Transformer for Sequential Patterns
```python
# Use a sequence model to capture temporal dependencies:
# Input: last 20 candles (100 minutes) as a sequence
# Output: probability of next move direction
# This can capture patterns XGBoost misses (like head-and-shoulders)
```

---

## Phase 4: RAG Enhancement (Target: 60-62%)
**Effort: 2 days | Impact: MEDIUM**

### 4.1 Weighted Historical Similarity
```python
# Give MORE weight to recent similar patterns (recency bias):
similarity_score = cosine_sim * recency_weight
# Where recency_weight decays exponentially for older patterns
```

### 4.2 Regime-Filtered RAG
```python
# Only retrieve similar patterns from the SAME market regime:
# e.g., if current ATR percentile is high (volatile), only look at volatile history
results = rag.retrieve_similar(
    current_row=row,
    where_filter={"regime": "volatile", "timestamp": {"$lt": current_ts}}
)
```

### 4.3 Expand Embedding Input
```python
# Current: "RSI 65.2, MACD histogram 0.03, ..."
# Better:  Include multi-timeframe context, session info, day-of-week
text = (f"RSI {rsi:.1f}, MACD {macd:.2f}, session={get_session(ts)}, "
        f"day={ts.day_name()}, trend_15m={'up' if trend else 'down'}, "
        f"volatility_regime={'high' if vol > 0.7 else 'low'}")
```

---

## Phase 5: Confidence Filtering (Target: "Effective" 70%+)
**Effort: 1 day | Impact: GAME CHANGER**

### 5.1 Only Trade High-Confidence Signals
```python
# Instead of trading EVERY signal, only trade when model is confident:
probabilities = model.predict_proba(X)
confidence = np.max(probabilities, axis=1)

# Only act on signals with > 60% confidence
high_conf_mask = confidence > 0.60
filtered_preds = preds[high_conf_mask]
# This typically boosts effective accuracy from 55% to 65-75%!
```

### 5.2 RAG Consensus Filter
```python
# Only trade when RAG historical similarity AGREES with model prediction:
if model_says_up AND rag_win_rate > 0.6:
    execute_trade()  # High conviction
else:
    skip()  # Low conviction
```

> **THIS is how you get to "90% accuracy" in practice** — not by predicting everything,
> but by only trading when your system has very high conviction.
> A fund that trades 100 times at 52% accuracy makes less than one that trades
> 20 times at 75% accuracy.

---

## Priority Ranking (What to do first)
| Priority | Phase | Expected Gain | Effort |
|----------|-------|---------------|--------|
| 1 | Phase 5: Confidence Filtering | +15-20% effective accuracy | 1 day |
| 2 | Phase 1: Feature Engineering | +3-5% raw accuracy | 1-2 days |
| 3 | Phase 2: Smarter Targets | +2-3% raw accuracy | 1 day |
| 4 | Phase 4: RAG Enhancement | +1-3% raw accuracy | 2 days |
| 5 | Phase 3: Model Architecture | +1-3% raw accuracy | 2-3 days |

---

## Summary
- **Raw accuracy ceiling**: ~62-65% (realistic for 15m gold prediction, improved from 5m)
- **Effective accuracy with confidence filtering**: 70-80%
- **The secret**: Trade LESS, but trade BETTER
- **Timeline**: 1-2 weeks for all phases

*Updated: 2026-07-04 | Migrated to 15-min candles | Previous baseline (5m): 52.28%*
