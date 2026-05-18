from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from typing import Dict, Optional
import xgboost as xgb
import pandas as pd
import numpy as np
import psutil
from datetime import datetime
from rag.rag_engine import MarketRAG
import os

# API Security
API_KEY = os.getenv("XAUUSD_API_KEY", "gold-standard-2026")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid API Key")

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="XAUUSD Predictor API v2.1")

# Configure CORS Middleware for React Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize models
rag = None
model = None
model_features = []

@app.on_event("startup")
def load_models():
    global rag, model, model_features
    try:
        rag = MarketRAG()
        model = xgb.XGBClassifier()
        # Try JSON first, then BST fallback
        model_path = "models/xauusd_model.json"
        if not os.path.exists(model_path):
            model_path = "models/xgb_model.json"
        model.load_model(model_path)
        print(f"[API] Model loaded from {model_path}")
        
        # Load features from metadata_latest.json
        metadata_path = "models/metadata_latest.json"
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                model_features = meta.get("features", [])
            print(f"[API] Loaded {len(model_features)} active features from metadata.")
        else:
            # Fallback to hardcoded list matching train_final.py
            model_features = [
                'body_ratio', 'returns_roll_std_10', 'lower_wick_ratio', 'upper_wick_ratio',
                'momentum_10', 'returns_roll_mean_10', 'atr_percentile', 'returns', 'atr',
                'ema_cross', 'macd_signal', 'rsi_roll_mean_10', 'bb_squeeze', 'macd_hist_roll_mean_10',
                'rsi_roll_std_10', 'macd', 'returns_lag1', 'macd_hist', 'bb_width', 'trend_alignment'
            ]
            print("[API] Warning: metadata_latest.json not found. Using fallback features list.")
    except Exception as e:
        print(f"[API] Warning: Model load failed: {e}")

class PredictRequest(BaseModel):
    current_indicators: Dict[str, float] = Field(..., description="Map of indicator names to values")
    macro_snippet: str = Field("no major news", description="Optional text context")
    timestamp: Optional[str] = Field(None, description="Optional ISO timestamp")
    confidence_threshold: float = Field(0.55, ge=0.5, le=0.9, description="Minimum confidence to trade")

def normalize_to_utc_naive(ts_str: str) -> str:
    """
    Timezone Auto-Fix: Converts any timezone string (IST, EST, etc.) 
    to UTC-Naive to match the Master Database.
    """
    try:
        ts = pd.to_datetime(ts_str)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC').tz_localize(None)
        else:
            ts = ts.tz_localize(None)
        return ts.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Timezone normalization failed: {e}")
        return ts_str

@app.post("/predict")
@limiter.limit("10/minute")
def predict(request: Request, req: PredictRequest, api_key: APIKey = Depends(get_api_key)):
    if model is None or rag is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    
    indicators = req.current_indicators
    
    # Required keys (v2.0 — normalized features only)
    required = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'atr', 'bb_width', 'returns', 'close']
    for k in required:
        if k not in indicators:
            raise HTTPException(status_code=400, detail=f"Missing required indicator: {k}")
        # NaN Handling (R3 mitigation)
        if np.isnan(indicators[k]):
            print(f"[API] Warning: NaN detected in {k}. Filling with 0.0")
            indicators[k] = 0.0
            
    # Input Validation: Price > 0
    if indicators['close'] <= 0:
        raise HTTPException(status_code=400, detail="Invalid price level: close must be > 0")
            
    # Auto-Fix Timezone
    if req.timestamp:
        normalized_time = normalize_to_utc_naive(req.timestamp)
    else:
        normalized_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    # Detect current volatility regime for filtered RAG (Phase 4.2)
    atr_pct = indicators.get('atr_percentile', 0.5)
    if atr_pct > 0.7:
        regime = "high_vol"
    elif atr_pct > 0.3:
        regime = "normal_vol"
    else:
        regime = "low_vol"
    
    # Query RAG with recency weighting + regime filter (Phase 4)
    rag_res = rag.retrieve_similar(
        current_row=indicators,
        current_timestamp_str=normalized_time,
        macro_snippet=req.macro_snippet,
        top_k=5,
        regime_filter=regime,
        recency_weight=0.15
    )
    
    # Build feature vector dynamically aligned with active model features
    aligned_dict = {}
    for f in model_features:
        val = indicators.get(f, 0.0)
        # NaN and Inf handling
        if pd.isna(val) or not np.isfinite(val):
            val = 0.0
        aligned_dict[f] = val
        
    features = pd.DataFrame([aligned_dict])
    
    # Predict with explicit column names
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    direction = "UP" if pred == 1 else "DOWN"
    confidence = float(max(proba))
    
    # Phase 5: Confidence-based trade signal
    if confidence >= 0.70:
        signal_strength = "STRONG"
    elif confidence >= 0.60:
        signal_strength = "MODERATE"
    elif confidence >= req.confidence_threshold:
        signal_strength = "WEAK"
    else:
        signal_strength = "NO_TRADE"
        direction = "NEUTRAL"
    
    # TP / SL based on ATR
    current_close = indicators['close']
    atr = indicators['atr']
    if direction == "UP":
        tp = current_close + (1.5 * atr)
        sl = current_close - (1.0 * atr)
    elif direction == "DOWN":
        tp = current_close - (1.5 * atr)
        sl = current_close + (1.0 * atr)
    else:
        tp = current_close
        sl = current_close

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "signal_strength": signal_strength,
        "should_trade": signal_strength != "NO_TRADE",
        "tp": round(tp, 3),
        "sl": round(sl, 3),
        "rag_insight": {
            "sim_win_rate": rag_res['sim_win_rate'],
            "sim_avg_return": rag_res['sim_avg_return'],
            "regime_used": rag_res.get('regime_used', 'auto'),
        },
        "similar_patterns": rag_res['similar_patterns']
    }

@app.get("/health")
def health():
    ram = psutil.virtual_memory()
    return {
        "status": "ok",
        "system": {
            "ram_used_gb": round(ram.used / (1024**3), 2),
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "cpu_percent": psutil.cpu_percent(),
        },
        "models": {
            "predictor_loaded": model is not None,
            "rag_loaded": rag is not None,
            "active_shards": list(rag._collections.keys()) if rag else [],
            "version": "2.1-Production"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
