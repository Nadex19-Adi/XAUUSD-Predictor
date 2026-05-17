import pandas as pd
import numpy as np
import xgboost as xgb
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def train_master_model():
    print("=== XAUUSD Master Training v2.0 (Advanced Features + Confidence Filter) ===")
    
    # 1. Load features
    feature_path = "data/xauusd_features.csv"
    if not os.path.exists(feature_path):
        print("Error: Features not found. Run feature engineering first.")
        return

    print("Loading features...")
    df = pd.read_csv(feature_path)
    
    # 2. Advanced Feature Set (ALL normalized — no raw prices)
    feature_cols = [
        # Core oscillators
        'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr', 'bb_width', 'returns',
        # Momentum (Phase 1.1)
        'rsi_roc', 'momentum_5', 'momentum_10', 'momentum_30', 'macd_hist_roc',
        # Volume (Phase 1.1)
        'volume_ratio', 'volume_roc',
        # Multi-timeframe (Phase 1.2)
        'ema_cross', 'trend_alignment', 'bb_position',
        # Volatility regime (Phase 1.3)
        'atr_percentile', 'bb_squeeze', 'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
    ]
    
    # Check which features actually exist
    available_features = [f for f in feature_cols if f in df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        print(f"  [NOTE] Missing features (skipped): {missing}")
        print(f"  [NOTE] Re-run feature engineering to generate them.")
    
    feature_cols = available_features
    print(f"  Using {len(feature_cols)} features: {feature_cols}")
    
    # 3. Target Selection — use 3-bar (15-min) target for less noise
    target_col = 'target_3bar' if 'target_3bar' in df.columns else 'next_5m_direction'
    print(f"  Target: {target_col}")
    
    # Drop NaN
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 4. 80/20 Temporal Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    
    # 5. Regularized XGBoost
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.65,
        colsample_bytree=0.65,
        min_child_weight=15,
        reg_alpha=0.3,
        reg_lambda=2.0,
        tree_method='hist',
        device='cpu',
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining on 80% data...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    # 6. Full Evaluation
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    overfit_gap = train_acc - test_acc
    
    print(f"\n[ALL SIGNALS — {len(y_test)} trades]:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Precision:      {test_prec:.4f}")
    print(f"  Recall:         {test_rec:.4f}")
    print(f"  Overfit Gap:    {overfit_gap:.4f}")
    
    # =====================================================================
    # PHASE 5: CONFIDENCE FILTERING
    # Only count predictions where model confidence > threshold
    # =====================================================================
    print(f"\n{'='*60}")
    print("CONFIDENCE FILTERING (Phase 5 — High-Conviction Trades Only)")
    print(f"{'='*60}")
    
    confidence = np.max(test_probs, axis=1)
    
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        mask = confidence >= threshold
        n_trades = mask.sum()
        
        if n_trades < 100:
            print(f"  [{threshold:.0%} conf] Too few trades ({n_trades}), skipped.")
            continue
            
        filtered_preds = test_preds[mask]
        filtered_actual = y_test.values[mask]
        
        filtered_acc = accuracy_score(filtered_actual, filtered_preds)
        filtered_prec = precision_score(filtered_actual, filtered_preds, zero_division=0)
        trade_pct = n_trades / len(y_test) * 100
        
        print(f"  [{threshold:.0%} conf] Acc: {filtered_acc:.4f} | "
              f"Prec: {filtered_prec:.4f} | "
              f"Trades: {n_trades}/{len(y_test)} ({trade_pct:.1f}%)")
    
    # 7. Feature Importance
    print(f"\n[TOP 10 FEATURES]:")
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    for fname, fimp in feat_imp[:10]:
        bar = "#" * int(fimp * 100)
        print(f"  {fname:25s} {fimp:.4f} {bar}")
    
    # 8. Save with Versioning (Production Gap Mitigation)
    os.makedirs("models", exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M')
    model_name = f"xauusd_model_{ts}.json"
    model_path = os.path.join("models", model_name)
    model.save_model(model_path)
    
    # Save a 'latest' symlink/copy for the API
    latest_path = "models/xauusd_model.json"
    import shutil
    shutil.copy(model_path, latest_path)
    
    # Metadata artifact
    metadata = {
        "version": "2.1",
        "timestamp": ts,
        "features": feature_cols,
        "metrics": {
            "test_accuracy": float(test_acc),
            "test_precision": float(test_prec),
            "overfit_gap": float(overfit_gap)
        },
        "model_file": model_name
    }
    import json
    with open(f"models/metadata_{ts}.json", "w") as f:
        json.dump(metadata, f, indent=4)
    with open("models/metadata_latest.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Save comprehensive metrics
    with open("metrics_master.txt", "w", encoding="utf-8") as f:
        f.write(f"XAUUSD Master Model v2.0 — Training Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Target: {target_col}\n")
        f.write(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}\n\n")
        f.write(f"[ALL SIGNALS]\n")
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Test Accuracy:  {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall:    {test_rec:.4f}\n")
        f.write(f"Overfit Gap:    {overfit_gap:.4f}\n\n")
        f.write(f"[CONFIDENCE FILTERED]\n")
        for threshold in [0.55, 0.60, 0.65, 0.70]:
            mask = confidence >= threshold
            n_trades = mask.sum()
            if n_trades >= 100:
                filtered_acc = accuracy_score(y_test.values[mask], test_preds[mask])
                f.write(f"{threshold:.0%} confidence: Acc={filtered_acc:.4f}, Trades={n_trades}\n")
        f.write(f"\n[TOP FEATURES]\n")
        for fname, fimp in feat_imp[:10]:
            f.write(f"  {fname}: {fimp:.4f}\n")
        f.write(f"\nTraining Time: {(end_time - start_time)/60:.2f} minutes\n")
    
    print(f"\n[SAVED] Model: {model_path}")
    print(f"[SAVED] Metrics: metrics_master.txt")
    print(f"Training Time: {(end_time - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    train_master_model()
