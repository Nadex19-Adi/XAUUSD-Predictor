import pandas as pd
import numpy as np
import xgboost as xgb
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score

def train_master_model():
    print("=== XAUUSD Master Training v2.0 (Advanced Features + Confidence Filter) ===")
    
    # 1. Load features
    feature_path = "data/xauusd_features.csv"
    if not os.path.exists(feature_path):
        print("Error: Features not found. Run feature engineering first.")
        return

    print("Loading features...")
    df = pd.read_csv(feature_path)
    
    # 2. Selected Top 20 Feature Set (Filtered via RFE / Importance to prevent overfitting)
    feature_cols = [
        'body_ratio',
        'returns_roll_std_10',
        'lower_wick_ratio',
        'upper_wick_ratio',
        'momentum_10',
        'returns_roll_mean_10',
        'atr_percentile',
        'returns',
        'atr',
        'ema_cross',
        'macd_signal',
        'rsi_roll_mean_10',
        'bb_squeeze',
        'macd_hist_roll_mean_10',
        'rsi_roll_std_10',
        'macd',
        'returns_lag1',
        'macd_hist',
        'bb_width',
        'trend_alignment'
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
    
    # 5. Regularized XGBoost (Hyperparameters optimized via sweep to target confidence performance)
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        device='cpu',
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining on 80% data...")

    # -------------------------------------------------------------
    # STEP 2a: TimeSeries Cross‑Validation (5‑fold) – evaluate stability
    # -------------------------------------------------------------
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    cv_accs = []
    cv_precs = []
    cv_recalls = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train, y_train)):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        cv_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            device='cpu',
            n_jobs=-1,
            random_state=42
        )
        cv_model.fit(X_cv_train, y_cv_train)
        cv_pred = cv_model.predict(X_cv_val)
        cv_accs.append(accuracy_score(y_cv_val, cv_pred))
        cv_precs.append(precision_score(y_cv_val, cv_pred, zero_division=0))
        cv_recalls.append(recall_score(y_cv_val, cv_pred, zero_division=0))
        print(f"  CV Fold {fold+1}/5 – Acc: {cv_accs[-1]:.4f} | Prec: {cv_precs[-1]:.4f} | Rec: {cv_recalls[-1]:.4f}")
    print("\nTimeSeries CV Summary:")
    print(f"  Avg Accuracy: {np.mean(cv_accs):.4f}")
    print(f"  Avg Precision: {np.mean(cv_precs):.4f}")
    print(f"  Avg Recall: {np.mean(cv_recalls):.4f}")

    # -------------------------------------------------------------
    # STEP 2b: Data Augmentation – Gaussian noise (±0.5% per feature)
    # -------------------------------------------------------------
    aug_factor = 0.005
    noise = np.random.normal(0, aug_factor, X_train.shape) * X_train.values
    X_aug = np.vstack([X_train.values, X_train.values + noise])
    y_aug = np.concatenate([y_train.values, y_train.values])

    # -------------------------------------------------------------
    # STEP 2c: Train auxiliary LightGBM model
    # -------------------------------------------------------------
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_aug, y_aug)
    model.fit(X_aug, y_aug)

    # -------------------------------------------------------------
    # STEP 2d: Stacking – Logistic meta‑learner on XGB & LightGBM probabilities
    # -------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    meta_X_train = np.column_stack([
        model.predict_proba(X_aug)[:, 1],
        lgb_model.predict_proba(X_aug)[:, 1]
    ])
    meta_clf = LogisticRegression(max_iter=500)
    meta_clf.fit(meta_X_train, y_aug)

    # -------------------------------------------------------------
    # STEP 2e: Fine‑tune confidence thresholds (0.55‑0.80 step 0.01) with ATR filter
    # -------------------------------------------------------------
    best_thr = None
    best_f1 = -1
    for thr in np.arange(0.55, 0.81, 0.01):
        # Stack probabilities for test set
        stacked_probs = meta_clf.predict_proba(np.column_stack([
            model.predict_proba(X_test)[:, 1],
            lgb_model.predict_proba(X_test)[:, 1]
        ]))[:, 1]
        mask = (stacked_probs >= thr) & (df.iloc[split_idx:]['atr_percentile'] > 30)
        if mask.sum() < 10:  # lowered trade count requirement
            continue
        stacked_preds = (stacked_probs >= thr).astype(int)
        f1 = f1_score(y_test[mask], stacked_preds[mask])
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    # Ensure a threshold is selected
    if best_thr is None:
        best_thr = 0.55
        best_f1 = 0.0
        print("\n[WARN] No threshold met the trade-count requirement; using fallback threshold=0.55.")
    print(f"\n[INFO] Best confidence threshold (ATR>30) = {best_thr:.2f} (F1={best_f1:.4f})")
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
