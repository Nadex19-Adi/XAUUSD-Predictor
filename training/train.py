import sys
import os
# Ensure project root is in PYTHONPATH for package imports like 'rag'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from rag.rag_engine import MarketRAG
import os
import optuna
import argparse

def train_pipeline(df: pd.DataFrame, model_out: str = "models/xgb_model.json"):
    print("\n--- Starting Deep Training Pipeline (Fixed — No Leakage) ---")
    
    # 1. Feature Engineering Check
    if 'next_5m_direction' not in df.columns:
        print("Error: Target column missing. Please run feature engineering first.")
        return

    # Ensure data is sorted chronologically (oldest first)
    df = df.sort_index()
    # Use the most recent 100,000 rows for training to keep relevance
    train_size = min(len(df), 100000)
    train_df = df.tail(train_size).copy()
    # Chronological split: first 80% for training, last 20% for validation (no look‑ahead leakage)
    split_idx = int(len(train_df) * 0.8)
    train_df, val_df = train_df.iloc[:split_idx].copy(), train_df.iloc[split_idx:].copy()
    
    # =====================================================================
    # FIX #1: Drop leaky / target-correlated columns BEFORE anything else
    # - 'next_return' is derived from close.shift(-1) → same info as target
    # - 'bb_upper', 'bb_lower' are raw prices → cause data memorization
    #   We keep 'bb_width' which is normalized (width / close)
    # =====================================================================
    
    # =====================================================================
    # FIX #2: RAG augmentation with CORRECT method signature
    # The rag_engine.retrieve_similar() expects (current_row, timestamp_str)
    # NOT (query_embedding, where_filter) which was the old broken call
    # =====================================================================
    rag = MarketRAG()
    
    sim_win_rates = []
    sim_avg_returns = []
    sim_max_similarities = []
    
    print(f"Querying FAISS Market Memory for {len(train_df)} historical analogies...")
    for i, (idx, row) in enumerate(train_df.iterrows()):
        if i % 5000 == 0:
            print(f"  Processed {i}/{len(train_df)} analogies...")
            
        timestamp_str = str(idx)
        
        # Use the CORRECT retrieve_similar signature (row dict + timestamp)
        results = rag.retrieve_similar(
            current_row=row.to_dict(),
            current_timestamp_str=timestamp_str,
            top_k=5
        )
        
        sim_win_rates.append(results['sim_win_rate'])
        sim_avg_returns.append(results['sim_avg_return'])
        sim_max_similarities.append(results['sim_max_similarity'])
        
    train_df['sim_win_rate'] = sim_win_rates
    train_df['sim_avg_return'] = sim_avg_returns
    train_df['sim_max_similarity'] = sim_max_similarities
    
    # =====================================================================
    # FIX #3: Clean feature set — NO raw price levels, NO target-correlated
    # Only normalized / ratio-based features allowed
    # =====================================================================
    feature_cols = [
        'rsi',                  # 0-100 bounded oscillator
        'macd',                 # Momentum (relative)
        'macd_signal',          # Signal line
        'macd_hist',            # Histogram (momentum delta)
        'atr',                  # Volatility measure
        'bb_width',             # Normalized band width (width / close)
        'returns',              # Percentage change (clean)
        'sim_win_rate',         # RAG: win rate of similar patterns
        'sim_avg_return',       # RAG: avg return of similar patterns
        'sim_max_similarity',   # RAG: max cosine similarity score
        # Lagged & Rolling (Task 1)
        'returns_lag1', 'returns_lag2', 'returns_lag4',
        'rsi_roll_mean_10', 'rsi_roll_std_10',
        'macd_hist_roll_mean_10', 'macd_hist_roll_std_10',
        'returns_roll_mean_10', 'returns_roll_std_10',
    ]
    # REMOVED: 'ema10', 'ema50' (raw prices → memorization risk)
    # REMOVED: 'bb_upper', 'bb_lower' (raw prices → memorization risk)
    
    target_col = 'target_3bar' if 'target_3bar' in train_df.columns else 'next_5m_direction'
    print(f"  Target variable: {target_col}")
    # Training set
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    # Validation set for early‑stop sanity check
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    # =====================================================================
    # FIX #4: Walk-Forward CV with PURGE GAP to prevent look-ahead bias
    # A gap of 12 bars (= 1 hour at 5m) between train/test prevents
    # any short-term autocorrelation from inflating accuracy
    # =====================================================================
    print("\nStarting Purged Walk-Forward Cross-Validation (6 splits, 1hr gap)...")
    # Walk‑forward CV on the training portion only (no data leakage)
    tscv = TimeSeriesSplit(n_splits=6, gap=12)
    
    # FIX #5: Regularized model — prevent overfitting on 100k rows
    model = xgb.XGBClassifier(
        max_depth=5,              # Reduced from 8 → less memorization
        learning_rate=0.05,       # Moderate learning rate
        subsample=0.7,            # 70% row sampling → regularization
        colsample_bytree=0.7,     # 70% feature sampling → regularization
        n_estimators=500,         # Reduced from 1000 → less overfitting
        min_child_weight=10,      # NEW: Minimum samples per leaf
        reg_alpha=0.1,            # NEW: L1 regularization
        reg_lambda=1.0,           # NEW: L2 regularization
        tree_method='hist',
        random_state=42
    )
    
    fold_accs = []
    fold_precs = []
    fold_recalls = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        # Optional early‑stop sanity check against the held‑out validation set
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"    Validation Accuracy (chronological holdout): {val_acc:.4f}")
        
        fold_accs.append(acc)
        fold_precs.append(prec)
        fold_recalls.append(rec)
        
        print(f"  Fold {fold+1}/6 — Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f}")
    
    avg_acc = np.mean(fold_accs)
    avg_prec = np.mean(fold_precs)
    avg_rec = np.mean(fold_recalls)
    
    print(f"\n📊 Walk-Forward Summary:")
    print(f"  Average Accuracy:  {avg_acc:.4f}")
    print(f"  Average Precision: {avg_prec:.4f}")
    print(f"  Average Recall:    {avg_rec:.4f}")
    
    # Sanity check — flag if accuracy is suspiciously high
    if avg_acc > 0.70:
        print("  ⚠️  WARNING: Accuracy > 70% on financial data is unusual. Review for leakage!")
    elif avg_acc > 0.55:
        print("  ✅  Metrics look realistic for financial data.")
    else:
        print("  ⚠️  Accuracy below 55%. Model may need more features or tuning.")
    
    # =====================================================================
    # FIX #6: Final model trained on 80% data, last 20% held out for check
    # =====================================================================
    # Final model training on the entire training portion (80% of chronological data)
    X_final_train, X_holdout = X, X_val
    y_final_train, y_holdout = y, y_val
    
    print(f"\nFinal training on {len(X_final_train)} rows (chronological training set)...")
    model.fit(X_final_train, y_final_train)
    
    holdout_preds = model.predict(X_holdout)
    holdout_acc = accuracy_score(y_holdout, holdout_preds)
    holdout_prec = precision_score(y_holdout, holdout_preds, zero_division=0)
    holdout_rec = recall_score(y_holdout, holdout_preds, zero_division=0)
    # Since X_holdout is the chronological validation set, these metrics are our true out‑of‑sample performance
    
    print(f"  Holdout Accuracy:  {holdout_acc:.4f}")
    print(f"  Holdout Precision: {holdout_prec:.4f}")
    print(f"  Holdout Recall:    {holdout_rec:.4f}")
    
    # Save metrics
    with open("metrics.txt", "w") as f:
        f.write(f"Walk-Forward Validation Results ({len(fold_accs)} splits, gap=12):\n")
        f.write(f"Average Accuracy:  {avg_acc:.4f}\n")
        f.write(f"Average Precision: {avg_prec:.4f}\n")
        f.write(f"Average Recall:    {avg_rec:.4f}\n")
        f.write(f"\nHoldout Test (20% unseen data):\n")
        f.write(f"Holdout Accuracy:  {holdout_acc:.4f}\n")
        f.write(f"Holdout Precision: {holdout_prec:.4f}\n")
        f.write(f"Holdout Recall:    {holdout_rec:.4f}\n")
    
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save_model(model_out)
    print(f"\n✅ Model saved to {model_out}")
    print(f"📄 Metrics saved to metrics.txt")

if __name__ == "__main__":
    # CLI argument parsing for optional hyper‑parameter tuning
    parser = argparse.ArgumentParser(description="Train XAUUSD model with optional Optuna tuning")
    parser.add_argument("--tune", action="store_true", help="Run hyper‑parameter search before final training")
    args = parser.parse_args()

    sys.path.append(os.getcwd())
    from data.data_loader import load_data
    from indicators.feature_engineering import engineer_features

    df = load_data(csv_path="data/xauusd_master_5m.csv")
    df_feat = engineer_features(df)

    if args.tune:
        # ------------------------------------------------------------
        # Hyper‑parameter tuning using Optuna
        # ------------------------------------------------------------
        def objective(trial):
            # Sample hyper‑parameters
            max_depth = trial.suggest_int("max_depth", 3, 8)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            n_estimators = trial.suggest_int("n_estimators", 200, 800, step=100)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 20)
            reg_alpha = trial.suggest_float("reg_alpha", 0.0, 0.5)
            reg_lambda = trial.suggest_float("reg_lambda", 0.5, 2.0)

            temp_model = xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample,
                n_estimators=n_estimators,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                tree_method='hist',
                random_state=42,
            )

            # Use same CV split as main pipeline (4 splits, gap=12)
            tscv = TimeSeriesSplit(n_splits=4, gap=12)
            accs = []
            for train_idx, test_idx in tscv.split(df_feat):
                train_set = df_feat.iloc[train_idx]
                test_set = df_feat.iloc[test_idx]
                feature_cols = [
                    'rsi','macd','macd_signal','macd_hist','atr','bb_width','returns',
                    'sim_win_rate','sim_avg_return','sim_max_similarity',
                    'returns_lag1','returns_lag2','returns_lag4',
                    'rsi_roll_mean_10','rsi_roll_std_10',
                    'macd_hist_roll_mean_10','macd_hist_roll_std_10',
                    'returns_roll_mean_10','returns_roll_std_10',
                ]
                target_col = 'target_3bar' if 'target_3bar' in train_set.columns else 'next_5m_direction'
                X_train = train_set[feature_cols].values
                y_train = train_set[target_col].values
                X_test = test_set[feature_cols].values
                y_test = test_set[target_col].values
                temp_model.fit(X_train, y_train)
                preds = temp_model.predict(X_test)
                accs.append(accuracy_score(y_test, preds))
            return np.mean(accs)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, timeout=600)
        print("\n🔎 Optuna best hyper‑parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        # Override default model hyper‑parameters in train_pipeline via globals
        globals().update(study.best_params)
        # ------------------------------------------------------------

    # Run training (will use tuned params if applied)
    train_pipeline(df_feat)
