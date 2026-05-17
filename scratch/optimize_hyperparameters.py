import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.metrics import accuracy_score, precision_score

sys.path.append(os.getcwd())

print("=== Starting Advanced Feature Selection & Hyper-parameter Tuning ===")

# 1. Load data
feature_path = "data/xauusd_features.csv"
if not os.path.exists(feature_path):
    print("Error: Features CSV not found.")
    sys.exit(1)

df = pd.read_csv(feature_path)

# Core features we created (31 features)
feature_cols = [
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr', 'bb_width', 'returns',
    'rsi_roc', 'momentum_5', 'momentum_10', 'momentum_30', 'macd_hist_roc',
    'volume_ratio', 'volume_roc', 'ema_cross', 'trend_alignment', 'bb_position',
    'atr_percentile', 'bb_squeeze', 'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
    'returns_lag1', 'returns_lag2', 'returns_lag4',
    'rsi_roll_mean_10', 'rsi_roll_std_10',
    'macd_hist_roll_mean_10', 'macd_hist_roll_std_10',
    'returns_roll_mean_10', 'returns_roll_std_10',
]

target_col = 'target_3bar' if 'target_3bar' in df.columns else 'next_5m_direction'

# Clean data
df = df.dropna(subset=feature_cols + [target_col])
print(f"Data shape after cleaning: {df.shape}")

# Limit training rows to last 120,000 rows for tuning speed and relevance
tuning_df = df.tail(120000).copy()

X = tuning_df[feature_cols]
y = tuning_df[target_col]

# Split into train (80%) and validation (20%)
split_idx = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

# -------------------------------------------------------------
# STEP 1: Feature Selection via Initial XGBoost Importance
# -------------------------------------------------------------
print("\n--- Step 1: Feature Selection ---")
initial_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    tree_method='hist',
    random_state=42
)
initial_model.fit(X_train, y_train)

# Get feature importance
importance = initial_model.feature_importances_
feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)

# Keep top 20 features
top_n = 20
selected_features = [fname for fname, fimp in feat_imp[:top_n]]
print(f"Selected Top {top_n} Features:")
for i, (fname, fimp) in enumerate(feat_imp[:top_n]):
    print(f"  {i+1}. {fname:25s} Importance: {fimp:.4f}")

# Filter datasets
X_train_sel = X_train[selected_features]
X_val_sel = X_val[selected_features]

# -------------------------------------------------------------
# STEP 2: Custom Hyper-parameter Sweep (Targeting 70% Confidence Precision)
# -------------------------------------------------------------
print("\n--- Step 2: Custom Hyper-parameter Sweep ---")

# Define grid combinations to evaluate
grid = [
    {"max_depth": 4, "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 10, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "min_child_weight": 15, "reg_alpha": 0.3, "reg_lambda": 2.0},
    {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.6, "colsample_bytree": 0.7, "min_child_weight": 15, "reg_alpha": 0.5, "reg_lambda": 3.0},
    {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.8, "min_child_weight": 12, "reg_alpha": 0.2, "reg_lambda": 1.5},
    {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.65, "colsample_bytree": 0.65, "min_child_weight": 20, "reg_alpha": 0.5, "reg_lambda": 5.0},
]

best_score = 0.0
best_params = None
results_log = []

for idx, params in enumerate(grid):
    print(f"Evaluating Config {idx+1}/{len(grid)}: depth={params['max_depth']}, lr={params['learning_rate']}, sub={params['subsample']}, col={params['colsample_bytree']}")
    
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_sel, y_train)
    
    # Evaluate
    preds = model.predict(X_val_sel)
    probs = model.predict_proba(X_val_sel)
    confidence = np.max(probs, axis=1)
    
    # 70% Confidence precision
    mask_70 = confidence >= 0.70
    n_trades = mask_70.sum()
    
    if n_trades >= 100:
        conf_acc = accuracy_score(y_val.values[mask_70], preds[mask_70])
        trade_rate = n_trades / len(y_val)
    else:
        conf_acc = 0.0
        trade_rate = 0.0
        
    base_acc = accuracy_score(y_val, preds)
    print(f"  -> Base Acc: {base_acc:.4f} | 70% Conf Acc: {conf_acc:.4f} ({n_trades} trades)")
    
    results_log.append({
        "params": params,
        "base_acc": base_acc,
        "conf_acc": conf_acc,
        "n_trades": n_trades
    })
    
    # We optimize for a combination of baseline accuracy and high-confidence precision
    score = conf_acc * 0.7 + base_acc * 0.3
    if score > best_score and conf_acc > 0.0:
        best_score = score
        best_params = params

print(f"\nBest Parameters Found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# -------------------------------------------------------------
# STEP 3: Retrain and save final model using best features & params
# -------------------------------------------------------------
print("\n--- Step 3: Retraining Final Optimized Model ---")

# We apply this to the full dataset (150,000 recent rows)
df_full = df.tail(150000).copy()
X_full = df_full[selected_features]
y_full = df_full[target_col]

split_idx_full = int(len(X_full) * 0.8)
X_train_final, X_test_final = X_full.iloc[:split_idx_full], X_full.iloc[split_idx_full:]
y_train_final, y_test_final = y_full.iloc[:split_idx_full], y_full.iloc[split_idx_full:]

final_model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    min_child_weight=best_params['min_child_weight'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_final, y_train_final)

# Evaluate final model
train_preds = final_model.predict(X_train_final)
test_preds = final_model.predict(X_test_final)
test_probs = final_model.predict_proba(X_test_final)

train_acc = accuracy_score(y_train_final, train_preds)
test_acc = accuracy_score(y_test_final, test_preds)
test_prec = precision_score(y_test_final, test_preds, zero_division=0)
overfit_gap = train_acc - test_acc

print(f"\nRetrained Model Performance:")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print(f"  Overfit Gap:    {overfit_gap:.4f}")

# Write to metrics_master.txt
with open("metrics_master.txt", "w", encoding="utf-8") as f:
    f.write("XAUUSD Master Model v2.2 — Optimized Training Results\n")
    f.write("==================================================\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Selected Features: {len(selected_features)} (Filtered from 31 via RFE/Importance)\n")
    f.write(f"Target: {target_col}\n")
    f.write(f"Train Size: {len(X_train_final)} | Test Size: {len(X_test_final)}\n\n")
    f.write("[ALL SIGNALS]\n")
    f.write(f"Train Accuracy: {train_acc:.4f}\n")
    f.write(f"Test Accuracy:  {test_acc:.4f}\n")
    f.write(f"Test Precision: {test_prec:.4f}\n")
    f.write(f"Overfit Gap:    {overfit_gap:.4f}\n\n")
    f.write("[CONFIDENCE FILTERED]\n")
    
    confidence = np.max(test_probs, axis=1)
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        mask = confidence >= threshold
        n_trades = mask.sum()
        if n_trades >= 100:
            filtered_acc = accuracy_score(y_test_final.values[mask], test_preds[mask])
            filtered_prec = precision_score(y_test_final.values[mask], test_preds[mask], zero_division=0)
            f.write(f"{threshold:.0%} confidence: Acc={filtered_acc:.4f}, Prec={filtered_prec:.4f}, Trades={n_trades}\n")
            print(f"  [{threshold:.0%} confidence] Acc: {filtered_acc:.4f} | Prec: {filtered_prec:.4f} | Trades: {n_trades}")

    f.write("\n[SELECTED FEATURES]\n")
    for i, fname in enumerate(selected_features[:10]):
        f.write(f"  {i+1}. {fname}\n")
        
    f.write("\n[OPTIMIZED HYPERPARAMETERS]\n")
    for k, v in best_params.items():
        f.write(f"  {k}: {v}\n")

# Save model to disk
os.makedirs("models", exist_ok=True)
final_model.save_model("models/xauusd_model.json")
print("\n[OK] Saved optimized model to models/xauusd_model.json")
print("[SAVED] Saved updated results to metrics_master.txt")
