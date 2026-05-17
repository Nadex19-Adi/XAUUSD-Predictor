# ✅ XAUUSD Predictor – Accuracy‑Boost To‑Do List

## 🎯 Goal
Raise overall test accuracy and high‑confidence precision of the XAUUSD direction model.

---

### 📅 Milestones & Tasks

| # | Task | Details | Owner | Status |
|---|------|---------|-------|--------|
| 1 | **Add lagged & rolling features** | Create 1‑, 2‑, 4‑bar returns, rolling mean/std/skew for each indicator in `data/xauusd_features.csv`. | You | ✅ Completed (Adds 3-bar shifts, rolling mean/std of RSI, returns, macd hist) |
| 2 | **Feature selection via RFE / SHAP** | Run Recursive Feature Elimination or SHAP on a validation split, keep top 15‑20 features. | You | ✅ Completed (Reduced 31 features to Top 20 using XGBoost feature importance ranking) |
| 3 | **Switch target to 3‑bar direction** | Update `target_col` logic (`target_3bar`) and re‑run training. | You | ✅ Completed (Integrated target_3bar in feature engineering and training scripts) |
| 4 | **Hyper‑parameter sweep** | Use `optuna` or `scikit‑optimize` to tune `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `reg_alpha`, `reg_lambda`. Include early stopping. | You | ✅ Completed (Custom parameter sweep optimized depth, subsample, and regularization parameters) |
| 5 | **Time‑Series Cross‑validation** | Replace the single 80/20 split with `TimeSeriesSplit(n_splits=5)`. Track mean validation accuracy. | You | ⬜ Not started |
| 6 | **Train an auxiliary model (LightGBM)** | Train a LightGBM classifier with the same features, then stack predictions via a logistic meta‑learner. | You | ⬜ Not started |
| 7 | **Data augmentation** | Add Gaussian noise (±0.5 % of each feature) to create synthetic rows, append to training set. | You | ⬜ Not started |
| 8 | **Fine‑tune confidence thresholds** | Grid‑search thresholds 0.55‑0.80 (step 0.01) to maximize F1‑score / risk‑adjusted return. | You | ⬜ Not started |
| 9 | **Volatility filter** | Only accept predictions when `ATR percentile > 30`. Combine with confidence filter. | You | ⬜ Not started |
|10 | **Document results** | After each experiment, log metrics to `metrics_master.txt` and update the report (run `generate_report.py`). | You | ✅ Completed (Logged results to metrics_master.txt and compiled them in skill report format.docx) |

---

## 📌 How to Use This List
1. Open the file `todo_boost_accuracy.md` in the project root.
2. Tick a box (✅) when a task is completed, or add notes.
3. Ask the assistant to generate code, run experiments, or update the report.

---

### 🔄 Follow‑up Workflow
When you finish a task, tell me “Task X done” and I’ll:
- Mark it as ✅ in this file.
- Suggest the next step or required code changes.
- Keep the list synchronized.

Let’s get started – which task would you like to tackle first?
