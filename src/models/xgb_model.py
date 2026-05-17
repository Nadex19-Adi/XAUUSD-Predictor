import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from src.core.config import settings

class XGBPredictor:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.1
        )
        self.feature_cols = [
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr', 'ema10', 'ema50',
            'bb_upper', 'bb_lower', 'bb_width', 'returns',
            'sim_win_rate', 'sim_avg_return', 'sim_max_similarity'
        ]

    def train_walk_forward(self, df: pd.DataFrame, n_splits: int = 5):
        """
        Trains the model using Purged Walk-Forward validation.
        """
        X = df[self.feature_cols].values
        y = df['next_5m_direction'].values
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        accuracies = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            accuracies.append(acc)
            
        print(f"Mean walk-forward accuracy: {np.mean(accuracies):.4f}")
        return np.mean(accuracies)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features[self.feature_cols])
        
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features[self.feature_cols])
        
    def save(self):
        self.model.save_model(settings.XGB_MODEL_PATH)
        
    def load(self):
        self.model.load_model(settings.XGB_MODEL_PATH)
