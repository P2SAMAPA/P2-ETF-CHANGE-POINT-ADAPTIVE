"""
Lightweight predictive model (Ridge regression with built-in CV).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import config


def create_features(returns: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Create lagged return features for a given ETF.
    """
    col = f"{ticker}_ret"
    if col not in returns.columns:
        return pd.DataFrame()
    series = returns[col]
    features = pd.DataFrame(index=series.index)
    for lag in config.LOOKBACK_FEATURES:
        features[f"lag_{lag}"] = series.shift(lag)
    return features.dropna()


def prepare_training_data(returns: pd.DataFrame, tickers: list, start_date: pd.Timestamp, end_date: pd.Timestamp):
    """
    Build feature matrix X and target vector y for all ETFs over the given window.
    Target is next-day return.
    """
    mask = (returns.index >= start_date) & (returns.index <= end_date)
    window_returns = returns.loc[mask]

    X_list, y_list = [], []
    for ticker in tickers:
        col = f"{ticker}_ret"
        if col not in window_returns.columns:
            continue
        features = create_features(window_returns, ticker)
        if features.empty:
            continue
        target = window_returns[col].shift(-1).loc[features.index]
        valid = target.notna()
        if valid.sum() < 10:
            continue
        X_list.append(features[valid])
        y_list.append(target[valid])

    if not X_list:
        return None, None, None
    X = pd.concat(X_list)
    y = pd.concat(y_list)
    return X, y, X.columns.tolist()


class AdaptiveRidgeModel:
    """Ridge regression model with feature scaling and cross-validated alpha."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)

    def predict_next_returns(self, returns: pd.DataFrame, tickers: list, current_date: pd.Timestamp) -> dict:
        """
        Predict next-day return for each ETF given data up to current_date.
        """
        predictions = {}
        for ticker in tickers:
            col = f"{ticker}_ret"
            if col not in returns.columns:
                continue
            features = create_features(returns.loc[:current_date], ticker)
            if features.empty:
                continue
            latest = features.iloc[-1:]
            if latest.isna().any().any():
                continue
            try:
                pred = self.predict(latest)[0]
                predictions[ticker] = pred
            except:
                continue
        return predictions
