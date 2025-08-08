# ml_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def create_ml_features(df):
    df = df.copy()
    # ensure indicators present
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['vol_1'] = df['volume'].pct_change().fillna(0)
    # use earlier indicators: rsi_14, macd_hist, sma_10, sma_50
    return df.dropna()

def make_label(df, horizon=3, threshold=0.002):
    # label=1 if future horizon return > threshold
    future_ret = df['close'].shift(-horizon) / df['close'] - 1
    return (future_ret > threshold).astype(int)

def train_ml_model(df):
    X = df[['ret_1','ret_3','vol_1','rsi_14','macd_hist']].copy()
    y = make_label(df, horizon=3, threshold=0.005)
    # split chronological
    split = int(len(X)*0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    # get predicted probabilities on full set
    X_full_s = scaler.transform(X)
    probs = model.predict_proba(X_full_s)[:,1]
    signals = (probs > 0.5).astype(int)

    df['ml_signal'] = signals
    return model, scaler, df
