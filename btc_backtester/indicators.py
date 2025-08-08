# indicators.py
import pandas as pd
import numpy as np

def add_basic_indicators(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)

    # SMA / EMA
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # RSI (14)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, adjust=False).mean()
    ma_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df

def sma_crossover_signal(df, short='sma_10', long='sma_50', rsi_threshold=None):
    """
    Generate a 0/1 long-only signal:
    - 1 when short SMA is above long SMA
    - Optionally require RSI to be below a threshold
    """
    s = (df[short] > df[long]).astype(int)
    if rsi_threshold is not None:
        s = s.where(df['rsi_14'] < rsi_threshold, 0)
    return s

