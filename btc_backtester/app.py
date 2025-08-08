# app.py
import streamlit as st
import pandas as pd
from data_fetch import fetch_ohlcv_yfinance
from indicators import add_basic_indicators, sma_crossover_signal
from backtester import backtest_vectorized
import matplotlib.pyplot as plt

st.title("BTC Backtest App (pandas)")

symbol = st.text_input("yfinance symbol", "BTC-USD")
start = st.date_input("Start date", value=pd.to_datetime("2021-01-01"))
interval = st.selectbox("Interval", ["1d", "1h", "30m"])
short = st.slider("SMA short", 5, 100, 10)
long = st.slider("SMA long", 10, 300, 50)

if st.button("Run backtest"):
    df = fetch_ohlcv_yfinance(symbol, interval=interval, start=start.isoformat())
    df = add_basic_indicators(df)
    signal = sma_crossover_signal(df, short=f'sma_{short}' if False else 'sma_10', long=f'sma_{long}' if False else 'sma_50') 
    # (for simplicity: use add_basic_indicators outputs; in final app compute sma with chosen windows)
    st.write("Data shape:", df.shape)
    st.write(df.head())
    st.write(df.tail())
    st.write("Signal shape:", signal.shape)

    out = backtest_vectorized(df, signal)
if 'error' in out['metrics']:
    st.error(out['metrics']['error'])
else:
    st.write(out['metrics'])
    fig, ax = plt.subplots()
    out['equity'].plot(ax=ax)
    st.pyplot(fig)

    st.write(out['metrics'])
    fig, ax = plt.subplots()
    out['equity'].plot(ax=ax)
    st.pyplot(fig)
