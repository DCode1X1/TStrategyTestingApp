# data_fetch.py
import pandas as pd
import ccxt
import time
import yfinance as yf

def fetch_ohlcv_binance(symbol='BTC/USDT', timeframe='1h', since=None, limit=1000):
    ex = ccxt.binance({'enableRateLimit': True})
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def fetch_ohlcv_yfinance(symbol='BTC-USD', interval='1h', start='2020-01-01', end=None):
    df = yf.download(symbol, interval=interval, start=start, end=end, progress=False)
    # yfinance returns columns ['Open','High','Low','Close','Volume']
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'})
    df.index = pd.to_datetime(df.index)
    return df[['open','high','low','close','volume']]
