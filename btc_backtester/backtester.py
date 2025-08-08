import numpy as np
import pandas as pd

def infer_periods_per_year(df):
    freq_s = df.index.to_series().diff().dt.total_seconds().median()
    if np.isnan(freq_s) or freq_s == 0:
        return 365
    return 365 * 24 * 3600 / freq_s

def backtest_vectorized(df, signal, init_capital=10000, fee=0.00075, slippage=0.0005):
    """
    Vectorized long-only backtest.
    """
    # --- SAFETY CHECKS ---
    if df is None or df.empty:
        return {
            'equity': pd.Series(dtype=float),
            'strategy_returns': pd.Series(dtype=float),
            'metrics': {'error': 'No price data provided.'},
            'trades': pd.DataFrame()
        }
    if 'close' not in df.columns:
        return {
            'equity': pd.Series(dtype=float),
            'strategy_returns': pd.Series(dtype=float),
            'metrics': {'error': 'No close column in dataframe.'},
            'trades': pd.DataFrame()
        }

    signal = signal.reindex(df.index).fillna(0).astype(int)
    if len(df) < 2:
        return {
            'equity': pd.Series(dtype=float),
            'strategy_returns': pd.Series(dtype=float),
            'metrics': {'error': 'Not enough rows to run backtest.'},
            'trades': pd.DataFrame()
        }

    # Avoid lookahead
    positions = signal.shift(1).fillna(0)

    # Returns
    ret = df['close'].pct_change().fillna(0)
    strat_ret = positions * ret

    # Costs
    trades = positions.diff().abs().fillna(0)
    cost_per_trade = fee + slippage
    strat_ret = strat_ret - trades * cost_per_trade

    equity = (1 + strat_ret).cumprod() * init_capital
    if equity.empty:
        return {
            'equity': pd.Series(dtype=float),
            'strategy_returns': pd.Series(dtype=float),
            'metrics': {'error': 'Equity curve is empty.'},
            'trades': pd.DataFrame()
        }

    # Metrics
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = (df.index[-1] - df.index[0]).total_seconds() / (365 * 24 * 3600)
    CAGR = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    periods_per_year = infer_periods_per_year(df)
    ann_sharpe = (strat_ret.mean() / (strat_ret.std() + 1e-9)) * np.sqrt(periods_per_year)

    cum = (1 + strat_ret).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    trades_log = []
    pos = 0
    entry_price = None
    entry_time = None
    for t, p in positions.iteritems():
        price = df.at[t, 'close']
        if p == 1 and pos == 0:
            pos = 1
            entry_price = price
            entry_time = t
        elif p == 0 and pos == 1:
            pos = 0
            exit_price = price
            exit_time = t
            ret_trade = exit_price / entry_price - 1
            trades_log.append({
                'entry_time': entry_time, 'exit_time': exit_time,
                'entry_price': entry_price, 'exit_price': exit_price,
                'return': ret_trade
            })

    if pos == 1:
        exit_price = df['close'].iloc[-1]
        exit_time = df.index[-1]
        ret_trade = exit_price / entry_price - 1
        trades_log.append({
            'entry_time': entry_time, 'exit_time': exit_time,
            'entry_price': entry_price, 'exit_price': exit_price,
            'return': ret_trade
        })

    trades_df = pd.DataFrame(trades_log)

    metrics = {
        'total_return': total_return,
        'CAGR': CAGR,
        'annualized_sharpe': ann_sharpe,
        'max_drawdown': max_dd,
        'trades': len(trades_df),
        'win_rate': (trades_df['return'] > 0).mean() if len(trades_df) else np.nan,
        'avg_trade_return': trades_df['return'].mean() if len(trades_df) else np.nan
    }

    return {
        'equity': equity,
        'strategy_returns': strat_ret,
        'metrics': metrics,
        'trades': trades_df
    }
