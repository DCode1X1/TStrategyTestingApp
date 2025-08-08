# optimize.py
import optuna
from indicators import add_basic_indicators
from backtester import backtest_vectorized

def objective(trial, df):
    short = trial.suggest_int('short', 5, 50)
    long = trial.suggest_int('long', 20, 200)
    if short >= long:
        return -999
    df2 = df.copy()
    df2['sma_short'] = df2['close'].rolling(short).mean()
    df2['sma_long'] = df2['close'].rolling(long).mean()
    signal = (df2['sma_short'] > df2['sma_long']).astype(int)
    out = backtest_vectorized(df2, signal)
    return float(out['metrics']['annualized_sharpe'] or -999)

def run_optuna(df, n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, df), n_trials=n_trials)
    return study
