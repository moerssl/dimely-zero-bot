import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys

# ───────────── PARAMETERS ─────────────
START_DATE   = "2025-06-01"
END_DATE     = "2025-06-10"
WINDOW       = 30      # minutes for rolling correlation
MAX_LAG      = 5       # max lag in minutes
MIN_CORR     = 0.6     # min peak correlation
LAG_RANGE    = (1, 5)  # acceptable lag window in minutes
ENTRY_MOVE   = 0.0015  # SPY 2-min move threshold (0.15%)
EXIT_PROFIT  = 0.002   # +0.2%
EXIT_STOP    = -0.0015 # −0.15%
EXIT_TIMEOUT = 15      # minutes
PRICE_CUTOFF = 50      # $50 max price for entry universe

# ───────────── TOP 50 SPX CONTRIBUTORS ─────────────
SPX_TOP50 = [
    "MSFT","NVDA","AAPL","AMZN","META","AVGO","TSLA","BRK-B","GOOG","GOOGL",
    "WMT","JPM","LLY","V","ORCL","NFLX","MA","XOM","COST","PG",
    "JNJ","HD","BAC","PLTR","ABBV","KO","PM","UNH","IBM","CSCO",
    "GE","TMUS","CRM","WFC","CVX","ABT","AMD","MS","LIN","AXP",
    "DIS","INTU","MCD","NOW","T","GS","MRK","UBER","RTX","ISRG"
]

def exit_with(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)

# ───────────── UTILITIES ─────────────
def get_current_prices(tickers):
    prices = {}
    try:
        data = yf.Tickers(" ".join(tickers))
    except Exception as e:
        exit_with(f"Failed to initialize yfinance Tickers: {e}")
    for t in tickers:
        try:
            info = data.tickers[t].info
            price = info.get("regularMarketPrice") or info.get("previousClose")
            if price is None:
                print(f"Warning: no price for {t}, skipping.", file=sys.stderr)
            else:
                prices[t] = price
        except Exception as e:
            print(f"Warning: error fetching price for {t}: {e}", file=sys.stderr)
    return prices

def download_1m_data(tickers, start, end):
    out = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, interval="1m",
                             progress=False)
        except Exception as e:
            print(f"Warning: download failed for {t}: {e}", file=sys.stderr)
            continue
        if df.empty:
            print(f"Warning: no data for {t} in {start}–{end}.", file=sys.stderr)
            continue
        df.index = pd.to_datetime(df.index)
        out[t] = df[['Open','High','Low','Close','Volume']].copy()
    return out

def rolling_lag_correlation(spy_close, stock_close, window, max_lag):
    if len(spy_close) < window or len(stock_close) < window + max_lag:
        return pd.Series([], dtype=float), pd.Series([], dtype=float)
    lags = range(0, max_lag+1)
    times = spy_close.index[window:]
    corrs, lagged = [], []
    for t in times:
        spy_w = spy_close.loc[t - timedelta(minutes=window):t].values
        best = (-1, None)
        for lag in lags:
            stock_w = stock_close.shift(lag).loc[t - timedelta(minutes=window):t].values
            if len(stock_w)==len(spy_w):
                c = np.corrcoef(spy_w, stock_w)[0,1]
                if c > best[0]:
                    best = (c, lag)
        corrs.append(best[0])
        lagged.append(best[1])
    return pd.Series(corrs, index=times), pd.Series(lagged, index=times)

def screen_laggers(spy_df, stock_dfs):
    keep = []
    spy_close = spy_df['Close']
    for sym, df in stock_dfs.items():
        corr_s, lag_s = rolling_lag_correlation(spy_close, df['Close'], WINDOW, MAX_LAG)
        if corr_s.empty:
            print(f"Warning: insufficient data to corr-test {sym}.", file=sys.stderr)
            continue
        peak_t = corr_s.idxmax()
        if (corr_s.loc[peak_t] >= MIN_CORR and
            LAG_RANGE[0] <= lag_s.loc[peak_t] <= LAG_RANGE[1]):
            keep.append(sym)
    return keep

def backtest_spy_lead(spy_df, stock_dfs, universe):
    trades = []
    spy_ret = spy_df['Close'].pct_change()
    signal = (spy_ret + spy_ret.shift(-1)) >= ENTRY_MOVE
    if signal.sum() == 0:
        print("No SPY triggers in period.", file=sys.stderr)
    for t in signal[signal].index:
        for sym in universe:
            df = stock_dfs.get(sym)
            if df is None or t not in df.index:
                continue
            pos = df.index.get_loc(t) + 1
            if pos >= len(df):
                continue
            t_entry = df.index[pos]
            price_entry = df.at[t_entry, 'Open']
            future = df.loc[t_entry:t_entry + timedelta(minutes=EXIT_TIMEOUT)]
            fut_ret = future['Close'] / price_entry - 1
            profit_hits = fut_ret[fut_ret >= EXIT_PROFIT]
            stop_hits   = fut_ret[fut_ret <= EXIT_STOP]
            times = []
            if not profit_hits.empty:
                times.append(profit_hits.index[0])
            if not stop_hits.empty:
                times.append(stop_hits.index[0])
            t_exit = min(times) if times else future.index[-1]
            price_exit = future.at[t_exit, 'Close']
            trades.append({
                'symbol': sym,
                'entry_time': t_entry,
                'exit_time': t_exit,
                'entry_price': price_entry,
                'exit_price': price_exit,
                'return': price_exit/price_entry - 1
            })
    return pd.DataFrame(trades)

# ───────────── MAIN WORKFLOW ─────────────
if __name__ == "__main__":
    # 1) Filter SPX top50 by current price < $50
    prices = get_current_prices(SPX_TOP50)
    under50 = [t for t, p in prices.items() if p < PRICE_CUTOFF]
    if not under50:
        exit_with(f"No top-50 SPX names trade under ${PRICE_CUTOFF}.")
    print(f"Tickers under ${PRICE_CUTOFF}: {under50}\n")

    # 2) Download 1-min data for SPY + filtered tickers
    symbols = ["SPY"] + under50
    data = download_1m_data(symbols, START_DATE, END_DATE)
    if "SPY" not in data:
        exit_with("Missing SPY data; cannot proceed.")
    spy_df = data.pop("SPY")
    if not data:
        exit_with("No stock data downloaded; cannot backtest.")

    # 3) Screen for laggers
    laggers = screen_laggers(spy_df, data)
    if not laggers:
        exit_with("No laggers found; adjust parameters or data window.")
    print("Laggers selected for backtest:", laggers, "\n")

    # 4) Backtest strategy
    results = backtest_spy_lead(spy_df, data, laggers)
    if results.empty:
        exit_with("Strategy generated no trades in period.")

    # 5) Metrics
    wins = results['return'] > 0
    win_rate = wins.mean()
    avg_ret  = results['return'].mean()
    sharpe   = avg_ret / results['return'].std() * np.sqrt(252*6.5*60)
    cum = results['return'].cumsum()
    max_dd = (cum - cum.cummax()).min()

    print(f"Trades: {len(results)}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Avg return: {avg_ret:.4f}")
    print(f"Sharpe (annualized): {sharpe:.2f}")
    print(f"Max drawdown: {max_dd:.4f}")
