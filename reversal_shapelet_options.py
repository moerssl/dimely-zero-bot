#!/usr/bin/env python3
import sys, json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

def load_and_clean(path):
    # 1) load + parse UNIX‐sec “date” → timestamp
    raw = json.load(open(path))
    df  = pd.DataFrame(raw)
    df['timestamp'] = pd.to_datetime(df['date'].astype(int), unit='s')
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 2) drop unwanted cols if present
    drop = ['trend','final_signal','tech_signal','strategy','sentiment']
    df = df.drop(columns=[c for c in drop if c in df.columns])
    return df

def add_technical_indicators(df):
    close = df['close']
    high  = df.get('high', close)
    low   = df.get('low',  close)
    vol   = df.get('volume', None)

    # moving averages
    for w in (5,10,20):
        df[f'SMA_{w}'] = close.rolling(w).mean()
        df[f'EMA_{w}'] = close.ewm(span=w, adjust=False).mean()

    # RSI(14)
    delta = close.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    ema_up   = up.ewm(alpha=1/14, adjust=False).mean()
    ema_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = ema_up/ema_down
    df['RSI_14'] = 100 - 100/(1+rs)

    # Bollinger‐band width (20,±2σ)
    m20 = close.rolling(20).mean()
    s20 = close.rolling(20).std()
    df['BB_width'] = (m20 + 2*s20) - (m20 - 2*s20)

    # ATR(14)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    TR  = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    df['ATR_14'] = TR.rolling(14).mean()

    # OBV
    if vol is not None:
        df['OBV'] = (np.sign(close.diff()) * vol).fillna(0).cumsum()

    return df

def label_reversals(df, H=10, delta=0.005):
    """Label a bar t=1 if it’s a local high+roll or local low+pop."""
    n = len(df)
    call = np.zeros(n, dtype=int)
    put  = np.zeros(n, dtype=int)
    p = df['close'].values
    for t in range(H, n-H):
        past   = (p[t]   - p[t-H]) / p[t-H]
        future = (p[t+H] - p[t])   / p[t]
        if  past> delta and future< -delta:
            call[t] = 1
        elif past< -delta and future>  delta:
            put[t]  = 1
    df['call_signal'] = call
    df['put_signal']  = put
    return df

def train_tree_and_extract(df, features, target):
    # drop any NA rows
    df2 = df.dropna(subset=features + [target])
    X, y = df2[features], df2[target]
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05)
    clf.fit(X,y)
    rules = export_text(clf, feature_names=features)
    imps  = sorted(zip(features, clf.feature_importances_),
                   key=lambda x:x[1], reverse=True)
    return rules, imps

def main(path):
    df = load_and_clean(path)
    df = add_technical_indicators(df)
    df = label_reversals(df)

    # pick all indicator columns
    forbid = {'date','timestamp','open','high','low','close','volume','call_signal','put_signal', "call"}
    features = [c for c in df.columns if c not in forbid]

    print(f"\nCandidate indicators ({len(features)}): {features}\n")

    # 1) Sell‐call tree
    print("🔹 Rules to SELL CALLS (price topped + rolling):")
    call_rules, call_imps = train_tree_and_extract(df, features, 'call_signal')
    print(call_rules)
    print("Feature importances:", call_imps, "\n")

    # 2) Sell‐put tree
    print("🔹 Rules to SELL PUTS (price bottomed + popping):")
    put_rules, put_imps = train_tree_and_extract(df, features, 'put_signal')
    print(put_rules)
    print("Feature importances:", put_imps, "\n")

if __name__=='__main__':
    file = "spy.json"
    main(file)
