import math
import numpy as np
import pandas as pd
import talib
from scipy.stats import norm
import pytz
from scipy.signal import argrelextrema

from util.config import CONFIG
from util.StrategyEnum import StrategyEnum

# 1. Datetime parsing
def parse_datetime(df, timezone="Europe/Berlin"):
    #df["datetime"] = pd.to_datetime(df["date"], format="%Y%m%d %H:%M:%S %Z")

    # Assuming df["date"] contains the epoch data
    # Step 1: Convert epoch to datetime
    try:
        df["datetime"] = pd.to_datetime(df["date"], unit="s", utc=True)

        # Step 2: Convert to New York time
        new_york_tz = pytz.timezone("America/New_York")
        df["datetime"] = df["datetime"].dt.tz_convert(new_york_tz)
        """

        # Optional: If you want the datetime column as a string without timezone info
        # df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["time"] = df["hour"] * 60 + ":"+ df["minute"]"
        """
    except Exception as e:
        print("Error parsing datetime: ", e)

    return df

# 2. Calculate TA indicators using TA-Lib
def calc_ta_indicators(df, cfg):
    # Convert columns once to float arrays
    open_arr = df["open"].values.astype(float)
    high_arr = df["high"].values.astype(float)
    low_arr = df["low"].values.astype(float)
    close_arr = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float)
    
    df["SMA5"] = talib.SMA(close_arr, timeperiod=cfg["ta"]["sma_timeperiod"])
    df["EMA5"] = talib.EMA(close_arr, timeperiod=5)
    df["EMA2"] = talib.EMA(close_arr, timeperiod=2)
    df["EMA8"] = talib.EMA(close_arr, timeperiod=8)
    df["EMA13"] = talib.EMA(close_arr, timeperiod=13)
    df["EMA20"] = talib.EMA(close_arr, timeperiod=20)
    df["EMA50"] = talib.EMA(close_arr, timeperiod=50)

    df["RSI"] = talib.RSI(close_arr, timeperiod=cfg["ta"]["rsi_timeperiod"])
    df["STOCH_K"], df["STOCH_d"] = talib.STOCH(high_arr,low_arr,close_arr)
    
    macd, macdsignal, macdhist = talib.MACD(
        close_arr,
        fastperiod=cfg["ta"]["macd"]["fastperiod"],
        slowperiod=cfg["ta"]["macd"]["slowperiod"],
        signalperiod=cfg["ta"]["macd"]["signalperiod"]
    )
    df["MACD"] = macd
    df["MACDSignal"] = macdsignal
    df["MACDHist"] = macdhist

    upperband, middleband, lowerband = talib.BBANDS(
        close_arr,
        timeperiod=cfg["ta"]["bbands"]["timeperiod"],
        nbdevup=cfg["ta"]["bbands"]["nbdevup"],
        nbdevdn=cfg["ta"]["bbands"]["nbdevdn"],
        matype=cfg["ta"]["bbands"]["matype"],
        
    )
    df["bb_up"] = np.round(upperband, 2)
    df["bb_mid"] = np.round(middleband, 2)
    df["bb_low"] = np.round(lowerband, 2)
    
    df["doji"] = talib.CDLDOJI(open_arr, high_arr, low_arr, close_arr)
    df["hammer"] = talib.CDLHAMMER(open_arr, high_arr, low_arr, close_arr)
    df["engulfing"] = talib.CDLENGULFING(open_arr, high_arr, low_arr, close_arr)
    df["adx"] = talib.ADX(high_arr, low_arr, close_arr, timeperiod=cfg["ta"]["adx_timeperiod"])
    df["PLUS_DI"] = talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=cfg["ta"]["adx_timeperiod"])
    df["MINUS_DI"] = talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=cfg["ta"]["adx_timeperiod"])

    
    # Create a new column for the gap
    df['night_gap'] = (df['open'] - df['close'].shift(1)).round(2)

    # VWAP (rolling)
    #df["VWAP"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    # use variables defined above
    df["VWAP"] = (volume * (high_arr + low_arr + close_arr) / 3).cumsum() / volume.cumsum()

    # Dynamic RSI thresholds based on recent extremes
    df["rsi_dynamic_thresh_high"] = df["RSI"].rolling(window=10).max() * 0.95
    df["rsi_dynamic_thresh_low"] = df["RSI"].rolling(window=10).min() * 1.05



    # Improved volume spike detection
    df["vol_spike"] = (df["volume"] > df["volume"].rolling(window=20).mean() * 2).astype(int)

    # Replace NaN IV values with the last valid observation
    df["IV_close"] = df["IV_close"].fillna(method="ffill")

    # If NaN persists (e.g., beginning of the dataset), replace it with the rolling mean
    df["IV_close"] = df["IV_close"].fillna(df["IV_close"].rolling(window=10, min_periods=1).mean())

    df["IV_rank"] = (df["IV_close"] - df["IV_close"].rolling(window=50).min()) / \
                    (df["IV_close"].rolling(window=50).max() - df["IV_close"].rolling(window=50).min())

    
    return df

# 3. Compute ATR and volatility-based measures
def calc_atr_and_volatility(df, cfg):
    close_arr = df["close"].values.astype(float)
    high_arr = df["high"].values.astype(float)
    low_arr = df["low"].values.astype(float)
    
    atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=cfg["ta"]["atr_timeperiod"])
    atr_percent = (atr / close_arr) * 100
    df["ATR"] = atr
    df["ATR_percent"] = atr_percent
        # Adjusted ATR scaling based on historical quartile
    df["atr_adjusted"] = df["ATR_percent"].rolling(window=20).quantile(0.25)
    return df, atr, atr_percent





def calc_technical_signal(df, atr, cfg):
    """
    Compute tech signals:
      - SellCall on bearish setups (price falling)
      - SellPut  on bullish setups (price rising)
      - SellIronCondor on low-vol neutrality/squeeze 
    Tie-breaking uses “method 2”: pick the highest vote ≥ its own threshold.
    Assumes numpy (np), pandas (pd), talib and StrategyEnum are in scope.
    """
    # 1) Cast series
    high    = df["high"].astype(float)
    low     = df["low"].astype(float)
    close   = df["close"].astype(float)
    rsi     = df["RSI"].astype(float)
    atr_pct = df["ATR_percent"].astype(float)
    adx     = df["adx"].astype(float)
    pdi     = df["PLUS_DI"].astype(float)
    mdi     = df["MINUS_DI"].astype(float)
    vwap    = df["VWAP"].astype(float)
    macdh   = df["MACDHist"].astype(float)
    ema5    = df["EMA5"].astype(float)
    ema20   = df["EMA20"].astype(float)
    vol     = df["volume"].astype(float)

    upper   = df["bb_up"]
    lower   = df["bb_low"]
    middle  = df["bb_mid"]

    # 2) Config shortcuts
    sig = cfg["signal"]
    lo, hi     = sig["rsi_oversold"],   sig["rsi_overbought"]
    ct, pt, it = sig["call_thresh"],    sig["put_thresh"],    sig["iron_condor_thresh"]

    # 3) RSI cross-inside events
    rsi_bull = ((rsi.shift(1) <  lo) & (rsi >= lo)).astype(int)  # crosses up into >30
    rsi_bear = ((rsi.shift(1) >  hi) & (rsi <= hi)).astype(int)  # crosses down into <70

    # 4) Bollinger proximities
    near_low  = (low  < lower + atr*sig["atr_multiplier"]).astype(int)
    near_high = (high > upper - atr*sig["atr_multiplier"]).astype(int)
    near_mid  = ((close - middle).abs() < sig["iron_condor_body_thresh"] * atr).astype(int)

    # 5) ADX + DI direction
    adx_ok  = (adx > sig["adx_thresh"]).astype(int)
    di_up   = (pdi > mdi).astype(int)
    di_down = (mdi > pdi).astype(int)

    # 6) VWAP, MACD, EMA‐crosses
    above_vwap = (close > vwap).astype(int)
    below_vwap = (close < vwap).astype(int)
    macd_pos   = (macdh > 0).astype(int)
    macd_neg   = (macdh < 0).astype(int)
    ema_up     = ((ema5 > ema20) & (ema5.shift(1) <= ema20.shift(1))).astype(int)
    ema_down   = ((ema5 < ema20) & (ema5.shift(1) >= ema20.shift(1))).astype(int)

    # 7) Candlestick + volume spike
    engulf = talib.CDLENGULFING(df["open"], high, low, close).fillna(0).astype(int)
    bull_cndl = ((engulf > 0) & (vol > vol.rolling(sig["vol_window"]).mean() * sig["vol_mult"])).astype(int)
    bear_cndl = ((engulf < 0) & (vol > vol.rolling(sig["vol_window"]).mean() * sig["vol_mult"])).astype(int)

    # 8) Iron-condor filters: dynamic & static ATR + BB squeeze
    rolling_atr_q = atr_pct.rolling(sig["atr_window"], min_periods=1) \
                               .quantile(sig["atr_quantile"])
    low_vol_dyn = (atr_pct < rolling_atr_q).astype(int)
    low_vol_stat= (atr_pct < sig["iron_condor_atr_thresh"]).astype(int)
    bb_width    = (upper - lower) / middle
    squeeze     = (bb_width < sig["bb_squeeze_thresh"]).astype(int)

    # 9) Build vote-stacks
    call_votes = (
        near_low
      + rsi_bear
      + bear_cndl
      + adx_ok * di_down
      + below_vwap
      + macd_neg
      + ema_down
    )

    put_votes = (
        near_high
      + rsi_bull
      + bull_cndl
      + adx_ok * di_up
      + above_vwap
      + macd_pos
      + ema_up
    )

    ic_votes = (
        low_vol_dyn
      + low_vol_stat
      + near_mid
      + ((rsi > lo) & (rsi < hi)).astype(int)
      + squeeze
    )

    # 10) Method 2: tie-break via argmax over vote columns
    votes_df = pd.DataFrame({
        "call": call_votes,
        "put":  put_votes,
        "ic":   ic_votes
    }, index=df.index)

    # zero-out any votes below their own threshold
    votes_df["call"] = votes_df["call"].where(votes_df["call"] >= ct, -1)
    votes_df["put"]  = votes_df["put"].where(votes_df["put"]   >= pt, -1)
    votes_df["ic"]   = votes_df["ic"].where(votes_df["ic"]     >= it, -1)

    # pick the column with the highest (>= thresh) vote
    winner = votes_df.idxmax(axis=1)

    mapping = {
        "call": StrategyEnum.SellCall,
        "put":  StrategyEnum.SellPut,
        "ic":   StrategyEnum.SellIronCondor
    }

    df["tech_signal"] = winner.map(mapping).fillna(StrategyEnum.Hold)

    # 11) Optional regime-gates: IV & ATR filters
    if "implied_vol" in df and "iv_threshold" in sig:
        iv = df["implied_vol"].astype(float)
        df.loc[iv < sig["iv_threshold"], "tech_signal"] = StrategyEnum.Hold

    if "atr_threshold" in sig:
        df.loc[atr_pct < sig["atr_threshold"], "tech_signal"] = StrategyEnum.Hold

    return df

"""

def calc_technical_signal(df, atr, cfg):
    try:

        # 1. Ensure IV columns are forward‐filled and compute IV_rank if missing
        if "IV_close" in df.columns:
            df["IV_close"] = df["IV_close"].fillna(method="ffill")
            df["IV_close"] = df["IV_close"].fillna(df["IV_close"].rolling(window=10, min_periods=1).mean())
            if "IV_rank" not in df.columns:
                iv_min = df["IV_close"].rolling(window=50, min_periods=1).min()
                iv_max = df["IV_close"].rolling(window=50, min_periods=1).max()
                df["IV_rank"] = (df["IV_close"] - iv_min) / (iv_max - iv_min + 1e-9)
        else:
            df["IV_rank"] = 0.0

        # 2. Bollinger Squeeze: narrow band detection
        #    Squeeze if (upper - lower) / middle < threshold
        bb_mid = df["bb_mid"].astype(float)
        bb_up = df["bb_up"].astype(float)
        bb_low = df["bb_low"].astype(float)
        bb_width_ratio = (bb_up - bb_low) / (bb_mid + 1e-9)  # avoid division by zero
        df["bb_squeeze"] = (bb_width_ratio < cfg["signal"]["bb_squeeze_thresh"]).astype(int)

        # 3. ADX Trend Filter: identify if market has a defined trend
        #    We'll require ADX > adx_thresh to consider directional signals
        adx = df["adx"].astype(float).fillna(0.0)
        trend_strong = (adx >= cfg["signal"]["adx_thresh"]).astype(int)

        # 4. Directional Bias via MACD and EMAs
        #    - For SellCall: market is weak (bearish): MACD < 0, EMA short < EMA mid, and ATR‐normalized
        #    - For SellPut: market is strong (bullish): MACD > 0, EMA short > EMA mid
        macd = df["MACD"].astype(float).fillna(0.0)
        ema_short = df["EMA5"].astype(float).fillna(method="ffill")
        ema_mid = df["EMA20"].astype(float).fillna(method="ffill")

        sell_call_trend = (
            (macd < 0).astype(int)
            & (ema_short < ema_mid).astype(int)
            & (trend_strong == 1)
        )
        sell_put_trend = (
            (macd > 0).astype(int)
            & (ema_short > ema_mid).astype(int)
            & (trend_strong == 1)
        )

        # 5. RSI Confirmation
        rsi = df["RSI"].astype(float).fillna(50.0)
        call_rsi_ok = (rsi > cfg["signal"]["rsi_oversold"]).astype(int)   # avoid oversold bounces
        put_rsi_ok = (rsi < cfg["signal"]["rsi_overbought"]).astype(int)   # avoid overbought pullbacks

        # 6. Bollinger‐Band Reversion
        #    If previous close was near outer band and current price has moved ~50% back to mid‐band
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        prev_bb_up = bb_up.shift(1)
        prev_bb_low = bb_low.shift(1)
        atr_val = pd.Series(atr, index=df.index).fillna(method="ffill")

        prev_near_upper = (prev_close >= prev_bb_up - cfg["signal"]["atr_multiplier"] * atr_val).astype(int)
        curr_dist_up_mid = close - bb_mid
        max_up_diff = bb_up - bb_mid
        call_bb_reversion = ((prev_near_upper == 1) & (curr_dist_up_mid < 0.5 * max_up_diff)).astype(int)

        prev_near_lower = (prev_close <= prev_bb_low + cfg["signal"]["atr_multiplier"] * atr_val).astype(int)
        curr_dist_low_mid = bb_mid - close
        max_low_diff = bb_mid - bb_low
        put_bb_reversion = ((prev_near_lower == 1) & (curr_dist_low_mid < 0.5 * max_low_diff)).astype(int)

        # 7. Volume Spike: leverage existing 'vol_spike' (1 if volume > vol_mult * rolling mean)
        vol_spike = df["vol_spike"].astype(int)

        # 8. Common “non‐directional” Votes
        #    Count: candlestick patterns, volume spike, squeeze, and high IV
        doji = (df.get("doji", 0) != 0).astype(int)
        hammer = (df.get("hammer", 0) != 0).astype(int)
        engulfing = (df.get("engulfing", 0) != 0).astype(int)
        candle_pattern = ((doji | hammer | engulfing) > 0).astype(int)

        iv_high = (df["IV_rank"] > cfg["signal"]["volatility_min"]).astype(int)
        common_votes = candle_pattern + vol_spike + df["bb_squeeze"].astype(int) + iv_high

        # 9. ATR‐Normalized Volatility Filter
        #    Only consider options trades when ATR_pct > threshold or in relative bottom quantile
        atr_pct = df["ATR_percent"].astype(float).fillna(0.0)
        low_atr_condition = (
            (atr_pct >= cfg["signal"]["atr_threshold"]).astype(int)
            | (df["atr_adjusted"] <= df["atr_adjusted"].rolling(window=cfg["signal"]["atr_window"], min_periods=1).quantile(cfg["signal"]["atr_quantile"]))
        ).astype(int)

        # 10. IV Threshold: require a minimum IV for selling premium
        iv_condition = (df["IV_rank"] >= cfg["signal"]["iv_threshold"]).astype(int)

        # 11. Iron Condor Neutral Signal
        #     Looking for low volatility, price near middle band, and RSI in mid‐range
        near_mid = (np.abs(close - bb_mid) < cfg["signal"]["iron_condor_body_thresh"] * atr_val).astype(int)
        rsi_in_mid = ((rsi > df["rsi_dynamic_thresh_low"]) & (rsi < df["rsi_dynamic_thresh_high"])).astype(int)
        low_volatility = (df["atr_adjusted"] < cfg["signal"]["iron_condor_atr_thresh"]).astype(int)
        ic_votes = low_volatility + near_mid + rsi_in_mid + df["bb_squeeze"].astype(int)

        # 12. Combine directional and common votes into scalar scores
        #     Weights can be tuned; here is a balanced approach:
        directional_call_score = (
            2.0 * sell_call_trend
            + 1.5 * call_bb_reversion
            + 1.0 * call_rsi_ok
            + 0.5 * trend_strong
        )
        directional_put_score = (
            2.0 * sell_put_trend
            + 1.5 * put_bb_reversion
            + 1.0 * put_rsi_ok
            + 0.5 * trend_strong
        )
        common_score = 1.0 * candle_pattern + 1.0 * vol_spike + 1.0 * df["bb_squeeze"].astype(int) + 1.0 * iv_high

        # 13. Thresholds from config
        common_min = cfg["signal"].get("common_threshold", 2)  # require at least 2 “common” votes
        call_thresh = cfg["signal"]["call_thresh"]
        put_thresh = cfg["signal"]["put_thresh"]
        ic_thresh = cfg["signal"]["iron_condor_thresh"]
        tie_tolerance = cfg.get("tie_tolerance", 0.5)

        # 14. Decide final signal row‐by‐row
        def decide_strategy(idx):
            c_score = common_score.iloc[idx]
            if c_score < common_min:
                return StrategyEnum.Hold

            # Check IV and ATR conditions
            if (iv_condition.iloc[idx] == 0) or (low_atr_condition.iloc[idx] == 0):
                # If IV or ATR conditions fail, default to Hold
                return StrategyEnum.Hold

            d_call = directional_call_score.iloc[idx]
            d_put = directional_put_score.iloc[idx]
            d_ic = ic_votes.iloc[idx]

            # If only one directional side is valid
            if (d_call >= call_thresh) and (d_put < put_thresh):
                return StrategyEnum.SellCall
            if (d_put >= put_thresh) and (d_call < call_thresh):
                return StrategyEnum.SellPut

            # If both directions exceed their thresholds
            if (d_call >= call_thresh) and (d_put >= put_thresh):
                if abs(d_call - d_put) < tie_tolerance:
                    # Ambiguous: prefer neutral iron condor if also strong
                    if d_ic >= ic_thresh:
                        return StrategyEnum.SellIronCondor
                    else:
                        return StrategyEnum.Hold
                else:
                    return StrategyEnum.SellCall if d_call > d_put else StrategyEnum.SellPut

            # If neither directional side qualifies but neutral is strong
            if d_ic >= ic_thresh:
                return StrategyEnum.SellIronCondor

            return StrategyEnum.Hold

        # 15. Apply vectorized scoring columns for debugging/analysis
        df["directional_call_score"] = directional_call_score
        df["directional_put_score"] = directional_put_score
        df["common_score"] = common_score
        df["ic_votes"] = ic_votes

        # 16. Compute final 'tech_signal' column
        df["tech_signal"] = [
            decide_strategy(i) for i in range(len(df))
        ]
    except Exception as e:
        print("Error calculating technical signal: ", e)
        # Fallback to default strategy if any error occurs
        df["tech_signal"] = StrategyEnum.Hold

    return df
"""

def detect_signals(df):
    EMA_FAST = "EMA8"
    EMA_SLOW = "EMA13"
    VWAP     = "VWAP"

    df["ema_bull"] = (df[EMA_FAST] > df[EMA_SLOW]) & (df[EMA_FAST] > df[EMA_FAST].shift(1))
    df["ema_bear"] = (df[EMA_FAST] < df[EMA_SLOW]) & (df[EMA_FAST] < df[EMA_FAST].shift(1))
    # Predicting bullish cross soon (fast EMA approaching slow EMA)
    df["ema_bull_soon"] = (df[EMA_FAST] > df[EMA_SLOW].shift(1)) & (df[EMA_FAST].shift(1) <= df[EMA_SLOW].shift(2))
    df["ema_bear_soon"] = (df[EMA_FAST] < df[EMA_SLOW].shift(1)) & (df[EMA_FAST].shift(1) >= df[EMA_SLOW].shift(2))


    df["macd_bull"] = df["MACD"] > df["MACDSignal"]
    df["macd_bear"] = df["MACD"] < df["MACDSignal"]

    df["macd_hist_slope"] = df["MACDHist"] - df["MACDHist"].shift(1)
    df["macd_hist_bull"] = df["macd_hist_slope"] > 0
    df["macd_hist_bear"] = df["macd_hist_slope"] < 0

    df["rsi_slope"] = df["RSI"] - df["RSI"].shift(1)
    df["rsi_bull"] = df["rsi_slope"] > 0
    df["rsi_bear"] = df["rsi_slope"] < 0

    df["adx_strong"] = df["adx"] > 20

    df["rsi_exhaustion_bull"] = (df["RSI"] > 70) & df["rsi_bear"]
    df["rsi_exhaustion_bear"] = (df["RSI"] < 30) & df["rsi_bull"]

    df["macd_exhaustion_bull"] = df["macd_bull"] & (df["macd_hist_slope"] < 0)
    df["macd_exhaustion_bear"] = df["macd_bear"] & (df["macd_hist_slope"] > 0)

    df["adx_exhaustion"] = (df["adx"] > 20) & (df["adx"].diff() < 0)

    df["volatility_low"] = df["ATR_percent"] < 0.3  # configurable threshold
    
    df["price_above_vwap"] = df["close"] > df[VWAP]
    df["price_below_vwap"] = df["close"] < df[VWAP]

    # 2) EMA-fast crossing VWAP
    df["ema_vwap_bull"]      = (df[EMA_FAST] > df[VWAP]) & (df[EMA_FAST].shift(1) <= df[VWAP].shift(1))
    df["ema_vwap_bear"]      = (df[EMA_FAST] < df[VWAP]) & (df[EMA_FAST].shift(1) >= df[VWAP].shift(1))

    # 3) EMA-fast “soon” crossing VWAP
    df["ema_vwap_bull_soon"] = (df[EMA_FAST] > df[VWAP].shift(1)) & (df[EMA_FAST].shift(1) <= df[VWAP].shift(2))
    df["ema_vwap_bear_soon"] = (df[EMA_FAST] < df[VWAP].shift(1)) & (df[EMA_FAST].shift(1) >= df[VWAP].shift(2))

    return df


def classify_momentum(df):
    # 1) populate all flags
    df = detect_signals(df)

    # 2) list out every bullish/bearish momentum signal
    bull_signals = [
        "ema_bull",
        "ema_bull_soon",
        "ema_vwap_bull",
        "ema_vwap_bull_soon",
        "macd_bull",
        "macd_hist_bull",
        "rsi_bull",
        "adx_strong",
        "price_above_vwap"
    ]

    bear_signals = [
        "ema_bear",
        "ema_bear_soon",
        "ema_vwap_bear",
        "ema_vwap_bear_soon",
        "macd_bear",
        "macd_hist_bear",
        "rsi_bear",
        "adx_strong",
        "price_below_vwap"
    ]

    # 3) sum them up
    df["bullish_votes"] = df[bull_signals].sum(axis=1).astype(int)
    df["bearish_votes"] = df[bear_signals].sum(axis=1).astype(int)

    # 4) exhaustion still overrides
    df["bullish_exhaustion"] = (
        df["rsi_exhaustion_bull"].astype(int) +
        df["macd_exhaustion_bull"].astype(int) +
        df["adx_exhaustion"].astype(int)
    )
    df["bearish_exhaustion"] = (
        df["rsi_exhaustion_bear"].astype(int) +
        df["macd_exhaustion_bear"].astype(int) +
        df["adx_exhaustion"].astype(int)
    )

    # 5) final sentiment logic
    df["sentiment"] = np.select(
        [
            # enough bullish votes, low vol, no exhaustion
            (df["bullish_votes"] >= 6) & df["volatility_low"] & (df["bullish_exhaustion"] == 0),
            # enough bearish votes, low vol, no exhaustion
            (df["bearish_votes"] >= 6) & df["volatility_low"] & (df["bearish_exhaustion"] == 0),
            # exhaustion states
            (df["bullish_exhaustion"] >= 2),
            (df["bearish_exhaustion"] >= 2),
        ],
        ["bullish", "bearish", "bullish_exhaustion", "bearish_exhaustion"],
        default="neutral"
    )

    # --- now mark your spreads & final strategy ---
    df["uptrend_confirm"]   = (df["close"] > df["VWAP"]) & (df["EMA8"] > df["EMA13"])
    df["downtrend_confirm"] = (df["close"] < df["VWAP"]) & (df["EMA8"] < df["EMA13"])

    df["sell_put_spread"]  = (df["sentiment"]=="bullish") & df["uptrend_confirm"]
    df["sell_call_spread"] = (df["sentiment"]=="bearish") & df["downtrend_confirm"]

    # 4) assign StrategyEnum
    df["strategy"] = StrategyEnum.Hold
    df.loc[df["sell_put_spread"],  "strategy"] = StrategyEnum.SellPut
    df.loc[df["sell_call_spread"], "strategy"] = StrategyEnum.SellCall

    # optional: when sentiment is neutral but vol is low → iron condor
    df.loc[
      (df["sentiment"]=="neutral") & df["volatility_low"],
      "strategy"
    ] = StrategyEnum.SellIronCondor

    return df

# 5. Determine trend using SMA5
def calc_trend(df):
    close_arr = df["close"].values.astype(float)
    sma5 = df["SMA5"].values
    df["trend"] = np.where(close_arr > sma5, "UP", "DOWN")
    try:
        df = classify_momentum(df)
    except Exception as e:
        print("Error classifying momentum: ", e)
    return df

# 6. Time-based indicators
def calc_time_indicators(df, cfg):
    trading_open = cfg["trading_time"]["open_min"]
    total_minutes = cfg["trading_time"]["total_minutes"]
    df["minutes_since_open"] = (df["datetime"].dt.hour * 60 + df["datetime"].dt.minute - trading_open).clip(lower=0)
    remaining_minutes = (total_minutes - df["minutes_since_open"]).clip(lower=0)
    df["remaining_intervals"] = (remaining_minutes // cfg["trading_time"]["candle_interval"])
    return df

# 7. (Existing) checkDayRange function (could be further vectorized)
def checkDayRange(df: pd.DataFrame) -> pd.DataFrame:
    df['new_day'] = df['remaining_intervals'] > df['remaining_intervals'].shift(1)
    df['day_group'] = df['new_day'].cumsum()
    df['day_high'] = df.groupby("day_group")['high'].transform('max')
    df['day_low'] = df.groupby("day_group")['low'].transform('min')
    remaining_highs = []
    remaining_lows = []
    for i in range(len(df)):
        day_group = df.loc[i, 'day_group']
        same_day_data = df[df['day_group'] == day_group]
        remaining_data = same_day_data[same_day_data.index >= i]
        remaining_highs.append(remaining_data['high'].max())
        remaining_lows.append(remaining_data['low'].min())
    df['day_high_remaining'] = remaining_highs
    df['day_low_remaining'] = remaining_lows
    max_day_group = df['day_group'].max()
    df.loc[df['day_group'] == max_day_group, 'day_high_remaining_today'] = df.loc[df['day_group'] == max_day_group, 'day_high_remaining']
    df.loc[df['day_group'] == max_day_group, 'day_high_remaining'] = np.nan
    df.loc[df['day_group'] == max_day_group, 'day_low_remaining_today'] = df.loc[df['day_group'] == max_day_group, 'day_low_remaining']
    df.loc[df['day_group'] == max_day_group, 'day_low_remaining'] = np.nan
    return df

# 8. Volatility scaling to compute remaining standard deviation
def calc_daily_volatility(df, cfg):
    returns = df["close"].pct_change()
    five_min_std = returns.std()
    if np.isnan(five_min_std) or five_min_std == 0:
        five_min_std = cfg["signal"]["volatility_min"]
    daily_vol = five_min_std * np.sqrt(78)
    current_price = df["close"]
    daily_abs_std = current_price * daily_vol
    remaining_std_dev = daily_abs_std * np.sqrt(df["remaining_intervals"] / 78)
    remaining_std_dev = np.maximum(remaining_std_dev, current_price * five_min_std)
    df["remaining_std_dev"] = remaining_std_dev
    return df, current_price, remaining_std_dev

def getStepSize(symbol):
    multiplier = 1.0
    if symbol == "SPX":
        multiplier = 5.0

    return multiplier

# 9. Assign strikes (note: relies on self.get_closest_delta_row; adjust as needed)
def assign_strikes(df, current_price, default_offset_put, default_offset_call, getStrikeFunc = None, symbol = None):
    if "put_strike" not in df.columns:
        df["put_strike"] = np.nan
    if "call_strike" not in df.columns:
        df["call_strike"] = np.nan
    # Here, we assume get_closest_delta_row is available as a method in the calling object.
    
    """
    if (getStrikeFunc is not None):
        latestPutRow = getStrikeFunc(symbol, 0.15, "P")
        latestCallRow = getStrikeFunc(symbol, 0.15, "C")
    else:
        latestCallRow = None
        latestPutRow = None
    if latestPutRow is not None:
        df.iloc[-1, df.columns.get_loc("put_strike")] = latestPutRow["Strike"]
    if latestCallRow is not None:
        df.iloc[-1, df.columns.get_loc("call_strike")] = latestCallRow["Strike"]
    """
    step = getStepSize(symbol)

    df["put_strike"] = (current_price - default_offset_put).apply(adjust_low, args=(step,))
    #df["put_strike"] = df["put_strike"].apply(adjust_low, args=(5,))
    df["call_strike"] = (current_price + default_offset_call).apply(adjust_high, args=(step,))
    return df

# 10. Compute probability-based indicators using Z-scores
def calc_probabilities(df, current_price, remaining_std_dev):
    try:
        # Coerce values to numeric (NaNs will be preserved)
        df["call_strike"] = pd.to_numeric(df["call_strike"], errors="coerce")
        df["put_strike"] = pd.to_numeric(df["put_strike"], errors="coerce")

        # If IV_close exists in the DataFrame, coerce it and use it if not NaN.
        # Otherwise, or for any NaN values, use remaining_std_dev.
        if "IV_close" in df.columns:
            df["IV_close"] = pd.to_numeric(df["IV_close"], errors="coerce")
            # Create a column 'std_dev' using IV_close when available, else fallback
            df["std_dev"] = df["IV_close"].mul(df["close"]).where(df["IV_close"].notna(), remaining_std_dev)

        else:
            df["std_dev"] = remaining_std_dev

        # Initialize columns with NaN as default
        df["call_p"] = np.nan
        df["put_p"] = np.nan
        df["c_dist_p"] = np.nan
        df["p_dist_p"] = np.nan

        # Create a boolean mask for valid rows
        valid_mask = (
            df["call_strike"].notna() & 
            df["put_strike"].notna() & 
            pd.notna(current_price) & 
            df["std_dev"]
        )

        if valid_mask.any():
            # Compute z-scores for valid rows using the index from df
            z_scores_call = (df.loc[valid_mask, "call_strike"] - current_price) /   df.loc[valid_mask, "std_dev"]
            z_scores_put = (df.loc[valid_mask, "put_strike"] - current_price) /  df.loc[valid_mask, "std_dev"]

            # Clip z-scores to limits
            z_scores_call = np.clip(z_scores_call, -4, 4)
            z_scores_put = np.clip(z_scores_put, -4, 4)

            # Calculate probabilities as Series with the proper index
            call_probs = pd.Series(norm.cdf(z_scores_call).round(4), index=z_scores_call.index)
            put_probs = pd.Series((1 - norm.cdf(z_scores_put)).round(4), index=z_scores_put.index)

            df.loc[valid_mask, "call_p"] = call_probs
            df.loc[valid_mask, "put_p"] = put_probs

            # Calculate percentage distance probabilities (again, ensuring Series with the correct index)
            c_dist = ((df.loc[valid_mask, "call_strike"] - current_price) / current_price * 100).round(2)
            p_dist = ((current_price - df.loc[valid_mask, "put_strike"]) / current_price * 100).round(2)

            df.loc[valid_mask, "c_dist"] = pd.Series(c_dist, index=c_dist.index)
            df.loc[valid_mask, "p_dist"] = pd.Series(p_dist, index=p_dist.index)

    except Exception as e:
        print("Error calculating probabilities: ", e)
    return df




# 11. Combine technical signal with probability-based signals
def combine_signals(df):
    if (df.empty):
        return df
    try:
        tech_signal = df["tech_signal"].values
        prob_theshold = CONFIG["signal"]["prob_threshold"]
        atr_thresh = CONFIG["signal"]["atr_threshold"]
        previous_tech_signal = np.roll(tech_signal, shift=1)
        previous_tech_signal[0] = StrategyEnum.Hold  # default for first row
        final_signal = np.where(
            (previous_tech_signal == StrategyEnum.SellPut) &
            (tech_signal == StrategyEnum.SellPut) &
            (df["put_p"].values >= prob_theshold),
            StrategyEnum.SellPut,
            np.where(
                (previous_tech_signal == StrategyEnum.SellCall) &
                (tech_signal == StrategyEnum.SellCall) &
                (df["call_p"].values >= prob_theshold), 
                StrategyEnum.SellCall,
                np.where(
                    (previous_tech_signal == StrategyEnum.SellIronCondor) &
                    (tech_signal == StrategyEnum.SellIronCondor) &
                    (df["call_p"].values >= prob_theshold) &
                    (df["put_p"].values >= prob_theshold),
                    StrategyEnum.SellIronCondor,
                    StrategyEnum.Hold
                )
            )
        )
        df["final_signal"] = final_signal
    except Exception as e:
        print("Error combining signals: ", e)
        df["final_signal"] = StrategyEnum.Hold
    return df

# 12. Adjust strikes based on previous values
def adjust_strikes(df, symbol = None):
    step = getStepSize(symbol)
    
    df['day_high_remaining_strike'] = df['day_high_remaining'].apply(adjust_high, args=(step,))
    df['day_low_remaining_strike'] = df['day_low_remaining'].apply(adjust_low, args=(step,))
    
    
    if "final_signal" not in df.columns:
        return df

    df["call_strike"] = np.where(
        df["final_signal"] != StrategyEnum.Hold,
        np.maximum(df["call_strike"], df["call_strike"].shift(1, fill_value=df["call_strike"].iloc[0])),
        df["call_strike"]
    )
    df["put_strike"] = np.where(
        df["final_signal"] != StrategyEnum.Hold,
        np.minimum(df["put_strike"], df["put_strike"].shift(1, fill_value=df["put_strike"].iloc[0])),
        df["put_strike"]
    )
    return df

def adjust_high(x, m = 5):
    try:
        # if nan just return x
        if pd.isna(x):
            return x
        return math.ceil(x / m) * m
    except Exception:
        return x

def adjust_low(x, m = 5):
    try:
        # if nan just return x
        if pd.isna(x):
            return x
        return math.floor(x / m) * m
    except Exception:
        return x

# 13. Main addIndicators function (to be used as a method)
def addIndicators(self, reqId):
    symbol = self.market_data_req_ids[reqId]["symbol"]
    if symbol not in self.candleData:
        self.addToActionLog(f"No candle data available for symbol: {symbol}")
    else:
        addIndicatorsForSymbol(self, symbol)
def addIndicatorsForSymbol(self, symbol):
    with self.candleLock:
        self.candleData[symbol] = addIndicatorsOn(self.candleData[symbol])

def determineStrikeOffsets(self, symbol):
    if symbol in CONFIG["offsets"]:
        call_offset = CONFIG["offsets"][symbol]["call"]
        put_offset = CONFIG["offsets"][symbol]["put"]
    else:
        call_offset = CONFIG["default_strike_offset_call"]
        put_offset = CONFIG["default_strike_offset_put"]
    return call_offset, put_offset

def addIndicatorsOn(self, df, symbol = None):
    df = parse_datetime(df)  # Step 1
    df = calc_ta_indicators(df, CONFIG)  # Step 2
    df, atr, atr_percent = calc_atr_and_volatility(df, CONFIG)  # Step 3
    df = calc_technical_signal(df, atr, CONFIG)  # Step 4
    df = calc_trend(df)  # Step 5
    df = calc_time_indicators(df, CONFIG)  # Step 6
    df = checkDayRange(df)  # Step 7
    df, current_price, remaining_std_dev = calc_daily_volatility(df, CONFIG)  # Step 8

    call_offset, put_offset = determineStrikeOffsets(self, symbol)

    if (symbol is not None):
        df = assign_strikes(df, current_price, put_offset, call_offset,  None, symbol)  # Step 9 (uses self.get_closest_delta_row)
        df = calc_probabilities(df, current_price, remaining_std_dev)  # Step 10
    df = combine_signals(df)  # Step 11
    df = adjust_strikes(df, symbol)  # Step 12
    return df

def aggregate_rsi_ranges(df, rsi_column, smoothing_window=5, order=10,
                        percentile_low=10, percentile_high=90):
    """
    Aggregate RSI turning points into a single overbought range and a single oversold range.
    
    Parameters:
    df: DataFrame with your data.
    rsi_column: Name of the RSI column.
    smoothing_window: Window size for a rolling average to smooth the RSI.
    order: Number of points on each side to consider for local extrema detection.
    percentile_low: Lower percentile to compute the aggregated range.
    percentile_high: Upper percentile to compute the aggregated range.
    
    Returns:
    A tuple: (overbought_range, oversold_range)
    where each range is a tuple (low_value, high_value) representing the aggregated zone.
    """
    data = df.copy()
    # Smooth the RSI values; this reduces noise and gives a more robust set of turning points.
    data['smoothed_rsi'] = data[rsi_column].rolling(window=smoothing_window, min_periods=1).mean()
    values = data['smoothed_rsi'].values

    # Identify local maxima (potential overbought turning points)
    local_max_idx = argrelextrema(values, np.greater, order=order)[0]
    # Identify local minima (potential oversold turning points)
    local_min_idx = argrelextrema(values, np.less, order=order)[0]
    
    # Debugging information: report how many turning points were found.
    print("Local maxima (overbought candidates) found:", len(local_max_idx))
    print("Local minima (oversold candidates) found:", len(local_min_idx))
    
    # Aggregate the detected extreme values.
    if len(local_max_idx) > 0:
        overbought_values = data[rsi_column].iloc[local_max_idx].values
        # Use percentiles to ignore any outliers:
        ob_low = np.percentile(overbought_values, percentile_low)
        ob_high = np.percentile(overbought_values, percentile_high)
        overbought_range = (ob_low, ob_high)
    else:
        overbought_range = (None, None)
        
    if len(local_min_idx) > 0:
        oversold_values = data[rsi_column].iloc[local_min_idx].values
        os_low = np.percentile(oversold_values, percentile_low)
        os_high = np.percentile(oversold_values, percentile_high)
        # For oversold values (which are lower numbers) the "range" is defined similarly.
        oversold_range = (os_low, os_high)
    else:
        oversold_range = (None, None)
        
    return overbought_range, oversold_range

def calc_rsi_stuff(df, rsi_column, smoothing_window=5, order=10,
                        percentile_low=10, percentile_high=90):
    # Get the aggregated ranges for overbought and oversold zones.
    overbought_range, oversold_range = aggregate_rsi_ranges(
        df, 
        rsi_column,
        smoothing_window, 
        order,
        percentile_low,
        percentile_high
    )

    # --- Turn the results into a readable DataFrame ---
    return pd.DataFrame({
        'Condition': ['Overbought', 'Oversold'],
        'Low': [overbought_range[0], oversold_range[0]],
        'High': [overbought_range[1], oversold_range[1]]
    })