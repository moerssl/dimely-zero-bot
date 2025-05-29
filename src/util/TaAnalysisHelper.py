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
    df["adx"] = talib.ADX(high_arr, low_arr, close_arr, timeperiod=cfg["ta"]["adx_timeperiod"])
    df["PLUS_DI"] = talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=cfg["ta"]["adx_timeperiod"])
    df["MINUS_DI"] = talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=cfg["ta"]["adx_timeperiod"])

    
    # Create a new column for the gap
    df['night_gap'] = df['open'] - df['close'].shift(1)

    # VWAP (rolling)
    #df["VWAP"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    # use variables defined above
    df["VWAP"] = (volume * (high_arr + low_arr + close_arr) / 3).cumsum() / volume.cumsum()


    
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
    return df, atr, atr_percent

# 4. Calculate technical signal from Bollinger bands and RSI
def calc_technical_signal(df, atr, cfg):
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    upper = df["bb_up"]
    lower = df["bb_low"]
    middle = df["bb_mid"]
    rsi = df["RSI"]
    atr_percent = df["ATR_percent"]

    # Candlestick signal: treat as one vote
    candlestick_signal = ((df["doji"] > 0) | (df["hammer"] > 0)).astype(int)

    # Signal components
    near_upper = (high > (upper - atr * cfg["signal"]["atr_multiplier"])).astype(int)
    near_lower = (low < (lower + atr * cfg["signal"]["atr_multiplier"])).astype(int)
    near_middle = ((close - middle).abs() < cfg["signal"]["iron_condor_body_thresh"] * atr).astype(int)

    # Vote counts
    call_votes = (
        near_upper +
        (rsi > cfg["signal"]["rsi_overbought"]).astype(int) +
        candlestick_signal
    )

    put_votes = (
        near_lower +
        (rsi < cfg["signal"]["rsi_oversold"]).astype(int) +
        candlestick_signal
    )

    ic_votes = (
        (atr_percent < cfg["signal"]["iron_condor_atr_thresh"]).astype(int) +
        near_middle +
        ((rsi > cfg["signal"]["rsi_oversold"]) & (rsi < cfg["signal"]["rsi_overbought"])).astype(int)
    )

    # Assign final strategy
    df["tech_signal"] = np.select(
        [
            call_votes >= 2,
            put_votes >= 2,
            ic_votes >= 3
        ],
        [
            StrategyEnum.SellCall,
            StrategyEnum.SellPut,
            StrategyEnum.SellIronCondor
        ],
        default=StrategyEnum.Hold
    )

    return df


import numpy as np

def detect_signals(df):
    df["ema_bull"] = (df["EMA20"] > df["EMA50"]) & (df["EMA20"] > df["EMA20"].shift(1))
    df["ema_bear"] = (df["EMA20"] < df["EMA50"]) & (df["EMA20"] < df["EMA20"].shift(1))

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
    return df



def classify_momentum(df):
    df = detect_signals(df)

    # Bullische Stimmen (PUT Credit Spreads)
    df["bullish_votes"] = (
        df["ema_bull"].astype(int) +
        df["macd_bull"].astype(int) +
        df["macd_hist_bull"].astype(int) +
        df["rsi_bull"].astype(int) +
        df["adx_strong"].astype(int)
    )

    # Bärische Stimmen (CALL Credit Spreads)
    df["bearish_votes"] = (
        df["ema_bear"].astype(int) +
        df["macd_bear"].astype(int) +
        df["macd_hist_bear"].astype(int) +
        df["rsi_bear"].astype(int) +
        df["adx_strong"].astype(int)
    )

    # Momentum-Erschöpfung berücksichtigen
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

    # Finales Sentiment unter Berücksichtigung der Erschöpfung
    df["sentiment"] = np.select(
        [
            (df["bullish_votes"] >= 4) & df["volatility_low"] & (df["bullish_exhaustion"] == 0),
            (df["bearish_votes"] >= 4) & df["volatility_low"] & (df["bearish_exhaustion"] == 0),
            (df["bullish_exhaustion"] >= 2),
            (df["bearish_exhaustion"] >= 2),
        ],
        ["bullish", "bearish", "bullish_exhaustion", "bearish_exhaustion"],
        default="neutral"
    )

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
    tech_signal = df["tech_signal"].values
    prob_theshold = CONFIG["signal"]["prob_threshold"]
    previous_tech_signal = np.roll(tech_signal, shift=1)
    previous_tech_signal[0] = StrategyEnum.Hold  # default for first row
    final_signal = np.where(
        (previous_tech_signal == StrategyEnum.SellPut) &
        (tech_signal == StrategyEnum.Hold) &
        (df["put_p"].values >= prob_theshold) &
        (df["ATR_percent"].values < 0.3),
        StrategyEnum.SellPut,
        np.where(
            (previous_tech_signal == StrategyEnum.SellCall) &
            (tech_signal == StrategyEnum.Hold) &
            (df["call_p"].values >= prob_theshold) &
            (df["ATR_percent"].values < 0.3),
            StrategyEnum.SellCall,
            np.where(
                (previous_tech_signal == StrategyEnum.SellIronCondor) &
                (tech_signal == StrategyEnum.Hold) &
                (df["call_p"].values >= prob_theshold) &
                (df["put_p"].values >= prob_theshold),
                StrategyEnum.SellIronCondor,
                StrategyEnum.Hold
            )
        )
    )
    df["final_signal"] = final_signal
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
        return math.ceil(x / m) * m
    except Exception:
        return x

def adjust_low(x, m = 5):
    try:
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