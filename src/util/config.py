import talib

CONFIG = {
    "ta": {
        "sma_timeperiod": 5,
        "ema_timeperiod": 5,
        "rsi_timeperiod": 14,
        "macd": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "bbands": {"timeperiod": 40, "nbdevup": 2, "nbdevdn": 2, "matype": talib.MA_Type.EMA},
        "adx_timeperiod": 5,
        "atr_timeperiod": 78,
    },
    "signal": {
        "atr_multiplier": 0.75,
        "rsi_overbought": 65,
        "rsi_oversold": 35,
        "iron_condor_atr_thresh": 0.2,
        "iron_condor_body_thresh": 0.25,
        "prob_threshold": 0.70,
        "volatility_min": 0.001,
    },
    "trading_time": {
        "open_min": 930,     # minutes (15:30 in minutes)
        "total_minutes": 390,  # trading day minutes
        "candle_interval": 5,  # minutes per candle
    },
    "tpFromLimit": 20,
    "slFromLimit": 200,
    "default_strike_offset": 35,
    "default_strike_offset_put": 50,
    "default_strike_offset_call": 35,
    "default_wing_span": 10,
    "default_late_ic_windspan": 5,
    "default_target_premium": 1,
    "offsets": {
        "SPY": {
            "call": 5,
            "put": 5,
            "wing_span": 5,
            "late_ic_windspan": 2,
            "target_premium": 0.8,
        }
    }
}