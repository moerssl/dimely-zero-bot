import talib

CONFIG = {
    "ta": {
        "sma_timeperiod": 5,
        "ema_timeperiod": 5,
        "rsi_timeperiod": 12,
        "macd": {"fastperiod": 7, "slowperiod": 14, "signalperiod": 8},
        "bbands": {"timeperiod": 10, "nbdevup": 2, "nbdevdn": 2, "matype": talib.MA_Type.EMA},
        "adx_timeperiod": 14,
        "atr_timeperiod": 14,
    },
    "signal": {
        "atr_multiplier": 0.75,
        "rsi_overbought": 65,
        "rsi_oversold": 35,
        "iron_condor_atr_thresh": 0.2,
        "iron_condor_body_thresh": 0.25,
        "prob_threshold": 0.70,
        "volatility_min": 0.001,
        "common_threshold": 1,
        "call_thresh": 2,
        "put_thresh": 2,
        "iron_condor_thresh": 4,
        "adx_thresh": 20,
        "iv_threshold": 0.0075,  # Minimum implied volatility threshold for signal generation
        "atr_window": 9,
        "atr_quantile": 0.25,
        "atr_threshold": 0.15,  # Minimum ATR threshold for signal generation
        "vol_window": 9,  # Rolling window for volume spike detection
        "vol_mult": 2.0,  # Multiplier for defining a volume spike
        "bb_squeeze_thresh": 0.05,  # Narrow Bollinger bandwidth threshold for squeeze detection
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
    "call_delta": 0.16,
    "put_delta": 0.16,
    "default_wing_span": 10,
    "default_late_ic_windspan": 5,
    "default_target_premium": 1,
    "default_orb_length": 60,  # minutes
    "offsets": {
        "SPX": {
            "call": 0.5,
            "put": 0.5,
            "call_delta": 0.4,
            "put_delta": 0.4,
            "wing_span": 5,
            "late_ic_windspan": 2,
            "target_premium": 0.4,
            "orbLength": 60,  # minutes
        },
        "SPY": {
            "call": 5,
            "put": 5,
            "call_delta": 0.16,
            "put_delta": 0.16,
            "wing_span": 5,
            "late_ic_windspan": 2,
            "target_premium": 0.4,
            "orbLength": 15,  # minutes
        },
        "IWM": {
            "call": 0.51,
            "put": 0.51,
            "call_delta": 0.4,
            "put_delta": 0.4,
            "wing_span": 3,
            "late_ic_windspan": 2,
            "target_premium": 0.2,
        },
        "QQQ": {
            "call": 0.01,
            "put": 0.01,
            "call_delta": 0.2,
            "put_delta": 0.2,
            "wing_span": 4,
            "late_ic_windspan": 0.01,
            "target_premium": 0.2,
        },

    }
}