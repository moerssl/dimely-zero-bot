def get_chart_config():
    return {
        "subplots": [

            {
                "name": "MACD",
                "height": 0.1,  # Fraction of total chart height
                "yaxis": "y2",  # Secondary y-axis for this subplot
                "traces": [
                    {
                        "type": "line",
                        "name": "MACD",
                        "columns": {"x": "datetime", "y": "MACD"},
                        "style": {"color": "red", "width": 2}
                    },
                    {
                        "type": "line",
                        "name": "MACD",
                        "columns": {"x": "datetime", "y": "MACDSignal"},
                        "style": {"color": "green", "width": 2}
                    },
                    {
                        "type": "bar",
                        "name": "MACD",
                        "columns": {"x": "datetime", "y": "MACDHist"},
                        "style": {"color": "purple", "width": 2}
                    }
                ]
            },
            {
                "name": "ATR%",
                "height": 0.1,  # Fraction of total chart height
                "yaxis": "y2",  # Secondary y-axis for this subplot
                "traces": [
                    {
                        "type": "line",
                        "name": "ATR%",
                        "columns": {"x": "datetime", "y": "ATR_percent"},
                        "style": {"color": "purple", "width": 2}
                    }
                ]
            },
            {
                "name": "IV",
                "height": 0.1,  # Fraction of total chart height
                "yaxis": "y2",  # Secondary y-axis for this subplot
                "traces": [
                    {
                        "type": "line",
                        "name": "IV High",
                        "columns": {"x": "datetime", "y": "IV_high"},
                        "style": {"color": "purple", "width": 2}
                    },
                    {
                        "type": "line",
                        "name": "HV High",
                        "columns": {"x": "datetime", "y": "HV_high"},
                        "style": {"color": "purple", "width": 2}
                    },

                ]
            },
                        {
                "name": "RSI | ADX",
                "height": 0.1,  # Fraction of total chart height
                "yaxis": "y2",  # Secondary y-axis for this subplot
                "lines": [
                    {
                        "val": 70,
                        "style": {"color": "red", "width": 2, "dash": "dash"}
                    },
                    {
                        "val": 30,
                        "style": {"color": "green", "width": 2, "dash": "dash"}
                    }
                ],
                "traces": [
                    {
                        "type": "line",
                        "name": "RSI",
                        "columns": {"x": "datetime", "y": "RSI"},
                        "style": {"color": "red", "width": 2}
                    },
                    {
                        "type": "line",
                        "name": "ADX",
                        "columns": {"x": "datetime", "y": "adx"},
                        "style": {"color": "purple", "width": 2}
                    },
                    {
                        "type": "bar",
                        "name": "hammer",
                        "columns": {"x": "datetime", "y": "hammer"},
                        "style": {"color": "rgba(255,0,255,0.4)", "width": 2}
                    },
                    {
                        "type": "bar",
                        "name": "Doji",
                        "columns": {"x": "datetime", "y": "doji"},
                        "style": {"color": "rgba(0,0,255,0.4)", "width": 2}
                    },
                ]
            },
            {
                "name": "Price and Bollinger Bands",
                "height": 0.6,  # Fraction of total chart height
                "yaxis": "y",  # Main y-axis for this subplot
                "traces": [
                    {
                        "type": "candlestick",
                        "name": "Candlestick",
                        "columns": {"x": "datetime", "open": "open", "high": "high", "low": "low", "close": "close"}
                    },
                    {
                        "type": "line",
                        "name": "SMA5",
                        "columns": {"x": "datetime", "y": "SMA5"},
                        "style": {"color": "rgba(0, 0, 0, 0.5)", "width": 2}
                    },
                                        {
                        "type": "line",
                        "name": "EMA5",
                        "columns": {"x": "datetime", "y": "EMA5"},
                        "style": {"color": "rgba(255, 140, 255, 1)", "width": 2}
                    },
                    {
                        "type": "line",
                        "name": "BB Upper",
                        "columns": {"x": "datetime", "y": "bb_up"},
                        "style": {"color": "rgba(255, 0, 0, 0.5)", "width": 2}
                    },
                    {
                        "type": "line",
                        "name": "BB Lower",
                        "columns": {"x": "datetime", "y": "bb_low"},
                        "style": {"color": "rgba(0, 0, 255, 0.5)", "width": 2}
                    },
                    {
                        "type": "line",
                        "name": "BB Mid",
                        "columns": {"x": "datetime", "y": "bb_mid"},
                        "style": {"color": "rgba(0, 125, 255, 0.5)", "width": 1}
                    },
                   
                    {
                        "type": "line",
                        "name": "High Remain",
                        "columns": {"x": "datetime", "y": "day_high_remaining_strike"},
                        "style": {"color": "rgba(0, 0, 0, 0.5)", "width": 2, "dash": "dash"}
                    },
                    {
                        "type": "line",
                        "name": "Low Remain",
                        "columns": {"x": "datetime", "y": "day_low_remaining_strike"},
                        "style": {"color": "rgba(0, 0, 0, 0.5)", "width": 2, "dash": "dash"}
                    },

                                        {
                        "type": "line",
                        "name": "High Remain Today",
                        "columns": {"x": "datetime", "y": "day_high_remaining_today"},
                        "style": {"color": "rgba(0, 0, 0, 1)", "width": 2, "dash": "dot"}
                    },
                    {
                        "type": "line",
                        "name": "Low Remain Today",
                        "columns": {"x": "datetime", "y": "day_low_remaining_today"},
                        "style": {"color": "rgba(0, 0, 0, 1)", "width": 2, "dash": "dot"}
                    },
                    {
                        "type": "line",
                        "name": "Put Strike",
                        "columns": {"x": "datetime", "y": "put_strike"},
                        "style": {"color": "rgba(0, 0, 0, 0.5)", "width": 1}
                    },
                    {
                        "type": "line",
                        "name": "High Prediction",
                        "columns": {"x": "datetime", "y": "predicted_day_high_remaining"},
                        "style": {"color": "green", "width": 2}
                    },
                    {
                        "type": "line",
                        "name": "Low Prediction",
                        "columns": {"x": "datetime", "y": "predicted_day_low_remaining"},
                        "style": {"color": "red", "width": 2}    
                    },
                    {
                        "type": "line",
                        "name": "Call Strike",
                        "columns": {"x": "datetime", "y": "call_strike"},
                        "style": {"color": "rgba(0, 0, 0, 0.5)", "width": 1}
                    },

                ]
            }


        ],
        "layout": {
            "title": "Dynamic Chart with Multiple Subplots",
            "legend_title": "Legend",
            "height": 1400  # Total chart height in pixels
        }
    }


"""
 
"""