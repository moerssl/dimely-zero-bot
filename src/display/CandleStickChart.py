# candlestick_chart.py

import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from threading import RLock
import time
from util.StrategyEnum import StrategyEnum

class CandlestickChart:
    def __init__(self, title):
        """
        Initializes the CandlestickChart class with the chart's title.
        """
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None
        self.title = title       
        self.latestData = None 
        self.updateLock = RLock()
        self._initialized = False

    def _initialize_chart(self):
        """
        Sets up the chart layout and subplots. This method is now called in the main thread,
        and plt.show(block=False) is used to ensure the GUI event loop runs in the main thread.
        """
        plt.ion()  # Enable interactive mode for real-time updates
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(
            4, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [4, 1, 1, 1]}, sharex=True
        )
        self.fig.tight_layout()
        self._initialized = True
        # Call plt.show(block=False) in the main thread to start the GUI event loop.
        plt.show(block=False)

    def update(self, raw: pd.DataFrame):
        """
        Updates the chart with the provided DataFrame. This method is thread-safe.
        """
        with self.updateLock:
            if "remaining_intervals" not in raw.columns:
                return
            
            data: pd.DataFrame = raw
            if data.equals(self.latestData):
                return
            
            if len(data) <= 5:
                return
            
            if not self._initialized:
                self._initialize_chart()
            
            self.latestData = data

            try:
                # Prepare x-axis data and formatted date strings
                data['date_num'] = pd.to_datetime(data['datetime']).dt.strftime('%Y-%m-%d %H:%M')
                data['x'] = range(len(data))
                ohlc = list(zip(data['x'], data['open'], data['high'], data['low'], data['close']))

                # Clear previous plots
                self.ax1.clear()
                self.ax2.clear()
                self.ax3.clear()
                self.ax4.clear()

                candle_width = 0.8

                # Draw candlestick chart and additional lines on ax1
                candlestick_ohlc(self.ax1, ohlc, width=candle_width, colorup='green', colordown='red', alpha=0.8)
                self.ax1.plot(data['x'], data['bb_mid'], label='Bollinger Mid', color='orange', linewidth=1.5)
                self.ax1.plot(data['x'], data['SMA5'], '--', label='SMA5', color='orange', linewidth=1)
                self.ax1.plot(data['x'], data['SMA50'], '--', label='SMA50', color='purple', linewidth=1)
                self.ax1.plot(data['x'], data['bb_up'], '--', color='blue', label='Bollinger Upper')
                self.ax1.plot(data['x'], data['bb_low'], '--', color='blue', label='Bollinger Lower')
                self.ax1.plot(data['x'], data['call_strike'],  color='green', label='Call Strike', linewidth=1)
                self.ax1.plot(data['x'], data['put_strike'],  color='green', label='Put Strike', linewidth=1)

                # Draw background shading based on signals
                def drawBackground(col, alpha):
                    for i, signal in enumerate(data[col]):
                        if signal == StrategyEnum.SellPut:
                            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                                ax.axvspan(i - 0.4, i + 0.4, color='red', alpha=alpha)
                        elif signal == StrategyEnum.SellCall:
                            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                                ax.axvspan(i - 0.4, i + 0.4, color='green', alpha=alpha)
                        elif signal == StrategyEnum.SellIronCondor:
                            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                                ax.axvspan(i - 0.4, i + 0.4, color='purple', alpha=alpha)

                drawBackground("temp_signal", 0.05)
                drawBackground("final_signal", 0.2)

                for i, signal in enumerate(data["narrow_bands"]):
                    if signal:
                        for ax in [self.ax2, self.ax3, self.ax4]:
                            ax.axvspan(i - 0.4, i + 0.4, color='black', alpha=0.2)

                self.ax1.set_title(self.title)
                self.ax1.legend()
                self.ax1.grid()
                ticks = data['x'][::max(1, len(data) // 10)]
                tick_labels = data['date_num'][::max(1, len(data) // 10)]
                self.ax1.set_xticks(ticks)
                self.ax1.set_xticklabels(tick_labels, rotation=45)

                # RSI Plot on ax2
                self.ax2.plot(data['x'], data['RSI'], color='purple', label='RSI')
                self.ax2.axhline(66, color='red', linestyle='--', label='Overbought')
                self.ax2.axhline(33, color='green', linestyle='--', label='Oversold')
                self.ax2.set_title('Relative Strength Index (RSI)')
                self.ax2.set_ylim(0, 100)
                self.ax2.legend()
                self.ax2.grid()
                self.ax2.set_xticks(ticks)
                self.ax2.set_xticklabels(tick_labels, rotation=45)

                # MACD Plot on ax3
                self.ax3.plot(data['x'], data['MACD'], color='blue', label='MACD')
                self.ax3.plot(data['x'], data['MACDSignal'], color='red', label='Signal')
                self.ax3.bar(data['x'], data['MACDHist'], color='gray', alpha=0.5, label='Histogram')
                self.ax3.set_title('MACD')
                self.ax3.legend()
                self.ax3.grid()
                self.ax3.set_xticks(ticks)
                self.ax3.set_xticklabels(tick_labels, rotation=45)

                # ATR Percent Plot on ax4
                self.ax4.plot(data['x'], data['ATR_percent'], color='red', label='ATR%')
                self.ax4.set_xticks(ticks)
                self.ax4.set_xticklabels(tick_labels, rotation=45)
                self.ax4.legend()
                self.ax4.grid()

                self.add_lines(data)

                # Redraw the figure (using draw_idle to safely schedule a redraw)
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

            except Exception as e:
                print(f"An error occurred: {e}")
                raise

    def add_lines(self, data: pd.DataFrame):
        """
        Adds vertical lines for day changes and horizontal lines for signals.
        """
        try:
            day_changes = data[data['remaining_intervals'].diff() > 0].index
            if not len(day_changes):
                return
            for x in day_changes:
                for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                    ax.axvline(x=x, color='orange', linestyle='--', linewidth=2)

            for i, signal in enumerate(data['final_signal']):
                if i >= len(data) or pd.isna(data['remaining_intervals'].iloc[i]):
                    continue
                try: 
                    end_of_day = min(i + data['remaining_intervals'][i] - 1, len(data) - 1)
                except Exception:
                    continue

                if signal == StrategyEnum.SellCall:
                    self.ax1.hlines(y=data['call_strike'][i], xmin=i, xmax=end_of_day,
                                    colors='green', linewidth=0.8, alpha=0.8)
                elif signal == StrategyEnum.SellPut:
                    self.ax1.hlines(y=data['put_strike'][i], xmin=i, xmax=end_of_day,
                                    colors='red', linewidth=0.8, alpha=0.8)
                elif signal == StrategyEnum.SellIronCondor:
                    self.ax1.hlines(y=data['call_strike'][i], xmin=i, xmax=end_of_day,
                                    colors='purple', linewidth=0.8, alpha=0.8)
                    self.ax1.hlines(y=data['put_strike'][i], xmin=i, xmax=end_of_day,
                                    colors='purple', linewidth=0.8, alpha=0.8)
        except Exception as e:
            print(f"An error occurred in add_lines: {e}")
            raise
