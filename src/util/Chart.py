import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Chart:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.data = pd.DataFrame(columns=["datetime", "open", "high", "low", "close"])
        self.running = True

    def initialize(self):
        """Initialize the chart layout."""
        self.ax.set_title("Live Candlestick Chart")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.xticks(rotation=45)

    def update(self, data):
        """Update the chart with new candlestick data."""
        if not data.empty:
            self.ax.clear()  # Clear existing chart
            self.ax.set_title("Live Candlestick Chart")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price")
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.xticks(rotation=45)

            # Plot candlestick bars manually
            for _, row in data.iterrows():
                color = "green" if row["close"] >= row["open"] else "red"
                self.ax.plot([row["datetime"], row["datetime"]], [row["low"], row["high"]], color="black", linewidth=1)
                self.ax.plot([row["datetime"], row["datetime"]], [row["open"], row["close"]], color=color, linewidth=4)

            # Redraw the figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def start(self, data_source):
        """Run the chart updater in the main thread."""
        plt.ion()  # Turn on interactive mode
        self.initialize()
        while self.running:
            new_data = data_source()
            self.update(new_data)
            plt.pause(1)  