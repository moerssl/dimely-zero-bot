import threading
import asyncio
from time import sleep
import pandas as pd
import curses
from wakepy import keep
# Enable eager execution if not already enabled.
import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

# Ensure TensorFlow detects and uses the GPU (if available).
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth set to True.")
    except Exception as e:
        print("Error setting GPU memory growth:", e)

# Import your IBApp and related modules.
from ib.IBApp import IBApp
from display.tiled import display_data_tiled
from display.dash_app import start_dash_app
from util.Chart import Chart

# Import the predictor class.
from data.HighLowPredictorParallel import HighLowPredictorParallel as HighLowPredictor

# Global lock to ensure only one prediction is running at a time.
prediction_lock = threading.Lock()

def merge_predictions_into_ibapp(app: IBApp, predicted_df, symbol="SPX"):
    """
    Merge the prediction columns from predicted_df into IBApp’s candleData for the given symbol.
    This overrides any existing prediction columns.
    """
    if symbol not in app.candleData:
        app.addToActionLog(f"No candle data available for symbol: {symbol}")
        return
    with app.candleLock:
        app.candleData[symbol]["predicted_day_high_remaining"] = predicted_df["predicted_day_high_remaining"]
        app.candleData[symbol]["predicted_day_low_remaining"] = predicted_df["predicted_day_low_remaining"]
    app.addToActionLog(f"Predictions updated for symbol {symbol}")

def correct_predictions_in_ibapp(app: IBApp, predictor: HighLowPredictor, symbol="SPX"):
    """
    After merging predictions, run a correction step to fine-tune the model based on feedback data.
    Uses candleLock to extract the relevant data and triggers predictor.learn_from_errors.
    """
    with app.candleLock:
        feedback_df = app.candleData[symbol].copy()
    feedback_df = feedback_df.dropna(subset=["day_high_remaining", "day_low_remaining"])
    if feedback_df.empty:
        app.addToActionLog("No feedback data available for model correction.")
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(predictor.learn_from_errors(feedback_df, error_threshold=0.5, fine_tune_epochs=3))
        app.addToActionLog("Performed correction based on feedback data.")
    except Exception as e:
        app.addToActionLog(f"Error during correction: {e}")
    finally:
        loop.close()

def update_chart_predictions(app: IBApp, predictor: HighLowPredictor):
    """
    Retrieve the latest chart data for symbol "SPX", generate predictions via the predictor,
    merge the predictions into IBApp’s candleData (using candleLock), then run the correction step.
    This function is wrapped with prediction_lock to ensure only one prediction runs at a time.
    """
    # Use the global prediction_lock to ensure only one prediction runs at once.
    with prediction_lock:
        symbol = "SPX"
        if symbol not in app.candleData:
            app.addToActionLog(f"No candle data available for symbol: {symbol}")
            return
        df = app.candleData[symbol].copy()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            predicted_df = loop.run_until_complete(predictor.predict(df))
        except Exception as e:
            app.addToActionLog(f"Error during prediction: {e}")
            return
        finally:
            loop.close()
        merge_predictions_into_ibapp(app, predicted_df, symbol)
        correct_predictions_in_ibapp(app, predictor, symbol)

async def main():
    # Initialize IBApp and connect.
    app = IBApp()
    print("Connecting to TWS/IB Gateway...")
    app.connect("127.0.0.1", 7497, clientId=1)  # Use your connection parameters.
    
    def run_loop():
        app.run()
    api_thread = threading.Thread(target=run_loop, daemon=True)
    api_thread.start()
    sleep(1)
    
    print("Fetching positions and orders...")
    app.fetch_positions()
    app.reqAllOpenOrders()
    
    symbols = [("VIX", "IND", "CBOE"), ("SPX", "IND", "CBOE")]
    app.request_market_data(symbols)
    app.fetch_options_data(symbols)
    app.reqHistoricalDataFor("SPX", "IND", "CBOE")
    
    dash_thread = threading.Thread(target=start_dash_app, args=(app,), daemon=True)
    dash_thread.start()
    
    # Wait until IBApp has enough chart data.
    print("Waiting for IBApp to have enough chart data...")
    initial_data = app.get_chart_data()  # For symbol "SPX"
    while (
        initial_data is None
        or initial_data.empty
        or "datetime" not in initial_data.columns
        or "day_high_remaining_strike" not in initial_data.columns
        or pd.isnull(initial_data["IV_close"].iloc[-1])  # Check for NaN or None
    ):
        print("Chart data not yet available. Waiting...")
        sleep(5)
        initial_data = app.get_chart_data()
        
    print("Chart data available. Initializing predictor...")
    predictor = HighLowPredictor(initial_data)
    
    # Ensure the model is trained (if adequate labeled historical data exists).
    print("Training the predictor model...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(predictor.train(epochs=5, batch_size=128))
    except Exception as e:
        print("Training error:", e)
    finally:
        loop.close()
    
    # Start a background thread to update predictions periodically.
    def prediction_updater():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            update_chart_predictions(app, predictor)
            sleep(1)  # Update every 5 seconds.
    prediction_thread = threading.Thread(target=prediction_updater, daemon=True)
    prediction_thread.start()

    curses.wrapper(display_data_tiled, app)
    app.disconnect()

if __name__ == "__main__":
    with keep.running():
        asyncio.run(main())