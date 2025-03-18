import pandas as pd
import numpy as np
import math
import time
import threading
import logging
from sklearn.preprocessing import StandardScaler  # Using StandardScaler to avoid clipping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class AdaptiveDataframeProcessor:
    def __init__(self, rolling_window=15, sequence_length=30, model_save_interval=30, adaptation_threshold=0.05):
        self.df = pd.DataFrame()
        self.features = []
        self.high_model = None
        self.low_model = None
        self.rolling_window = rolling_window
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()  # For features
        self.target_scaler_high = StandardScaler()  # For high target
        self.target_scaler_low = StandardScaler()   # For low target
        self.model_save_interval = model_save_interval  # Seconds for periodic saving
        self.adaptation_threshold = adaptation_threshold    # Error threshold for adaptation
        self.load_model()  # Attempt to load pre-trained models
        self.start_model_saving_thread()  # Start background saving thread

    def update_dataframe(self, new_data):
        """Update the working DataFrame with a copy of new_data."""
        self.df = new_data.copy()

    def add_features(self):
        """
        Add technical indicator and time-based features.
        Assumes the DataFrame has at least these columns:
          'datetime', 'remaining_intervals', 'close', 'RSI', 'SMA5', 'ATR_percent',
          'MACDHist', 'bb_up', 'bb_mid', 'bb_low', 'day_high_remaining', 'day_low_remaining'
        If 'high' and 'low' columns exist (with 5‑minute candles), compute the intraday range
        using the cumulative daily high and low.
        """
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            market_open = self.df['datetime'].dt.normalize() + pd.Timedelta(hours=9, minutes=30)
            self.df['minutes_since_open'] = (self.df['datetime'] - market_open).dt.total_seconds() / 60.0
            self.df['minutes_since_open'] = self.df['minutes_since_open'].clip(lower=0)
            self.df['minutes_until_close'] = 390 - self.df['minutes_since_open']
            self.df['normalized_time'] = self.df['minutes_since_open'] / 390.0
            self.df['sin_time'] = np.sin(2 * np.pi * self.df['normalized_time'])
            self.df['cos_time'] = np.cos(2 * np.pi * self.df['normalized_time'])
            self.df['session_part'] = pd.cut(
                self.df['normalized_time'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['open', 'mid', 'close']
            )
            self.df['session_part_num'] = self.df['session_part'].map({'open': 0, 'mid': 1, 'close': 2})
            self.df['date'] = self.df['datetime'].dt.date

        self.df['time_to_close'] = self.df['remaining_intervals'] / self.df['remaining_intervals'].max()
        self.df['rolling_volatility'] = self.df['close'].rolling(window=5).std()
        self.df['rolling_volatility'] = self.df['rolling_volatility'].fillna(method='ffill').fillna(method='bfill')

        if 'high' in self.df.columns and 'low' in self.df.columns:
            if 'date' not in self.df.columns and 'datetime' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['datetime']).dt.date
            self.df['cumulative_high'] = self.df.groupby('date')['high'].cummax()
            self.df['cumulative_low'] = self.df.groupby('date')['low'].cummin()
            self.df['intraday_range'] = self.df['cumulative_high'] - self.df['cumulative_low']
        else:
            self.df['intraday_range'] = np.nan

        base_features = [
            'close', 'time_to_close', 'rolling_volatility',
            'RSI', 'SMA5', 'ATR_percent', 'MACDHist', 'bb_up', 'bb_mid', 'bb_low'
        ]
        extra_features = []
        if 'minutes_since_open' in self.df.columns:
            extra_features += ['minutes_since_open', 'minutes_until_close', 'normalized_time', 'sin_time', 'cos_time', 'session_part_num']
        if 'intraday_range' in self.df.columns:
            extra_features.append('intraday_range')
        self.features = base_features + extra_features

        self.df[self.features] = self.df[self.features].fillna(method='ffill').fillna(method='bfill')

    def prepare_data_for_lstm(self, for_training=True):
        """
        Prepare sequences for LSTM training or prediction.
        When for_training=True, only rows with non-null day_high_remaining and day_low_remaining are used,
        and the targets are adjusted and scaled. When for_training=False, all rows are used to generate
        prediction sequences.
        """
        if for_training:
            df_train = self.df.dropna(subset=['day_high_remaining', 'day_low_remaining']).copy()
            if df_train.empty:
                raise ValueError("No complete days available for training.")

            def adjust_high(x):
                try:
                    return math.ceil(x / 5) * 5 + 5
                except Exception:
                    return x

            def adjust_low(x):
                try:
                    return math.floor(x / 5) * 5 - 5
                except Exception:
                    return x

            df_train['adjusted_day_high'] = df_train['day_high_remaining'].apply(adjust_high)
            df_train['adjusted_day_low'] = df_train['day_low_remaining'].apply(adjust_low)

            features = df_train[self.features].values
            features_scaled = self.scaler.fit_transform(features)
            target_high = df_train['adjusted_day_high'].values.reshape(-1, 1)
            target_low = df_train['adjusted_day_low'].values.reshape(-1, 1)
            target_high_scaled = self.target_scaler_high.fit_transform(target_high)
            target_low_scaled = self.target_scaler_low.fit_transform(target_low)

            X, y_high, y_low = [], [], []
            for i in range(self.sequence_length, len(features_scaled)):
                X.append(features_scaled[i - self.sequence_length:i])
                y_high.append(target_high_scaled[i])
                y_low.append(target_low_scaled[i])
            X = np.array(X)
            y_high = np.array(y_high)
            y_low = np.array(y_low)
            print("Training data prepared: X shape:", X.shape, "y_high shape:", y_high.shape, "y_low shape:", y_low.shape)
            return X, y_high, y_low
        else:
            features = self.df[self.features].values
            if hasattr(self.scaler, 'mean_'):
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = self.scaler.fit_transform(features)
            X = []
            for i in range(self.sequence_length, len(features_scaled)):
                X.append(features_scaled[i - self.sequence_length:i])
            X = np.array(X)
            print("Prediction data prepared: X shape:", X.shape)
            return X, None, None

    def train_lstm_model(self, X_train, y_high_train, y_low_train):
        """Build and train two LSTM models (one for high and one for low predictions)."""
        high_model = Sequential()
        high_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        high_model.add(LSTM(units=50))
        high_model.add(Dense(units=1))
        high_model.compile(optimizer='adam', loss='mean_squared_error')
        high_model.fit(X_train, y_high_train, epochs=10, batch_size=32)

        low_model = Sequential()
        low_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        low_model.add(LSTM(units=50))
        low_model.add(Dense(units=1))
        low_model.compile(optimizer='adam', loss='mean_squared_error')
        low_model.fit(X_train, y_low_train, epochs=10, batch_size=32)

        return high_model, low_model

    def train_model(self):
        """Train the models using complete days only."""
        X_train, y_high_train, y_low_train = self.prepare_data_for_lstm(for_training=True)
        self.high_model, self.low_model = self.train_lstm_model(X_train, y_high_train, y_low_train)
        logger.info("Model training complete.")

    def predict_high_low_prices(self):
        """
        Generate predictions for day_high_remaining (high) and day_low_remaining (low) for all rows.
        """
        if (self.high_model is None or self.low_model is None or 
            not hasattr(self.target_scaler_high, 'scale_') or 
            not hasattr(self.target_scaler_low, 'scale_')):
            logger.info("Models or target scalers not fitted. Training now using complete days.")
            self.train_model()

        X_pred, _, _ = self.prepare_data_for_lstm(for_training=False)
        predicted_high = self.high_model.predict(X_pred)
        predicted_low = self.low_model.predict(X_pred)
        logger.info(f"Raw predicted high shape: {predicted_high.shape}, raw predicted low shape: {predicted_low.shape}")

        predicted_high = self.target_scaler_high.inverse_transform(predicted_high)
        predicted_low = self.target_scaler_low.inverse_transform(predicted_low)
        logger.info("Inverse transformed predictions obtained.")

        pred_high_full = np.empty((len(self.df), 1))
        pred_high_full[:] = np.nan
        pred_low_full = np.empty((len(self.df), 1))
        pred_low_full[:] = np.nan
        pred_high_full[self.sequence_length:] = predicted_high
        pred_low_full[self.sequence_length:] = predicted_low

        self.df['predicted_high'] = pred_high_full
        self.df['predicted_low'] = pred_low_full

        self.adapt_models()
        return self.df

    def adapt_models(self):
        """
        Adapt (retrain) the models if a significant proportion of predictions have directional errors.
        
        Errors occur in the following cases:
        - Predicted high is lower than the actual high.
        - Predicted low is higher than the actual low.
        - Predicted high is lower than predicted low (invalid state).
        
        Uses 'day_high_remaining' if available, otherwise falls back to 'day_high_remaining_today'.
        Same logic applies to low values.
        """
        if 'predicted_high' not in self.df.columns or 'predicted_low' not in self.df.columns:
            return

        # Select actual highs and lows based on availability
        actual_day_high = np.where(
            self.df['day_high_remaining'].notna(), 
            self.df['day_high_remaining'], 
            self.df['day_high_remaining_today']
        )

        actual_day_low = np.where(
            self.df['day_low_remaining'].notna(), 
            self.df['day_low_remaining'], 
            self.df['day_low_remaining_today']
        )

        # Calculate errors
        high_error = np.where(self.df['predicted_high'] < actual_day_high,
                            actual_day_high - self.df['predicted_high'], 0)
        
        low_error = np.where(self.df['predicted_low'] > actual_day_low,
                            self.df['predicted_low'] - actual_day_low, 0)

        # Additional error conditions
        high_error_candle = np.where(self.df['predicted_high'] < self.df['high'],
                                    self.df['high'] - self.df['predicted_high'], 0)
        
        low_error_candle = np.where(self.df['predicted_low'] > self.df['low'],
                                    self.df['predicted_low'] - self.df['low'], 0)

        prediction_invalid = self.df['predicted_high'] < self.df['predicted_low']

        # Combine all error conditions
        total_high_error = high_error + high_error_candle
        total_low_error = low_error + low_error_candle

        high_error_exceeded = total_high_error > self.adaptation_threshold
        low_error_exceeded = total_low_error > self.adaptation_threshold
        invalid_prediction_error = prediction_invalid.any()

        # Retrain if errors exceed the threshold
        if high_error_exceeded.mean() > 0.1 or low_error_exceeded.mean() > 0.1 or invalid_prediction_error:
            logger.info("Prediction errors exceeded threshold. Adapting models...")
            self.retrain_models_with_errors(total_high_error, total_low_error)


    def retrain_models_with_errors(self, high_error, low_error):
        """Retrain the LSTM models using the latest available training data."""
        self.df['high_error'] = high_error
        self.df['low_error'] = low_error

        X_train, y_high_train, y_low_train = self.prepare_data_for_lstm(for_training=True)
        self.high_model, self.low_model = self.train_lstm_model(X_train, y_high_train, y_low_train)

    def save_model(self):
        """Save the trained models in the native Keras format (.keras)."""
        if self.high_model and self.low_model:
            self.high_model.save("high_model.keras")
            self.low_model.save("low_model.keras")
            logger.info("Models saved successfully.")

    def load_model(self):
        """Attempt to load pre-trained models from disk."""
        try:
            self.high_model = load_model("high_model.keras")
            self.low_model = load_model("low_model.keras")
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.info("Error loading models. They will be trained anew. Exception: %s", e)

    def start_model_saving_thread(self):
        """
        Start a background thread that adapts and saves the models every model_save_interval seconds,
        if data is present.
        """
        def save_model_periodically():
            while True:
                if not self.df.empty:
                    self.adapt_models()
                    self.save_model()
                    print("Model saved at", time.ctime())
                time.sleep(self.model_save_interval)
        thread = threading.Thread(target=save_model_periodically)
        thread.daemon = True
        thread.start()

    def evaluate_predictions(self):
        """
        Evaluate predictions by comparing them with actual target columns.
        Computes mean squared error (ignoring NaN rows).
        """
        self.df['high_error'] = self.df['predicted_high'] - self.df['day_high_remaining']
        self.df['low_error'] = self.df['predicted_low'] - self.df['day_low_remaining']
        mse_high = np.nanmean(self.df['high_error'] ** 2)
        mse_low = np.nanmean(self.df['low_error'] ** 2)
        print(f"Mean Squared Error for High Prediction: {mse_high}")
        print(f"Mean Squared Error for Low Prediction: {mse_low}")

    def plot_predictions(self):
        """Plot actual vs predicted day high and day low values."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['day_high_remaining'], label='Actual Day High', color='blue')
        plt.plot(self.df['predicted_high'], label='Predicted High', color='orange')
        plt.plot(self.df['day_low_remaining'], label='Actual Day Low', color='green')
        plt.plot(self.df['predicted_low'], label='Predicted Low', color='red')
        plt.legend()
        plt.title('Actual vs. Predicted Day High and Day Low')
        plt.show()

    def merge_predictions_into(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the prediction columns ('predicted_high' and 'predicted_low') from the internal DataFrame
        into an external DataFrame, overriding or adding the columns if they don't exist.
        """
        # If the columns don't exist in original_df, create them
        if 'predicted_high' not in original_df.columns:
            original_df['predicted_high'] = None  # Or any default value you'd like to set
        
        if 'predicted_low' not in original_df.columns:
            original_df['predicted_low'] = None  # Or any default value you'd like to set

        if 'predicted_high' not in self.df.columns or 'predicted_low' not in self.df.columns:
            return original_df


        original_df['predicted_high'] = self.df['predicted_high']
        original_df['predicted_low'] = self.df['predicted_low']
        
        print("Predictions merged and overridden in the original DataFrame.")
        return original_df
