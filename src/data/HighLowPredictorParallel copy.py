import os
import asyncio
import pandas as pd
import numpy as np
import tensorflow as tf

# Enable GPU memory growth if available.
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU memory growth set to True.")
    except Exception as e:
        print("Error setting GPU memory growth:", e)

# Force eager execution so tensors can be immediately converted to numpy values.
tf.config.run_functions_eagerly(True)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

class HighLowPredictorParallel:
    def __init__(self, df: pd.DataFrame, model_path: str = "high_low_model_p.keras"):
        """
        Initialize the predictor with a historical DataFrame and a model file path.
        The model is expected to learn offsets:
          offset_high = day_high_remaining - current_candle_high
          offset_low  = current_candle_low - day_low_remaining
        If a saved model exists and its input shape matches the current feature set,
        it is loaded; otherwise, a new model is built.
        """
        if df is None or df.empty:
            raise ValueError("The input DataFrame cannot be None or empty.")
        self.df = df.copy()

        # Ensure 'datetime' column exists.
        if "datetime" not in self.df.columns:
            raise ValueError("The DataFrame must have a 'datetime' column.")
        try:
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        except Exception as e:
            raise ValueError(f"Error parsing 'datetime': {e}")
        self.df.sort_values("datetime", inplace=True)

        # Define feature and target columns.
        # The features are the input indicators (including current candle high and low).
        self.feature_columns = [
            "open", "high", "low", "close", "bb_up", "bb_mid", "bb_low", "adx",
            "ATR", "SMA5", "EMA5", "MACDHist", "RSI", "doji", "hammer", "IV_open", "IV_close", "IV_high", "IV_low", "IV_volume", "IV_wap"
        ]
        # The targets remain as the absolute future values.
        self.target_columns = ["day_high_remaining_strike", "day_low_remaining_strike"]

        # Ensure all required columns are present.
        for col in self.feature_columns + self.target_columns:
            if col not in self.df.columns:
                print(f"Warning: Missing column '{col}'. Filling with NaN values.")
                self.df[col] = np.nan

        self.scaler = StandardScaler()
        self.model_path = model_path
        self.model = None

        # Try: load an existing model.
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path, compile=False)
                print(f"Loaded model from {self.model_path}")
                # Check if the loaded model's input shape matches current feature count.
                if self.model.input_shape[-1] != len(self.feature_columns):
                    print(f"Loaded model expects {self.model.input_shape[-1]} features, but {len(self.feature_columns)} provided. Discarding model.")
                    self.model = None
                else:
                    # Recompile with our new loss (we can use simple MSE here since our output is offset).
                    self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            except Exception as e:
                print(f"Failed to load model: {e}. A new model will be initialized.")
                self.model = None
        else:
            print("Model file not found. A new model will be initialized.")
            self.model = None

        # Build a new model if necessary.
        if self.model is None:
            self.model = self.build_model(len(self.feature_columns))
            print("Initialized a new model in __init__.")
    
    def build_model(self, input_shape):
        """
        Build and compile a new model using reparameterization.
        The model outputs two nonnegative offsets (via ReLU):
            offset_high (to be added to current candle high)
            offset_low  (to be subtracted from current candle low)
        """
        # Using ReLU to ensure non-negative offsets.
        model = Sequential([
            Dense(128, activation="relu", input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(32, activation="relu"),
            Dense(2, activation="relu")  # Predict nonnegative offsets.
        ])
        # We use standard mean squared error since targets will be computed as offsets.
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model
    
    async def preprocess_data(self, df: pd.DataFrame = None, fit_scaler: bool = False):
        """
        Preprocess the data and compute training targets as offsets.
        For each row, the training targets become:
            t_high_offset = day_high_remaining - current_candle_high
            t_low_offset  = current_candle_low - day_low_remaining
        """
        if df is None:
            df = self.df
        # Drop rows missing the target values.
        df = df.dropna(subset=self.target_columns)
        if df.empty:
            print("Warning: No valid rows after filtering for targets. Returning empty arrays.")
            return np.empty((0, len(self.feature_columns))), np.empty((0, 2))
        
        X = df[self.feature_columns].fillna(0).values
        # Original targets are absolute values.
        y_abs = df[self.target_columns].values
        # Use the current candle's "high" and "low" as the baseline (from the feature columns).
        current_high = df["high"].values
        current_low  = df["low"].values
        
        # Compute offsets.
        t_high_offset = y_abs[:, 0] - current_high  # Future high - current high (should be >= 0).
        t_low_offset  = current_low - y_abs[:, 1]     # Current low - future low (should be >= 0).
        y_offsets = np.stack([t_high_offset, t_low_offset], axis=1)
        
        if fit_scaler or not hasattr(self.scaler, "scale_"):
            self.scaler.fit(X)
        
        loop = asyncio.get_running_loop()
        X_scaled = await loop.run_in_executor(None, self.scaler.transform, X)
        return X_scaled, y_offsets
    
    async def train(self, epochs: int = 10, batch_size: int = 32, fit_scaler: bool = True):
        """
        Asynchronously train the model using offset targets.
        """
        loop = asyncio.get_running_loop()
        X_scaled, y_offsets = await self.preprocess_data(self.df, fit_scaler=fit_scaler)
        if X_scaled.size == 0 or y_offsets.size == 0:
            print("No data available for training. Skipping training.")
            return
        
        def train_model():
            from tensorflow.keras.callbacks import EarlyStopping
            early_stopping = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
            self.model.fit(X_scaled, y_offsets, epochs=epochs, batch_size=batch_size,
                           verbose=1, callbacks=[early_stopping])
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")
        
        await loop.run_in_executor(None, train_model)
    
    async def predict(self, new_data: pd.DataFrame):
        """
        Generate predictions for new_data and reparameterize to absolute values.
        
        The model outputs predicted offsets. We then compute:
            predicted_day_high = current_candle_high + predicted_offset_high
            predicted_day_low  = current_candle_low  - predicted_offset_low
        
        Assumes new_data has columns "high" and "low" representing the current candle.
        """
        loop = asyncio.get_running_loop()
        for col in self.feature_columns:
            if col not in new_data.columns:
                new_data[col] = 0
        
        def scale_features():
            X = new_data[self.feature_columns].fillna(0).values
            if not hasattr(self.scaler, "scale_"):
                X_fit = self.df[self.feature_columns].fillna(0).values
                self.scaler.fit(X_fit)
            return self.scaler.transform(X)
        
        X_scaled = await loop.run_in_executor(None, scale_features)
        
        def make_predictions():
            return self.model.predict(X_scaled)
        
        predicted_offsets = await loop.run_in_executor(None, make_predictions)
        # Get current candle's high and low from new_data.
        current_high = new_data["high"].values
        current_low = new_data["low"].values
        
        # Reconstruct absolute predictions from offsets.
        predicted_day_high = current_high + predicted_offsets[:, 0]
        predicted_day_low = current_low - predicted_offsets[:, 1]
        
        new_data["predicted_day_high_remaining"] = predicted_day_high
        new_data["predicted_day_low_remaining"] = predicted_day_low
        return new_data
    
    async def update_data(self, new_rows: pd.DataFrame):
        """
        Update internal data with new_rows, dropping duplicate dates.
        """
        new_rows = new_rows.copy()
        if "datetime" in new_rows.columns and not np.issubdtype(new_rows["datetime"].dtype, np.datetime64):
            new_rows["datetime"] = pd.to_datetime(new_rows["datetime"])
        combined_df = pd.concat([self.df, new_rows])
        combined_df = combined_df.drop_duplicates(subset="datetime", keep="last")
        self.df = combined_df.sort_values("datetime")
        print(f"Data updated. New data shape: {self.df.shape}")
    
    async def learn_from_errors(self, feedback_df: pd.DataFrame, error_threshold: float = 0.5, fine_tune_epochs: int = 5):
        """
        Fine-tune the model using feedback data on which predicted offsets (converted
        to absolute predictions) differ significantly from the actual values.
        For each sample, convert the absolute feedback into target offsets:
            t_high_offset = actual_day_high - current_high
            t_low_offset  = current_low - actual_day_low
        Fine-tune on samples where either error exceeds error_threshold.
        """
        feedback_df = feedback_df.copy()
        for col in self.feature_columns + self.target_columns:
            if col not in feedback_df.columns:
                feedback_df[col] = 0
        loop = asyncio.get_running_loop()
        # Preprocess data to get offsets as targets.
        X_scaled, y_offsets = await self.preprocess_data(feedback_df)
        if X_scaled.size == 0 or y_offsets.size == 0:
            print("No valid feedback data found. Skipping learning from errors.")
            return

        def predict_and_calculate_error():
            predictions = self.model.predict(X_scaled)
            error_high = np.abs(predictions[:, 0] - y_offsets[:, 0])
            error_low  = np.abs(predictions[:, 1] - y_offsets[:, 1])
            return predictions, error_high, error_low

        predictions, error_high, error_low = await loop.run_in_executor(None, predict_and_calculate_error)
        print("Average predicted high offset error:", np.mean(error_high))
        print("Average predicted low offset error:", np.mean(error_low))
        
        def select_samples():
            mask = (error_high > error_threshold) | (error_low > error_threshold)
            return np.where(mask)[0]
        
        mispred_indices = await loop.run_in_executor(None, select_samples)
        if len(mispred_indices) == 0:
            print("No significant errors detected; skipping fine-tuning.")
            return
        
        X_mispred = X_scaled[mispred_indices]
        y_mispred = y_offsets[mispred_indices]
        print(f"Fine-tuning on {len(mispred_indices)} mispredicted samples.")
        
        def fine_tune():
            self.model.fit(X_mispred, y_mispred, epochs=fine_tune_epochs, batch_size=16, verbose=1)
            self.model.save(self.model_path)
            print("Model fine-tuned and saved.")
        
        await loop.run_in_executor(None, fine_tune)
