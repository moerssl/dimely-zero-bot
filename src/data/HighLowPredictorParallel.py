import os
import shutil
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

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import StandardScaler

class HighLowPredictorParallel:
    def __init__(self, df: pd.DataFrame, model_path: str = "high_low_model_p.keras"):
        """
        Initialize the predictor with a historical DataFrame and a model file path.
        The model learns two offset values:
          offset_high = day_high_remaining - current_candle_high
          offset_low  = current_candle_low - day_low_remaining
        In addition, a subset of feature columns (offset_features) will be transformed
        into offsets relative to the close price.
        
        If a saved model exists and its output structure matches our expected dictionary
        with keys 'high_output' and 'low_output', it is loaded; otherwise, the old model is
        backed up and a new model is built.
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
        self.feature_columns = [
            "open", "high", "low", "close", "bb_up", "bb_mid", "bb_low", "adx",
            "ATR", "SMA5", "EMA5", "MACDHist", "RSI", "doji", "hammer", 
            "IV_open", "IV_close", "IV_high", "IV_low", "IV_volume", "IV_wap"
        ]
        self.target_columns = ["day_high_remaining_strike", "day_low_remaining_strike"]

        # Define which feature columns should be transformed into offsets relative to "close".
        self.offset_features = ["bb_up", "bb_mid", "bb_low", "SMA5", "EMA5", "ATR"]

        # Ensure all required columns are present.
        for col in self.feature_columns + self.target_columns:
            if col not in self.df.columns:
                print(f"Warning: Missing column '{col}'. Filling with NaN values.")
                self.df[col] = np.nan

        self.scaler = StandardScaler()
        self.model_path = model_path
        self.model = None

        # Try loading an existing model.
        if os.path.exists(self.model_path):
            try:
                loaded_model = load_model(self.model_path, compile=False)
                # Check if loaded_model has two outputs named correctly and matching input features.
                if (not hasattr(loaded_model, "output_names") or 
                    set(loaded_model.output_names) != {"high_output", "low_output"} or 
                    loaded_model.input_shape[-1] != len(self.feature_columns)):
                    # Backup the existing model.
                    backup_path = self.model_path + ".bak"
                    shutil.move(self.model_path, backup_path)
                    print(f"Existing model does not match expected structure. Backed up to {backup_path}.")
                    self.model = None
                else:
                    self.model = loaded_model
                    self.model.compile(
                        optimizer="adam",
                        loss={"high_output": "mse", "low_output": "mse"},
                        loss_weights={"high_output": 1.0, "low_output": 1.0},
                        metrics={"high_output": "mae", "low_output": "mae"}
                    )
                    print(f"Loaded model from {self.model_path}")
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
        Build and compile a new model with a shared base and two output branches for high and low offsets.
        Outputs are defined as a dictionary with keys 'high_output' and 'low_output'.
        """
        inputs = Input(shape=(input_shape,))
        x = Dense(128, activation="relu")(inputs)
        x = Dropout(0.2)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(32, activation="relu")(x)
        
        high_output = Dense(1, activation="relu", name="high_output")(x)
        low_output = Dense(1, activation="relu", name="low_output")(x)
        
        model = Model(inputs=inputs, outputs={"high_output": high_output, "low_output": low_output})
        model.compile(
            optimizer="adam",
            loss={"high_output": "mse", "low_output": "mse"},
            loss_weights={"high_output": 1.0, "low_output": 1.0},
            metrics={"high_output": "mae", "low_output": "mae"}
        )
        return model
    
    def _transform_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        For each column in self.offset_features, convert its absolute value into an offset from the close price.
        """
        df_features = df_features.copy()
        # Only transform if "close" is present
        if "close" not in df_features.columns:
            raise ValueError("The 'close' column is required to compute offsets.")
        for col in self.offset_features:
            if col in df_features.columns:
                df_features[col] = df_features[col] - df_features["close"]
        return df_features
    
    async def preprocess_data(self, df: pd.DataFrame = None, fit_scaler: bool = False):
        """
        Preprocess the data and compute training targets as offsets.
        For each row:
            t_high_offset = day_high_remaining - current_candle_high
            t_low_offset  = current_candle_low - day_low_remaining
        Also, transform specified feature columns into offsets from the close.
        """
        if df is None:
            df = self.df
        df = df.dropna(subset=self.target_columns)
        if df.empty:
            print("Warning: No valid rows after filtering for targets. Returning empty arrays.")
            return np.empty((0, len(self.feature_columns))), np.empty((0, 2))
        
        # Create a copy for feature transformation.
        df_features = df[self.feature_columns].fillna(0)
        # Transform selected features into offsets relative to close.
        df_features = self._transform_features(df_features)
        X = df_features.values

        y_abs = df[self.target_columns].values
        current_high = df["high"].values
        current_low  = df["low"].values

        t_high_offset = y_abs[:, 0] - current_high  # Future high - current high.
        t_low_offset  = current_low - y_abs[:, 1]     # Current low - future low.
        y_offsets = np.stack([t_high_offset, t_low_offset], axis=1)
        
        if fit_scaler or not hasattr(self.scaler, "scale_"):
            self.scaler.fit(X)
        
        loop = asyncio.get_running_loop()
        X_scaled = await loop.run_in_executor(None, self.scaler.transform, X)
        return X_scaled, y_offsets
    
    async def train(self, epochs: int = 10, batch_size: int = 32, fit_scaler: bool = True):
        """
        Asynchronously train the model using offset targets.
        Targets are provided as a dictionary matching the model outputs.
        """
        loop = asyncio.get_running_loop()
        X_scaled, y_offsets = await self.preprocess_data(self.df, fit_scaler=fit_scaler)
        if X_scaled.size == 0 or y_offsets.size == 0:
            print("No data available for training. Skipping training.")
            return

        y_high = y_offsets[:, 0]
        y_low  = y_offsets[:, 1]

        def train_model():
            from tensorflow.keras.callbacks import EarlyStopping
            early_stopping = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
            self.model.fit(
                X_scaled,
                {"high_output": y_high, "low_output": y_low},
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[early_stopping]
            )
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")

        await loop.run_in_executor(None, train_model)
    
    async def predict(self, new_data: pd.DataFrame):
        """
        Generate predictions for new_data and reparameterize to absolute values.
        For each sample:
            predicted_day_high = current_candle_high + predicted_offset_high
            predicted_day_low  = current_candle_low  - predicted_offset_low
        This method handles outputs returned as a dict, a single array, or a list/tuple.
        """
        loop = asyncio.get_running_loop()
        for col in self.feature_columns:
            if col not in new_data.columns:
                new_data[col] = 0

        def scale_features():
            df_features = new_data[self.feature_columns].fillna(0)
            df_features = self._transform_features(df_features)
            X = df_features.values
            if not hasattr(self.scaler, "scale_"):
                X_fit = self.df[self.feature_columns].fillna(0)
                X_fit = self._transform_features(X_fit)
                self.scaler.fit(X_fit.values)
            return self.scaler.transform(X)

        X_scaled = await loop.run_in_executor(None, scale_features)

        def make_predictions():
            predictions = self.model.predict(X_scaled)
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            if isinstance(predictions, dict):
                pred_high = predictions["high_output"].flatten()
                pred_low = predictions["low_output"].flatten()
            elif isinstance(predictions, np.ndarray):
                if predictions.shape[1] != 2:
                    raise ValueError(f"Unexpected shape of model output array: {predictions.shape}")
                pred_high = predictions[:, 0].flatten()
                pred_low = predictions[:, 1].flatten()
            elif isinstance(predictions, (list, tuple)) and len(predictions) == 2:
                pred_high = predictions[0].flatten()
                pred_low = predictions[1].flatten()
            else:
                raise ValueError(f"Unexpected model output: {predictions}")
            return pred_high, pred_low

        pred_high, pred_low = await loop.run_in_executor(None, make_predictions)

        current_high = new_data["high"].values
        current_low = new_data["low"].values

        if pred_high.shape != current_high.shape:
            raise ValueError(f"Shape mismatch: pred_high {pred_high.shape} vs current_high {current_high.shape}")
        if pred_low.shape != current_low.shape:
            raise ValueError(f"Shape mismatch: pred_low {pred_low.shape} vs current_low {current_low.shape}")

        predicted_day_high = current_high + pred_high
        predicted_day_low = current_low - pred_low

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
    
    async def learn_from_errors(self, feedback_df: pd.DataFrame, 
                                error_threshold: float = None,
                                high_threshold: float = 5.0, 
                                low_threshold: float = 5.0, 
                                fine_tune_epochs: int = 5):
        """
        Fine-tune the model using feedback data.
        For each sample:
            t_high_offset = actual_day_high - current_candle_high
            t_low_offset  = current_candle_low - actual_day_low
        A single error_threshold, if provided, overrides both high and low thresholds.
        """
        if error_threshold is not None:
            high_threshold = error_threshold
            low_threshold = error_threshold
        
        feedback_df = feedback_df.copy()
        for col in self.feature_columns + self.target_columns:
            if col not in feedback_df.columns:
                feedback_df[col] = 0
        loop = asyncio.get_running_loop()
        X_scaled, y_offsets = await self.preprocess_data(feedback_df)
        if X_scaled.size == 0 or y_offsets.size == 0:
            print("No valid feedback data found. Skipping learning from errors.")
            return

        def predict_and_calculate_error():
            predictions = self.model.predict(X_scaled)
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            if isinstance(predictions, dict):
                pred_high = predictions["high_output"].flatten()
                pred_low = predictions["low_output"].flatten()
            elif isinstance(predictions, np.ndarray):
                if predictions.shape[1] != 2:
                    raise ValueError(f"Unexpected shape of model output array: {predictions.shape}")
                pred_high = predictions[:, 0].flatten()
                pred_low = predictions[:, 1].flatten()
            elif isinstance(predictions, (list, tuple)) and len(predictions) == 2:
                pred_high = predictions[0].flatten()
                pred_low = predictions[1].flatten()
            else:
                raise ValueError(f"Unexpected model output: {predictions}")
            error_high = np.abs(pred_high - y_offsets[:, 0])
            error_low = np.abs(pred_low - y_offsets[:, 1])
            return predictions, error_high, error_low

        predictions, error_high, error_low = await loop.run_in_executor(None, predict_and_calculate_error)
        print("Average predicted high offset error:", np.mean(error_high))
        print("Average predicted low offset error:", np.mean(error_low))
        
        def select_samples():
            mask_high = error_high > high_threshold
            mask_low = error_low > low_threshold
            mask = mask_high | mask_low
            return np.where(mask)[0]

        mispred_indices = await loop.run_in_executor(None, select_samples)
        if len(mispred_indices) == 0:
            print("No significant errors detected; skipping fine-tuning.")
            return

        X_mispred = X_scaled[mispred_indices]
        y_mispred = y_offsets[mispred_indices]
        print(f"Fine-tuning on {len(mispred_indices)} mispredicted samples.")

        def fine_tune():
            y_high_mispred = y_mispred[:, 0]
            y_low_mispred = y_mispred[:, 1]
            self.model.fit(
                X_mispred,
                {"high_output": y_high_mispred, "low_output": y_low_mispred},
                epochs=fine_tune_epochs,
                batch_size=16,
                verbose=1
            )
            self.model.save(self.model_path)
            print("Model fine-tuned and saved.")

        await loop.run_in_executor(None, fine_tune)
