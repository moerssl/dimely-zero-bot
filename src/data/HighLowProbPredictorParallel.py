"""
HighLowProbPredictorParallel

This class is designed to predict the probability that the underlying asset’s
price will remain within a target range during the trading day. It is intended
to support trading strategies such as Short Put Credit Spreads, Short Call Credit
Spreads, and Short Iron Condors, where a win is defined as the underlying not 
breaching a strike level. The model outputs two probability values:
    - prob_high: The probability that the high price is not breached.
    - prob_low:  The probability that the low price is not breached.

Features are transformed so that certain absolute indicators (like Bollinger Bands
and moving averages) are expressed as offsets from the current close price.

Public methods:
    - train_async(epochs, batch_size, fit_scaler): Asynchronously trains the model on new data.
    - predict_async(new_data): Asynchronously returns probability predictions, appended to new_data.
    - correct_predictions(X, y_actual): Fine-tunes the model on samples with errors.
    - update_data(new_rows): Incorporates new data into the internal DataFrame.
    - learn_from_errors(feedback_df, ...): Performs error-based fine-tuning.

This version uses asynchronous wrappers to support both batch training and real-time correction scenarios.
"""

import os
import shutil
import asyncio
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure GPU memory growth if available.
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info("GPU memory growth set to True.")
    except Exception as e:
        logger.error("Error setting GPU memory growth: %s", e)

# Force eager execution for debugging purposes.
tf.config.run_functions_eagerly(True)

class HighLowProbPredictorParallel:
    def __init__(self, df: pd.DataFrame, model_path: str = "high_low_prob_model_p.keras"):
        """
        Initializes the predictor with historical data and a model file path.
        
        The model is designed to learn two probability outputs:
          - prob_high: probability that the intraday high remains below a target strike.
          - prob_low: probability that the intraday low remains above a target strike.
        
        Feature columns include price data and technical indicators. Some features
        (listed in self.offset_features) are converted from absolute values into
        offsets from the close price.
        
        If a saved model exists but its structure does not match the expected format,
        it is backed up and a new model is built.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be None or empty.")
        self.df = df.copy()
        self._prepare_dataframe()

        # Define required columns.
        self.feature_columns = [
            "open", "high", "low", "close", "volume", "bb_up", "bb_mid", "bb_low",
            "adx", "ATR", "SMA5", "EMA5", "MACDHist", "RSI", "doji", "hammer",
            "IV_open", "IV_close", "IV_high", "IV_low", "IV_volume", "IV_wap"
        ]
        self.target_columns = ["day_high_remaining_strike", "day_low_remaining_strike"]
        # Columns to transform into offsets relative to "close"
        self.offset_features = ["bb_up", "bb_mid", "bb_low", "SMA5", "EMA5", "MACDHist", "RSI"]

        # Check for missing required columns and fill with NaN.
        for col in self.feature_columns + self.target_columns:
            if col not in self.df.columns:
                logger.warning("Missing column '%s'. Filling with NaN.", col)
                self.df[col] = np.nan

        # Initialize a scaler for feature normalization.
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.model = None

        # Attempt to load an existing model, or build a new one.
        self._load_or_build_model()

    def _prepare_dataframe(self):
        """Ensures 'datetime' column is parsed and data is sorted."""
        if "datetime" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'datetime' column.")
        try:
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        except Exception as e:
            raise ValueError(f"Error parsing 'datetime': {e}")
        self.df.sort_values("datetime", inplace=True)
    
    def _load_or_build_model(self):
        """
        Attempts to load a saved model. If the model exists but does not conform to the
        expected structure (e.g., output names or input shape do not match), it is backed up,
        and a new model is built.
        """
        if os.path.exists(self.model_path):
            try:
                loaded_model = keras.models.load_model(self.model_path, compile=False)
                # Check if the model outputs are in dictionary format with proper keys.
                if (not hasattr(loaded_model, "output_names") or 
                    set(loaded_model.output_names) != {"prob_high", "prob_low"} or 
                    loaded_model.input_shape[-1] != len(self.feature_columns)):
                    backup_path = self.model_path + ".bak"
                    shutil.move(self.model_path, backup_path)
                    logger.info("Model structure mismatch. Backed up existing model to %s", backup_path)
                    self.model = self._build_model()
                else:
                    self.model = loaded_model
                    self.model.compile(
                        optimizer="adam",
                        loss={"prob_high": "binary_crossentropy", "prob_low": "binary_crossentropy"},
                        loss_weights={"prob_high": 1.0, "prob_low": 1.0},
                        metrics={"prob_high": "accuracy", "prob_low": "accuracy"}
                    )
                    logger.info("Loaded model from %s", self.model_path)
            except Exception as e:
                logger.error("Error loading model: %s. Building a new model.", e)
                self.model = self._build_model()
        else:
            logger.info("No model file found. Building a new model.")
            self.model = self._build_model()
    
    def _build_model(self):
        """
        Builds a new probabilistic model with two outputs:
            - prob_high: probability that the intraday high remains safe.
            - prob_low: probability that the intraday low remains safe.
        
        The model is constructed with several dense layers and uses sigmoid activations.
        """
        inputs = keras.Input(shape=(len(self.feature_columns),), name="features")
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(32, activation="relu")(x)
        prob_high = layers.Dense(1, activation="sigmoid", name="prob_high")(x)
        prob_low = layers.Dense(1, activation="sigmoid", name="prob_low")(x)
        model = keras.Model(inputs=inputs, outputs={"prob_high": prob_high, "prob_low": prob_low})
        model.compile(optimizer="adam",
                      loss={"prob_high": "binary_crossentropy", "prob_low": "binary_crossentropy"},
                      loss_weights={"prob_high": 1.0, "prob_low": 1.0},
                      metrics={"prob_high": "accuracy", "prob_low": "accuracy"})
        return model

    def _transform_features(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Transforms feature columns by converting selected absolute indicators into offsets
        relative to the 'close' price. Other features are left as is, and missing values
        are filled with 0.
        """
        df_trans = df_features.copy()
        if "close" not in df_trans.columns:
            raise ValueError("Feature DataFrame must contain a 'close' column for offset transformation.")
        for col in self.offset_features:
            if col in df_trans.columns:
                df_trans[col] = df_trans[col] - df_trans["close"]
        return df_trans.fillna(0).to_numpy()
    
    def _transform_targets(self, df_targets: pd.DataFrame) -> list:
        """
        Transforms the target DataFrame into binary labels.
        
        Expected columns in df_targets:
            - high: the actual high price of the day.
            - low: the actual low price of the day.
            - entry_price: the entry price when the trade was initiated.
        
        For a short spread strategy, a win is achieved if:
            - The actual high is not above the entry price for puts.
            - The actual low is not below the entry price for calls.
        This method returns two numpy arrays representing binary outcomes.
        """
        df_t = df_targets.copy()
        if not {"high", "low", "entry_price"}.issubset(df_t.columns):
            raise ValueError("Target DataFrame must contain 'high', 'low', and 'entry_price' columns.")
        df_t["prob_high"] = (df_t["high"] <= df_t["entry_price"]).astype(int)
        df_t["prob_low"] = (df_t["low"] >= df_t["entry_price"]).astype(int)
        return [df_t["prob_high"].to_numpy(), df_t["prob_low"].to_numpy()]

    async def preprocess_data(self, df: pd.DataFrame = None, fit_scaler: bool = False) -> tuple:
        """
        Preprocesses the DataFrame by:
          1. Dropping rows with missing target values.
          2. Transforming feature columns (including offset conversion).
          3. Computing training targets as binary labels.
          4. Scaling the feature matrix using StandardScaler.
        
        Returns a tuple: (X_scaled, y_offsets)
        """
        if df is None:
            df = self.df
        df = df.dropna(subset=self.target_columns)
        if df.empty:
            logger.warning("No valid rows after filtering for targets. Returning empty arrays.")
            return np.empty((0, len(self.feature_columns))), np.empty((0, 2))
        df_features = df[self.feature_columns].fillna(0)
        X = self._transform_features(df_features)
        y_abs = self._transform_targets(df[self.target_columns + ["high", "low", "entry_price"]])
        if fit_scaler or not hasattr(self.scaler, "scale_"):
            self.scaler.fit(X)
        loop = asyncio.get_running_loop()
        X_scaled = await loop.run_in_executor(None, self.scaler.transform, X)
        y_offsets = np.stack([y_abs[0], y_abs[1]], axis=1)
        return X_scaled, y_offsets

    async def train_async(self, epochs: int = 10, batch_size: int = 32, fit_scaler: bool = True):
        """
        Asynchronous wrapper for training.
        """
        loop = asyncio.get_running_loop()
        X_scaled, y_offsets = await self.preprocess_data(self.df, fit_scaler=fit_scaler)
        if X_scaled.size == 0 or y_offsets.size == 0:
            logger.error("No data available for training. Skipping training.")
            return
        y_high = y_offsets[:, 0]
        y_low = y_offsets[:, 1]
        def train_model():
            callback = keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
            self.model.fit(X_scaled, {"prob_high": y_high, "prob_low": y_low},
                           epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[callback])
            self.model.save(self.model_path)
            logger.info("Model saved to %s", self.model_path)
        await loop.run_in_executor(None, train_model)

    async def predict_async(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Asynchronous prediction.
        
        Expects new_data to include the feature columns. Transforms features, scales them,
        and returns the predictions as probabilities for both high and low outcomes.
        
        Adds columns 'pred_prob_high' and 'pred_prob_low' to new_data.
        """
        loop = asyncio.get_running_loop()
        for col in self.feature_columns:
            if col not in new_data.columns:
                new_data[col] = 0
        def scale_features():
            df_features = new_data[self.feature_columns].fillna(0)
            transformed = self._transform_features(df_features)
            if not hasattr(self.scaler, "scale_"):
                X_fit = self.df[self.feature_columns].fillna(0)
                X_fit = self._transform_features(X_fit)
                self.scaler.fit(X_fit)
            return self.scaler.transform(transformed)
        X_scaled = await loop.run_in_executor(None, scale_features)
        def make_predictions():
            preds = self.model.predict(X_scaled)
            if hasattr(preds, 'numpy'):
                preds = preds.numpy()
            if isinstance(preds, list) and len(preds) == 2:
                pred_high = preds[0].flatten()
                pred_low = preds[1].flatten()
            elif isinstance(preds, np.ndarray) and preds.shape[1] == 2:
                pred_high = preds[:, 0].flatten()
                pred_low = preds[:, 1].flatten()
            else:
                raise ValueError(f"Unexpected model output: {preds}")
            return pred_high, pred_low
        pred_high, pred_low = await loop.run_in_executor(None, make_predictions)
        new_data = new_data.copy()
        new_data["pred_prob_high"] = pred_high
        new_data["pred_prob_low"] = pred_low
        return new_data

    async def update_data(self, new_rows: pd.DataFrame):
        """
        Incorporates new rows into the internal DataFrame, ensuring duplicates (based on 'datetime') are dropped.
        """
        new_rows = new_rows.copy()
        if "datetime" in new_rows.columns and not np.issubdtype(new_rows["datetime"].dtype, np.datetime64):
            new_rows["datetime"] = pd.to_datetime(new_rows["datetime"])
        combined_df = pd.concat([self.df, new_rows])
        combined_df = combined_df.drop_duplicates(subset="datetime", keep="last")
        self.df = combined_df.sort_values("datetime")
        logger.info("Data updated. New data shape: %s", self.df.shape)

    async def learn_from_errors(self, feedback_df: pd.DataFrame, error_threshold: float = None,
                                high_threshold: float = 0.5, low_threshold: float = 0.5,
                                fine_tune_epochs: int = 5):
        """
        Performs error-based fine-tuning of the model.
        
        For each sample in the feedback data, the binary targets are computed.
        Samples where the predicted probability deviates from the target beyond a given threshold
        are selected for fine-tuning.
        
        A single error_threshold, if provided, overrides both high and low thresholds.
        """
        if error_threshold is not None:
            high_threshold = error_threshold
            low_threshold = error_threshold
        feedback_df = feedback_df.copy()
        for col in self.feature_columns + self.target_columns + ["high", "low", "entry_price"]:
            if col not in feedback_df.columns:
                feedback_df[col] = np.nan
        loop = asyncio.get_running_loop()
        X_scaled, y_offsets = await self.preprocess_data(feedback_df)
        if X_scaled.size == 0 or y_offsets.size == 0:
            logger.warning("No valid feedback data found. Skipping error-based learning.")
            return
        def predict_and_calculate_error():
            preds = self.model.predict(X_scaled)
            if hasattr(preds, 'numpy'):
                preds = preds.numpy()
            if isinstance(preds, list) and len(preds) == 2:
                pred_high = preds[0].flatten()
                pred_low = preds[1].flatten()
            elif isinstance(preds, np.ndarray) and preds.shape[1] == 2:
                pred_high = preds[:, 0].flatten()
                pred_low = preds[:, 1].flatten()
            else:
                raise ValueError(f"Unexpected model output: {preds}")
            error_high = np.abs(pred_high - y_offsets[:, 0])
            error_low = np.abs(pred_low - y_offsets[:, 1])
            return error_high, error_low
        error_high, error_low = await loop.run_in_executor(None, predict_and_calculate_error)
        logger.info("Average high error: %f, low error: %f", np.mean(error_high), np.mean(error_low))
        def select_samples():
            mask = (error_high > high_threshold) | (error_low > low_threshold)
            return np.where(mask)[0]
        mispred_indices = await loop.run_in_executor(None, select_samples)
        if len(mispred_indices) == 0:
            logger.info("No significant errors detected; skipping fine-tuning.")
            return
        X_mispred = X_scaled[mispred_indices]
        y_mispred = y_offsets[mispred_indices]
        logger.info("Fine-tuning on %d mispredicted samples.", len(mispred_indices))
        def fine_tune():
            y_high_mispred = y_mispred[:, 0]
            y_low_mispred = y_mispred[:, 1]
            self.model.fit(X_mispred, {"prob_high": y_high_mispred, "prob_low": y_low_mispred},
                           epochs=fine_tune_epochs, batch_size=16, verbose=1)
            self.model.save(self.model_path)
            logger.info("Model fine-tuned and saved.")
        await loop.run_in_executor(None, fine_tune)
