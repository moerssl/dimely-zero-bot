import os
import asyncio
import pandas as pd
import numpy as np
import tensorflow as tf

# Force eager execution so that tensors can be converted to numpy values immediately.
tf.config.run_functions_eagerly(True)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

# Define a custom loss function that adds a penalty
# when the predicted high is not above the predicted low.
def custom_loss(y_true, y_pred):
    # Compute standard mean squared error.
    mse_obj = tf.keras.losses.MeanSquaredError()
    mse = mse_obj(y_true, y_pred)

    
    # Assume y_pred[:, 0] is predicted high and y_pred[:, 1] is predicted low.
    predicted_high = y_pred[:, 0]
    predicted_low  = y_pred[:, 1]
    
    # Define a margin that the high must exceed the low by.
    margin = 0.0  # Change to a positive number if you want a gap.
    
    # Calculate the penalty if predicted_high is not at least predicted_low + margin.
    penalty = tf.maximum(0.0, margin - (predicted_high - predicted_low))
    
    # Weight the penalty term.
    lambda_penalty = 1.0
    penalty_mean = tf.reduce_mean(penalty)
    
    return mse + lambda_penalty * penalty_mean

class HighLowPredictor:
    def __init__(self, df: pd.DataFrame, model_path: str = "high_low_model.h5"):
        """
        Initialize with a DataFrame (historical data) and a model path.
        If a saved model exists and its input shape matches the current
        feature set, it is loaded; otherwise, a new model is built immediately.
        """
        if df is None or df.empty:
            raise ValueError("The input DataFrame cannot be None or empty.")
        self.df = df.copy()
        
        # Ensure a 'datetime' column exists.
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
            "ATR", "SMA5", "EMA5", "MACDHist", "RSI", "doji", "hammer"
        ]
        self.target_columns = ["day_high_remaining", "day_low_remaining"]
        
        # Check for missing columns.
        for col in self.feature_columns + self.target_columns:
            if col not in self.df.columns:
                print(f"Warning: Missing column '{col}'. Filling with NaN values.")
                self.df[col] = np.nan
        
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.model = None
        
        # Try to load an existing model.
        if os.path.exists(self.model_path):
            try:
                # Do not compile immediately; we'll recompile with our custom loss.
                self.model = load_model(self.model_path, compile=False)
                print(f"Loaded model from {self.model_path}")
                # Check if the loaded model's input size matches the new feature count.
                if self.model.input_shape[-1] != len(self.feature_columns):
                    print(f"Loaded model expecting {self.model.input_shape[-1]} features, "
                          f"but {len(self.feature_columns)} provided. Discarding old model.")
                    self.model = None
                else:
                    self.model.compile(optimizer="adam", loss=custom_loss, metrics=["mae"])
            except Exception as e:
                print(f"Failed to load model from {self.model_path}: {e}. A new model will be initialized.")
                self.model = None
        else:
            print("Model file not found. A new model will be initialized.")
            self.model = None
        
        # If no appropriate model was loaded, build a new one.
        if self.model is None:
            self.model = self.build_model(len(self.feature_columns))
            print("Initialized a new model in __init__.")
    
    def build_model(self, input_shape):
        """
        Build and compile a new model with two outputs.
        """
        model = Sequential([
            Dense(128, activation="relu", input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(32, activation="relu"),
            Dense(2, activation="linear")
        ])
        model.compile(optimizer="adam", loss=custom_loss, metrics=["mae"])
        return model
    
    async def preprocess_data(self, df: pd.DataFrame = None):
        """
        Prepare the feature matrix and target vector.
        """
        if df is None:
            df = self.df
        df = df.dropna(subset=self.target_columns)
        if df.empty:
            print("Warning: No valid rows after filtering for targets. Returning empty arrays.")
            return np.empty((0, len(self.feature_columns))), np.empty((0, 2))
        X = df[self.feature_columns].fillna(0).values
        y = df[self.target_columns].values
        # Fit the scaler on current data and transform.
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    async def train(self, epochs: int = 10, batch_size: int = 32):
        """
        Train the model asynchronously using historical data.
        (Call this manually once you have enough labeled data.)
        """
        loop = asyncio.get_running_loop()
        X_scaled, y = await self.preprocess_data()
        if X_scaled.size == 0 or y.size == 0:
            print("No data available for training. Skipping training.")
            return
        
        def train_model():
            self.model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size, verbose=1)
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")
        
        await loop.run_in_executor(None, train_model)
    
    async def predict(self, new_data: pd.DataFrame):
        """
        Predict new values for high and low. Returns the DataFrame with two added columns.
        """
        loop = asyncio.get_running_loop()
        for col in self.feature_columns:
            if col not in new_data.columns:
                new_data[col] = 0
        
        def scale_features():
            X = new_data[self.feature_columns].fillna(0).values
            # If the scaler isn’t fitted yet, fit it on internal data.
            if not hasattr(self.scaler, "scale_"):
                X_fit = self.df[self.feature_columns].fillna(0).values
                self.scaler.fit(X_fit)
            return self.scaler.transform(X)
        
        X_scaled = await loop.run_in_executor(None, scale_features)
        
        def make_predictions():
            return self.model.predict(X_scaled)
        
        predictions = await loop.run_in_executor(None, make_predictions)
        new_data["predicted_day_high_remaining"] = predictions[:, 0]
        new_data["predicted_day_low_remaining"] = predictions[:, 1]
        return new_data
    
    async def update_data(self, new_rows: pd.DataFrame):
        """
        Update the internal DataFrame with the new rows.
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
        Fine-tune the model on feedback data where the error is too high.
        This version computes errors separately for high and low predictions.
        """
        feedback_df = feedback_df.copy()
        for col in self.feature_columns + self.target_columns:
            if col not in feedback_df.columns:
                feedback_df[col] = 0
        loop = asyncio.get_running_loop()
        X_scaled, y_true = await self.preprocess_data(feedback_df)
        if X_scaled.size == 0 or y_true.size == 0:
            print("No valid feedback data found. Skipping learning from errors.")
            return
        
        def predict_and_calculate_error():
            predictions = self.model.predict(X_scaled)
            error_high = np.abs(predictions[:, 0] - y_true[:, 0])
            error_low  = np.abs(predictions[:, 1] - y_true[:, 1])
            return predictions, error_high, error_low
        
        predictions, error_high, error_low = await loop.run_in_executor(None, predict_and_calculate_error)
        print("Average high prediction error:", np.mean(error_high))
        print("Average low prediction error:", np.mean(error_low))
        
        # Select samples if either the high error or the low error exceeds its threshold.
        def select_samples():
            mask = (error_high > error_threshold) | (error_low > error_threshold)
            return np.where(mask)[0]
        
        mispred_indices = await loop.run_in_executor(None, select_samples)
        if len(mispred_indices) == 0:
            print("No significant errors detected; skipping fine-tuning.")
            return
        
        X_mispred = X_scaled[mispred_indices]
        y_mispred = y_true[mispred_indices]
        print(f"Fine-tuning on {len(mispred_indices)} mispredicted samples.")
        
        def fine_tune():
            self.model.fit(X_mispred, y_mispred, epochs=fine_tune_epochs, batch_size=16, verbose=1)
            self.model.save(self.model_path)
            print("Model fine-tuned and saved.")
        
        await loop.run_in_executor(None, fine_tune)
