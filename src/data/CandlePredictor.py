import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import joblib

class CandlePredictor:
    """
    This class handles an intraday candle DataFrame that must include at least:
      - 'date', 'open', 'high', 'low'
    
    In my experience as a successful options trader, predicting the remaining daily range is best
    approached by fusing price action with volatility and technical indicators. In addition to the basic
    fields, the class will automatically compute additional technical indicators if a 'close' column is available,
    such as:
      - SMA_10, SMA_20,
      - RSI_14, and
      - ATR_14.
    
    Optional columns like 'IV_close', 'volume', or 'trend' (if provided) are automatically appended
    to the feature set.
    
    The model uses a MultiOutput XGBoost regressor (wrapped in scikit‑learn’s MultiOutputRegressor)
    to forecast the final day high and low. From these final predictions, it computes the remaining move,
    which is of particular interest when structuring options positions.
    """

    # The minimum required columns.
    REQUIRED_COLUMNS = ["date", "open", "high", "low"]

    def __init__(self, df: pd.DataFrame = None, xgb_params: dict = None):
        # Define optional indicator columns that might be provided in the raw data.
        self.optional_features = []
        for col in ["IV_close", "volume", "trend"]:
            self.optional_features.append(col)
        
        # If no data is provided, set defaults.
        if df is None or df.empty:
            print("Warning: No data provided at __init__; update data later using update_dataframe().")
            self.df = pd.DataFrame()
            self.market_close_time = None
            self.avg_daily_range = None
        else:
            missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            self.df = self._process_date_column(df.copy())
            self._enrich()  # computes candle duration and technical indicators if possible
            self.market_close_time = self._estimate_market_close_time()
            self.avg_daily_range = self._calculate_avg_daily_range()
        
        self.multiplier = 0.5
        # Use XGBoost as our base regressor, wrapped for multi-output.
        if xgb_params is None:
            xgb_params = {
                "objective": "reg:squarederror",
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": 42,
            }
        self.xgb_params = xgb_params
        self.ai_model = None

    def _process_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the 'date' column to datetime.
        (Optionally, you can add timezone conversion, e.g. to Berlin time.)
        Then sets 'day' (the date portion) and 'market_open' (first candle's open each day).
        """
        if df["date"].dtype == object:
            temp = pd.to_numeric(df["date"], errors="coerce")
            if temp.notna().all():
                df["datetime"] = pd.to_datetime(temp, unit="s")
            else:
                df["datetime"] = pd.to_datetime(df["date"])
        elif np.issubdtype(df["date"].dtype, np.number):
            df["datetime"] = pd.to_datetime(df["date"], unit="s")
        else:
            df["datetime"] = pd.to_datetime(df["date"])
        df.sort_values("datetime", inplace=True)
        df["day"] = df["datetime"].dt.date
        df["market_open"] = df.groupby("day")["open"].transform("first")
        return df

    def _enrich(self):
        """
        Adds the following to self.df:
         - 'candle_duration_min': Minutes between consecutive candles.
         - Technical indicators (if a 'close' column is present): SMA_10, SMA_20, RSI_14, ATR_14.
        """
        if self.df.empty:
            return
        # Calculate candle duration.
        self.df["candle_duration_min"] = self.df.groupby("day")["datetime"].diff().dt.total_seconds() / 60
        mode_val = self.df["candle_duration_min"].mode()
        common = mode_val.iloc[0] if not mode_val.empty else np.nan
        self.df["candle_duration_min"].fillna(common, inplace=True)
        
        # Compute technical indicators if 'close' is present.
        if "close" in self.df.columns:
            # Simple Moving Averages
            self.df["SMA_10"] = self.df["close"].rolling(window=10, min_periods=1).mean()
            self.df["SMA_20"] = self.df["close"].rolling(window=20, min_periods=1).mean()
            
            # RSI_14 (Relative Strength Index)
            delta = self.df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            rs = avg_gain / avg_loss
            self.df["RSI_14"] = 100 - (100 / (1 + rs))
            self.df["RSI_14"].fillna(50, inplace=True)  # Default neutral RSI
            
            # ATR_14 (Average True Range)
            high_low = self.df["high"] - self.df["low"]
            high_close = (self.df["high"] - self.df["close"].shift()).abs()
            low_close = (self.df["low"] - self.df["close"].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.df["ATR_14"] = true_range.rolling(window=14, min_periods=1).mean()

    def _estimate_market_close_time(self):
        """
        Determines the typical market close time from complete days,
        by taking the mode of each day’s last candle’s time.
        """
        if self.df.empty:
            return None
        close_times = self.df.groupby("day")["datetime"].max()
        return close_times.mode()[0].time()

    def _calculate_avg_daily_range(self):
        """
        Computes the average daily range (mean over days of (max(high) - min(low))).
        """
        if self.df.empty:
            return None
        daily = self.df.groupby("day").agg({"high": "max", "low": "min"}).reset_index()
        daily["range"] = daily["high"] - daily["low"]
        return daily["range"].mean()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds the full feature matrix for the AI model.
        Base features: time_ratio, high, low, market_open.
        Adds optional ones: technical indicators (SMA_10, SMA_20, RSI_14, ATR_14) and any from self.optional_features.
        Ensures that all feature columns are numeric.
        """
        df = df.copy()
        # First, compute time_ratio.
        df["market_open_dt"] = df.groupby("day")["datetime"].transform("first")
        close_delta = pd.Timedelta(
            hours=self.market_close_time.hour,
            minutes=self.market_close_time.minute,
            seconds=self.market_close_time.second
        )
        df["market_close_dt"] = pd.to_datetime(df["day"].astype(str)) + close_delta
        df["total_trading_time"] = (df["market_close_dt"] - df["market_open_dt"]).apply(lambda td: td.total_seconds())
        df["elapsed_time"] = (df["datetime"] - df["market_open_dt"]).apply(lambda td: td.total_seconds())
        df["time_ratio"] = df["elapsed_time"] / df["total_trading_time"]

        base = ["time_ratio", "high", "low", "market_open"]
        # Add technical indicators if present.
        tech = []
        for col in ["SMA_10", "SMA_20", "RSI_14", "ATR_14"]:
            if col in df.columns:
                tech.append(col)
        feat_cols = base + self.optional_features + tech
        
        # Select the features DataFrame.
        X = df[feat_cols].copy()
        # Convert all feature columns to numeric (this will convert object types such as 'volume' to numbers)
        X = X.apply(pd.to_numeric, errors="coerce")
        # Fill any missing values with forward fill or 0.
        X.fillna(method="ffill", inplace=True)
        X.fillna(0, inplace=True)
        return X


    def train_ai_model(self):
        """
        Trains the AI model to predict the additional (residual) move for the day.
        For each complete day, we define:
            residual_high = final_day_high - current_high,
            residual_low = current_low - final_day_low.
        We then build features (including time_ratio) and train the model
        to predict these residuals.
        """
        if self.df.empty:
            raise ValueError("No data available; update the DataFrame first.")
        df_all = self.df.copy()
        complete_days = df_all.groupby("day").filter(
            lambda grp: grp["datetime"].max().time() == self.market_close_time
        )["day"].unique()
        df_train = df_all[df_all["day"].isin(complete_days)].copy()
        
        # Compute observed final extremes per day.
        df_train["final_day_high"] = df_train.groupby("day")["high"].transform("max")
        df_train["final_day_low"] = df_train.groupby("day")["low"].transform("min")
        
        # For training, consider the last candle of each complete day:
        df_last = df_train.groupby("day").tail(1).copy()
        # Compute the residual moves
        df_last["residual_high"] = df_last["final_day_high"] - df_last["high"]
        df_last["residual_low"] = df_last["low"] - df_last["final_day_low"]
        
        # Build features for these last observations.
        X = self._build_features(df_last)
        y = df_last[["residual_high", "residual_low"]]
        
        from sklearn.multioutput import MultiOutputRegressor
        self.ai_model = MultiOutputRegressor(xgb.XGBRegressor(**self.xgb_params))
        self.ai_model.fit(X, y)
        print(f"Model trained on {len(X)} samples with targets residuals.")
        
    def predict_day_extremes_ai(self) -> dict:
        """
        Uses the trained residual model to predict the additional move.
        
        For the latest candle, it predicts:
        - predicted_residual_high: the additional upward move expected.
        - predicted_residual_low: the additional downward move expected.
        
        Then, the final remaining day predictions are:
        - remaining_day_high = current high + predicted_residual_high
        - remaining_day_low  = current low - predicted_residual_low
        """
        if self.ai_model is None:
            raise ValueError("AI model not trained; run train_ai_model() first.")
        if self.df.empty:
            raise ValueError("No data available; update the DataFrame first.")
        
        latest = self.df.iloc[-1]
        # Extract the latest candle into a one-row DataFrame.
        df_current = self.df[self.df["datetime"] == latest["datetime"]].copy()
        X_new = self._build_features(df_current)
        preds = self.ai_model.predict(X_new)[0]
        predicted_residual_high = preds[0]
        predicted_residual_low = preds[1]

        remaining_day_high = latest["high"] + predicted_residual_high
        remaining_day_low = latest["low"] - predicted_residual_low

        return {
            "predicted_remaining_day_high": remaining_day_high,
            "predicted_remaining_day_low": remaining_day_low
        }


    def predict_all_candles_ai(self) -> pd.DataFrame:
        """
        Predicts the additional move (residual) for every candle,
        then generates the final remaining day extremes by:
            remaining_day_high = current high + predicted residual_high,
            remaining_day_low  = current low - predicted residual_low.
        """
        if self.df.empty:
            raise ValueError("No data available; update the DataFrame first.")
        if self.ai_model is None:
            raise ValueError("AI model not trained; run train_ai_model() first.")
            
        df_new = self.df.copy().sort_values("datetime")
        # Compute time features as before.
        df_new["market_open_dt"] = df_new.groupby("day")["datetime"].transform("first")
        close_delta = pd.Timedelta(
            hours=self.market_close_time.hour,
            minutes=self.market_close_time.minute,
            seconds=self.market_close_time.second
        )
        df_new["market_close_dt"] = pd.to_datetime(df_new["day"].astype(str)) + close_delta
        df_new["total_trading_time"] = (df_new["market_close_dt"] - df_new["market_open_dt"]).apply(lambda td: td.total_seconds())
        df_new["elapsed_time"] = (df_new["datetime"] - df_new["market_open_dt"]).apply(lambda td: td.total_seconds())
        df_new["time_ratio"] = df_new["elapsed_time"] / df_new["total_trading_time"]
        
        # Build feature matrix.
        X = self._build_features(df_new)
        preds = self.ai_model.predict(X)
        # Predicted residual moves:
        df_new["predicted_residual_high"] = preds[:, 0]
        df_new["predicted_residual_low"] = preds[:, 1]
        
        # Final predicted extremes are computed by adding/subtracting the residual move:
        df_new["remaining_day_high"] = df_new["high"] + df_new["predicted_residual_high"]
        df_new["remaining_day_low"] = df_new["low"] - df_new["predicted_residual_low"]
        
        return df_new



    def evaluate_and_adjust_model(self):
        """
        Evaluates the model on complete days and adjusts it using residual errors.
        
        For each complete day (using the final candle as the representative observation):
        - observed_residual_high = final_day_high - current_high
        - observed_residual_low  = current_low - final_day_low
        
        (These targets capture how much extra move occurred after the current candle.)
        
        The model’s prediction for the residual is then compared to the observed residual;
        a correction factor is applied to generate adjusted targets, and the model is retrained.
        """
        if self.ai_model is None:
            raise ValueError("AI model not trained; run train_ai_model() first.")
        if self.df.empty:
            raise ValueError("No data available; update the DataFrame first.")

        df_all = self.df.copy()
        # Identify complete days: days where the last candle’s time equals the estimated market close.
        complete_days = df_all.groupby("day").filter(
            lambda grp: grp["datetime"].max().time() == self.market_close_time
        )["day"].unique()
        # For evaluation, take the last candle from each complete day.
        df_complete = df_all[df_all["day"].isin(complete_days)].copy()
        df_last = df_complete.groupby("day").tail(1).copy()

        # For each complete day, use all the candles to compute the observed final extremes.
        # (Here we recompute the final day's high and low.)
        df_last["final_day_high"] = df_complete.groupby("day")["high"].transform("max")
        df_last["final_day_low"] = df_complete.groupby("day")["low"].transform("min")

        # Compute residual targets from the last candle.
        df_last["observed_residual_high"] = df_last["final_day_high"] - df_last["high"]
        df_last["observed_residual_low"]  = df_last["low"] - df_last["final_day_low"]

        # Build feature matrix using the last candle of each day.
        X_last = self._build_features(df_last)
        preds = self.ai_model.predict(X_last)
        df_last["predicted_residual_high"] = preds[:, 0]
        df_last["predicted_residual_low"] = preds[:, 1]

        # Compute error: (observed - predicted). A positive error means the model underpredicted the extra move.
        df_last["error_high"] = df_last["observed_residual_high"] - df_last["predicted_residual_high"]
        df_last["error_low"]  = df_last["observed_residual_low"] - df_last["predicted_residual_low"]

        avg_error_high = df_last["error_high"].mean()
        avg_error_low = df_last["error_low"].mean()

        # Apply a correction factor so the adjusted target nudges the model in the right direction.
        correction_factor = 0.1  # This parameter can be tuned.
        df_last["adjusted_residual_high"] = df_last["observed_residual_high"] + correction_factor * df_last["error_high"]
        df_last["adjusted_residual_low"] = df_last["observed_residual_low"] + correction_factor * df_last["error_low"]

        # Build new training targets and retrain the model.
        X_new = self._build_features(df_last)
        y_new = df_last[["adjusted_residual_high", "adjusted_residual_low"]]
        from sklearn.multioutput import MultiOutputRegressor
        # Reinitialize (or update) the model with the same XGBoost parameters.
        self.ai_model = MultiOutputRegressor(xgb.XGBRegressor(**self.xgb_params))
        self.ai_model.fit(X_new, y_new)

        print(f"Model adjusted: avg residual high error = {avg_error_high:.4f}, avg residual low error = {avg_error_low:.4f}")


    def update_dataframe(self, new_df: pd.DataFrame, mode: str = "append"):
        """
        Updates the internal DataFrame with new data.
          - mode 'append': new data is added.
          - mode 'replace': current data is overwritten.
        The new DataFrame must include the required columns.
        After processing via _process_date_column(), the DataFrame is enriched and key values are recalculated.
        """
        if new_df is None or new_df.empty:
            raise ValueError("The new DataFrame provided is empty.")
        missing = [col for col in self.REQUIRED_COLUMNS if col not in new_df.columns]
        if missing:
            raise ValueError(f"New DataFrame is missing required columns: {missing}")
        new_data = self._process_date_column(new_df.copy())
        if mode == "append":
            self.df = pd.concat([self.df, new_data], ignore_index=True)
        elif mode == "replace":
            self.df = new_data.copy()
        else:
            raise ValueError("Mode must be 'append' or 'replace'.")
        self.df.sort_values("datetime", inplace=True)
        self._enrich()
        self.market_close_time = self._estimate_market_close_time()
        self.avg_daily_range = self._calculate_avg_daily_range()
        print(f"DataFrame updated. Current number of records: {len(self.df)}")

    def save_model(self, filename="model.pkl"):
        """
        Saves the trained ai_model to disk.
        """
        if self.ai_model is None: 
            raise ValueError("No model to save. Train the model first.")
        joblib.dump(self.ai_model, filename)
        print(f"Model saved to {filename}.")

    def load_model(self, filename="model.pkl"):
        """
        Loads a previously saved ai_model from disk.
        """
        try:
            self.ai_model = joblib.load(filename)
            print(f"Model loaded from {filename}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {filename} not found.")