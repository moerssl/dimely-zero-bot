import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class AdaptiveDataframeProcessor:
    def __init__(self, rolling_window=15):
        self.df = pd.DataFrame()
        self.features = []
        self.high_model = None
        self.low_model = None
        self.rolling_window = rolling_window

    def update_dataframe(self, new_data):
        """
        Update the internal dataframe with new data. A copy is made to avoid modifying the caller’s data.
        """
        self.df = new_data.copy()
        print("Dataframe updated.")

    def add_features(self):
        """
        Add features to the internal dataframe.
        """
        if self.df.empty:
            raise ValueError("The dataframe is empty. Please update it with new data first.")

        required_columns = ['close', 'remaining_intervals']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.df['rolling_std_15'] = self.df['close'].rolling(window=self.rolling_window).std()
        self.df['momentum'] = self.df['close'] - self.df['close'].shift(5)
        self.df['time_to_close'] = self.df['remaining_intervals'] / self.df['remaining_intervals'].max()

        self.features = ['rolling_std_15', 'momentum', 'time_to_close']
        print("Features added.")

    def train_models(self, high_target: str, low_target: str):
        """
        Train RandomForest models to predict high and low prices.
        """
        if not self.features:
            raise ValueError("No features available for training. Please add features first.")

        required_columns = self.features + [high_target, low_target]
        for col in required_columns:
            if col not in self.df.columns:
                self.df[col] = np.nan

        data = self.df.dropna(subset=required_columns)
        if data.empty:
            raise ValueError("Not enough data to train models after dropping NaNs.")

        X = data[self.features]

        self.high_model = RandomForestRegressor(random_state=42)
        self.high_model.fit(X, data[high_target])

        self.low_model = RandomForestRegressor(random_state=42)
        self.low_model.fit(X, data[low_target])
        print("Models trained.")

    def predict_high_low_prices(self):
        """
        Use trained models to predict high and low prices on the internal dataframe.
        The predicted values are stored in the columns 'predicted_high' and 'predicted_low'.
        """
        if self.high_model is None or self.low_model is None:
            raise ValueError("Models are not trained yet. Train the models before making predictions.")
        if not self.features:
            raise ValueError("No features available for prediction. Please add features first.")

        feature_matrix = self.df[self.features].fillna(0).replace([np.inf, -np.inf], 0)

        self.df['predicted_high'] = self.high_model.predict(feature_matrix)
        self.df['predicted_low'] = self.low_model.predict(feature_matrix)
        print("Predictions made.")
        return self.df[['predicted_high', 'predicted_low']]

    def evaluate_predictions(self, high_actual: str, low_actual: str):
        """
        Evaluate predictions by comparing predicted values with the actual high/low prices.
        """
        if 'predicted_high' not in self.df.columns or 'predicted_low' not in self.df.columns:
            raise ValueError("No predictions found. Run predictions before evaluation.")

        self.df['high_prediction_correct'] = self.df['predicted_high'] >= self.df[high_actual]
        self.df['low_prediction_correct'] = self.df['predicted_low'] <= self.df[low_actual]
        print("Evaluation completed.")

    def adapt_models(self):
        """
        Adapt (retrain) the models using the rows where the predictions were incorrect.
        """
        if 'high_prediction_correct' not in self.df.columns or 'low_prediction_correct' not in self.df.columns:
            raise ValueError("Evaluation must be performed before adapting models.")

        incorrect_highs = self.df[self.df['high_prediction_correct'] == False]
        incorrect_lows = self.df[self.df['low_prediction_correct'] == False]

        if incorrect_highs.empty and incorrect_lows.empty:
            print("No incorrect predictions found. No adaptation needed.")
            return

        additional_data = pd.concat([incorrect_highs, incorrect_lows])
        additional_data = additional_data.dropna(subset=['predicted_high', 'predicted_low'])
        if additional_data.empty:
            print("No additional data available for adaptation after dropping NaNs.")
            return

        X_additional = additional_data[self.features]
        y_high_additional = additional_data['predicted_high']
        y_low_additional = additional_data['predicted_low']

        self.high_model = RandomForestRegressor(random_state=42)
        self.high_model.fit(X_additional, y_high_additional)

        self.low_model = RandomForestRegressor(random_state=42)
        self.low_model.fit(X_additional, y_low_additional)
        print("Models adapted using incorrect predictions as new training data.")

    def merge_predictions_into(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the prediction columns ('predicted_high' and 'predicted_low') from the internal dataframe 
        into the provided original dataframe. This is done based on the index alignment.
        
        Parameters:
            original_df (pd.DataFrame): The external dataframe that you want to update with predictions.
            
        Returns:
            pd.DataFrame: A copy of original_df with the prediction columns merged in.
        """
        if 'predicted_high' not in self.df.columns or 'predicted_low' not in self.df.columns:
            raise ValueError("No predictions found in the internal dataframe.")

        # Make a copy of the external dataframe to avoid modifying it directly
        merged_df = original_df.copy()
        # Join based on the index. This ensures that row 0 in self.df matches row 0 in merged_df.
        updated_predictions = self.df[['predicted_high', 'predicted_low']]
        merged_df = merged_df.join(updated_predictions, how='left')
        print("Predictions merged into the original dataframe.")
        return merged_df
