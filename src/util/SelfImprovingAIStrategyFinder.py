from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import random

class SelfImprovingAIStrategyFinder:
    def __init__(self, dataframe):
        """
        Initialize with the given dataframe.
        """
        self.data = dataframe.copy()
        self.models = {
            'calls': None,
            'puts': None,
            'iron_condors': None
        }
        self.feature_conditions = {
            'calls': None,
            'puts': None,
            'iron_condors': None
        }
        self.previous_runs = {
            'calls': [],
            'puts': [],
            'iron_condors': []
        }
        self.generated_indicators = []  # To log and display generated indicators

    def generate_indicators(self):
        """
        Dynamically create new indicators by combining existing ones and normalizing absolute price values.
        """
        print("Generating indicators")
        indicators_log = []

        # Normalize Bollinger Band distances relative to price
        self.data['Rel_to_BB_Up_%'] = (self.data['bb_up'] - self.data['close']) / self.data['close'] * 100
        indicators_log.append("Rel_to_BB_Up_% = (bb_up - close) / close * 100")
        
        self.data['Rel_to_BB_Low_%'] = (self.data['close'] - self.data['bb_low']) / self.data['close'] * 100
        indicators_log.append("Rel_to_BB_Low_% = (close - bb_low) / close * 100")
        
        # ATR (volatility) based normalization
        self.data['Norm_to_BB_Up_ATR'] = (self.data['bb_up'] - self.data['close']) / (self.data['ATR'] + 1e-6)
        indicators_log.append("Norm_to_BB_Up_ATR = (bb_up - close) / ATR")
        
        self.data['Norm_to_BB_Low_ATR'] = (self.data['close'] - self.data['bb_low']) / (self.data['ATR'] + 1e-6)
        indicators_log.append("Norm_to_BB_Low_ATR = (close - bb_low) / ATR")

        # Add normalized differences between SMA values
        self.data['Rel_SMA_diff'] = (self.data['SMA5'] - self.data['SMA50']) / self.data['SMA50']
        indicators_log.append("Rel_SMA_diff = (SMA5 - SMA50) / SMA50")

        # RSI change indicator
        self.data['RSI_change'] = self.data['RSI'].diff()
        indicators_log.append("RSI_change = RSI.diff()")
        
        # Relative volatility signal (normalize bandwidth)
        self.data['volatility_signal'] = self.data['ATR_percent'] * self.data['band_width'] / self.data['close']
        indicators_log.append("volatility_signal = (ATR_percent * band_width) / close")

        # Use `narrow_bands` boolean directly (no change, already boolean)
        indicators_log.append("narrow_bands is retained as boolean")

        # Log all indicators
        self.generated_indicators = indicators_log
        print("Generated Indicators and Formulas:")
        for formula in indicators_log:
            print(f"  - {formula}")

    def prepare_data(self, trade_type):
        """
        Prepare the dataset for training based on the trade type.
        """
        # Define the win condition based on the trade type
        if trade_type == 'calls':
            self.data['Win_calls'] = (self.data['day_high_remaining'] < self.data['call_strike']).astype(int)
            target_column = 'Win_calls'
        elif trade_type == 'puts':
            self.data['Win_puts'] = (self.data['day_low_remaining'] > self.data['put_strike']).astype(int)
            target_column = 'Win_puts'
        elif trade_type == 'iron_condors':
            self.data['Win_iron_condors'] = ((self.data['day_low_remaining'] > self.data['put_strike']) & 
                                             (self.data['day_high_remaining'] < self.data['call_strike'])).astype(int)
            target_column = 'Win_iron_condors'
        else:
            raise ValueError("Invalid trade_type. Choose from 'calls', 'puts', or 'iron_condors'.")

        # Exclude unnecessary columns
        exclude_columns = ['call_strike', 'put_strike', 'day_high_remaining', 'day_low_remaining', 'close',
                           'count', 'x']  # Ignore absolute prices and unnecessary columns

        features = [col for col in self.data.columns if col not in exclude_columns + [target_column]]
        X = self.data[features].fillna(0)  # Replace NaNs with 0
        y = self.data[target_column]
        return X, y

    def train_model(self, trade_type):
        """
        Train a Gradient Boosting Classifier for the given trade type.
        """
        X, y = self.prepare_data(trade_type)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {trade_type.capitalize()}: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))

        # Store the model
        self.models[trade_type] = model

        # Generate feature conditions (e.g., RSI >= 40.2 or ATR <= 0.5)
        self.feature_conditions[trade_type] = self.get_feature_conditions(X_train, model)

        # Save this run's results for future learning
        self.previous_runs[trade_type].append({
            'accuracy': accuracy,
            'feature_conditions': self.feature_conditions[trade_type],
            'generated_indicators': self.generated_indicators
        })

    def get_feature_conditions(self, X, model):
        """
        Generate human-readable feature conditions (>= or <= thresholds) from the trained model.
        """
        feature_conditions = {}
        feature_importances = model.feature_importances_
        for idx, feature in enumerate(X.columns):
            if feature_importances[idx] > 0:  # Only include features with importance
                threshold_direction = ">=" if random.choice([True, False]) else "<="  # Randomly assign direction
                threshold_value = np.mean(X[feature])  # Use mean as a representative threshold for now
                feature_conditions[feature] = f"{threshold_direction} {threshold_value:.2f}"
        return feature_conditions

    def print_feature_conditions(self, trade_type):
        """
        Print the feature conditions for entering a trade based on the trained model.
        """
        if self.feature_conditions[trade_type] is None:
            raise ValueError(f"No feature conditions found for {trade_type}. Train the model first.")
        print(f"Feature Conditions for {trade_type.capitalize()}:")
        for feature, condition in self.feature_conditions[trade_type].items():
            print(f"  {feature} {condition}")

    def predict(self, trade_type, new_data):
        """
        Predict the win condition for new data using the trained model.
        """
        if self.models[trade_type] is None:
            raise ValueError(f"No model found for {trade_type}. Train the model first.")
        
        features = [col for col in new_data.columns if col in self.models[trade_type].feature_importances_]
        X_new = new_data[features].fillna(0)  # Ensure same features as training
        predictions = self.models[trade_type].predict(X_new)
        return predictions

if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('SPX.csv', parse_dates=['datetime', 'date_num'])

    # Drop unnecessary columns
    data = data.drop(columns=['date', 'datetime', 'date_num', 'count', 'x', 'final_signal', 'temp_signal', 'tech_signal'])

    # Map categorical values
    data['trend'] = data['trend'].map({'UP': 1, 'DOWN': 0})

    # Initialize the class
    ai_finder = SelfImprovingAIStrategyFinder(data)

    # Generate indicators
    ai_finder.generate_indicators()

    # Train models
    ai_finder.train_model(trade_type='calls')
    ai_finder.train_model(trade_type='puts')
    ai_finder.train_model(trade_type='iron_condors')

    # Print feature conditions
    ai_finder.print_feature_conditions(trade_type='calls')
    ai_finder.print_feature_conditions(trade_type='puts')
    ai_finder.print_feature_conditions(trade_type='iron_condors')
