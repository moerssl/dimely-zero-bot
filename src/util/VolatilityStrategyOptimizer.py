import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats

class VolatilityStrategyOptimizer:
    def __init__(self, df, feature_columns=None):
        self.df = self._enhance_features(df.copy())
        self.features = feature_columns or self._get_default_features()
        self.strategy_rules = {
            'Call': {'analysis': None, 'rules': None, 'performance': None},
            'Put': {'analysis': None, 'rules': None, 'performance': None},
            'IronCondor': {'analysis': None, 'rules': None, 'performance': None}
        }
        
    def _enhance_features(self, df):
        """Add advanced technical features including Bollinger Band metrics"""
        # Calculate Bollinger Band distances
        df['bb_upper_dist_pct'] = ((df['bb_up'] - df['close']) / df['close']) * 100
        df['bb_mid_dist_pct'] = ((df['bb_mid'] - df['close']) / df['close']) * 100
        df['bb_lower_dist_pct'] = ((df['bb_low'] - df['close']) / df['close']) * 100
        
        # Calculate distances in ATR multiples
        df['bb_upper_dist_atr'] = (df['bb_up'] - df['close']) / df['ATR']
        df['bb_mid_dist_atr'] = (df['bb_mid'] - df['close']) / df['ATR']
        df['bb_lower_dist_atr'] = (df['bb_low'] - df['close']) / df['ATR']
        
        # Add volatility-adjusted features
        df['MACDHist_atr'] = df['MACDHist'] / df['ATR']
        df['RSI_volatility_ratio'] = df['RSI'] / df['ATR_percent']
        
        return df.sort_values('datetime').dropna()

    def _get_default_features(self):
        """Return updated list of technical features"""
        return [
            'RSI', 'ATR_percent', 'MACDHist', 'SMA5', 'SMA50',
            'band_width', 'volume', 'minutes_since_open',
            'bb_upper_dist_pct', 'bb_mid_dist_pct', 'bb_lower_dist_pct',
            'bb_upper_dist_atr', 'bb_mid_dist_atr', 'bb_lower_dist_atr',
            'MACDHist_atr', 'RSI_volatility_ratio'
        ]

    def _calculate_success(self):
        """Identify successful trades using remaining day levels"""
        self.df['Call_Success'] = (self.df['call_strike'] > self.df['day_high_remaining']).astype(int)
        self.df['Put_Success'] = (self.df['put_strike'] < self.df['day_low_remaining']).astype(int)
        self.df['IronCondor_Success'] = self.df['Call_Success'] & self.df['Put_Success']
        return self

    def _analyze_feature_distributions(self, strategy):
        """Statistical analysis of feature differences"""
        success = self.df[self.df[f'{strategy}_Success'] == 1][self.features]
        failure = self.df[self.df[f'{strategy}_Success'] == 0][self.features]
        
        results = {}
        for col in self.features:
            t_stat, p_val = stats.ttest_ind(success[col], failure[col], nan_policy='omit')
            results[col] = {
                'success_mean': success[col].mean(),
                'failure_mean': failure[col].mean(),
                'effect_size': success[col].mean() - failure[col].mean(),
                'p_value': p_val
            }
        return pd.DataFrame(results).T

    def _generate_rules(self, strategy, n_rules=3):
        """Generate trading rules based on feature analysis"""
        analysis = self.strategy_rules[strategy]['analysis']
        significant = analysis[analysis['p_value'] < 0.05].sort_values('effect_size', key=abs, ascending=False)
        
        rules = []
        for feature in significant.index[:n_rules]:
            success_q = np.percentile(self.df[self.df[f'{strategy}_Success'] == 1][feature], 75)
            failure_q = np.percentile(self.df[self.df[f'{strategy}_Success'] == 0][feature], 25)
            
            if significant.loc[feature, 'effect_size'] > 0:
                threshold = (success_q + failure_q) / 2
                rules.append(f"{feature} > {threshold:.2f}")
            else:
                threshold = (failure_q + success_q) / 2
                rules.append(f"{feature} < {threshold:.2f}")
                
        return rules

    def _backtest_rules(self, strategy, rules):
        """Backtest generated trading rules"""
        if not rules:
            return {'success_rate': 0, 'return': 0, 'trades': 0}
            
        filter_condition = " & ".join(rules)
        selected = self.df.query(filter_condition)
        
        if len(selected) == 0:
            return {'success_rate': 0, 'return': 0, 'trades': 0}
            
        success_rate = selected[f'{strategy}_Success'].mean()
        avg_return = selected[f'{strategy}_return'].mean()
        
        return {
            'success_rate': success_rate,
            'return': avg_return,
            'trades': len(selected),
            'rules': rules
        }

    def optimize(self):
        """Complete optimization pipeline"""
        self._calculate_success()
        
        for strategy in ['Call', 'Put', 'IronCondor']:
            # Feature analysis
            feature_analysis = self._analyze_feature_distributions(strategy)
            
            # Machine learning validation
            X = self.df[self.features].fillna(0)
            y = self.df[f'{strategy}_Success']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)
            feature_analysis['importance'] = model.feature_importances_
            
            # Rule generation and backtesting
            rules = self._generate_rules(strategy)
            performance = self._backtest_rules(strategy, rules)
            
            self.strategy_rules[strategy] = {
                'analysis': feature_analysis,
                'model': model,
                'performance': performance
            }
            
        return self

    def visualize_features(self, strategy, n_features=5):
        """Visualize key predictive features"""
        analysis = self.strategy_rules[strategy]['analysis']
        top_features = analysis.sort_values('importance', ascending=False).index[:n_features]
        
        plt.figure(figsize=(15, 3*n_features))
        for i, feature in enumerate(top_features, 1):
            plt.subplot(n_features, 1, i)
            
            success = self.df[self.df[f'{strategy}_Success'] == 1][feature]
            failure = self.df[self.df[f'{strategy}_Success'] == 0][feature]
            
            plt.hist(success, bins=50, alpha=0.5, label='Success')
            plt.hist(failure, bins=50, alpha=0.5, label='Failure')
            plt.title(f"{feature} Distribution (Strategy: {strategy})")
            plt.legend()
            
        plt.tight_layout()
        plt.show()

    def get_strategies(self):
        """Return formatted trading strategies"""
        strategies = {}
        for strategy in self.strategy_rules:
            perf = self.strategy_rules[strategy]['performance']
            rules = "\nAND ".join(perf['rules'])
            strategies[strategy] = (
                f"Strategy: {strategy}\n"
                f"Success Rate: {perf['success_rate']:.1%}\n"
                f"Average Return: {perf['return']:.2f}\n"
                f"Conditions:\n{rules}"
            )
        return strategies

# Example Usage
if __name__ == "__main__":
    # Load your data (example structure)
    data = pd.read_csv('SPX.csv', parse_dates=['datetime'])
    
    # Initialize and optimize strategies
    optimizer = VolatilityStrategyOptimizer(data)
    optimizer.optimize()
    
    # Display results
    strategies = optimizer.get_strategies()
    for strategy in strategies:
        print(strategies[strategy])
        print("\n" + "="*50 + "\n")
        
    # Visualize key features for Call strategy
    optimizer.visualize_features('Call')