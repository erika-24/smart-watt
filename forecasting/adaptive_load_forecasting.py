import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from typing import List, Dict, Optional, Any, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveLoadForecaster:
    """
    Adaptive load forecasting that learns from historical data and adapts to changing patterns
    """
    
    def __init__(self, model_path: str = "models"):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.rf_model = None
        self.gb_model = None
        self.model_accuracy = {}
        self.best_model = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Try to load pre-trained models if they exist
        try:
            self.rf_model = joblib.load(f"{model_path}/rf_load_model.pkl")
            self.gb_model = joblib.load(f"{model_path}/gb_load_model.pkl")
            self.scaler = joblib.load(f"{model_path}/load_scaler.pkl")
            
            # Load model accuracy metrics if available
            try:
                self.model_accuracy = joblib.load(f"{model_path}/model_accuracy.pkl")
                # Determine best model based on accuracy
                if self.model_accuracy:
                    self.best_model = min(self.model_accuracy.items(), key=lambda x: x[1]['rmse'])[0]
            except:
                pass
                
            logger.info(f"Loaded pre-trained models. Best model: {self.best_model}")
        except:
            logger.info("No pre-trained models found")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess historical load data for forecasting
        """
        # Convert to DataFrame if it's a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Ensure we have a datetime index
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
        
        # Ensure the data is sorted by timestamp
        data = data.sort_index()
        
        # Resample to regular intervals if needed
        data = data.resample('15T').mean().fillna(method='ffill')
        
        # Add time-based features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['is_weekend'] = data.index.dayofweek >= 5
        data['day_of_year'] = data.index.dayofyear
        data['week_of_year'] = data.index.isocalendar().week
        
        # Add lag features
        data['load_lag_1h'] = data['load_power'].shift(4)  # 4 x 15min = 1h
        data['load_lag_2h'] = data['load_power'].shift(8)  # 8 x 15min = 2h
        data['load_lag_1d'] = data['load_power'].shift(96)  # 96 x 15min = 1d
        data['load_lag_1w'] = data['load_power'].shift(672)  # 672 x 15min = 1w
        
        # Add rolling statistics
        data['load_rolling_mean_1h'] = data['load_power'].rolling(window=4).mean()
        data['load_rolling_std_1h'] = data['load_power'].rolling(window=4).std()
        data['load_rolling_mean_1d'] = data['load_power'].rolling(window=96).mean()
        
        # Fill NaN values
        data.fillna(method='bfill', inplace=True)
        
        return data
    
    def train_models(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train multiple models and evaluate their performance
        
        Args:
            historical_data: DataFrame with historical load data
            
        Returns:
            Dictionary with model accuracy metrics
        """
        # Preprocess data
        data = self.preprocess_data(historical_data)
        
        # Define features and target
        features = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'day_of_year', 'week_of_year',
            'load_lag_1h', 'load_lag_2h', 'load_lag_1d', 'load_lag_1w',
            'load_rolling_mean_1h', 'load_rolling_std_1h', 'load_rolling_mean_1d'
        ]
        
        # Check if all features are available
        available_features = [f for f in features if f in data.columns]
        
        X = data[available_features].values
        y = data['load_power'].values
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        logger.info("Training Random Forest model...")
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting model
        logger.info("Training Gradient Boosting model...")
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_pred = self.rf_model.predict(X_test_scaled)
        gb_pred = self.gb_model.predict(X_test_scaled)
        
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mae = mean_absolute_error(y_test, rf_pred)
        
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
        gb_mae = mean_absolute_error(y_test, gb_pred)
        
        # Save model accuracy metrics
        self.model_accuracy = {
            'random_forest': {'rmse': rf_rmse, 'mae': rf_mae},
            'gradient_boosting': {'rmse': gb_rmse, 'mae': gb_mae}
        }
        
        # Determine best model
        if rf_rmse <= gb_rmse:
            self.best_model = 'random_forest'
        else:
            self.best_model = 'gradient_boosting'
        
        # Save models and scaler
        joblib.dump(self.rf_model, f"{self.model_path}/rf_load_model.pkl")
        joblib.dump(self.gb_model, f"{self.model_path}/gb_load_model.pkl")
        joblib.dump(self.scaler, f"{self.model_path}/load_scaler.pkl")
        joblib.dump(self.model_accuracy, f"{self.model_path}/model_accuracy.pkl")
        
        logger.info(f"Models trained and saved. Best model: {self.best_model}")
        logger.info(f"Random Forest RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")
        logger.info(f"Gradient Boosting RMSE: {gb_rmse:.4f}, MAE: {gb_mae:.4f}")
        
        return self.model_accuracy
    
    def forecast(self, 
                historical_data: pd.DataFrame, 
                forecast_horizon: int, 
                time_step: int,
                method: str = None) -> List[float]:
        """
        Generate load forecast using the best model or specified method
        
        Args:
            historical_data: DataFrame with historical load data
            forecast_horizon: Number of hours to forecast
            time_step: Time step in minutes
            method: Forecasting method to use (random_forest, gradient_boosting, or None for best)
            
        Returns:
            List of forecasted load values
        """
        # If no method specified, use the best model
        if method is None:
            method = self.best_model or 'random_forest'
        
        # If models are not trained, train them
        if (method == 'random_forest' and self.rf_model is None) or \
           (method == 'gradient_boosting' and self.gb_model is None):
            self.train_models(historical_data)
        
        # Preprocess data
        data = self.preprocess_data(historical_data)
        
        # Define features
        features = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'day_of_year', 'week_of_year',
            'load_lag_1h', 'load_lag_2h', 'load_lag_1d', 'load_lag_1w',
            'load_rolling_mean_1h', 'load_rolling_std_1h', 'load_rolling_mean_1d'
        ]
        
        # Check if all features are available
        available_features = [f for f in features if f in data.columns]
        
        # Calculate number of intervals
        intervals = int((forecast_horizon * 60) / time_step)
        
        # Generate forecast
        forecast = []
        last_data = data.iloc[-1:].copy()
        
        for i in range(intervals):
            # Update time features for the next interval
            next_time = last_data.index[-1] + timedelta(minutes=time_step)
            next_row = pd.DataFrame(index=[next_time])
            next_row['hour'] = next_time.hour
            next_row['day_of_week'] = next_time.dayofweek
            next_row['month'] = next_time.month
            next_row['is_weekend'] = next_time.dayofweek >= 5
            next_row['day_of_year'] = next_time.dayofyear
            next_row['week_of_year'] = next_time.isocalendar().week
            
            # Use last known values for lag features
            next_row['load_lag_1h'] = last_data['load_power'].values[0]
            next_row['load_lag_2h'] = data[data.index <= next_time - timedelta(hours=2)]['load_power'].iloc[-1] if len(data[data.index <= next_time - timedelta(hours=2)]) > 0 else last_data['load_power'].values[0]
            next_row['load_lag_1d'] = data[data.index <= next_time - timedelta(days=1)]['load_power'].iloc[-1] if len(data[data.index <= next_time - timedelta(days=1)]) > 0 else last_data['load_power'].values[0]
            next_row['load_lag_1w'] = data[data.index <= next_time - timedelta(days=7)]['load_power'].iloc[-1] if len(data[data.index <= next_time - timedelta(days=7)]) > 0 else last_data['load_power'].values[0]
            
            # Calculate rolling statistics
            next_row['load_rolling_mean_1h'] = last_data['load_power'].values[0]  # Simplified
            next_row['load_rolling_std_1h'] = 0.1  # Simplified
            next_row['load_rolling_mean_1d'] = data['load_power'].mean()  # Simplified
            
            # Make prediction
            X = next_row[available_features].values
            X_scaled = self.scaler.transform(X)
            
            if method == 'random_forest':
                prediction = self.rf_model.predict(X_scaled)[0]
            elif method == 'gradient_boosting':
                prediction = self.gb_model.predict(X_scaled)[0]
            else:
                raise ValueError(f"Unknown forecasting method: {method}")
            
            # Add prediction to forecast
            forecast.append(prediction)
            
            # Update last_data for next iteration
            next_row['load_power'] = prediction
            last_data = next_row
        
        return forecast
    
    def forecast_from_current(self, 
                             current_load: float, 
                             forecast_horizon: int, 
                             time_step: int) -> List[float]:
        """
        Generate a load forecast based on current load when no historical data is available
        
        Args:
            current_load: Current load power in kW
            forecast_horizon: Number of hours to forecast
            time_step: Time step in minutes
            
        Returns:
            List of forecasted load values
        """
        # Calculate number of intervals
        intervals = int((forecast_horizon * 60) / time_step)
        
        # Get current time
        now = datetime.now()
        
        # Create a synthetic forecast based on typical residential patterns
        forecast = []
        
        for i in range(intervals):
            # Calculate time for this interval
            interval_time = now + timedelta(minutes=i * time_step)
            interval_hour = interval_time.hour
            interval_day = interval_time.weekday()
            
            # Base load factor on time of day and day of week
            if interval_day >= 5:  # Weekend
                if 0 <= interval_hour < 7:
                    # Night (low load)
                    load_factor = 0.6
                elif 7 <= interval_hour < 10:
                    # Morning (medium load)
                    load_factor = 0.9 + 0.2 * np.sin(np.pi * (interval_hour - 7) / 3)
                elif 10 <= interval_hour < 18:
                    # Day (medium-high load)
                    load_factor = 1.1
                elif 18 <= interval_hour < 23:
                    # Evening (high load)
                    load_factor = 1.3 + 0.3 * np.sin(np.pi * (interval_hour - 18) / 5)
                else:
                    # Late night (decreasing load)
                    load_factor = 0.8
            else:  # Weekday
                if 0 <= interval_hour < 6:
                    # Night (low load)
                    load_factor = 0.5
                elif 6 <= interval_hour < 9:
                    # Morning peak
                    load_factor = 1.2 + 0.4 * np.sin(np.pi * (interval_hour - 6) / 3)
                elif 9 <= interval_hour < 16:
                    # Daytime (medium load)
                    load_factor = 0.9
                elif 16 <= interval_hour < 22:
                    # Evening peak
                    load_factor = 1.5 + 0.5 * np.sin(np.pi * (interval_hour - 16) / 6)
                else:
                    # Late evening (decreasing load)
                    load_factor = 0.7
            
            # Add some randomness
            load_factor *= (0.9 + 0.2 * np.random.random())
            
            # Calculate load based on current load and factor
            load = current_load * load_factor
            
            # Ensure non-negative value
            load = max(0, load)
            
            forecast.append(load)
        
        return forecast

