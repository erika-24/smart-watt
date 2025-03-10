import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import List, Dict, Optional, Any, Union

class LoadForecaster:
    """
    Class for forecasting load demand using various methods:
    1. Statistical methods (ARIMA, SARIMA)
    2. Machine learning (Random Forest)
    3. Pattern-based forecasting
    4. Hybrid approaches
    """
    
    def __init__(self, model_path: str = "models"):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.rf_model = None
        self.arima_model = None
        self.sarima_model = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Try to load pre-trained models if they exist
        try:
            self.rf_model = joblib.load(f"{model_path}/rf_load_model.pkl")
            self.scaler = joblib.load(f"{model_path}/load_scaler.pkl")
            print("Loaded pre-trained Random Forest model")
        except:
            print("No pre-trained Random Forest model found")
    
    def preprocess_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess historical load data for forecasting
        """
        # Convert to DataFrame if it's a dictionary
        if isinstance(historical_data, dict):
            historical_data = pd.DataFrame(historical_data)
        
        # Ensure we have a datetime index
        if 'timestamp' in historical_data.columns:
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
            historical_data.set_index('timestamp', inplace=True)
        
        # Resample to regular intervals if needed
        historical_data = historical_data.resample('15T').mean().fillna(method='ffill')
        
        # Add time-based features
        historical_data['hour'] = historical_data.index.hour
        historical_data['day_of_week'] = historical_data.index.dayofweek
        historical_data['month'] = historical_data.index.month
        historical_data['is_weekend'] = historical_data.index.dayofweek >= 5
        
        # Add lag features
        historical_data['load_lag_1h'] = historical_data['load'].shift(4)  # 4 x 15min = 1h
        historical_data['load_lag_1d'] = historical_data['load'].shift(96)  # 96 x 15min = 1d
        historical_data['load_lag_1w'] = historical_data['load'].shift(672)  # 672 x 15min = 1w
        
        # Fill NaN values
        historical_data.fillna(method='bfill', inplace=True)
        
        return historical_data
    
    def train_random_forest(self, historical_data: pd.DataFrame) -> None:
        """
        Train a Random Forest model for load forecasting
        """
        # Preprocess data
        data = self.preprocess_data(historical_data)
        
        # Define features and target
        features = ['hour', 'day_of_week', 'month', 'is_weekend', 
                   'load_lag_1h', 'load_lag_1d', 'load_lag_1w']
        X = data[features].values
        y = data['load'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest model
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_scaled, y)
        
        # Save model and scaler
        joblib.dump(self.rf_model, f"{self.model_path}/rf_load_model.pkl")
        joblib.dump(self.scaler, f"{self.model_path}/load_scaler.pkl")
        
        print("Trained and saved Random Forest model")
    
    def train_arima(self, historical_data: pd.DataFrame) -> None:
        """
        Train an ARIMA model for load forecasting
        """
        # Preprocess data
        data = self.preprocess_data(historical_data)
        
        # Train ARIMA model (p=1, d=1, q=1) as a simple example
        # In practice, you would use auto_arima or grid search to find optimal parameters
        self.arima_model = ARIMA(data['load'], order=(1, 1, 1))
        self.arima_model = self.arima_model.fit()
        
        print("Trained ARIMA model")
    
    def train_sarima(self, historical_data: pd.DataFrame) -> None:
        """
        Train a SARIMA model for load forecasting
        """
        # Preprocess data
        data = self.preprocess_data(historical_data)
        
        # Train SARIMA model with daily seasonality (96 periods for 15-min data)
        # (p,d,q) x (P,D,Q,s) = (1,1,1) x (1,1,1,96)
        self.sarima_model = SARIMAX(
            data['load'], 
            order=(1, 1, 1), 
            seasonal_order=(1, 1, 1, 96)
        )
        self.sarima_model = self.sarima_model.fit(disp=False)
        
        print("Trained SARIMA model")
    
    def forecast_random_forest(self, 
                              historical_data: pd.DataFrame, 
                              forecast_horizon: int, 
                              time_step: int) -> List[float]:
        """
        Generate load forecast using Random Forest model
        """
        # Train model if not already trained
        if self.rf_model is None:
            self.train_random_forest(historical_data)
        
        # Preprocess data
        data = self.preprocess_data(historical_data)
        
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
            
            # Use last known values for lag features
            next_row['load_lag_1h'] = last_data['load'].values[0]
            next_row['load_lag_1d'] = data[data.index <= next_time - timedelta(days=1)]['load'].iloc[-1] if len(data[data.index <= next_time - timedelta(days=1)]) > 0 else last_data['load'].values[0]
            next_row['load_lag_1w'] = data[data.index <= next_time - timedelta(days=7)]['load'].iloc[-1] if len(data[data.index <= next_time - timedelta(days=7)]) > 0 else last_data['load'].values[0]
            
            # Make prediction
            features = ['hour', 'day_of_week', 'month', 'is_weekend', 
                       'load_lag_1h', 'load_lag_1d', 'load_lag_1w']
            X = next_row[features].values
            X_scaled = self.scaler.transform(X)
            prediction = self.rf_model.predict(X_scaled)[0]
            
            # Add prediction to forecast
            forecast.append(prediction)
            
            # Update last_data for next iteration
            next_row['load'] = prediction
            last_data = next_row
        
        return forecast
    
    def forecast_arima(self, 
                      historical_data: pd.DataFrame, 
                      forecast_horizon: int, 
                      time_step: int) -> List[float]:
        """
        Generate load forecast using ARIMA model
        """
        # Train model if not already trained
        if self.arima_model is None:
            self.train_arima(historical_data)
        
        # Calculate number of intervals
        intervals = int((forecast_horizon * 60) / time_step)
        
        # Generate forecast
        forecast = self.arima_model.forecast(steps=intervals)
        
        # Ensure non-negative values
        forecast = [max(0, value) for value in forecast]
        
        return forecast
    
    def forecast_sarima(self, 
                       historical_data: pd.DataFrame, 
                       forecast_horizon: int, 
                       time_step: int) -> List[float]:
        """
        Generate load forecast using SARIMA model
        """
        # Train model if not already trained
        if self.sarima_model is None:
            self.train_sarima(historical_data)
        
        # Calculate number of intervals
        intervals = int((forecast_horizon * 60) / time_step)
        
        # Generate forecast
        forecast = self.sarima_model.forecast(steps=intervals)
        
        # Ensure non-negative values
        forecast = [max(0, value) for value in forecast]
        
        return forecast
    
    def forecast_pattern_based(self, 
                              historical_data: pd.DataFrame, 
                              forecast_horizon: int, 
                              time_step: int) -> List[float]:
        """
        Generate load forecast using pattern-based approach
        """
        # Preprocess data
        data = self.preprocess_data(historical_data)
        
        # Calculate number of intervals
        intervals = int((forecast_horizon * 60) / time_step)
        
        # Get current time
        now = datetime.now()
        current_hour = now.hour
        current_day = now.weekday()
        
        # Generate forecast based on similar days
        forecast = []
        
        for i in range(intervals):
            # Calculate time for this interval
            interval_time = now + timedelta(minutes=i * time_step)
            interval_hour = interval_time.hour
            interval_day = interval_time.weekday()
            
            # Find similar days in historical data
            if interval_day >= 5:  # Weekend
                similar_days = data[data.index.dayofweek >= 5]
            else:  # Weekday
                similar_days = data[data.index.dayofweek < 5]
            
            # Filter for the same hour
            similar_hours = similar_days[similar_days.index.hour == interval_hour]
            
            if len(similar_hours) > 0:
                # Use average load for this hour on similar days
                prediction = similar_hours['load'].mean()
            else:
                # Fallback to overall average
                prediction = data['load'].mean()
            
            forecast.append(prediction)
        
        return forecast
    
    def forecast_hybrid(self, 
                       historical_data: pd.DataFrame, 
                       forecast_horizon: int, 
                       time_step: int) -> List[float]:
        """
        Generate load forecast using a hybrid approach (ensemble of methods)
        """
        # Get forecasts from different methods
        try:
            rf_forecast = self.forecast_random_forest(historical_data, forecast_horizon, time_step)
        except:
            rf_forecast = None
        
        try:
            arima_forecast = self.forecast_arima(historical_data, forecast_horizon, time_step)
        except:
            arima_forecast = None
        
        try:
            sarima_forecast = self.forecast_sarima(historical_data, forecast_horizon, time_step)
        except:
            sarima_forecast = None
        
        pattern_forecast = self.forecast_pattern_based(historical_data, forecast_horizon, time_step)
        
        # Combine forecasts (simple average of available methods)
        forecast = []
        intervals = int((forecast_horizon * 60) / time_step)
        
        for i in range(intervals):
            values = []
            
            if rf_forecast is not None and i < len(rf_forecast):
                values.append(rf_forecast[i])
            
            if arima_forecast is not None and i < len(arima_forecast):
                values.append(arima_forecast[i])
            
            if sarima_forecast is not None and i < len(sarima_forecast):
                values.append(sarima_forecast[i])
            
            if i < len(pattern_forecast):
                values.append(pattern_forecast[i])
            
            # Average of available forecasts
            if values:
                forecast.append(sum(values) / len(values))
            else:
                # Fallback to a simple heuristic
                forecast.append(1.0)  # Default 1 kW
        
        return forecast
    
    def forecast(self, 
                historical_data: pd.DataFrame, 
                forecast_horizon: int, 
                time_step: int, 
                method: str = "hybrid") -> List[float]:
        """
        Generate load forecast using the specified method
        """
        if method == "random_forest":
            return self.forecast_random_forest(historical_data, forecast_horizon, time_step)
        elif method == "arima":
            return self.forecast_arima(historical_data, forecast_horizon, time_step)
        elif method == "sarima":
            return self.forecast_sarima(historical_data, forecast_horizon, time_step)
        elif method == "pattern":
            return self.forecast_pattern_based(historical_data, forecast_horizon, time_step)
        elif method == "hybrid":
            return self.forecast_hybrid(historical_data, forecast_horizon, time_step)
        else:
            raise ValueError(f"Unknown forecasting method: {method}")
    
    def forecast_from_current(self, 
                             current_load: float, 
                             forecast_horizon: int, 
                             time_step: int) -> List[float]:
        """
        Generate a simple load forecast based on current load and typical patterns
        """
        # Calculate number of intervals
        intervals = int((forecast_horizon * 60) / time_step)
        
        # Get current time
        now = datetime.now()
        current_hour = now.hour
        
        # Create a synthetic forecast based on typical residential patterns
        forecast = []
        
        for i in range(intervals):
            # Calculate time for this interval
            interval_time = now + timedelta(minutes=i * time_step)
            interval_hour = interval_time.hour
            
            # Base load factor on time of day
            if 0 <= interval_hour < 6:
                # Night (low load)
                load_factor = 0.6
            elif 6 <= interval_hour < 9:
                # Morning peak
                load_factor = 1.2 + 0.3 * np.sin(np.pi * (interval_hour - 6) / 3)
            elif 9 <= interval_hour < 16:
                # Daytime (medium load)
                load_factor = 0.9
            elif 16 <= interval_hour < 22:
                # Evening peak
                load_factor = 1.5 + 0.5 * np.sin(np.pi * (interval_hour - 16) / 6)
            else:
                # Late evening (decreasing load)
                load_factor = 1.0
            
            # Add some randomness
            load_factor *= (0.9 + 0.2 * np.random.random())
            
            # Calculate load based on current load and factor
            load = current_load * load_factor
            
            # Ensure non-negative value
            load = max(0, load)
            
            forecast.append(load)
        
        return forecast

