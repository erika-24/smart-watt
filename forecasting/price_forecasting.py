import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import List, Dict, Optional, Any, Union
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PriceForecaster:
    """
    Class for forecasting electricity prices using various methods:
    1. Statistical methods (ARIMA)
    2. Machine learning (Random Forest)
    3. Pattern-based forecasting
    4. Time-of-use pricing
    5. Nordpool integration
    """
    
    def __init__(self, model_path: str = "models", nordpool_client=None):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.rf_model = None
        self.arima_model = None
        self.nordpool_client = nordpool_client
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Try to load pre-trained models if they exist
        try:
            self.rf_model = joblib.load(f"{model_path}/rf_price_model.pkl")
            self.scaler = joblib.load(f"{model_path}/price_scaler.pkl")
            logger.info("Loaded pre-trained Random Forest model for price forecasting")
        except:
            logger.info("No pre-trained Random Forest model found for price forecasting")
    
    def preprocess_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess historical price data for forecasting
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
        historical_data['price_lag_1h'] = historical_data['price'].shift(4)  # 4 x 15min = 1h
        historical_data['price_lag_1d'] = historical_data['price'].shift(96)  # 96 x 15min = 1d
        historical_data['price_lag_1w'] = historical_data['price'].shift(672)  # 672 x 15min = 1w
        
        # Fill NaN values
        historical_data.fillna(method='bfill', inplace=True)
        
        return historical_data
    
    def train_random_forest(self, historical_data: pd.DataFrame) -> None:
        """
        Train a Random Forest model for price forecasting
        """
        # Preprocess data
        data = self.preprocess_data(historical_data)
        
        # Define features and target
        features = ['hour', 'day_of_week', 'month', 'is_weekend', 
                   'price_lag_1h', 'price_lag_1d', 'price_lag_1w']
        X = data[features].values
        y = data['price'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest model
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_scaled, y)
        
        # Save model and scaler
        joblib.dump(self.rf_model, f"{self.model_path}/rf_price_model.pkl")
        joblib.dump(self.scaler, f"{self.model_path}/price_scaler.pkl")
        
        logger.info("Trained and saved Random Forest model for price forecasting")
    
    def train_arima(self, historical_data: pd.DataFrame) -> None:
        """
        Train an ARIMA model for price forecasting
        """
        # Preprocess data
        data = self.preprocess_data(historical_data)
        
        # Train ARIMA model (p=1, d=1, q=1) as a simple example
        # In practice, you would use auto_arima or grid search to find optimal parameters
        self.arima_model = ARIMA(data['price'], order=(1, 1, 1))
        self.arima_model = self.arima_model.fit()
        
        logger.info("Trained ARIMA model for price forecasting")
    
    def forecast_random_forest(self, 
                              historical_data: pd.DataFrame, 
                              forecast_horizon: int, 
                              time_step: int) -> List[float]:
        """
        Generate price forecast using Random Forest model
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
            next_row['price_lag_1h'] = last_data['price'].values[0]
            next_row['price_lag_1d'] = data[data.index <= next_time - timedelta(days=1)]['price'].iloc[-1] if len(data[data.index <= next_time - timedelta(days=1)]) > 0 else last_data['price'].values[0]
            next_row['price_lag_1w'] = data[data.index <= next_time - timedelta(days=7)]['price'].iloc[-1] if len(data[data.index <= next_time - timedelta(days=7)]) > 0 else last_data['price'].values[0]
            
            # Make prediction
            features = ['hour', 'day_of_week', 'month', 'is_weekend', 
                       'price_lag_1h', 'price_lag_1d', 'price_lag_1w']
            X = next_row[features].values
            X_scaled = self.scaler.transform(X)
            prediction = self.rf_model.predict(X_scaled)[0]
            
            # Add prediction to forecast
            forecast.append(prediction)
            
            # Update last_data for next iteration
            next_row['price'] = prediction
            last_data = next_row
        
        return forecast
    
    def forecast_arima(self, 
                      historical_data: pd.DataFrame, 
                      forecast_horizon: int, 
                      time_step: int) -> List[float]:
        """
        Generate price forecast using ARIMA model
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
    
    def forecast_pattern_based(self, 
                              historical_data: pd.DataFrame, 
                              forecast_horizon: int, 
                              time_step: int) -> List[float]:
        """
        Generate price forecast using pattern-based approach
        """
        # Preprocess data
        data = self.preprocess_data(historical_data)
        
        # Calculate number of intervals
        intervals = int((forecast_horizon * 60) / time_step)
        
        # Get current time
        now = datetime.now()
        
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
                # Use average price for this hour on similar days
                prediction = similar_hours['price'].mean()
            else:
                # Fallback to overall average
                prediction = data['price'].mean()
            
            forecast.append(prediction)
        
        return forecast
    
    def forecast_tou_pricing(self, 
                            forecast_horizon: int, 
                            time_step: int) -> Dict[str, List[float]]:
        """
        Generate price forecast using Time-of-Use pricing
        """
        # Calculate number of intervals
        intervals = int((forecast_horizon * 60) / time_step)
        
        # Get current time
        now = datetime.now()
        
        # Define Time-of-Use pricing periods
        # Example: Off-peak, mid-peak, on-peak
        off_peak_price = 0.08  # $/kWh
        mid_peak_price = 0.12  # $/kWh
        on_peak_price = 0.24   # $/kWh
        
        # Generate import and export price forecasts
        import_forecast = []
        export_forecast = []
        
        for i in range(intervals):
            # Calculate time for this interval
            interval_time = now + timedelta(minutes=i * time_step)
            interval_hour = interval_time.hour
            interval_day = interval_time.weekday()
            
            # Determine price based on time of day
            if interval_day >= 5:  # Weekend
                # Weekends are off-peak
                import_price = off_peak_price
            else:  # Weekday
                if 7 <= interval_hour < 11 or 17 <= interval_hour < 21:
                    # Morning and evening peaks
                    import_price = on_peak_price
                elif 11 <= interval_hour < 17:
                    # Mid-day
                    import_price = mid_peak_price
                else:
                    # Night
                    import_price = off_peak_price
            
            # Export price is typically lower than import price
            export_price = import_price * 0.3  # 30% of import price
            
            import_forecast.append(import_price)
            export_forecast.append(export_price)
        
        return {
            "import": import_forecast,
            "export": export_forecast
        }
    
    async def forecast_nordpool(self, forecast_horizon: int, time_step: int) -> Dict[str, List[float]]:
        """
        Generate price forecast using Nordpool data
        
        Args:
            forecast_horizon: Number of hours to forecast
            time_step: Time step in minutes
            
        Returns:
            Dictionary with import and export price forecasts
        """
        if self.nordpool_client is None:
            logger.warning("Nordpool client not available, using TOU pricing instead")
            return self.forecast_tou_pricing(forecast_horizon, time_step)
        
        try:
            # Get price forecast from Nordpool
            price_forecast = await self.nordpool_client.get_price_forecast(hours=forecast_horizon)
            
            # Calculate number of intervals
            intervals = int((forecast_horizon * 60) / time_step)
            
            # Interpolate hourly prices to the requested time step
            import_prices = price_forecast["import"]
            export_prices = price_forecast["export"]
            
            # If we have hourly prices but need finer granularity
            if time_step < 60 and len(import_prices) > 0:
                # Create hourly timestamps
                now = datetime.now().replace(minute=0, second=0, microsecond=0)
                hourly_times = [now + timedelta(hours=i) for i in range(len(import_prices))]
                
                # Create hourly dataframe
                hourly_df = pd.DataFrame({
                    'timestamp': hourly_times,
                    'import': import_prices,
                    'export': export_prices
                })
                hourly_df.set_index('timestamp', inplace=True)
                
                # Resample to requested time step
                step_df = hourly_df.resample(f'{time_step}T').interpolate(method='linear')
                
                # Extract resampled prices
                import_prices = step_df['import'].values[:intervals].tolist()
                export_prices = step_df['export'].values[:intervals].tolist()
            
            # Ensure we have enough prices for the requested intervals
            if len(import_prices) < intervals:
                # Pad with the last known price
                last_import = import_prices[-1] if import_prices else 0.1
                last_export = export_prices[-1] if export_prices else 0.03
                
                import_prices.extend([last_import] * (intervals - len(import_prices)))
                export_prices.extend([last_export] * (intervals - len(export_prices)))
            
            return {
                "import": import_prices[:intervals],
                "export": export_prices[:intervals]
            }
        
        except Exception as e:
            logger.error(f"Error getting Nordpool forecast: {e}")
            # Fallback to TOU pricing
            return self.forecast_tou_pricing(forecast_horizon, time_step)
    
    async def forecast(self, 
                historical_data: Optional[pd.DataFrame] = None, 
                forecast_horizon: int = 24, 
                time_step: int = 15, 
                method: str = "nordpool") -> Dict[str, List[float]]:
        """
        Generate price forecast using the specified method
        
        Args:
            historical_data: Historical price data (optional)
            forecast_horizon: Number of hours to forecast
            time_step: Time step in minutes
            method: Forecasting method to use
            
        Returns:
            Dictionary with import and export price forecasts
        """
        if method == "nordpool" and self.nordpool_client is not None:
            return await self.forecast_nordpool(forecast_horizon, time_step)
        elif method == "random_forest" and historical_data is not None:
            forecast = self.forecast_random_forest(historical_data, forecast_horizon, time_step)
            return {"import": forecast, "export": [price * 0.3 for price in forecast]}
        elif method == "arima" and historical_data is not None:
            forecast = self.forecast_arima(historical_data, forecast_horizon, time_step)
            return {"import": forecast, "export": [price * 0.3 for price in forecast]}
        elif method == "pattern" and historical_data is not None:
            forecast = self.forecast_pattern_based(historical_data, forecast_horizon, time_step)
            return {"import": forecast, "export": [price * 0.3 for price in forecast]}
        elif method == "tou":
            return self.forecast_tou_pricing(forecast_horizon, time_step)
        else:
            # Default to Time-of-Use pricing if no historical data or unknown method
            return self.forecast_tou_pricing(forecast_horizon, time_step)

