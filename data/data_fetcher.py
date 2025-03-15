import pandas as pd
import numpy as np
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Fetches energy data from various sources (Home Assistant, ESPHome, Solcast, etc.)
    """
    
    def __init__(self):
        """Initialize the data fetcher with API credentials"""
        # Home Assistant API
        self.ha_url = os.environ.get("HOME_ASSISTANT_URL")
        self.ha_token = os.environ.get("HOME_ASSISTANT_TOKEN")
        
        # ESPHome API
        self.esphome_url = os.environ.get("ESPHOME_API_URL")
        
        # Solcast API
        self.solcast_api_key = os.environ.get("SOLCAST_API_KEY")
        
        # Check if credentials are available
        if not self.ha_url or not self.ha_token:
            logger.warning("Home Assistant credentials not found in environment variables")
        
        if not self.esphome_url:
            logger.warning("ESPHome API URL not found in environment variables")
        
        if not self.solcast_api_key:
            logger.warning("Solcast API key not found in environment variables")
    
    def get_current_energy_data(self) -> Dict[str, Any]:
        """
        Get current energy data from Home Assistant
        
        Returns:
            Dictionary with current energy data
        """
        try:
            if not self.ha_url or not self.ha_token:
                return self._generate_mock_current_data()
            
            # Headers for Home Assistant API
            headers = {
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json",
            }
            
            # Get current state of energy-related entities
            entities = [
                "sensor.grid_power",
                "sensor.solar_power",
                "sensor.battery_power",
                "sensor.load_power",
                "sensor.battery_soc"
            ]
            
            states = {}
            for entity in entities:
                response = requests.get(f"{self.ha_url}/api/states/{entity}", headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    states[entity] = float(data["state"]) if data["state"] not in ["unknown", "unavailable"] else 0
                else:
                    logger.warning(f"Failed to get state for {entity}: {response.status_code}")
                    states[entity] = 0
            
            # Map entity names to our standard format
            return {
                "grid_power": states.get("sensor.grid_power", 0),
                "pv_power": states.get("sensor.solar_power", 0),
                "battery_power": states.get("sensor.battery_power", 0),
                "load_power": states.get("sensor.load_power", 0),
                "battery_soc": states.get("sensor.battery_soc", 50),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error fetching current energy data: {e}")
            # Fall back to mock data
            return self._generate_mock_current_data()
    
    def get_historical_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Get historical energy data from Home Assistant
        
        Args:
            start_time: Start time for historical data
            end_time: End time for historical data
            
        Returns:
            DataFrame with historical energy data
        """
        try:
            if not self.ha_url or not self.ha_token:
                return self._generate_mock_historical_data(start_time, end_time)
            
            # Headers for Home Assistant API
            headers = {
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json",
            }
            
            # Get historical data for energy-related entities
            entities = [
                "sensor.grid_power",
                "sensor.solar_power",
                "sensor.battery_power",
                "sensor.load_power"
            ]
            
            # Format timestamps for HA API
            start_str = start_time.isoformat()
            end_str = end_time.isoformat()
            
            all_data = {}
            for entity in entities:
                payload = {
                    "entity_id": entity,
                    "start_time": start_str,
                    "end_time": end_str,
                    "minimal_response": True
                }
                
                response = requests.post(
                    f"{self.ha_url}/api/history/period", 
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0 and len(data[0]) > 0:
                        # Extract timestamps and states
                        timestamps = [datetime.fromisoformat(item["last_updated"]) for item in data[0]]
                        states = [float(item["state"]) if item["state"] not in ["unknown", "unavailable"] else 0 for item in data[0]]
                        
                        all_data[entity] = pd.Series(states, index=timestamps)
                else:
                    logger.warning(f"Failed to get history for {entity}: {response.status_code}")
            
            # Combine all series into a DataFrame
            if all_data:
                df = pd.DataFrame(all_data)
                # Rename columns to our standard format
                df = df.rename(columns={
                    "sensor.grid_power": "grid_power",
                    "sensor.solar_power": "pv_power",
                    "sensor.battery_power": "battery_power",
                    "sensor.load_power": "load_power"
                })
                return df
            else:
                logger.warning("No historical data retrieved from Home Assistant")
                return self._generate_mock_historical_data(start_time, end_time)
        
        except Exception as e:
            logger.error(f"Error fetching historical energy data: {e}")
            # Fall back to mock data
            return self._generate_mock_historical_data(start_time, end_time)
    
    def get_pv_forecast(self, start_time: datetime, days: int = 1) -> pd.DataFrame:
        """
        Get solar PV forecast from Solcast
        
        Args:
            start_time: Start time for forecast
            days: Number of days to forecast
            
        Returns:
            DataFrame with PV forecast data
        """
        try:
            if not self.solcast_api_key:
                return self._generate_mock_pv_forecast(start_time, days)
            
            # Get site details (latitude, longitude, capacity)
            # In a real implementation, these would be stored in a config file or database
            latitude = 37.7749
            longitude = -122.4194
            capacity = 5.0  # kW
            
            # Solcast API endpoint
            url = f"https://api.solcast.com.au/rooftop_sites/forecasts?latitude={latitude}&longitude={longitude}&capacity={capacity}&api_key={self.solcast_api_key}&format=json"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                forecasts = data.get("forecasts", [])
                
                if forecasts:
                    # Convert to DataFrame
                    df = pd.DataFrame(forecasts)
                    
                    # Convert period_end to datetime and set as index
                    df["period_end"] = pd.to_datetime(df["period_end"])
                    df = df.set_index("period_end")
                    
                    # Filter to requested time range
                    end_time = start_time + timedelta(days=days)
                    df = df[(df.index >= start_time) & (df.index < end_time)]
                    
                    # Rename columns to our standard format
                    df = df.rename(columns={"pv_estimate": "pv_power"})
                    
                    return df
                else:
                    logger.warning("No forecast data received from Solcast")
            else:
                logger.warning(f"Failed to get PV forecast from Solcast: {response.status_code}")
            
            # Fall back to mock data
            return self._generate_mock_pv_forecast(start_time, days)
        
        except Exception as e:
            logger.error(f"Error fetching PV forecast: {e}")
            # Fall back to mock data
            return self._generate_mock_pv_forecast(start_time, days)
    
    def get_load_forecast(self, start_time: datetime, days: int = 1) -> pd.DataFrame:
        """
        Get load forecast based on historical data and day-ahead predictions
        
        Args:
            start_time: Start time for forecast
            days: Number of days to forecast
            
        Returns:
            DataFrame with load forecast data
        """
        try:
            # Get historical data for the same days of the week in the past few weeks
            # This is a simplified approach; a real implementation would use more sophisticated
            # forecasting methods like ARIMA, Prophet, or machine learning models
            
            # Get data for the past 4 weeks
            historical_start = start_time - timedelta(days=28)
            historical_end = start_time
            
            historical_data = self.get_historical_data(historical_start, historical_end)
            
            if historical_data.empty:
                return self._generate_mock_load_forecast(start_time, days)
            
            # Get the day of the week for the start time
            start_day = start_time.weekday()
            
            # Filter historical data to the same day of the week
            historical_data['day_of_week'] = historical_data.index.dayofweek
            same_day_data = historical_data[historical_data['day_of_week'] == start_day]
            
            if same_day_data.empty:
                return self._generate_mock_load_forecast(start_time, days)
            
            # Group by hour of day and calculate average load
            same_day_data['hour'] = same_day_data.index.hour
            hourly_avg = same_day_data.groupby('hour')['load_power'].mean()
            
            # Create forecast DataFrame
            forecast_index = pd.date_range(start=start_time, periods=24*days, freq='H')
            forecast_df = pd.DataFrame(index=forecast_index)
            
            # Fill with hourly averages
            forecast_df['hour'] = forecast_df.index.hour
            forecast_df['load_power'] = forecast_df['hour'].map(hourly_avg)
            
            # Add some random variation (±10%)
            np.random.seed(42)  # For reproducibility
            variation = np.random.uniform(0.9, 1.1, len(forecast_df))
            forecast_df['load_power'] = forecast_df['load_power'] * variation
            
            # Drop the hour column
            forecast_df = forecast_df.drop(columns=['hour'])
            
            return forecast_df
        
        except Exception as e:
            logger.error(f"Error generating load forecast: {e}")
            # Fall back to mock data
            return self._generate_mock_load_forecast(start_time, days)
    
    def get_devices(self) -> List[Dict[str, Any]]:
        """
        Get list of controllable devices from Home Assistant
        
        Returns:
            List of device dictionaries
        """
        try:
            if not self.ha_url or not self.ha_token:
                return self._generate_mock_devices()
            
            # Headers for Home Assistant API
            headers = {
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json",
            }
            
            # Get all switches and climate devices
            response = requests.get(f"{self.ha_url}/api/states", headers=headers)
            
            if response.status_code == 200:
                all_entities = response.json()
                
                devices = []
                
                # Process switches (on/off devices)
                for entity in all_entities:
                    entity_id = entity["entity_id"]
                    
                    # Check if it's a controllable device
                    if entity_id.startswith(("switch.", "light.", "climate.", "water_heater.", "fan.")):
                        device_type = "unknown"
                        if "switch.washing_machine" in entity_id or "switch.dishwasher" in entity_id:
                            device_type = "appliance"
                        elif "switch.ev_charger" in entity_id or "switch.car_charger" in entity_id:
                            device_type = "charger"
                        elif "light." in entity_id:
                            device_type = "lighting"
                        elif "climate." in entity_id or "fan." in entity_id:
                            device_type = "hvac"
                        elif "water_heater." in entity_id:
                            device_type = "heating"
                        
                        # Get power consumption if available
                        power = 0
                        power_entity_id = f"sensor.{entity_id.split('.')[1]}_power"
                        power_response = requests.get(f"{self.ha_url}/api/states/{power_entity_id}", headers=headers)
                        if power_response.status_code == 200:
                            power_data = power_response.json()
                            if power_data["state"] not in ["unknown", "unavailable"]:
                                power = float(power_data["state"])
                        
                        # Create device entry
                        device = {
                            "id": entity_id,
                            "name": entity["attributes"].get("friendly_name", entity_id),
                            "type": device_type,
                            "status": "online",
                            "power": power,
                            "isOn": entity["state"] in ["on", "heat", "cool"],
                            "icon": self._get_device_icon(device_type)
                        }
                        
                        # Add schedule if available
                        if "schedule" in entity["attributes"]:
                            device["schedule"] = entity["attributes"]["schedule"]
                        
                        devices.append(device)
                
                return devices
            else:
                logger.warning(f"Failed to get devices from Home Assistant: {response.status_code}")
                return self._generate_mock_devices()
        
        except Exception as e:
            logger.error(f"Error fetching devices: {e}")
            # Fall back to mock data
            return self._generate_mock_devices()
    
    def _generate_mock_current_data(self) -> Dict[str, Any]:
        """Generate mock current energy data"""
        hour = datetime.now().hour
        
        # Solar production follows a bell curve during daylight hours
        solar = 0
        if 6 <= hour <= 18:
            solar = 2.0 * np.sin(((hour - 6) * np.pi) / 12)
        
        # Load varies throughout the day
        load = 0.8 + 0.5 * np.sin(((hour - 8) * 2 * np.pi) / 24)
        
        # Battery charges during solar production, discharges at night
        battery = 0
        if solar > load:
            battery = min(solar - load, 1.0)  # Charging
        elif hour >= 18 or hour <= 6:
            battery = -min(load, 1.0)  # Discharging
        
        # Grid balances the system
        grid = load - solar - battery
        
        # Battery SOC varies between 20% and 90%
        battery_soc = 20 + 70 * (0.5 + 0.5 * np.sin(((hour - 12) * np.pi) / 12))
        
        return {
            "grid_power": float(grid),
            "pv_power": float(solar),
            "battery_power": float(battery),
            "load_power": float(load),
            "battery_soc": float(battery_soc),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_mock_historical_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Generate mock historical energy data"""
        # Calculate number of hours
        hours = int((end_time - start_time).total_seconds() / 3600) + 1
        
        # Create date range
        date_range = pd.date_range(start=start_time, periods=hours, freq='H')
        
        # Create DataFrame
        df = pd.DataFrame(index=date_range)
        
        # Generate data for each hour
        for i, dt in enumerate(date_range):
            hour = dt.hour
            
            # Solar production follows a bell curve during daylight hours
            solar = 0
            if 6 <= hour <= 18:
                solar = 2.0 * np.sin(((hour - 6) * np.pi) / 12)
            
            # Add some day-to-day variation
            day_factor = 0.8 + 0.4 * np.sin(dt.day_of_year * np.pi / 180)
            solar *= day_factor
            
            # Load varies throughout the day
            load = 0.8 + 0.5 * np.sin(((hour - 8) * 2 * np.pi) / 24)
            
            # Add some random variation
            load *= (0.9 + 0.2 * np.random.random())
            
            # Battery charges during solar production, discharges at night
            battery = 0
            if solar > load:
                battery = min(solar - load, 1.0)  # Charging
            elif hour >= 18 or hour <= 6:
                battery = -min(load, 1.0)  # Discharging
            
            # Grid balances the system
            grid = load - solar - battery
            
            # Add to DataFrame
            df.loc[dt, 'load_power'] = load
            df.loc[dt, 'pv_power'] = solar
            df.loc[dt, 'battery_power'] = battery
            df.loc[dt, 'grid_power'] = grid
        
        return df
    
    def _generate_mock_pv_forecast(self, start_time: datetime, days: int = 1) -> pd.DataFrame:
        """Generate mock PV forecast data"""
        # Create date range with hourly intervals
        hours = days * 24
        date_range = pd.date_range(start=start_time, periods=hours, freq='H')
        
        # Create DataFrame
        df = pd.DataFrame(index=date_range)
        
        # Generate forecast for each hour
        for i, dt in enumerate(date_range):
            hour = dt.hour
            
            # Solar production follows a bell curve during daylight hours
            solar = 0
            if 6 <= hour <= 18:
                solar = 2.0 * np.sin(((hour - 6) * np.pi) / 12)
            
            # Add some day-to-day variation
            day_factor = 0.8 + 0.4 * np.sin(dt.day_of_year * np.pi / 180)
            solar *= day_factor
            
            # Add to DataFrame
            df.loc[dt, 'pv_power'] = solar
        
        return df
    
    def _generate_mock_load_forecast(self, start_time: datetime, days: int = 1) -> pd.DataFrame:
        """Generate mock load forecast data"""
        # Create date range with hourly intervals
        hours = days * 24
        date_range = pd.date_range(start=start_time, periods=hours, freq='H')
        
        # Create DataFrame
        df = pd.DataFrame(index=date_range)
        
        # Generate forecast for each hour
        for i, dt in enumerate(date_range):
            hour = dt.hour
            
            # Load varies throughout the day
            # Morning peak, evening peak, lower at night
            if 6 <= hour < 9:  # Morning peak
                load = 1.2 + 0.3 * np.sin(((hour - 6) * np.pi) / 3)
            elif 17 <= hour < 22:  # Evening peak
                load = 1.5 + 0.5 * np.sin(((hour - 17) * np.pi) / 5)
            elif 22 <= hour or hour < 6:  # Night
                load = 0.5 + 0.2 * np.random.random()
            else:  # Daytime
                load = 0.8 + 0.3 * np.random.random()
            
            # Add some day-of-week variation
            day_of_week = (start_time.weekday() + (dt - start_time).days) % 7
            if day_of_week >= 5:  # Weekend
                load *= 1.2  # Higher load on weekends
            
            # Add to DataFrame
            df.loc[dt, 'load_power'] = load
        
        return df
    
    def _generate_mock_devices(self) -> List[Dict[str, Any]]:
        """Generate mock devices data for low wattage devices"""
        return [
            {
                "id": "1",
                "name": "RGB LED Strip",
                "type": "lighting",
                "status": "online",
                "power": 12,  # 12W
                "isOn": True,
                "icon": "lightbulb",
                "schedule": [{"start": "18:00", "end": "23:00", "power": 0.012}],
            },
            {
                "id": "2",
                "name": "Desk Fan",
                "type": "fan",
                "status": "online",
                "power": 25,  # 25W
                "isOn": True,
                "icon": "fan",
            },
            {
                "id": "3",
                "name": "PWM Water Pump",
                "type": "pump",
                "status": "offline",
                "power": 0,
                "isOn": False,
                "icon": "droplet",
                "schedule": [{"start": "08:00", "end": "08:30", "power": 0.035}],
            },
            {
                "id": "4",
                "name": "Servo Motor",
                "type": "motor",
                "status": "online",
                "power": 0,
                "isOn": False,
                "icon": "rotate-cw",
                "schedule": [{"start": "14:00", "end": "14:10", "power": 0.005}],
            },
            {
                "id": "5",
                "name": "Temperature Sensor",
                "type": "sensor",
                "status": "online",
                "power": 0.5,  # 0.5W
                "isOn": True,
                "icon": "thermometer",
            },
        ]
    
    def _get_device_icon(self, device_type: str) -> str:
        """Get icon name for device type"""
        icons = {
            "lighting": "lightbulb",
            "fan": "fan",
            "pump": "droplet",
            "motor": "rotate-cw",
            "sensor": "thermometer",
            "controller": "cpu",
        }
        return icons.get(device_type, "zap")

