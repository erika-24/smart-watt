import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataStorage:
    """
    Handles data storage and retrieval for the SmartWatt system
    """
    
    def __init__(self, data_dir: str = "data/storage"):
        """
        Initialize the data storage
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Define file paths
        self.energy_data_file = os.path.join(self.data_dir, "energy_data.json")
        self.forecast_data_file = os.path.join(self.data_dir, "forecast_data.json")
        self.optimization_results_file = os.path.join(self.data_dir, "optimization_results.json")
        self.devices_file = os.path.join(self.data_dir, "devices.json")
        self.system_config_file = os.path.join(self.data_dir, "system_config.json")
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize data files if they don't exist"""
        # Energy data
        if not os.path.exists(self.energy_data_file):
            with open(self.energy_data_file, 'w') as f:
                json.dump([], f)
        
        # Forecast data
        if not os.path.exists(self.forecast_data_file):
            with open(self.forecast_data_file, 'w') as f:
                json.dump([], f)
        
        # Optimization results
        if not os.path.exists(self.optimization_results_file):
            with open(self.optimization_results_file, 'w') as f:
                json.dump([], f)
        
        # Devices
        if not os.path.exists(self.devices_file):
            with open(self.devices_file, 'w') as f:
                json.dump([], f)
        
        # System config
        if not os.path.exists(self.system_config_file):
            # Default system configuration
            default_config = {
                "battery_capacity": 10.0,  # kWh
                "battery_max_power": 5.0,  # kW
                "battery_efficiency": 0.95,  # 95%
                "pv_capacity": 8.0,  # kW
                "grid_connection_capacity": 10.0,  # kW
            }
            with open(self.system_config_file, 'w') as f:
                json.dump(default_config, f)
    
    def save_energy_data(self, data: Dict[str, Any]):
        """
        Save energy data point
        
        Args:
            data: Energy data point
        """
        try:
            # Load existing data
            energy_data = self.get_energy_data()
            
            # Add new data point
            energy_data.append(data)
            
            # Keep only the last 7 days of data
            seven_days_ago = datetime.now() - timedelta(days=7)
            filtered_data = [
                point for point in energy_data 
                if datetime.fromisoformat(point["timestamp"]) >= seven_days_ago
            ]
            
            # Save to file
            with open(self.energy_data_file, 'w') as f:
                json.dump(filtered_data, f)
        
        except Exception as e:
            logger.error(f"Error saving energy data: {e}")
    
    def get_energy_data(self) -> List[Dict[str, Any]]:
        """
        Get all energy data
        
        Returns:
            List of energy data points
        """
        try:
            with open(self.energy_data_file, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error getting energy data: {e}")
            return []
    
    def save_forecast_data(self, data: Dict[str, Any]):
        """
        Save forecast data
        
        Args:
            data: Forecast data
        """
        try:
            with open(self.forecast_data_file, 'w') as f:
                json.dump(data, f)
        
        except Exception as e:
            logger.error(f"Error saving forecast data: {e}")
    
    def get_forecast_data(self) -> Dict[str, Any]:
        """
        Get forecast data
        
        Returns:
            Forecast data
        """
        try:
            with open(self.forecast_data_file, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error getting forecast data: {e}")
            return {}
    
    def save_optimization_result(self, result: Dict[str, Any]):
        """
        Save optimization result
        
        Args:
            result: Optimization result
        """
        try:
            # Load existing results
            results = self.get_optimization_results()
            
            # Add new result
            results.append(result)
            
            # Keep only the last 30 results
            if len(results) > 30:
                results = results[-30:]
            
            # Save to file
            with open(self.optimization_results_file, 'w') as f:
                json.dump(results, f)
        
        except Exception as e:
            logger.error(f"Error saving optimization result: {e}")
    
    def get_optimization_results(self) -> List[Dict[str, Any]]:
        """
        Get all optimization results
        
        Returns:
            List of optimization results
        """
        try:
            with open(self.optimization_results_file, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error getting optimization results: {e}")
            return []
    
    def get_optimization_result_by_id(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Get optimization result by ID
        
        Args:
            result_id: Optimization result ID
            
        Returns:
            Optimization result or None if not found
        """
        try:
            results = self.get_optimization_results()
            for result in results:
                if result.get("id") == result_id:
                    return result
            return None
        
        except Exception as e:
            logger.error(f"Error getting optimization result by ID: {e}")
            return None
    
    def save_devices(self, devices: List[Dict[str, Any]]):
        """
        Save devices
        
        Args:
            devices: List of devices
        """
        try:
            with open(self.devices_file, 'w') as f:
                json.dump(devices, f)
        
        except Exception as e:
            logger.error(f"Error saving devices: {e}")
    
    def get_devices(self) -> List[Dict[str, Any]]:
        """
        Get all devices
        
        Returns:
            List of devices
        """
        try:
            with open(self.devices_file, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error getting devices: {e}")
            return []
    
    def update_device(self, device_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update device
        
        Args:
            device_id: Device ID
            updates: Updates to apply
            
        Returns:
            Updated device or None if not found
        """
        try:
            devices = self.get_devices()
            for i, device in enumerate(devices):
                if device.get("id") == device_id:
                    devices[i] = {**device, **updates}
                    self.save_devices(devices)
                    return devices[i]
            return None
        
        except Exception as e:
            logger.error(f"Error updating device: {e}")
            return None
    
    def get_system_config(self) -> Dict[str, Any]:
        """
        Get system configuration
        
        Returns:
            System configuration
        """
        try:
            with open(self.system_config_file, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error getting system config: {e}")
            return {
                "battery_capacity": 10.0,  # kWh
                "battery_max_power": 5.0,  # kW
                "battery_efficiency": 0.95,  # 95%
                "pv_capacity": 8.0,  # kW
                "grid_connection_capacity": 10.0,  # kW
            }
    
    def update_system_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update system configuration
        
        Args:
            updates: Updates to apply
            
        Returns:
            Updated system configuration
        """
        try:
            config = self.get_system_config()
            config = {**config, **updates}
            with open(self.system_config_file, 'w') as f:
                json.dump(config, f)
            return config
        
        except Exception as e:
            logger.error(f"Error updating system config: {e}")
            return self.get_system_config()

