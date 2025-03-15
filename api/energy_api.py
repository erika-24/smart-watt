from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import asyncio
import uuid

from data.data_fetcher import DataFetcher
from optimization.linear_program import EnergyOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Energy Management API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data fetcher
data_fetcher = DataFetcher()

# Store for optimization results
optimization_results = {}

# Background optimization tasks
optimization_tasks = {}

# Models
class OptimizationParams(BaseModel):
    optimizationMode: str
    timeHorizon: str
    batteryConstraints: Dict[str, Any]
    gridConstraints: Dict[str, Any]

class OptimizationResult(BaseModel):
    id: str
    cost: float
    selfConsumption: float
    peakGridPower: float
    batteryCycles: float
    timestamp: str
    scheduleData: List[Dict[str, Any]]
    deviceScheduleData: List[Dict[str, Any]]

# Routes
@app.get("/api/energy/current")
async def get_current_energy_data():
    """Get current energy data"""
    try:
        data = await data_fetcher.get_current_energy_data()
        
        # Convert to format expected by frontend
        result = []
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            
            # Scale data based on time of day
            time_factor = 1.0
            if hour == datetime.now().hour:
                time_factor = 1.0
            elif abs(hour - datetime.now().hour) <= 1:
                time_factor = 0.9
            elif abs(hour - datetime.now().hour) <= 2:
                time_factor = 0.8
            else:
                time_factor = 0.7
            
            # Generate hourly data based on current values
            hour_data = {
                "time": hour_str,
                "consumption": data["load_power"] * time_factor,
                "solar": 0,
                "battery": 0,
                "grid": 0
            }
            
            # Solar only during daylight
            if 6 <= hour <= 18:
                solar_factor = np.sin(((hour - 6) * np.pi) / 12)
                hour_data["solar"] = data["pv_power"] * solar_factor
            
            # Battery and grid balance the system
            if hour_data["solar"] > hour_data["consumption"]:
                hour_data["battery"] = min(hour_data["solar"] - hour_data["consumption"], data["battery_power"])
                hour_data["grid"] = hour_data["consumption"] - hour_data["solar"] - hour_data["battery"]
            else:
                deficit = hour_data["consumption"] - hour_data["solar"]
                hour_data["battery"] = -min(deficit * 0.7, abs(data["battery_power"]))
                hour_data["grid"] = deficit + hour_data["battery"]
            
            result.append(hour_data)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in get_current_energy_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/energy/forecast")
async def get_energy_forecast(date: str = None):
    """Get energy forecast data"""
    try:
        # Parse date or use today
        if date:
            forecast_date = datetime.fromisoformat(date)
        else:
            forecast_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get PV and load forecasts
        pv_forecast = await data_fetcher.get_pv_forecast(forecast_date)
        load_forecast = await data_fetcher.get_load_forecast(forecast_date)
        
        # Combine forecasts
        combined = pd.concat([pv_forecast, load_forecast], axis=1)
        
        # Generate optimized load (simplified version)
        combined['optimizedLoad'] = combined['load_power'] * 0.9
        
        # Convert to list of dictionaries
        result = []
        for dt, row in combined.iterrows():
            result.append({
                "time": dt.strftime("%H:%M"),
                "load": float(row['load_power']),
                "solar": float(row['pv_power']),
                "optimizedLoad": float(row['optimizedLoad'])
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error in get_energy_forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/energy/history")
async def get_energy_history(timeRange: str = "day", date: str = None):
    """Get historical energy data"""
    try:
        # Parse date or use today
        if date:
            history_date = datetime.fromisoformat(date)
        else:
            history_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Determine time range
        if timeRange == "day":
            start_time = history_date
            end_time = start_time + timedelta(days=1)
        elif timeRange == "week":
            start_time = history_date - timedelta(days=history_date.weekday())
            end_time = start_time + timedelta(days=7)
        elif timeRange == "month":
            start_time = history_date.replace(day=1)
            next_month = start_time.month + 1 if start_time.month < 12 else 1
            next_year = start_time.year + 1 if start_time.month == 12 else start_time.year
            end_time = datetime(next_year, next_month, 1)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid timeRange: {timeRange}")
        
        # Get historical data
        historical_data = await data_fetcher.get_historical_data(start_time, end_time)
        
        # Process based on time range
        if timeRange == "day":
            # Hourly data
            result = []
            for dt, row in historical_data.iterrows():
                result.append({
                    "time": dt.strftime("%H:%M"),
                    "consumption": float(row['load_power']),
                    "solar": float(row['pv_power']),
                    "grid": float(row['grid_power'])
                })
        elif timeRange == "week":
            # Daily data
            historical_data = historical_data.resample('D').mean()
            result = []
            for dt, row in historical_data.iterrows():
                result.append({
                    "date": dt.strftime("%a"),
                    "consumption": float(row['load_power'] * 24),
                    "solar": float(row['pv_power'] * 24),
                    "grid": float(row['grid_power'] * 24)
                })
        elif timeRange == "month":
            # Weekly data
            historical_data = historical_data.resample('W').mean()
            result = []
            for i, (dt, row) in enumerate(historical_data.iterrows()):
                result.append({
                    "date": f"Week {i+1}",
                    "consumption": float(row['load_power'] * 24 * 7),
                    "solar": float(row['pv_power'] * 24 * 7),
                    "grid": float(row['grid_power'] * 24 * 7)
                })
        
        return result
    
    except Exception as e:
        logger.error(f"Error in get_energy_history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/energy/breakdown")
async def get_energy_breakdown():
    """Get energy breakdown data"""
    try:
        # Get current data
        current_data = await data_fetcher.get_current_energy_data()
        
        # Get historical data for the past day
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        historical_data = await data_fetcher.get_historical_data(start_time, end_time)
        
        # Calculate energy totals
        if not historical_data.empty:
            total_load = historical_data['load_power'].sum()
            total_pv = historical_data['pv_power'].sum()
            total_battery_discharge = abs(historical_data[historical_data['battery_power'] < 0]['battery_power'].sum())
            total_grid_import = historical_data[historical_data['grid_power'] > 0]['grid_power'].sum()
            
            # Energy sources breakdown
            sources = [
                {"name": "Solar", "value": float(total_pv)},
                {"name": "Battery", "value": float(total_battery_discharge)},
                {"name": "Grid", "value": float(total_grid_import)}
            ]
            
            # Energy consumption breakdown (estimated)
            # In a real implementation, this would come from device-level monitoring
            hvac_pct = 0.35
            appliances_pct = 0.25
            lighting_pct = 0.15
            ev_charging_pct = 0.15
            other_pct = 0.10
            
            consumption = [
                {"name": "HVAC", "value": float(total_load * hvac_pct)},
                {"name": "Appliances", "value": float(total_load * appliances_pct)},
                {"name": "Lighting", "value": float(total_load * lighting_pct)},
                {"name": "EV Charging", "value": float(total_load * ev_charging_pct)},
                {"name": "Other", "value": float(total_load * other_pct)}
            ]
        else:
            # Fallback if no historical data
            sources = [
                {"name": "Solar", "value": 10.5},
                {"name": "Battery", "value": 4.2},
                {"name": "Grid", "value": 5.3}
            ]
            
            consumption = [
                {"name": "HVAC", "value": 7.5},
                {"name": "Appliances", "value": 5.2},
                {"name": "Lighting", "value": 2.8},
                {"name": "EV Charging", "value": 3.1},
                {"name": "Other", "value": 1.4}
            ]
        
        return {"sources": sources, "consumption": consumption}
    
    except Exception as e:
        logger.error(f"Error in get_energy_breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/devices")
async def get_devices():
    """Get list of devices"""
    try:
        devices = await data_fetcher.get_devices()
        return devices
    
    except Exception as e:
        logger.error(f"Error in get_devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/devices/{id}/toggle")
async def toggle_device(id: str, data: Dict[str, bool]):
    """Toggle device state"""
    try:
        is_on = data.get("isOn", False)
        
        # In a real implementation, this would communicate with Home Assistant
        # to toggle the device state
        
        # For now, return a mock response
        device_types = {
            "1": {"name": "RGB LED Strip", "type": "lighting", "icon": "lightbulb", "power": 12},
            "2": {"name": "Desk Fan", "type": "fan", "icon": "fan", "power": 25},
            "3": {"name": "PWM Water Pump", "type": "pump", "icon": "droplet", "power": 35},
            "4": {"name": "Servo Motor", "type": "motor", "icon": "rotate-cw", "power": 5},
            "5": {"name": "Temperature Sensor", "type": "sensor", "icon": "thermometer", "power": 0.5}
        }
        
        device_info = device_types.get(id, {"name": f"Device {id}", "type": "unknown", "icon": "plug", "power": 100})
        
        return {
            "id": id,
            "name": device_info["name"],
            "type": device_info["type"],
            "status": "online",
            "power": device_info["power"] if is_on else 0,
            "isOn": is_on,
            "icon": device_info["icon"]
        }
    
    except Exception as e:
        logger.error(f"Error in toggle_device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_optimization_task(task_id: str, params: OptimizationParams):
    """Background task to run optimization"""
    try:
        # Parse optimization parameters
        optimization_mode = params.optimizationMode
        time_horizon_hours = int(params.timeHorizon)
        
        battery_constraints = params.batteryConstraints
        grid_constraints = params.gridConstraints
        
        # Get forecast data
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        days_needed = (time_horizon_hours + 23) // 24  # Ceiling division
        
        pv_forecast = await data_fetcher.get_pv_forecast(start_time, days=days_needed)
        load_forecast = await data_fetcher.get_load_forecast(start_time, days=days_needed)
        
        # Combine forecasts
        forecast_data = pd.concat([pv_forecast, load_forecast], axis=1)
        
        # Limit to requested time horizon
        forecast_data = forecast_data.iloc[:time_horizon_hours]
        
        # Get current battery state
        current_data = await data_fetcher.get_current_energy_data()
        battery_soc = current_data.get("battery_soc", 50) / 100.0  # Convert to 0-1 scale
        
        # Get devices
        devices = await data_fetcher.get_devices()
        
        # Prepare device parameters for optimization
        device_params = []
        for device in devices:
            if device["type"] == "charger" and device["id"] == "3":  # EV Charger
                device_params.append({
                    "id": device["id"],
                    "name": device["name"],
                    "type": "ev_charger",
                    "arrival_time": 19,  # 7 PM
                    "departure_time": 7,  # 7 AM
                    "energy_needed": 10.0,  # kWh
                    "max_power": 7.2  # kW
                })
            elif device["type"] == "appliance" and device["id"] == "4":  # Washing Machine
                device_params.append({
                    "id": device["id"],
                    "name": device["name"],
                    "type": "shiftable",
                    "start_window": 8,  # 8 AM
                    "end_window": 18,  # 6 PM
                    "duration": 2,  # 2 hours
                    "power": 0.8  # kW
                })
            elif device["type"] == "hvac" and device["id"] == "5":  # Smart Thermostat
                device_params.append({
                    "id": device["id"],
                    "name": device["name"],
                    "type": "thermal",
                    "temp_min": 68,  # °F
                    "temp_max": 78,  # °F
                    "temp_init": 72,  # °F
                    "max_power": 1.5  # kW
                })
        
        # Prepare battery parameters
        battery_params = {
            "capacity": 10.0,  # kWh
            "max_power": 5.0,  # kW
            "efficiency": 0.95,
            "min_soc": battery_constraints["minSoc"] / 100.0 if battery_constraints["enabled"] else 0.1,
            "max_soc": 0.9,
            "initial_soc": battery_soc
        }
        
        # Prepare grid parameters
        grid_params = {
            "max_power": grid_constraints["maxPower"] if grid_constraints["enabled"] else 10.0,
            "import_price": 0.15,  # $/kWh
            "export_price": 0.05  # $/kWh
        }
        
        # Create optimizer with weights based on optimization mode
        optimizer = EnergyOptimizer()
        weights = optimizer.get_optimization_mode_weights(optimization_mode)
        optimizer.cost_weight = weights[0]
        optimizer.self_consumption_weight = weights[1]
        optimizer.peak_shaving_weight = weights[2]
        optimizer.battery_cycle_weight = weights[3]
        
        # Run optimization
        result = optimizer.optimize(
            forecast_data=forecast_data,
            battery_params=battery_params,
            grid_params=grid_params,
            device_params=device_params
        )
        
        # Store result
        optimization_results[task_id] = result
        
    except Exception as e:
        logger.error(f"Error in optimization task {task_id}: {e}")
        optimization_results[task_id] = {"status": "error", "message": str(e)}
    
    finally:
        # Clean up task
        if task_id in optimization_tasks:
            del optimization_tasks[task_id]

@app.post("/api/optimization/run")
async def run_optimization(params: OptimizationParams, background_tasks: BackgroundTasks):
    """Run energy optimization"""
    try:
        # Generate task ID
        task_id = f"opt-{uuid.uuid4()}"
        
        # Start optimization in background
        background_tasks.add_task(run_optimization_task, task_id, params)
        optimization_tasks[task_id] = datetime.now()
        
        # For demo purposes, we'll simulate the optimization with a mock result
        # In a real implementation, this would be calculated by the optimizer
        
        # Wait for optimization to complete (with timeout)
        timeout = 10  # seconds
        start_time = datetime.now()
        while task_id not in optimization_results and (datetime.now() - start_time).total_seconds() < timeout:
            await asyncio.sleep(0.5)
        
        if task_id in optimization_results:
            result = optimization_results[task_id]
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("message", "Optimization failed"))
            return result
        else:
            # If timeout, return mock data
            logger.warning(f"Optimization timeout for task {task_id}, returning mock data")
            
            # Generate mock optimization result
            mock_result = {
                "id": task_id,
                "cost": 3.75,
                "selfConsumption": 85,
                "peakGridPower": 2.1,
                "batteryCycles": 0.8,
                "timestamp": datetime.now().isoformat(),
                "scheduleData": [],
                "deviceScheduleData": [
                    {
                        "id": "3",
                        "name": "EV Charger",
                        "type": "charger",
                        "schedule": [
                            {
                                "start": params.optimizationMode == "cost" ? "01:00" : "22:00",
                                "end": params.optimizationMode == "cost" ? "05:00" : "02:00",
                                "power": 7.2
                            }
                        ]
                    },
                    {
                        "id": "4",
                        "name": "Washing Machine",
                        "type": "appliance",
                        "schedule": [
                            {
                                "start": params.optimizationMode == "self_consumption" ? "12:00" : "14:00",
                                "end": params.optimizationMode == "self_consumption" ? "13:30" : "15:30",
                                "power": 0.8
                            }
                        ]
                    }
                ]
            }
            
            return mock_result
    
    except Exception as e:
        logger.error(f"Error in run_optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/optimization/results/{id}")
async def get_optimization_results(id: str):
    """Get optimization results by ID"""
    try:
        if id in optimization_results:
            return optimization_results[id]
        else:
            raise HTTPException(status_code=404, detail=f"Optimization results not found for ID: {id}")
    
    except Exception as e:
        logger.error(f"Error in get_optimization_results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimization/apply/{id}")
async def apply_optimization_schedule(id: str):
    """Apply optimization schedule"""
    try:
        if id not in optimization_results:
            raise HTTPException(status_code=404, detail=f"Optimization results not found for ID: {id}")
        
        # In a real implementation, this would communicate with Home Assistant
        # to apply the schedule to devices
        
        return {
            "success": True,
            "message": f"Schedule {id} applied successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in apply_optimization_schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

