from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from optimization.optimizer import EnergyOptimizer
from data.data_fetcher import DataFetcher
from data.data_storage import DataStorage
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from dotenv import load_dotenv
# Add these imports at the top of the file
from optimization.custom_optimizer import CustomEnergyOptimizer, GeneticEnergyOptimizer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("smartwatt.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smartwatt-secret-key")

# Initialize components
data_fetcher = DataFetcher()
data_storage = DataStorage()
optimizer = EnergyOptimizer()

# Store optimization results in memory (would use a database in production)
optimization_results = {}

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html', page='dashboard')

@app.route('/dashboard')
def dashboard():
    """Render the energy dashboard page"""
    return render_template('dashboard.html', page='dashboard')

@app.route('/devices')
def devices():
    """Render the devices page"""
    return render_template('devices.html', page='devices')

@app.route('/optimization')
def optimization():
    """Render the optimization page"""
    return render_template('optimization.html', page='optimization')

@app.route('/energy-flow')
def energy_flow():
    """Render the energy flow page"""
    return render_template('energy-flow.html', page='energy-flow')

@app.route('/settings')
def settings():
    """Render the settings page"""
    return render_template('settings.html', page='settings')

# API Routes

@app.route('/api/energy/current')
def get_current_energy_data():
    """Get current energy data"""
    try:
        # Get current data
        current_data = data_fetcher.get_current_energy_data()
        
        # Format for frontend
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
                "consumption": current_data["load_power"] * time_factor,
                "solar": 0,
                "battery": 0,
                "grid": 0
            }
            
            # Solar only during daylight
            if 6 <= hour <= 18:
                solar_factor = np.sin(((hour - 6) * np.pi) / 12)
                hour_data["solar"] = current_data["pv_power"] * solar_factor
            
            # Battery and grid balance the system
            if hour_data["solar"] > hour_data["consumption"]:
                hour_data["battery"] = min(hour_data["solar"] - hour_data["consumption"], current_data["battery_power"])
                hour_data["grid"] = hour_data["consumption"] - hour_data["solar"] - hour_data["battery"]
            else:
                deficit = hour_data["consumption"] - hour_data["solar"]
                hour_data["battery"] = -min(deficit * 0.7, abs(current_data["battery_power"]))
                hour_data["grid"] = deficit + hour_data["battery"]
            
            result.append(hour_data)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in get_current_energy_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/energy/forecast')
def get_energy_forecast():
    """Get energy forecast data"""
    try:
        # Parse date or use today
        date_str = request.args.get('date')
        if date_str:
            forecast_date = datetime.fromisoformat(date_str)
        else:
            forecast_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get PV and load forecasts
        pv_forecast = data_fetcher.get_pv_forecast(forecast_date)
        load_forecast = data_fetcher.get_load_forecast(forecast_date)
        
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
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in get_energy_forecast: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/energy/history')
def get_energy_history():
    """Get historical energy data"""
    try:
        # Parse parameters
        time_range = request.args.get('timeRange', 'day')
        date_str = request.args.get('date')
        
        # Parse date or use today
        if date_str:
            history_date = datetime.fromisoformat(date_str)
        else:
            history_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Determine time range
        if time_range == "day":
            start_time = history_date
            end_time = start_time + timedelta(days=1)
        elif time_range == "week":
            start_time = history_date - timedelta(days=history_date.weekday())
            end_time = start_time + timedelta(days=7)
        elif time_range == "month":
            start_time = history_date.replace(day=1)
            next_month = start_time.month + 1 if start_time.month < 12 else 1
            next_year = start_time.year + 1 if start_time.month == 12 else start_time.year
            end_time = datetime(next_year, next_month, 1)
        else:
            return jsonify({"error": f"Invalid timeRange: {time_range}"}), 400
        
        # Get historical data
        historical_data = data_fetcher.get_historical_data(start_time, end_time)
        
        # Process based on time range
        if time_range == "day":
            # Hourly data
            result = []
            for dt, row in historical_data.iterrows():
                result.append({
                    "time": dt.strftime("%H:%M"),
                    "consumption": float(row['load_power']),
                    "solar": float(row['pv_power']),
                    "grid": float(row['grid_power'])
                })
        elif time_range == "week":
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
        elif time_range == "month":
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
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in get_energy_history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/energy/breakdown')
def get_energy_breakdown():
    """Get energy breakdown data"""
    try:
        # Get current data
        current_data = data_fetcher.get_current_energy_data()
        
        # Get historical data for the past day
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        historical_data = data_fetcher.get_historical_data(start_time, end_time)
        
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
            lighting_pct = 0.35
            fans_pct = 0.25
            motors_pct = 0.15
            pumps_pct = 0.15
            sensors_pct = 0.10
            
            consumption = [
                {"name": "Lighting", "value": float(total_load * lighting_pct)},
                {"name": "Fans", "value": float(total_load * fans_pct)},
                {"name": "Motors", "value": float(total_load * motors_pct)},
                {"name": "Pumps", "value": float(total_load * pumps_pct)},
                {"name": "Sensors", "value": float(total_load * sensors_pct)}
            ]
        else:
            # Fallback if no historical data
            sources = [
                {"name": "Solar", "value": 0.5},
                {"name": "Battery", "value": 0.2},
                {"name": "Grid", "value": 0.3}
            ]
            
            consumption = [
                {"name": "Lighting", "value": 0.35},
                {"name": "Fans", "value": 0.25},
                {"name": "Motors", "value": 0.15},
                {"name": "Pumps", "value": 0.15},
                {"name": "Sensors", "value": 0.1}
            ]
        
        return jsonify({"sources": sources, "consumption": consumption})
    
    except Exception as e:
        logger.error(f"Error in get_energy_breakdown: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/devices')
def get_devices():
    """Get list of devices"""
    try:
        devices = data_fetcher.get_devices()
        return jsonify(devices)
    
    except Exception as e:
        logger.error(f"Error in get_devices: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/devices/<id>/toggle', methods=['POST'])
def toggle_device(id):
    """Toggle device state"""
    try:
        data = request.json
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
        
        return jsonify({
            "id": id,
            "name": device_info["name"],
            "type": device_info["type"],
            "status": "online",
            "power": device_info["power"] if is_on else 0,
            "isOn": is_on,
            "icon": device_info["icon"]
        })
    
    except Exception as e:
        logger.error(f"Error in toggle_device: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/run', methods=['POST'])
def run_optimization():
    """Run energy optimization"""
    try:
        # Get optimization parameters
        params = request.json
        optimization_mode = params.get("optimizationMode", "cost")
        time_horizon = int(params.get("timeHorizon", "24"))
        battery_constraints = params.get("batteryConstraints", {
            "enabled": True,
            "minSoc": 20,
            "maxCycles": 1
        })
        grid_constraints = params.get("gridConstraints", {
            "enabled": True,
            "maxPower": 5
        })
        
        # Generate task ID
        task_id = f"opt-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Get forecast data
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        days_needed = (time_horizon + 23) // 24  # Ceiling division
        
        pv_forecast = data_fetcher.get_pv_forecast(start_time, days=days_needed)
        load_forecast = data_fetcher.get_load_forecast(start_time, days=days_needed)
        
        # Combine forecasts
        forecast_data = pd.concat([pv_forecast, load_forecast], axis=1)
        
        # Limit to requested time horizon
        forecast_data = forecast_data.iloc[:time_horizon]
        
        # Get current battery state
        current_data = data_fetcher.get_current_energy_data()
        battery_soc = current_data.get("battery_soc", 50) / 100.0  # Convert to 0-1 scale
        
        # Get devices
        devices = data_fetcher.get_devices()
        
        # Prepare device parameters for optimization
        device_params = []
        for device in devices:
            if device["type"] == "pump" and device["id"] == "3":  # PWM Water Pump
                device_params.append({
                    "id": device["id"],
                    "name": device["name"],
                    "type": "shiftable",
                    "start_window": 8,  # 8 AM
                    "end_window": 18,  # 6 PM
                    "duration": 0.5,  # 30 minutes
                    "power": 0.035  # kW (35W)
                })
            elif device["type"] == "motor" and device["id"] == "4":  # Servo Motor
                device_params.append({
                    "id": device["id"],
                    "name": device["name"],
                    "type": "shiftable",
                    "start_window": 8,  # 8 AM
                    "end_window": 18,  # 6 PM
                    "duration": 0.2,  # 12 minutes
                    "power": 0.005  # kW (5W)
                })
            elif device["type"] == "fan" and device["id"] == "2":  # Desk Fan
                device_params.append({
                    "id": device["id"],
                    "name": device["name"],
                    "type": "thermal",
                    "temp_min": 24,  # °C
                    "temp_max": 28,  # °C
                    "temp_init": 26,  # °C
                    "max_power": 0.025  # kW (25W)
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
        
        # Set optimizer weights based on optimization mode
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
        
        # Return result
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in run_optimization: {e}")
        return jsonify({"error": str(e)}), 500

# Then add this route after the existing optimization route
@app.route('/api/optimization/custom', methods=['POST'])
def run_custom_optimization():
    """Run custom energy optimization"""
    try:
        # Get optimization parameters
        params = request.json
        optimization_mode = params.get("optimizationMode", "cost")
        time_horizon = int(params.get("timeHorizon", "24"))
        battery_constraints = params.get("batteryConstraints", {
            "enabled": True,
            "minSoc": 20,
            "maxCycles": 1
        })
        grid_constraints = params.get("gridConstraints", {
            "enabled": True,
            "maxPower": 5
        })
        algorithm = params.get("algorithm", "sgd")  # sgd, lbfgs, genetic
        
        # Generate task ID
        task_id = f"opt-custom-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Get forecast data
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        days_needed = (time_horizon + 23) // 24  # Ceiling division
        
        pv_forecast = data_fetcher.get_pv_forecast(start_time, days=days_needed)
        load_forecast = data_fetcher.get_load_forecast(start_time, days=days_needed)
        
        # Combine forecasts
        forecast_data = pd.concat([pv_forecast, load_forecast], axis=1)
        
        # Limit to requested time horizon
        forecast_data = forecast_data.iloc[:time_horizon]
        
        # Get current battery state
        current_data = data_fetcher.get_current_energy_data()
        battery_soc = current_data.get("battery_soc", 50) / 100.0  # Convert to 0-1 scale
        
        # Get devices
        devices = data_fetcher.get_devices()
        
        # Prepare device parameters for optimization
        device_params = []
        for device in devices:
            if device["type"] == "pump" and device["id"] == "3":  # PWM Water Pump
                device_params.append({
                    "id": device["id"],
                    "name": device["name"],
                    "type": "shiftable",
                    "start_window": 8,  # 8 AM
                    "end_window": 18,  # 6 PM
                    "duration": 0.5,  # 30 minutes
                    "power": 0.035  # kW (35W)
                })
            elif device["type"] == "motor" and device["id"] == "4":  # Servo Motor
                device_params.append({
                    "id": device["id"],
                    "name": device["name"],
                    "type": "shiftable",
                    "start_window": 8,  # 8 AM
                    "end_window": 18,  # 6 PM
                    "duration": 0.2,  # 12 minutes
                    "power": 0.005  # kW (5W)
                })
            elif device["type"] == "fan" and device["id"] == "2":  # Desk Fan
                device_params.append({
                    "id": device["id"],
                    "name": device["name"],
                    "type": "thermal",
                    "temp_min": 24,  # °C
                    "temp_max": 28,  # °C
                    "temp_init": 26,  # °C
                    "max_power": 0.025  # kW (25W)
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
        
        # Set optimizer weights based on optimization mode
        if optimization_mode == "cost":
            weights = (1.0, 0.1, 0.1, 0.1)
        elif optimization_mode == "self_consumption":
            weights = (0.1, 1.0, 0.1, 0.1)
        elif optimization_mode == "grid_independence":
            weights = (0.1, 0.3, 1.0, 0.1)
        elif optimization_mode == "battery_life":
            weights = (0.1, 0.1, 0.1, 1.0)
        else:
            weights = (1.0, 0.1, 0.1, 0.1)
        
        # Choose algorithm
        if algorithm == "sgd":
            custom_optimizer = CustomEnergyOptimizer(
                cost_weight=weights[0],
                self_consumption_weight=weights[1],
                peak_shaving_weight=weights[2],
                battery_cycle_weight=weights[3],
                use_sgd=True
            )
        elif algorithm == "lbfgs":
            custom_optimizer = CustomEnergyOptimizer(
                cost_weight=weights[0],
                self_consumption_weight=weights[1],
                peak_shaving_weight=weights[2],
                battery_cycle_weight=weights[3],
                use_sgd=False
            )
        elif algorithm == "genetic":
            custom_optimizer = GeneticEnergyOptimizer(
                cost_weight=weights[0],
                self_consumption_weight=weights[1],
                peak_shaving_weight=weights[2],
                battery_cycle_weight=weights[3]
            )
        else:
            return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400
        
        # Run optimization
        result = custom_optimizer.optimize(
            forecast_data=forecast_data,
            battery_params=battery_params,
            grid_params=grid_params,
            device_params=device_params
        )
        
        # Store result
        optimization_results[task_id] = result
        
        # Return result
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in run_custom_optimization: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/results/<id>')
def get_optimization_results(id):
    """Get optimization results by ID"""
    try:
        if id in optimization_results:
            return jsonify(optimization_results[id])
        else:
            return jsonify({"error": f"Optimization results not found for ID: {id}"}), 404
    
    except Exception as e:
        logger.error(f"Error in get_optimization_results: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/apply/<id>', methods=['POST'])
def apply_optimization_schedule(id):
    """Apply optimization schedule"""
    try:
        if id not in optimization_results:
            return jsonify({"error": f"Optimization results not found for ID: {id}"}), 404
        
        # In a real implementation, this would communicate with Home Assistant
        # to apply the schedule to devices
        
        return jsonify({
            "success": True,
            "message": f"Schedule {id} applied successfully",
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in apply_optimization_schedule: {e}")
        return jsonify({"error": str(e)}), 500

# Helper routes for generating charts
@app.route('/api/charts/energy-flow')
def get_energy_flow_chart():
    """Generate energy flow chart"""
    try:
        # Get current data
        current_data = data_fetcher.get_current_energy_data()
        
        # Create Plotly figure
        fig = make_subplots(rows=1, cols=1)
        
        # Add traces
        times = [f"{h:02d}:00" for h in range(24)]
        
        # Solar production (bell curve during daylight)
        solar_values = []
        for hour in range(24):
            if 6 <= hour <= 18:
                solar = current_data["pv_power"] * np.sin(((hour - 6) * np.pi) / 12)
            else:
                solar = 0
            solar_values.append(solar)
        
        # Load (varies throughout the day)
        load_values = []
        for hour in range(24):
            load = current_data["load_power"] * (0.7 + 0.3 * np.sin(((hour - 8) * 2 * np.pi) / 24))
            load_values.append(load)
        
        # Battery (charges during excess solar, discharges at night)
        battery_values = []
        for hour in range(24):
            if solar_values[hour] > load_values[hour]:
                battery = min(solar_values[hour] - load_values[hour], current_data["battery_power"])
            elif hour >= 18 or hour <= 6:
                battery = -min(load_values[hour] * 0.5, current_data["battery_power"])
            else:
                battery = 0
            battery_values.append(battery)
        
        # Grid (balances the system)
        grid_values = []
        for hour in range(24):
            grid = load_values[hour] - solar_values[hour] - battery_values[hour]
            grid_values.append(grid)
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=times, y=solar_values, name="Solar", fill='tozeroy', line=dict(color='gold')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=times, y=battery_values, name="Battery", line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=times, y=grid_values, name="Grid", line=dict(color='purple')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=times, y=load_values, name="Load", line=dict(color='red')),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Energy Flow",
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        # Convert to JSON
        chart_json = pio.to_json(fig)
        return chart_json
    
    except Exception as e:
        logger.error(f"Error in get_energy_flow_chart: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/charts/energy-breakdown')
def get_energy_breakdown_chart():
    """Generate energy breakdown chart"""
    try:
        # Get energy breakdown data
        breakdown_data = json.loads(get_energy_breakdown().data)
        
        # Create Plotly figures
        sources_fig = px.pie(
            breakdown_data["sources"], 
            values="value", 
            names="name",
            title="Energy Sources",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        consumption_fig = px.pie(
            breakdown_data["consumption"], 
            values="value", 
            names="name",
            title="Energy Consumption",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        # Convert to JSON
        sources_json = pio.to_json(sources_fig)
        consumption_json = pio.to_json(consumption_fig)
        
        return jsonify({
            "sources": json.loads(sources_json),
            "consumption": json.loads(consumption_json)
        })
    
    except Exception as e:
        logger.error(f"Error in get_energy_breakdown_chart: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5002))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=True)

