# import numpy as np
# import pandas as pd
# import cvxpy as cp
# from datetime import datetime, timedelta
# import logging
# from typing import Dict, List, Any, Optional, Tuple

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class EnergyOptimizer:
#     """
#     Energy optimization using linear programming techniques similar to EMHASS
#     """
    
#     def __init__(self, 
#                  cost_weight: float = 1.0,
#                  self_consumption_weight: float = 0.0,
#                  peak_shaving_weight: float = 0.0,
#                  battery_cycle_weight: float = 0.0):
#         """
#         Initialize the energy optimizer
        
#         Args:
#             cost_weight: Weight for cost minimization objective
#             self_consumption_weight: Weight for self-consumption maximization
#             peak_shaving_weight: Weight for peak shaving objective
#             battery_cycle_weight: Weight for battery cycle minimization
#         """
#         self.cost_weight = cost_weight
#         self.self_consumption_weight = self_consumption_weight
#         self.peak_shaving_weight = peak_shaving_weight
#         self.battery_cycle_weight = battery_cycle_weight
        
#     def optimize(self, 
#                  forecast_data: pd.DataFrame,
#                  battery_params: Dict[str, Any],
#                  grid_params: Dict[str, Any],
#                  device_params: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Run optimization using linear programming
        
#         Args:
#             forecast_data: DataFrame with forecasted load and PV production
#             battery_params: Battery parameters (capacity, efficiency, etc.)
#             grid_params: Grid parameters (max power, pricing, etc.)
#             device_params: List of controllable devices with parameters
            
#         Returns:
#             Dictionary with optimization results
#         """
#         try:
#             # Extract time steps and horizon
#             time_steps = len(forecast_data)
#             delta_t = 1.0  # Time step in hours
            
#             # Extract forecasts
#             pv_forecast = forecast_data['pv_power'].values
#             load_forecast = forecast_data['load_power'].values
            
#             # Extract battery parameters
#             battery_capacity = battery_params.get('capacity', 10.0)  # kWh
#             battery_max_power = battery_params.get('max_power', 5.0)  # kW
#             battery_efficiency = battery_params.get('efficiency', 0.95)
#             battery_min_soc = battery_params.get('min_soc', 0.1)
#             battery_max_soc = battery_params.get('max_soc', 0.9)
#             battery_initial_soc = battery_params.get('initial_soc', 0.5)
            
#             # Extract grid parameters
#             grid_max_power = grid_params.get('max_power', 10.0)  # kW
#             grid_export_price = grid_params.get('export_price', 0.05)  # $/kWh
            
#             # Time-of-use pricing if available, otherwise flat rate
#             if 'import_price_schedule' in grid_params:
#                 grid_import_price = np.array(grid_params['import_price_schedule'])
#             else:
#                 grid_import_price = np.ones(time_steps) * grid_params.get('import_price', 0.15)  # $/kWh
            
#             # Define optimization variables
#             p_grid = cp.Variable(time_steps)  # Grid power (positive = import, negative = export)
#             p_batt = cp.Variable(time_steps)  # Battery power (positive = charging, negative = discharging)
#             soc = cp.Variable(time_steps + 1)  # Battery state of charge
            
#             # Device power variables (if controllable devices are provided)
#             p_devices = {}
#             device_constraints = []
            
#             # Prepare device parameters for optimization
#             device_params = []
#             for device in device_params:
#                 if device.get('type') == 'shiftable':
#                     # For low wattage devices, we need to scale the power appropriately
#                     power = device.get('power', 0.01)  # Default to 10W
#                     duration = device.get('duration', 0.5)  # Default to 30 minutes
                    
#                     device_params.append({
#                         "id": device.get('id', 'unknown'),
#                         "name": device.get('name', 'Unknown Device'),
#                         "type": "shiftable",
#                         "start_window": device.get('start_window', 8),
#                         "end_window": device.get('end_window', 18),
#                         "duration": duration,
#                         "power": power
#                     })
#                 elif device.get('type') == 'thermal':
#                     # For fans and temperature-controlled devices
#                     max_power = device.get('max_power', 0.025)  # Default to 25W
                    
#                     device_params.append({
#                         "id": device.get('id', 'unknown'),
#                         "name": device.get('name', 'Unknown Device'),
#                         "type": "thermal",
#                         "temp_min": device.get('temp_min', 24),
#                         "temp_max": device.get('temp_max', 28),
#                         "temp_init": device.get('temp_init', 26),
#                         "max_power": max_power
#                     })
            
#             p_devices = {}
#             device_constraints = []
            
#             for device in device_params:
#                 device_id = device['id']
#                 p_devices[device_id] = cp.Variable(time_steps)
                
#                 # Device constraints
#                 if device['type'] == 'shiftable':
#                     # Shiftable load (e.g., washing machine, dishwasher)
#                     # Must run for a specific duration within a time window
#                     start_window = device.get('start_window', 0)
#                     end_window = device.get('end_window', time_steps - 1)
#                     duration = device.get('duration', 2)
#                     power = device.get('power', 1.0)
                    
#                     # Binary variable for each possible start time
#                     x_start = cp.Variable(time_steps, boolean=True)
                    
#                     # Constraint: can only start once
#                     device_constraints.append(cp.sum(x_start) == 1)
                    
#                     # Constraint: can only start within the window
#                     for t in range(time_steps):
#                         if t < start_window or t > end_window - duration + 1:
#                             device_constraints.append(x_start[t] == 0)
                    
#                     # Constraint: power profile based on start time
#                     for t in range(time_steps):
#                         # Sum over all possible start times that would make the device run at time t
#                         p_t = cp.sum([x_start[max(0, t - d)] * power for d in range(min(duration, t + 1))])
#                         device_constraints.append(p_devices[device_id][t] == p_t)
                
#                 elif device['type'] == 'thermal':
#                     # Thermal load (e.g., water heater, HVAC)
#                     # Has a temperature state that must be kept within bounds
#                     temp_min = device.get('temp_min', 55)  # Minimum temperature
#                     temp_max = device.get('temp_max', 65)  # Maximum temperature
#                     temp_init = device.get('temp_init', 60)  # Initial temperature
#                     temp_ambient = device.get('temp_ambient', 20)  # Ambient temperature
#                     thermal_resistance = device.get('thermal_resistance', 0.1)  # K/kW
#                     thermal_capacitance = device.get('thermal_capacitance', 0.2)  # kWh/K
#                     cop = device.get('cop', 3.0)  # Coefficient of performance
#                     max_power = device.get('max_power', 2.0)  # Maximum power
                    
#                     # Temperature state variable
#                     temp = cp.Variable(time_steps + 1)
                    
#                     # Initial temperature
#                     device_constraints.append(temp[0] == temp_init)
                    
#                     # Temperature dynamics
#                     for t in range(time_steps):
#                         temp_next = temp[t] + delta_t * (
#                             p_devices[device_id][t] * cop / thermal_capacitance - 
#                             (temp[t] - temp_ambient) / (thermal_resistance * thermal_capacitance)
#                         )
#                         device_constraints.append(temp[t+1] == temp_next)
                        
#                     # Temperature bounds
#                     for t in range(time_steps + 1):
#                         device_constraints.append(temp[t] >= temp_min)
#                         device_constraints.append(temp[t] <= temp_max)
                    
#                     # Power bounds
#                     for t in range(time_steps):
#                         device_constraints.append(p_devices[device_id][t] >= 0)
#                         device_constraints.append(p_devices[device_id][t] <= max_power)
                
#                 elif device['type'] == 'ev_charger':
#                     # EV charger
#                     arrival_time = device.get('arrival_time', 0)
#                     departure_time = device.get('departure_time', time_steps - 1)
#                     energy_needed = device.get('energy_needed', 10.0)  # kWh
#                     max_power = device.get('max_power', 7.2)  # kW
                    
#                     # Constraint: only charge when EV is present
#                     for t in range(time_steps):
#                         if t < arrival_time or t > departure_time:
#                             device_constraints.append(p_devices[device_id][t] == 0)
                    
#                     # Constraint: power bounds when EV is present
#                     for t in range(arrival_time, departure_time + 1):
#                         device_constraints.append(p_devices[device_id][t] >= 0)
#                         device_constraints.append(p_devices[device_id][t] <= max_power)
                    
#                     # Constraint: total energy delivered
#                     device_constraints.append(cp.sum(p_devices[device_id]) * delta_t == energy_needed)
                
#                 else:
#                     # Default: fixed load profile
#                     power_profile = device.get('power_profile', np.zeros(time_steps))
#                     for t in range(time_steps):
#                         device_constraints.append(p_devices[device_id][t] == power_profile[t])
            
#             # Calculate total controllable load
#             p_controllable_load = cp.sum([p_devices[d] for d in p_devices], axis=0) if p_devices else np.zeros(time_steps)
            
#             # Power balance constraint: PV + Grid + Battery = Load
#             # Positive grid = import, negative grid = export
#             # Positive battery = charging, negative battery = discharging
#             power_balance_constraints = []
#             for t in range(time_steps):
#                 power_balance_constraints.append(
#                     pv_forecast[t] + p_grid[t] - p_batt[t] == load_forecast[t] + p_controllable_load[t]
#                 )
            
#             # Battery constraints
#             battery_constraints = []
            
#             # SOC dynamics
#             for t in range(time_steps):
#                 # Charging efficiency when p_batt > 0, discharging efficiency when p_batt < 0
#                 charge_term = cp.maximum(p_batt[t], 0) * battery_efficiency
#                 discharge_term = cp.minimum(p_batt[t], 0) / battery_efficiency
                
#                 soc_next = soc[t] + (charge_term + discharge_term) * delta_t / battery_capacity
#                 battery_constraints.append(soc[t+1] == soc_next)
            
#             # Initial SOC
#             battery_constraints.append(soc[0] == battery_initial_soc)
            
#             # SOC bounds
#             for t in range(time_steps + 1):
#                 battery_constraints.append(soc[t] >= battery_min_soc)
#                 battery_constraints.append(soc[t] <= battery_max_soc)
            
#             # Battery power bounds
#             for t in range(time_steps):
#                 battery_constraints.append(p_batt[t] >= -battery_max_power)  # Discharge limit
#                 battery_constraints.append(p_batt[t] <= battery_max_power)   # Charge limit
            
#             # Grid constraints
#             grid_constraints = []
            
#             # Grid power bounds
#             for t in range(time_steps):
#                 grid_constraints.append(p_grid[t] >= -grid_max_power)  # Export limit
#                 grid_constraints.append(p_grid[t] <= grid_max_power)   # Import limit
            
#             # Define objective function components
            
#             # 1. Cost minimization
#             grid_import = cp.maximum(p_grid, 0)
#             grid_export = cp.minimum(p_grid, 0)
#             cost = cp.sum(grid_import * grid_import_price) - cp.sum(grid_export * grid_export_price)
            
#             # 2. Self-consumption maximization (minimize grid export)
#             self_consumption = cp.sum(cp.abs(grid_export))
            
#             # 3. Peak shaving (minimize maximum grid import)
#             peak_power = cp.max(grid_import)
            
#             # 4. Battery cycle minimization
#             # Approximate by minimizing the sum of absolute battery power
#             battery_cycles = cp.sum(cp.abs(p_batt)) / (2 * battery_capacity)
            
#             # Combined objective with weights
#             objective = (
#                 self.cost_weight * cost +
#                 self.self_consumption_weight * self_consumption +
#                 self.peak_shaving_weight * peak_power +
#                 self.battery_cycle_weight * battery_cycles
#             )
            
#             # Define and solve the problem
#             constraints = (
#                 power_balance_constraints + 
#                 battery_constraints + 
#                 grid_constraints + 
#                 device_constraints
#             )
            
#             problem = cp.Problem(cp.Minimize(objective), constraints)
#             problem.solve(solver=cp.ECOS)
            
#             if problem.status != 'optimal':
#                 logger.warning(f"Optimization problem status: {problem.status}")
#                 return {"status": "failed", "message": f"Optimization failed with status: {problem.status}"}
            
#             # Extract results
#             grid_power_result = p_grid.value
#             battery_power_result = p_batt.value
#             soc_result = soc.value
            
#             device_schedules = []
#             for device in device_params:
#                 device_id = device['id']
#                 device_power = p_devices[device_id].value
                
#                 # Create schedule entries for non-zero power periods
#                 schedule = []
#                 current_start = None
#                 current_power = None
                
#                 for t in range(time_steps):
#                     time_str = forecast_data.index[t].strftime('%H:%M')
#                     power = device_power[t]
                    
#                     if power > 0.01:  # Non-zero power (with small threshold for numerical issues)
#                         if current_start is None:
#                             current_start = time_str
#                             current_power = power
#                         elif abs(power - current_power) > 0.01:
#                             # Power level changed, end previous entry and start new one
#                             end_time = time_str
#                             schedule.append({
#                                 "start": current_start,
#                                 "end": end_time,
#                                 "power": float(current_power)
#                             })
#                             current_start = time_str
#                             current_power = power
#                     elif current_start is not None:
#                         # Power became zero, end the current entry
#                         end_time = time_str
#                         schedule.append({
#                             "start": current_start,
#                             "end": end_time,
#                             "power": float(current_power)
#                         })
#                         current_start = None
#                         current_power = None
                
#                 # Add the last entry if there's an open one
#                 if current_start is not None:
#                     end_time = forecast_data.index[-1].strftime('%H:%M')
#                     schedule.append({
#                         "start": current_start,
#                         "end": end_time,
#                         "power": float(current_power)
#                     })
                
#                 device_schedules.append({
#                     "id": device_id,
#                     "name": device.get('name', f"Device {device_id}"),
#                     "type": device.get('type', 'unknown'),
#                     "schedule": schedule
#                 })
            
#             # Calculate metrics
#             total_load = np.sum(load_forecast) + np.sum([p_devices[d].value for d in p_devices], axis=0).sum() if p_devices else np.sum(load_forecast)
#             total_pv = np.sum(pv_forecast)
#             total_grid_import = np.sum(np.maximum(grid_power_result, 0))
#             total_grid_export = np.sum(np.abs(np.minimum(grid_power_result, 0)))
            
#             # Self-consumption percentage
#             self_consumption_pct = 100 * (1 - total_grid_export / total_pv) if total_pv > 0 else 0
            
#             # Battery cycles
#             battery_cycles_value = np.sum(np.abs(battery_power_result)) / (2 * battery_capacity)
            
#             # Peak grid power
#             peak_grid_power_value = np.max(np.maximum(grid_power_result, 0))
            
#             # Cost calculation
#             cost_value = np.sum(np.maximum(grid_power_result, 0) * grid_import_price) - np.sum(np.abs(np.minimum(grid_power_result, 0)) * grid_export_price)
            
#             # Prepare schedule data for visualization
#             schedule_data = []
#             for t in range(time_steps):
#                 time_str = forecast_data.index[t].strftime('%H:%M')
                
#                 # Calculate optimized load (original load + controllable load)
#                 optimized_load = load_forecast[t]
#                 if p_devices:
#                     optimized_load += sum(p_devices[d].value[t] for d in p_devices)
                
#                 schedule_data.append({
#                     "time": time_str,
#                     "load": float(load_forecast[t]),
#                     "solar": float(pv_forecast[t]),
#                     "battery": float(battery_power_result[t]),
#                     "grid": float(grid_power_result[t]),
#                     "optimizedLoad": float(optimized_load)
#                 })
            
#             # Return results
#             return {
#                 "status": "success",
#                 "id": f"opt-{datetime.now().strftime('%Y%m%d%H%M%S')}",
#                 "cost": float(cost_value),
#                 "selfConsumption": float(self_consumption_pct),
#                 "peakGridPower": float(peak_grid_power_value),
#                 "batteryCycles": float(battery_cycles_value),
#                 "timestamp": datetime.now().isoformat(),
#                 "scheduleData": schedule_data,
#                 "deviceScheduleData": device_schedules
#             }
            
#         except Exception as e:
#             logger.error(f"Error in optimization: {e}", exc_info=True)
#             return {"status": "error", "message": str(e)}
    
#     def get_optimization_mode_weights(self, mode: str) -> Tuple[float, float, float, float]:
#         """
#         Get objective weights based on optimization mode
        
#         Args:
#             mode: Optimization mode (cost, self_consumption, grid_independence, battery_life)
            
#         Returns:
#             Tuple of weights (cost, self_consumption, peak_shaving, battery_cycle)
#         """
#         if mode == "cost":
#             return (1.0, 0.1, 0.1, 0.1)
#         elif mode == "self_consumption":
#             return (0.1, 1.0, 0.1, 0.1)
#         elif mode == "grid_independence":
#             return (0.1, 0.3, 1.0, 0.1)
#         elif mode == "battery_life":
#             return (0.1, 0.1, 0.1, 1.0)
#         else:
#             # Default to cost optimization
#             return (1.0, 0.1, 0.1, 0.1)

# import cvxpy as cp
# import numpy as np
# from datetime import datetime, timedelta
# import json
# import os
# # from .battery_model import create_battery_constraints, add_battery_to_objective
# import numpy as np
# import pandas as pd
# import cvxpy as cp
# from datetime import datetime, timedelta
# import logging
# from typing import Dict, List, Any, Optional, Tuple

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class EnergyOptimizer:
#     """
#     Energy optimization using linear programming techniques similar to EMHASS
#     """
    
#     def __init__(self, 
#                  forecast_data=None, devices=None, battery_capacity=10.0,
#                  cost_weight: float = 1.0,
#                  self_consumption_weight: float = 0.0,
#                  peak_shaving_weight: float = 0.0,
#                  battery_cycle_weight: float = 0.0):
#         """
#         Initialize the energy optimizer
        
#         Args:
#             cost_weight: Weight for cost minimization objective
#             self_consumption_weight: Weight for self-consumption maximization
#             peak_shaving_weight: Weight for peak shaving objective
#             battery_cycle_weight: Weight for battery cycle minimization
#         """
#         self.forecast_data = forecast_data or {}
#         self.devices = devices or []
#         self.battery_capacity = battery_capacity
#         self.time_periods = 24  # Default to 24 hours
#         self.period_duration = 1  # hours
#         self.cost_weight = cost_weight
#         self.self_consumption_weight = self_consumption_weight
#         self.peak_shaving_weight = peak_shaving_weight
#         self.battery_cycle_weight = battery_cycle_weight
        
#     def load_forecast_data(self, forecast_data):
#         """Load forecast data for optimization"""
#         self.forecast_data = forecast_data
        
#     def load_devices(self, devices):
#         """Load device data for optimization"""
#         self.devices = devices
        
#     def optimize(self, 
#                  forecast_data: pd.DataFrame=None,
#                  battery_params: Dict[str, Any]=None,
#                  grid_params: Dict[str, Any]=None,
#                  device_params: List[Dict[str, Any]]=None) -> Dict[str, Any]:
#         """
#         Run the optimization to create an optimal schedule
        
#         Args:
#             forecast_data: DataFrame with forecasted load and PV production
#             battery_params: Battery parameters (capacity, efficiency, etc.)
#             grid_params: Grid parameters (max power, pricing, etc.)
#             device_params: List of controllable devices with parameters
            
#         Returns:
#             Dictionary with optimization results
#         """
#         if forecast_data is None:
#             forecast_data = pd.DataFrame({
#                 'pv_power': self.forecast_data.get('solar_generation', [0] * self.time_periods),
#                 'load_power': self.forecast_data.get('base_load', [0.5] * self.time_periods)
#             })
        
#         if battery_params is None:
#             battery_params = {
#                 'capacity': self.battery_capacity,
#                 'max_power': 5.0,
#                 'efficiency': 0.95,
#                 'min_soc': 0.1,
#                 'max_soc': 0.9,
#                 'initial_soc': 0.5
#             }
        
#         if grid_params is None:
#             grid_params = {
#                 'max_power': 10.0,
#                 'export_price': 0.05,
#                 'import_price': 0.15
#             }
        
#         if device_params is None:
#             device_params = self.devices
            
#         try:
#             # Extract time steps and horizon
#             time_steps = len(forecast_data)
#             delta_t = 1.0  # Time step in hours
            
#             # Extract forecasts
#             pv_forecast = forecast_data['pv_power'].values
#             load_forecast = forecast_data['load_power'].values
            
#             # Extract battery parameters
#             battery_capacity = battery_params.get('capacity', 10.0)  # kWh
#             battery_max_power = battery_params.get('max_power', 5.0)  # kW
#             battery_efficiency = battery_params.get('efficiency', 0.95)
#             battery_min_soc = battery_params.get('min_soc', 0.1)
#             battery_max_soc = battery_params.get('max_soc', 0.9)
#             battery_initial_soc = battery_params.get('initial_soc', 0.5)
            
#             # Extract grid parameters
#             grid_max_power = grid_params.get('max_power', 10.0)  # kW
#             grid_export_price = grid_params.get('export_price', 0.05)  # $/kWh
            
#             # Time-of-use pricing if available, otherwise flat rate
#             if 'import_price_schedule' in grid_params:
#                 grid_import_price = np.array(grid_params['import_price_schedule'])
#             else:
#                 grid_import_price = np.ones(time_steps) * grid_params.get('import_price', 0.15)  # $/kWh
            
#             # Define optimization variables
#             p_grid = cp.Variable(time_steps)  # Grid power (positive = import, negative = export)
            
#             # DCP-compliant battery variables
#             # Instead of a single p_batt variable, use separate charging and discharging variables
#             p_batt_charge = cp.Variable(time_steps, nonneg=True)  # Battery charging power (always positive)
#             p_batt_discharge = cp.Variable(time_steps, nonneg=True)  # Battery discharging power (always positive)
#             soc = cp.Variable(time_steps + 1)  # Battery state of charge
            
#             # Device power variables (if controllable devices are provided)
#             p_devices = {}
#             device_constraints = []
            
#             # Prepare device parameters for optimization
#             device_params_processed = []
#             for device in device_params:
#                 if device.get('type') == 'shiftable':
#                     # For low wattage devices, we need to scale the power appropriately
#                     power = device.get('power', 0.01)  # Default to 10W
#                     duration = device.get('duration', 0.5)  # Default to 30 minutes
                    
#                     device_params_processed.append({
#                         "id": device.get('id', 'unknown'),
#                         "name": device.get('name', 'Unknown Device'),
#                         "type": "shiftable",
#                         "start_window": device.get('start_window', 8),
#                         "end_window": device.get('end_window', 18),
#                         "duration": duration,
#                         "power": power
#                     })
#                 elif device.get('type') == 'thermal':
#                     # For fans and temperature-controlled devices
#                     max_power = device.get('max_power', 0.025)  # Default to 25W
                    
#                     device_params_processed.append({
#                         "id": device.get('id', 'unknown'),
#                         "name": device.get('name', 'Unknown Device'),
#                         "type": "thermal",
#                         "temp_min": device.get('temp_min', 24),
#                         "temp_max": device.get('temp_max', 28),
#                         "temp_init": device.get('temp_init', 26),
#                         "max_power": max_power
#                     })
            
#             p_devices = {}
#             device_constraints = []
            
#             for device in device_params_processed:
#                 device_id = device['id']
#                 p_devices[device_id] = cp.Variable(time_steps)
                
#                 # Device constraints
#                 if device['type'] == 'shiftable':
#                     # Shiftable load (e.g., washing machine, dishwasher)
#                     # Must run for a specific duration within a time window
#                     start_window = device.get('start_window', 0)
#                     end_window = device.get('end_window', time_steps - 1)
#                     duration = device.get('duration', 2)
#                     power = device.get('power', 1.0)
                    
#                     # Binary variable for each possible start time
#                     x_start = cp.Variable(time_steps, boolean=True)
                    
#                     # Constraint: can only start once
#                     device_constraints.append(cp.sum(x_start) == 1)
                    
#                     # Constraint: can only start within the window
#                     for t in range(time_steps):
#                         if t < start_window or t > end_window - duration + 1:
#                             device_constraints.append(x_start[t] == 0)
                    
#                     # Constraint: power profile based on start time
#                     for t in range(time_steps):
#                         # Sum over all possible start times that would make the device run at time t
#                         p_t = cp.sum([x_start[max(0, t - d)] * power for d in range(min(duration, t + 1))])
#                         device_constraints.append(p_devices[device_id][t] == p_t)
                
#                 elif device['type'] == 'thermal':
#                     # Thermal load (e.g., water heater, HVAC)
#                     # Has a temperature state that must be kept within bounds
#                     temp_min = device.get('temp_min', 55)  # Minimum temperature
#                     temp_max = device.get('temp_max', 65)  # Maximum temperature
#                     temp_init = device.get('temp_init', 60)  # Initial temperature
#                     temp_ambient = device.get('temp_ambient', 20)  # Ambient temperature
#                     thermal_resistance = device.get('thermal_resistance', 0.1)  # K/kW
#                     thermal_capacitance = device.get('thermal_capacitance', 0.2)  # kWh/K
#                     cop = device.get('cop', 3.0)  # Coefficient of performance
#                     max_power = device.get('max_power', 2.0)  # Maximum power
                    
#                     # Temperature state variable
#                     temp = cp.Variable(time_steps + 1)
                    
#                     # Initial temperature
#                     device_constraints.append(temp[0] == temp_init)
                    
#                     # Temperature dynamics
#                     for t in range(time_steps):
#                         temp_next = temp[t] + delta_t * (
#                             p_devices[device_id][t] * cop / thermal_capacitance - 
#                             (temp[t] - temp_ambient) / (thermal_resistance * thermal_capacitance)
#                         )
#                         device_constraints.append(temp[t+1] == temp_next)
                        
#                     # Temperature bounds
#                     for t in range(time_steps + 1):
#                         device_constraints.append(temp[t] >= temp_min)
#                         device_constraints.append(temp[t] <= temp_max)
                    
#                     # Power bounds
#                     for t in range(time_steps):
#                         device_constraints.append(p_devices[device_id][t] >= 0)
#                         device_constraints.append(p_devices[device_id][t] <= max_power)
                
#                 elif device['type'] == 'ev_charger':
#                     # EV charger
#                     arrival_time = device.get('arrival_time', 0)
#                     departure_time = device.get('departure_time', time_steps - 1)
#                     energy_needed = device.get('energy_needed', 10.0)  # kWh
#                     max_power = device.get('max_power', 7.2)  # kW
                    
#                     # Constraint: only charge when EV is present
#                     for t in range(time_steps):
#                         if t < arrival_time or t > departure_time:
#                             device_constraints.append(p_devices[device_id][t] == 0)
                    
#                     # Constraint: power bounds when EV is present
#                     for t in range(arrival_time, departure_time + 1):
#                         device_constraints.append(p_devices[device_id][t] >= 0)
#                         device_constraints.append(p_devices[device_id][t] <= max_power)
                    
#                     # Constraint: total energy delivered
#                     device_constraints.append(cp.sum(p_devices[device_id]) * delta_t == energy_needed)
                
#                 else:
#                     # Default: fixed load profile
#                     power_profile = device.get('power_profile', np.zeros(time_steps))
#                     for t in range(time_steps):
#                         device_constraints.append(p_devices[device_id][t] == power_profile[t])
            
#             # Calculate total controllable load
#             p_controllable_load = cp.sum([p_devices[d] for d in p_devices], axis=0) if p_devices else np.zeros(time_steps)
            
#             # Power balance constraint: PV + Grid + Battery = Load
#             # Positive grid = import, negative grid = export
#             # Battery is now split into charging (positive) and discharging (positive)
#             power_balance_constraints = []
#             for t in range(time_steps):
#                 power_balance_constraints.append(
#                     pv_forecast[t] + p_grid[t] + p_batt_discharge[t] - p_batt_charge[t] == 
#                     load_forecast[t] + p_controllable_load[t]
#                 )
            
#             # Battery constraints
#             battery_constraints = []
            
#             # SOC dynamics - DCP-compliant version
#             for t in range(time_steps):
#                 # Charging adds energy with efficiency, discharging removes energy with efficiency
#                 soc_next = soc[t] + (battery_efficiency * p_batt_charge[t] - p_batt_discharge[t] / battery_efficiency) * delta_t / battery_capacity
#                 battery_constraints.append(soc[t+1] == soc_next)
            
#             # Initial SOC
#             battery_constraints.append(soc[0] == battery_initial_soc)
            
#             # SOC bounds
#             for t in range(time_steps + 1):
#                 battery_constraints.append(soc[t] >= battery_min_soc)
#                 battery_constraints.append(soc[t] <= battery_max_soc)
            
#             # Battery power bounds
#             for t in range(time_steps):
#                 battery_constraints.append(p_batt_charge[t] <= battery_max_power)   # Charge limit
#                 battery_constraints.append(p_batt_discharge[t] <= battery_max_power)  # Discharge limit
            
#             # Grid constraints
#             grid_constraints = []
            
#             # Grid power bounds
#             for t in range(time_steps):
#                 grid_constraints.append(p_grid[t] >= -grid_max_power)  # Export limit
#                 grid_constraints.append(p_grid[t] <= grid_max_power)   # Import limit
            
#             # Define objective function components
            
#             # 1. Cost minimization
#             grid_import = cp.maximum(p_grid, 0)
#             grid_export = cp.minimum(p_grid, 0)
#             cost = cp.sum(grid_import * grid_import_price) - cp.sum(grid_export * grid_export_price)
            
#             # 2. Self-consumption maximization (minimize grid export)
#             self_consumption = cp.sum(cp.abs(grid_export))
            
#             # 3. Peak shaving (minimize maximum grid import)
#             peak_power = cp.max(grid_import)
            
#             # 4. Battery cycle minimization
#             # Approximate by minimizing the sum of charging and discharging power
#             battery_cycles = cp.sum(p_batt_charge + p_batt_discharge) / (2 * battery_capacity)
            
#             # Combined objective with weights
#             objective = (
#                 self.cost_weight * cost +
#                 self.self_consumption_weight * self_consumption +
#                 self.peak_shaving_weight * peak_power +
#                 self.battery_cycle_weight * battery_cycles
#             )
            
#             # Define and solve the problem
#             constraints = (
#                 power_balance_constraints + 
#                 battery_constraints + 
#                 grid_constraints + 
#                 device_constraints
#             )
            
#             problem = cp.Problem(cp.Minimize(objective), constraints)
#             problem.solve(solver=cp.ECOS)
            
#             if problem.status != 'optimal':
#                 logger.warning(f"Optimization problem status: {problem.status}")
#                 return {"status": "failed", "message": f"Optimization failed with status: {problem.status}"}
            
#             # Extract results
#             grid_power_result = p_grid.value
            
#             # Reconstruct the original battery power from charge and discharge components
#             battery_power_result = p_batt_charge.value - p_batt_discharge.value
#             soc_result = soc.value
            
#             device_schedules = []
#             for device in device_params_processed:
#                 device_id = device['id']
#                 device_power = p_devices[device_id].value
                
#                 # Create schedule entries for non-zero power periods
#                 schedule = []
#                 current_start = None
#                 current_power = None
                
#                 for t in range(time_steps):
#                     time_str = forecast_data.index[t].strftime('%H:%M')
#                     power = device_power[t]
                    
#                     if power > 0.01:  # Non-zero power (with small threshold for numerical issues)
#                         if current_start is None:
#                             current_start = time_str
#                             current_power = power
#                         elif abs(power - current_power) > 0.01:
#                             # Power level changed, end previous entry and start new one
#                             end_time = time_str
#                             schedule.append({
#                                 "start": current_start,
#                                 "end": end_time,
#                                 "power": float(current_power)
#                             })
#                             current_start = time_str
#                             current_power = power
#                     elif current_start is not None:
#                         # Power became zero, end the current entry
#                         end_time = time_str
#                         schedule.append({
#                             "start": current_start,
#                             "end": end_time,
#                             "power": float(current_power)
#                         })
#                         current_start = None
#                         current_power = None
                
#                 # Add the last entry if there's an open one
#                 if current_start is not None:
#                     end_time = forecast_data.index[-1].strftime('%H:%M')
#                     schedule.append({
#                         "start": current_start,
#                         "end": end_time,
#                         "power": float(current_power)
#                     })
                
#                 device_schedules.append({
#                     "id": device_id,
#                     "name": device.get('name', f"Device {device_id}"),
#                     "type": device.get('type', 'unknown'),
#                     "schedule": schedule
#                 })
            
#             # Calculate metrics
#             total_load = np.sum(load_forecast) + np.sum([p_devices[d].value for d in p_devices], axis=0).sum() if p_devices else np.sum(load_forecast)
#             total_pv = np.sum(pv_forecast)
#             total_grid_import = np.sum(np.maximum(grid_power_result, 0))
#             total_grid_export = np.sum(np.abs(np.minimum(grid_power_result, 0)))
            
#             # Self-consumption percentage
#             self_consumption_pct = 100 * (1 - total_grid_export / total_pv) if total_pv > 0 else 0
            
#             # Battery cycles
#             battery_cycles_value = np.sum(np.abs(battery_power_result)) / (2 * battery_capacity)
            
#             # Peak grid power
#             peak_grid_power_value = np.max(np.maximum(grid_power_result, 0))
            
#             # Cost calculation
#             cost_value = np.sum(np.maximum(grid_power_result, 0) * grid_import_price) - np.sum(np.abs(np.minimum(grid_power_result, 0)) * grid_export_price)
            
#             # Prepare schedule data for visualization
#             schedule_data = []
#             for t in range(time_steps):
#                 time_str = forecast_data.index[t].strftime('%H:%M')
                
#                 # Calculate optimized load (original load + controllable load)
#                 optimized_load = load_forecast[t]
#                 if p_devices:
#                     optimized_load += sum(p_devices[d].value[t] for d in p_devices)
                
#                 schedule_data.append({
#                     "time": time_str,
#                     "load": float(load_forecast[t]),
#                     "solar": float(pv_forecast[t]),
#                     "battery": float(battery_power_result[t]),
#                     "grid": float(grid_power_result[t]),
#                     "optimizedLoad": float(optimized_load)
#                 })
            
#             # Return results
#             return {
#                 "status": "success",
#                 "id": f"opt-{datetime.now().strftime('%Y%m%d%H%M%S')}",
#                 "cost": float(cost_value),
#                 "selfConsumption": float(self_consumption_pct),
#                 "peakGridPower": float(peak_grid_power_value),
#                 "batteryCycles": float(battery_cycles_value),
#                 "timestamp": datetime.now().isoformat(),
#                 "scheduleData": schedule_data,
#                 "deviceScheduleData": device_schedules
#             }
            
#         except Exception as e:
#             logger.error(f"Error in optimization: {e}", exc_info=True)
#             return {"status": "error", "message": str(e)}
            
#     def generate_schedule(self):
#         """Generate an optimal schedule based on forecasts and device constraints"""
#         # Run optimization
        
#         # Create dummy dataframe for the old generate_schedule function
#         forecast_data = pd.DataFrame({
#             'pv_power': self.forecast_data.get('solar_generation', [0] * self.time_periods),
#             'load_power': self.forecast_data.get('base_load', [0.5] * self.time_periods)
#         })
        
#         optimization_results = self.optimize(
#             forecast_data=forecast_data,
#             battery_params={
#                 'capacity': self.battery_capacity,
#                 'max_power': 5.0,
#                 'efficiency': 0.95,
#                 'min_soc': 0.1,
#                 'max_soc': 0.9,
#                 'initial_soc': 0.5
#             },
#             grid_params={
#                 'max_power': 10.0,
#                 'export_price': 0.05,
#                 'import_price': 0.15
#             },
#             device_params=self.devices
#         )
        
#         if optimization_results.get("status") == "success":
#             # Create schedule with timestamps
#             start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
#             schedule = []
            
#             # Process device schedules
#             for device_data in optimization_results.get("deviceScheduleData", {}):
#                 device_schedule = []
#                 for schedule_entry in device_data["schedule"]:
#                     start_time_obj = datetime.strptime(schedule_entry["start"], "%H:%M")
#                     end_time_obj = datetime.strptime(schedule_entry["end"], "%H:%M")
                    
#                     # Calculate the start and end datetime objects for the current day
#                     start_datetime = start_time.replace(hour=start_time_obj.hour, minute=start_time_obj.minute)
#                     end_datetime = start_time.replace(hour=end_time_obj.hour, minute=end_time_obj.minute)
                    
#                     # Handle cases where the end time is before the start time (crosses midnight)
#                     if end_datetime <= start_datetime:
#                         end_datetime += timedelta(days=1)
                    
#                     device_schedule.append({
#                         "start": start_datetime.strftime("%Y-%m-%d %H:%M"),
#                         "end": end_datetime.strftime("%Y-%m-%d %H:%M"),
#                         "power_kw": schedule_entry["power"]
#                     })
                
#                 schedule.append({
#                     "device_id": device_data["id"],
#                     "name": device_data["name"],
#                     "schedule": device_schedule,
#                     "type": device_data["type"]
#                 })
            
#             # Add grid and battery data
#             grid_data = []
#             battery_data = []
#             for schedule_entry in optimization_results["scheduleData"]:
#                 time_str = start_time.replace(hour=int(schedule_entry["time"][:2]), minute=int(schedule_entry["time"][3:])).strftime("%Y-%m-%d %H:%M")
                
#                 grid_data.append({
#                     "time": time_str,
#                     "import_kw": max(schedule_entry["grid"], 0),
#                     "export_kw": abs(min(schedule_entry["grid"], 0))
#                 })
                
#                 battery_data.append({
#                     "time": time_str,
#                     "power_kw": schedule_entry["battery"]
#                 })
            
#             return {
#                 "status": "success",
#                 "total_cost": optimization_results["cost"],
#                 "self_consumption": optimization_results["selfConsumption"],
#                 "peak_grid_power": optimization_results["peakGridPower"],
#                 "battery_cycles": optimization_results["batteryCycles"],
#                 "device_schedules": schedule,
#                 "grid_data": grid_data,
#                 "battery_data": battery_data
#             }
#         else:
#             return {
#                 "status": "failed",
#                 "error": optimization_results.get("message", "Unknown optimization error")
#             }
    
#     def get_optimization_mode_weights(self, mode: str) -> Tuple[float, float, float, float]:
#         """
#         Get objective weights based on optimization mode
        
#         Args:
#             mode: Optimization mode (cost, self_consumption, grid_independence, battery_life)
            
#         Returns:
#             Tuple of weights (cost, self_consumption, peak_shaving, battery_cycle)
#         """
#         if mode == "cost":
#             return (1.0, 0.1, 0.1, 0.1)
#         elif mode == "self_consumption":
#             return (0.1, 1.0, 0.1, 0.1)
#         elif mode == "grid_independence":
#             return (0.1, 0.3, 1.0, 0.1)
#         elif mode == "battery_life":
#             return (0.1, 0.1, 0.1, 1.0)
#         else:
#             # Default to cost optimization
#             return (1.0, 0.1, 0.1, 0.1)

# import numpy as np
# import pandas as pd
# import cvxpy as cp
# from datetime import datetime, timedelta
# import logging
# from typing import Dict, List, Any, Optional, Tuple

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class EnergyOptimizer:
#     """
#     Energy optimization using linear programming techniques similar to EMHASS
#     """
    
#     def __init__(self, 
#                  forecast_data=None, devices=None, battery_capacity=10.0,
#                  cost_weight: float = 1.0,
#                  self_consumption_weight: float = 0.0,
#                  peak_shaving_weight: float = 0.0,
#                  battery_cycle_weight: float = 0.0):
#         """
#         Initialize the energy optimizer
        
#         Args:
#             cost_weight: Weight for cost minimization objective
#             self_consumption_weight: Weight for self-consumption maximization
#             peak_shaving_weight: Weight for peak shaving objective
#             battery_cycle_weight: Weight for battery cycle minimization
#         """
#         self.forecast_data = forecast_data or {}
#         self.devices = devices or []
#         self.battery_capacity = battery_capacity
#         self.time_periods = 24  # Default to 24 hours
#         self.period_duration = 1  # hours
#         self.cost_weight = cost_weight
#         self.self_consumption_weight = self_consumption_weight
#         self.peak_shaving_weight = peak_shaving_weight
#         self.battery_cycle_weight = battery_cycle_weight
        
#     def load_forecast_data(self, forecast_data):
#         """Load forecast data for optimization"""
#         self.forecast_data = forecast_data
        
#     def load_devices(self, devices):
#         """Load device data for optimization"""
#         self.devices = devices
        
#     def optimize(self, 
#                  forecast_data: pd.DataFrame=None,
#                  battery_params: Dict[str, Any]=None,
#                  grid_params: Dict[str, Any]=None,
#                  device_params: List[Dict[str, Any]]=None) -> Dict[str, Any]:
#         """
#         Run the optimization to create an optimal schedule
        
#         Args:
#             forecast_data: DataFrame with forecasted load and PV production
#             battery_params: Battery parameters (capacity, efficiency, etc.)
#             grid_params: Grid parameters (max power, pricing, etc.)
#             device_params: List of controllable devices with parameters
            
#         Returns:
#             Dictionary with optimization results
#         """
#         if forecast_data is None:
#             forecast_data = pd.DataFrame({
#                 'pv_power': self.forecast_data.get('solar_generation', [0] * self.time_periods),
#                 'load_power': self.forecast_data.get('base_load', [0.5] * self.time_periods)
#             })
        
#         if battery_params is None:
#             battery_params = {
#                 'capacity': self.battery_capacity,
#                 'max_power': 5.0,
#                 'efficiency': 0.95,
#                 'min_soc': 0.1,
#                 'max_soc': 0.9,
#                 'initial_soc': 0.5
#             }
        
#         if grid_params is None:
#             grid_params = {
#                 'max_power': 10.0,
#                 'export_price': 0.05,
#                 'import_price': 0.15
#             }
        
#         if device_params is None:
#             device_params = self.devices
            
#         try:
#             # Extract time steps and horizon - ensure it's an integer
#             time_steps = int(len(forecast_data))
#             delta_t = 1.0  # Time step in hours
            
#             # Extract forecasts
#             pv_forecast = forecast_data['pv_power'].values
#             load_forecast = forecast_data['load_power'].values
            
#             # Extract battery parameters
#             battery_capacity = float(battery_params.get('capacity', 10.0))  # kWh
#             battery_max_power = float(battery_params.get('max_power', 5.0))  # kW
#             battery_efficiency = float(battery_params.get('efficiency', 0.95))
#             battery_min_soc = float(battery_params.get('min_soc', 0.1))
#             battery_max_soc = float(battery_params.get('max_soc', 0.9))
#             battery_initial_soc = float(battery_params.get('initial_soc', 0.5))
            
#             # Extract grid parameters
#             grid_max_power = float(grid_params.get('max_power', 10.0))  # kW
#             grid_export_price = float(grid_params.get('export_price', 0.05))  # $/kWh
            
#             # Time-of-use pricing if available, otherwise flat rate
#             if 'import_price_schedule' in grid_params:
#                 grid_import_price = np.array(grid_params['import_price_schedule'], dtype=float)
#             else:
#                 grid_import_price = np.ones(time_steps, dtype=float) * float(grid_params.get('import_price', 0.15))  # $/kWh
            
#             # Define optimization variables
#             p_grid = cp.Variable(time_steps)  # Grid power (positive = import, negative = export)
            
#             # DCP-compliant battery variables
#             # Instead of a single p_batt variable, use separate charging and discharging variables
#             p_batt_charge = cp.Variable(time_steps, nonneg=True)  # Battery charging power (always positive)
#             p_batt_discharge = cp.Variable(time_steps, nonneg=True)  # Battery discharging power (always positive)
#             soc = cp.Variable(time_steps + 1)  # Battery state of charge
            
#             # Device power variables (if controllable devices are provided)
#             p_devices = {}
#             device_constraints = []
            
#             # Prepare device parameters for optimization
#             device_params_processed = []
#             for device in device_params:
#                 if device.get('type') == 'shiftable':
#                     # For low wattage devices, we need to scale the power appropriately
#                     power = float(device.get('power', 0.01))  # Default to 10W
#                     duration = float(device.get('duration', 0.5))  # Default to 30 minutes
                    
#                     device_params_processed.append({
#                         "id": device.get('id', 'unknown'),
#                         "name": device.get('name', 'Unknown Device'),
#                         "type": "shiftable",
#                         "start_window": int(device.get('start_window', 8)),
#                         "end_window": int(device.get('end_window', 18)),
#                         "duration": int(duration),  # Convert to integer for iteration
#                         "power": power
#                     })
#                 elif device.get('type') == 'thermal':
#                     # For fans and temperature-controlled devices
#                     max_power = float(device.get('max_power', 0.025))  # Default to 25W
                    
#                     device_params_processed.append({
#                         "id": device.get('id', 'unknown'),
#                         "name": device.get('name', 'Unknown Device'),
#                         "type": "thermal",
#                         "temp_min": float(device.get('temp_min', 24)),
#                         "temp_max": float(device.get('temp_max', 28)),
#                         "temp_init": float(device.get('temp_init', 26)),
#                         "max_power": max_power
#                     })
            
#             p_devices = {}
#             device_constraints = []
            
#             for device in device_params_processed:
#                 device_id = device['id']
#                 p_devices[device_id] = cp.Variable(time_steps)
                
#                 # Device constraints
#                 if device['type'] == 'shiftable':
#                     # Shiftable load (e.g., washing machine, dishwasher)
#                     # Must run for a specific duration within a time window
#                     start_window = int(device.get('start_window', 0))
#                     end_window = int(device.get('end_window', time_steps - 1))
#                     duration = int(device.get('duration', 2))
#                     power = float(device.get('power', 1.0))
                    
#                     # Binary variable for each possible start time
#                     x_start = cp.Variable(time_steps, boolean=True)
                    
#                     # Constraint: can only start once
#                     device_constraints.append(cp.sum(x_start) == 1)
                    
#                     # Constraint: can only start within the window
#                     for t in range(time_steps):
#                         if t < start_window or t > end_window - duration + 1:
#                             device_constraints.append(x_start[t] == 0)
                    
#                     # Constraint: power profile based on start time
#                     for t in range(time_steps):
#                         # Sum over all possible start times that would make the device run at time t
#                         p_t = cp.sum([x_start[max(0, t - d)] * power for d in range(min(duration, t + 1))])
#                         device_constraints.append(p_devices[device_id][t] == p_t)
                
#                 elif device['type'] == 'thermal':
#                     # Thermal load (e.g., water heater, HVAC)
#                     # Has a temperature state that must be kept within bounds
#                     temp_min = float(device.get('temp_min', 55))  # Minimum temperature
#                     temp_max = float(device.get('temp_max', 65))  # Maximum temperature
#                     temp_init = float(device.get('temp_init', 60))  # Initial temperature
#                     temp_ambient = float(device.get('temp_ambient', 20))  # Ambient temperature
#                     thermal_resistance = float(device.get('thermal_resistance', 0.1))  # K/kW
#                     thermal_capacitance = float(device.get('thermal_capacitance', 0.2))  # kWh/K
#                     cop = float(device.get('cop', 3.0))  # Coefficient of performance
#                     max_power = float(device.get('max_power', 2.0))  # Maximum power
                    
#                     # Temperature state variable
#                     temp = cp.Variable(time_steps + 1)
                    
#                     # Initial temperature
#                     device_constraints.append(temp[0] == temp_init)
                    
#                     # Temperature dynamics
#                     for t in range(time_steps):
#                         temp_next = temp[t] + delta_t * (
#                             p_devices[device_id][t] * cop / thermal_capacitance - 
#                             (temp[t] - temp_ambient) / (thermal_resistance * thermal_capacitance)
#                         )
#                         device_constraints.append(temp[t+1] == temp_next)
                        
#                     # Temperature bounds
#                     for t in range(time_steps + 1):
#                         device_constraints.append(temp[t] >= temp_min)
#                         device_constraints.append(temp[t] <= temp_max)
                    
#                     # Power bounds
#                     for t in range(time_steps):
#                         device_constraints.append(p_devices[device_id][t] >= 0)
#                         device_constraints.append(p_devices[device_id][t] <= max_power)
                
#                 elif device['type'] == 'ev_charger':
#                     # EV charger
#                     arrival_time = int(device.get('arrival_time', 0))
#                     departure_time = int(device.get('departure_time', time_steps - 1))
#                     energy_needed = float(device.get('energy_needed', 10.0))  # kWh
#                     max_power = float(device.get('max_power', 7.2))  # kW
                    
#                     # Constraint: only charge when EV is present
#                     for t in range(time_steps):
#                         if t < arrival_time or t > departure_time:
#                             device_constraints.append(p_devices[device_id][t] == 0)
                    
#                     # Constraint: power bounds when EV is present
#                     for t in range(arrival_time, departure_time + 1):
#                         device_constraints.append(p_devices[device_id][t] >= 0)
#                         device_constraints.append(p_devices[device_id][t] <= max_power)
                    
#                     # Constraint: total energy delivered
#                     device_constraints.append(cp.sum(p_devices[device_id]) * delta_t == energy_needed)
                
#                 else:
#                     # Default: fixed load profile
#                     power_profile = device.get('power_profile', np.zeros(time_steps))
#                     for t in range(time_steps):
#                         device_constraints.append(p_devices[device_id][t] == power_profile[t])
            
#             # Calculate total controllable load
#             p_controllable_load = cp.sum([p_devices[d] for d in p_devices], axis=0) if p_devices else np.zeros(time_steps)
            
#             # Power balance constraint: PV + Grid + Battery = Load
#             # Positive grid = import, negative grid = export
#             # Battery is now split into charging (positive) and discharging (positive)
#             power_balance_constraints = []
#             for t in range(time_steps):
#                 power_balance_constraints.append(
#                     pv_forecast[t] + p_grid[t] + p_batt_discharge[t] - p_batt_charge[t] == 
#                     load_forecast[t] + p_controllable_load[t]
#                 )
            
#             # Battery constraints
#             battery_constraints = []
            
#             # SOC dynamics - DCP-compliant version
#             for t in range(time_steps):
#                 # Charging adds energy with efficiency, discharging removes energy with efficiency
#                 soc_next = soc[t] + (battery_efficiency * p_batt_charge[t] - p_batt_discharge[t] / battery_efficiency) * delta_t / battery_capacity
#                 battery_constraints.append(soc[t+1] == soc_next)
            
#             # Initial SOC
#             battery_constraints.append(soc[0] == battery_initial_soc)
            
#             # SOC bounds
#             for t in range(time_steps + 1):
#                 battery_constraints.append(soc[t] >= battery_min_soc)
#                 battery_constraints.append(soc[t] <= battery_max_soc)
            
#             # Battery power bounds
#             for t in range(time_steps):
#                 battery_constraints.append(p_batt_charge[t] <= battery_max_power)   # Charge limit
#                 battery_constraints.append(p_batt_discharge[t] <= battery_max_power)  # Discharge limit
            
#             # Grid constraints
#             grid_constraints = []
            
#             # Grid power bounds
#             for t in range(time_steps):
#                 grid_constraints.append(p_grid[t] >= -grid_max_power)  # Export limit
#                 grid_constraints.append(p_grid[t] <= grid_max_power)   # Import limit
            
#             # Define objective function components
            
#             # 1. Cost minimization
#             grid_import = cp.maximum(p_grid, 0)
#             grid_export = cp.minimum(p_grid, 0)
#             cost = cp.sum(grid_import * grid_import_price) - cp.sum(grid_export * grid_export_price)
            
#             # 2. Self-consumption maximization (minimize grid export)
#             self_consumption = cp.sum(cp.abs(grid_export))
            
#             # 3. Peak shaving (minimize maximum grid import)
#             peak_power = cp.max(grid_import)
            
#             # 4. Battery cycle minimization
#             # Approximate by minimizing the sum of charging and discharging power
#             battery_cycles = cp.sum(p_batt_charge + p_batt_discharge) / (2 * battery_capacity)
            
#             # Combined objective with weights
#             objective = (
#                 self.cost_weight * cost +
#                 self.self_consumption_weight * self_consumption +
#                 self.peak_shaving_weight * peak_power +
#                 self.battery_cycle_weight * battery_cycles
#             )
            
#             # Define and solve the problem
#             constraints = (
#                 power_balance_constraints + 
#                 battery_constraints + 
#                 grid_constraints + 
#                 device_constraints
#             )
            
#             problem = cp.Problem(cp.Minimize(objective), constraints)
#             problem.solve(solver=cp.ECOS)
            
#             if problem.status != 'optimal':
#                 logger.warning(f"Optimization problem status: {problem.status}")
#                 return {"status": "failed", "message": f"Optimization failed with status: {problem.status}"}
            
#             # Extract results
#             grid_power_result = p_grid.value
            
#             # Reconstruct the original battery power from charge and discharge components
#             battery_power_result = p_batt_charge.value - p_batt_discharge.value
#             soc_result = soc.value
            
#             device_schedules = []
#             for device in device_params_processed:
#                 device_id = device['id']
#                 device_power = p_devices[device_id].value
                
#                 # Create schedule entries for non-zero power periods
#                 schedule = []
#                 current_start = None
#                 current_power = None
                
#                 for t in range(time_steps):
#                     time_str = forecast_data.index[t].strftime('%H:%M')
#                     power = device_power[t]
                    
#                     if power > 0.01:  # Non-zero power (with small threshold for numerical issues)
#                         if current_start is None:
#                             current_start = time_str
#                             current_power = power
#                         elif abs(power - current_power) > 0.01:
#                             # Power level changed, end previous entry and start new one
#                             end_time = time_str
#                             schedule.append({
#                                 "start": current_start,
#                                 "end": end_time,
#                                 "power": float(current_power)
#                             })
#                             current_start = time_str
#                             current_power = power
#                     elif current_start is not None:
#                         # Power became zero, end the current entry
#                         end_time = time_str
#                         schedule.append({
#                             "start": current_start,
#                             "end": end_time,
#                             "power": float(current_power)
#                         })
#                         current_start = None
#                         current_power = None
                
#                 # Add the last entry if there's an open one
#                 if current_start is not None:
#                     end_time = forecast_data.index[-1].strftime('%H:%M')
#                     schedule.append({
#                         "start": current_start,
#                         "end": end_time,
#                         "power": float(current_power)
#                     })
                
#                 device_schedules.append({
#                     "id": device_id,
#                     "name": device.get('name', f"Device {device_id}"),
#                     "type": device.get('type', 'unknown'),
#                     "schedule": schedule
#                 })
            
#             # Calculate metrics
#             total_load = np.sum(load_forecast) + np.sum([p_devices[d].value for d in p_devices], axis=0).sum() if p_devices else np.sum(load_forecast)
#             total_pv = np.sum(pv_forecast)
#             total_grid_import = np.sum(np.maximum(grid_power_result, 0))
#             total_grid_export = np.sum(np.abs(np.minimum(grid_power_result, 0)))
            
#             # Self-consumption percentage
#             self_consumption_pct = 100 * (1 - total_grid_export / total_pv) if total_pv > 0 else 0
            
#             # Battery cycles
#             battery_cycles_value = np.sum(np.abs(battery_power_result)) / (2 * battery_capacity)
            
#             # Peak grid power
#             peak_grid_power_value = np.max(np.maximum(grid_power_result, 0))
            
#             # Cost calculation
#             cost_value = np.sum(np.maximum(grid_power_result, 0) * grid_import_price) - np.sum(np.abs(np.minimum(grid_power_result, 0)) * grid_export_price)
            
#             # Prepare schedule data for visualization
#             schedule_data = []
#             for t in range(time_steps):
#                 time_str = forecast_data.index[t].strftime('%H:%M')
                
#                 # Calculate optimized load (original load + controllable load)
#                 optimized_load = load_forecast[t]
#                 if p_devices:
#                     optimized_load += sum(p_devices[d].value[t] for d in p_devices)
                
#                 schedule_data.append({
#                     "time": time_str,
#                     "load": float(load_forecast[t]),
#                     "solar": float(pv_forecast[t]),
#                     "battery": float(battery_power_result[t]),
#                     "grid": float(grid_power_result[t]),
#                     "optimizedLoad": float(optimized_load)
#                 })
            
#             # Return results
#             return {
#                 "status": "success",
#                 "id": f"opt-{datetime.now().strftime('%Y%m%d%H%M%S')}",
#                 "cost": float(cost_value),
#                 "selfConsumption": float(self_consumption_pct),
#                 "peakGridPower": float(peak_grid_power_value),
#                 "batteryCycles": float(battery_cycles_value),
#                 "timestamp": datetime.now().isoformat(),
#                 "scheduleData": schedule_data,
#                 "deviceScheduleData": device_schedules
#             }
            
#         except Exception as e:
#             logger.error(f"Error in optimization: {e}", exc_info=True)
#             return {"status": "error", "message": str(e)}
            
#     def generate_schedule(self):
#         """Generate an optimal schedule based on forecasts and device constraints"""
#         # Run optimization
        
#         # Create dummy dataframe for the old generate_schedule function
#         # Create index for the DataFrame
#         index = pd.date_range(
#             start=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
#             periods=self.time_periods,
#             freq='H'
#         )
        
#         forecast_data = pd.DataFrame({
#             'pv_power': self.forecast_data.get('solar_generation', [0] * self.time_periods),
#             'load_power': self.forecast_data.get('base_load', [0.5] * self.time_periods)
#         }, index=index)
        
#         optimization_results = self.optimize(
#             forecast_data=forecast_data,
#             battery_params={
#                 'capacity': self.battery_capacity,
#                 'max_power': 5.0,
#                 'efficiency': 0.95,
#                 'min_soc': 0.1,
#                 'max_soc': 0.9,
#                 'initial_soc': 0.5
#             },
#             grid_params={
#                 'max_power': 10.0,
#                 'export_price': 0.05,
#                 'import_price': 0.15
#             },
#             device_params=self.devices
#         )
        
#         if optimization_results.get("status") == "success":
#             # Create schedule with timestamps
#             start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
#             schedule = []
            
#             # Process device schedules
#             for device_data in optimization_results.get("deviceScheduleData", []):
#                 device_schedule = []
#                 for schedule_entry in device_data.get("schedule", []):
#                     try:
#                         start_time_obj = datetime.strptime(schedule_entry["start"], "%H:%M")
#                         end_time_obj = datetime.strptime(schedule_entry["end"], "%H:%M")
                        
#                         # Calculate the start and end datetime objects for the current day
#                         start_datetime = start_time.replace(hour=start_time_obj.hour, minute=start_time_obj.minute)
#                         end_datetime = start_time.replace(hour=end_time_obj.hour, minute=end_time_obj.minute)
                        
#                         # Handle cases where the end time is before the start time (crosses midnight)
#                         if end_datetime <= start_datetime:
#                             end_datetime += timedelta(days=1)
                        
#                         device_schedule.append({
#                             "start": start_datetime.strftime("%Y-%m-%d %H:%M"),
#                             "end": end_datetime.strftime("%Y-%m-%d %H:%M"),
#                             "power_kw": float(schedule_entry["power"])
#                         })
#                     except Exception as e:
#                         logger.error(f"Error processing schedule entry: {e}")
#                         continue
                
#                 schedule.append({
#                     "device_id": device_data["id"],
#                     "name": device_data["name"],
#                     "schedule": device_schedule,
#                     "type": device_data.get("type", "unknown")
#                 })
            
#             # Add grid and battery data
#             grid_data = []
#             battery_data = []
#             for schedule_entry in optimization_results.get("scheduleData", []):
#                 try:
#                     hour, minute = map(int, schedule_entry["time"].split(':'))
#                     time_str = start_time.replace(hour=hour, minute=minute).strftime("%Y-%m-%d %H:%M")
                    
#                     grid_data.append({
#                         "time": time_str,
#                         "import_kw": max(schedule_entry["grid"], 0),
#                         "export_kw": abs(min(schedule_entry["grid"], 0))
#                     })
                    
#                     battery_data.append({
#                         "time": time_str,
#                         "power_kw": schedule_entry["battery"]
#                     })
#                 except Exception as e:
#                     logger.error(f"Error processing schedule data entry: {e}")
#                     continue
            
#             return {
#                 "status": "success",
#                 "total_cost": optimization_results["cost"],
#                 "self_consumption": optimization_results["selfConsumption"],
#                 "peak_grid_power": optimization_results["peakGridPower"],
#                 "battery_cycles": optimization_results["batteryCycles"],
#                 "device_schedules": schedule,
#                 "grid_data": grid_data,
#                 "battery_data": battery_data
#             }
#         else:
#             return {
#                 "status": "failed",
#                 "error": optimization_results.get("message", "Unknown optimization error")
#             }
    
#     def get_optimization_mode_weights(self, mode: str) -> Tuple[float, float, float, float]:
#         """
#         Get objective weights based on optimization mode
        
#         Args:
#             mode: Optimization mode (cost, self_consumption, grid_independence, battery_life)
            
#         Returns:
#             Tuple of weights (cost, self_consumption, peak_shaving, battery_cycle)
#         """
#         if mode == "cost":
#             return (1.0, 0.1, 0.1, 0.1)
#         elif mode == "self_consumption":
#             return (0.1, 1.0, 0.1, 0.1)
#         elif mode == "grid_independence":
#             return (0.1, 0.3, 1.0, 0.1)
#         elif mode == "battery_life":
#             return (0.1, 0.1, 0.1, 1.0)
#         else:
#             # Default to cost optimization
#             return (1.0, 0.1, 0.1, 0.1)

import numpy as np
import pandas as pd
import cvxpy as cp
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyOptimizer:
    """
    Energy optimization using linear programming techniques similar to EMHASS
    """
    
    def __init__(self, 
                 forecast_data=None, devices=None, battery_capacity=10.0,
                 cost_weight: float = 1.0,
                 self_consumption_weight: float = 0.0,
                 peak_shaving_weight: float = 0.0,
                 battery_cycle_weight: float = 0.0):
        """
        Initialize the energy optimizer
        
        Args:
            cost_weight: Weight for cost minimization objective
            self_consumption_weight: Weight for self-consumption maximization
            peak_shaving_weight: Weight for peak shaving objective
            battery_cycle_weight: Weight for battery cycle minimization
        """
        self.forecast_data = forecast_data or {}
        self.devices = devices or []
        self.battery_capacity = battery_capacity
        self.time_periods = 24  # Default to 24 hours
        self.period_duration = 1  # hours
        self.cost_weight = cost_weight
        self.self_consumption_weight = self_consumption_weight
        self.peak_shaving_weight = peak_shaving_weight
        self.battery_cycle_weight = battery_cycle_weight
        
    def load_forecast_data(self, forecast_data):
        """Load forecast data for optimization"""
        self.forecast_data = forecast_data
        
    def load_devices(self, devices):
        """Load device data for optimization"""
        self.devices = devices
        
    def optimize(self, 
                 forecast_data: pd.DataFrame=None,
                 battery_params: Dict[str, Any]=None,
                 grid_params: Dict[str, Any]=None,
                 device_params: List[Dict[str, Any]]=None) -> Dict[str, Any]:
        """
        Run the optimization to create an optimal schedule
        
        Args:
            forecast_data: DataFrame with forecasted load and PV production
            battery_params: Battery parameters (capacity, efficiency, etc.)
            grid_params: Grid parameters (max power, pricing, etc.)
            device_params: List of controllable devices with parameters
            
        Returns:
            Dictionary with optimization results
        """
        if forecast_data is None:
            forecast_data = pd.DataFrame({
                'pv_power': self.forecast_data.get('solar_generation', [0] * self.time_periods),
                'load_power': self.forecast_data.get('base_load', [0.5] * self.time_periods)
            })
        
        if battery_params is None:
            battery_params = {
                'capacity': self.battery_capacity,
                'max_power': 5.0,
                'efficiency': 0.95,
                'min_soc': 0.1,
                'max_soc': 0.9,
                'initial_soc': 0.5
            }
        
        if grid_params is None:
            grid_params = {
                'max_power': 10.0,
                'export_price': 0.05,
                'import_price': 0.15
            }
        
        if device_params is None:
            device_params = self.devices
            
        try:
            # Extract time steps and horizon - ensure it's an integer
            time_steps = int(len(forecast_data))
            delta_t = 1.0  # Time step in hours
            
            # Extract forecasts
            pv_forecast = forecast_data['pv_power'].values
            load_forecast = forecast_data['load_power'].values
            
            # Extract battery parameters
            battery_capacity = float(battery_params.get('capacity', 10.0))  # kWh
            battery_max_power = float(battery_params.get('max_power', 5.0))  # kW
            battery_efficiency = float(battery_params.get('efficiency', 0.95))
            battery_min_soc = float(battery_params.get('min_soc', 0.1))
            battery_max_soc = float(battery_params.get('max_soc', 0.9))
            battery_initial_soc = float(battery_params.get('initial_soc', 0.5))
            
            # Extract grid parameters
            grid_max_power = float(grid_params.get('max_power', 10.0))  # kW
            grid_export_price = float(grid_params.get('export_price', 0.05))  # $/kWh
            
            # Time-of-use pricing if available, otherwise flat rate
            if 'import_price_schedule' in grid_params:
                grid_import_price = np.array(grid_params['import_price_schedule'], dtype=float)
            else:
                grid_import_price = np.ones(time_steps, dtype=float) * float(grid_params.get('import_price', 0.15))  # $/kWh
            
            # Define optimization variables
            p_grid = cp.Variable(time_steps)  # Grid power (positive = import, negative = export)
            
            # DCP-compliant battery variables
            # Instead of a single p_batt variable, use separate charging and discharging variables
            p_batt_charge = cp.Variable(time_steps, nonneg=True)  # Battery charging power (always positive)
            p_batt_discharge = cp.Variable(time_steps, nonneg=True)  # Battery discharging power (always positive)
            soc = cp.Variable(time_steps + 1)  # Battery state of charge
            
            # Device power variables (if controllable devices are provided)
            p_devices = {}
            device_constraints = []
            
            # Prepare device parameters for optimization
            device_params_processed = []
            for device in device_params:
                if device.get('type') == 'shiftable':
                    # For low wattage devices, we need to scale the power appropriately
                    power = float(device.get('power', 0.01))  # Default to 10W
                    duration = float(device.get('duration', 0.5))  # Default to 30 minutes
                    
                    device_params_processed.append({
                        "id": device.get('id', 'unknown'),
                        "name": device.get('name', 'Unknown Device'),
                        "type": "shiftable",
                        "start_window": int(device.get('start_window', 8)),
                        "end_window": int(device.get('end_window', 18)),
                        "duration": int(duration),  # Convert to integer for iteration
                        "power": power
                    })
                elif device.get('type') == 'thermal':
                    # For fans and temperature-controlled devices
                    max_power = float(device.get('max_power', 0.025))  # Default to 25W
                    
                    device_params_processed.append({
                        "id": device.get('id', 'unknown'),
                        "name": device.get('name', 'Unknown Device'),
                        "type": "thermal",
                        "temp_min": float(device.get('temp_min', 24)),
                        "temp_max": float(device.get('temp_max', 28)),
                        "temp_init": float(device.get('temp_init', 26)),
                        "max_power": max_power
                    })
            
            p_devices = {}
            device_constraints = []
            
            for device in device_params_processed:
                device_id = device['id']
                p_devices[device_id] = cp.Variable(time_steps)
                
                # Device constraints
                if device['type'] == 'shiftable':
                    # Shiftable load (e.g., washing machine, dishwasher)
                    # Must run for a specific duration within a time window
                    start_window = int(device.get('start_window', 0))
                    end_window = int(device.get('end_window', time_steps - 1))
                    duration = int(device.get('duration', 2))
                    power = float(device.get('power', 1.0))
                    
                    # Binary variable for each possible start time
                    x_start = cp.Variable(time_steps, boolean=True)
                    
                    # Constraint: can only start once
                    device_constraints.append(cp.sum(x_start) == 1)
                    
                    # Constraint: can only start within the window
                    for t in range(time_steps):
                        if t < start_window or t > end_window - duration + 1:
                            device_constraints.append(x_start[t] == 0)
                    
                    # Constraint: power profile based on start time
                    for t in range(time_steps):
                        # Sum over all possible start times that would make the device run at time t
                        p_t = cp.sum([x_start[max(0, t - d)] * power for d in range(min(duration, t + 1))])
                        device_constraints.append(p_devices[device_id][t] == p_t)
                
                elif device['type'] == 'thermal':
                    # Thermal load (e.g., water heater, HVAC)
                    # Has a temperature state that must be kept within bounds
                    temp_min = float(device.get('temp_min', 55))  # Minimum temperature
                    temp_max = float(device.get('temp_max', 65))  # Maximum temperature
                    temp_init = float(device.get('temp_init', 60))  # Initial temperature
                    temp_ambient = float(device.get('temp_ambient', 20))  # Ambient temperature
                    thermal_resistance = float(device.get('thermal_resistance', 0.1))  # K/kW
                    thermal_capacitance = float(device.get('thermal_capacitance', 0.2))  # kWh/K
                    cop = float(device.get('cop', 3.0))  # Coefficient of performance
                    max_power = float(device.get('max_power', 2.0))  # Maximum power
                    
                    # Temperature state variable
                    temp = cp.Variable(time_steps + 1)
                    
                    # Initial temperature
                    device_constraints.append(temp[0] == temp_init)
                    
                    # Temperature dynamics
                    for t in range(time_steps):
                        temp_next = temp[t] + delta_t * (
                            p_devices[device_id][t] * cop / thermal_capacitance - 
                            (temp[t] - temp_ambient) / (thermal_resistance * thermal_capacitance)
                        )
                        device_constraints.append(temp[t+1] == temp_next)
                        
                    # Temperature bounds
                    for t in range(time_steps + 1):
                        device_constraints.append(temp[t] >= temp_min)
                        device_constraints.append(temp[t] <= temp_max)
                    
                    # Power bounds
                    for t in range(time_steps):
                        device_constraints.append(p_devices[device_id][t] >= 0)
                        device_constraints.append(p_devices[device_id][t] <= max_power)
                
                elif device['type'] == 'ev_charger':
                    # EV charger
                    arrival_time = int(device.get('arrival_time', 0))
                    departure_time = int(device.get('departure_time', time_steps - 1))
                    energy_needed = float(device.get('energy_needed', 10.0))  # kWh
                    max_power = float(device.get('max_power', 7.2))  # kW
                    
                    # Constraint: only charge when EV is present
                    for t in range(time_steps):
                        if t < arrival_time or t > departure_time:
                            device_constraints.append(p_devices[device_id][t] == 0)
                    
                    # Constraint: power bounds when EV is present
                    for t in range(arrival_time, departure_time + 1):
                        device_constraints.append(p_devices[device_id][t] >= 0)
                        device_constraints.append(p_devices[device_id][t] <= max_power)
                    
                    # Constraint: total energy delivered
                    device_constraints.append(cp.sum(p_devices[device_id]) * delta_t == energy_needed)
                
                else:
                    # Default: fixed load profile
                    power_profile = device.get('power_profile', np.zeros(time_steps))
                    for t in range(time_steps):
                        device_constraints.append(p_devices[device_id][t] == power_profile[t])
            
            # Calculate total controllable load
            p_controllable_load = cp.sum([p_devices[d] for d in p_devices], axis=0) if p_devices else np.zeros(time_steps)
            
            # Power balance constraint: PV + Grid + Battery = Load
            # Positive grid = import, negative grid = export
            # Battery is now split into charging (positive) and discharging (positive)
            power_balance_constraints = []
            for t in range(time_steps):
                power_balance_constraints.append(
                    pv_forecast[t] + p_grid[t] + p_batt_discharge[t] - p_batt_charge[t] == 
                    load_forecast[t] + p_controllable_load[t]
                )
            
            # Battery constraints
            battery_constraints = []
            
            # SOC dynamics - DCP-compliant version
            for t in range(time_steps):
                # Charging adds energy with efficiency, discharging removes energy with efficiency
                soc_next = soc[t] + (battery_efficiency * p_batt_charge[t] - p_batt_discharge[t] / battery_efficiency) * delta_t / battery_capacity
                battery_constraints.append(soc[t+1] == soc_next)
            
            # Initial SOC
            battery_constraints.append(soc[0] == battery_initial_soc)
            
            # SOC bounds
            for t in range(time_steps + 1):
                battery_constraints.append(soc[t] >= battery_min_soc)
                battery_constraints.append(soc[t] <= battery_max_soc)
            
            # Battery power bounds
            for t in range(time_steps):
                battery_constraints.append(p_batt_charge[t] <= battery_max_power)   # Charge limit
                battery_constraints.append(p_batt_discharge[t] <= battery_max_power)  # Discharge limit
            
            # Grid constraints
            grid_constraints = []
            
            # Grid power bounds
            for t in range(time_steps):
                grid_constraints.append(p_grid[t] >= -grid_max_power)  # Export limit
                grid_constraints.append(p_grid[t] <= grid_max_power)   # Import limit
            
            # Define objective function components
            
            # 1. Cost minimization
            grid_import = cp.maximum(p_grid, 0)
            grid_export = cp.minimum(p_grid, 0)
            cost = cp.sum(grid_import * grid_import_price) - cp.sum(grid_export * grid_export_price)
            
            # 2. Self-consumption maximization (minimize grid export)
            self_consumption = cp.sum(cp.abs(grid_export))
            
            # 3. Peak shaving (minimize maximum grid import)
            peak_power = cp.max(grid_import)
            
            # 4. Battery cycle minimization
            # Approximate by minimizing the sum of charging and discharging power
            battery_cycles = cp.sum(p_batt_charge + p_batt_discharge) / (2 * battery_capacity)
            
            # Combined objective with weights
            objective = (
                self.cost_weight * cost +
                self.self_consumption_weight * self_consumption +
                self.peak_shaving_weight * peak_power +
                self.battery_cycle_weight * battery_cycles
            )
            
            # Define and solve the problem
            constraints = (
                power_balance_constraints + 
                battery_constraints + 
                grid_constraints + 
                device_constraints
            )
            
            problem = cp.Problem(cp.Minimize(objective), constraints)
            
            # Try different solvers in order of preference
            solved = False
            solver_error = None
            
            # Try GLPK_MI first (open-source MIP solver)
            try:
                logger.info("Attempting to solve with GLPK_MI solver...")
                problem.solve(solver=cp.GLPK_MI)
                solved = True
                logger.info(f"Problem solved with GLPK_MI. Status: {problem.status}")
            except Exception as e:
                solver_error = f"GLPK_MI failed: {str(e)}"
                logger.warning(solver_error)
            
            # If GLPK_MI failed, try CBC
            if not solved:
                try:
                    logger.info("Attempting to solve with CBC solver...")
                    problem.solve(solver=cp.CBC)
                    solved = True
                    logger.info(f"Problem solved with CBC. Status: {problem.status}")
                except Exception as e:
                    solver_error = f"{solver_error}; CBC failed: {str(e)}"
                    logger.warning(f"CBC failed: {str(e)}")
            
            # If CBC failed, try SCIP
            if not solved:
                try:
                    logger.info("Attempting to solve with SCIP solver...")
                    problem.solve(solver=cp.SCIP)
                    solved = True
                    logger.info(f"Problem solved with SCIP. Status: {problem.status}")
                except Exception as e:
                    solver_error = f"{solver_error}; SCIP failed: {str(e)}"
                    logger.warning(f"SCIP failed: {str(e)}")
            
            # If all MIP solvers failed, try relaxing the binary constraints and using ECOS
            if not solved:
                logger.warning("All MIP solvers failed. Attempting to solve with relaxed binary constraints using ECOS...")
                # This would require restructuring the model to use continuous variables instead of binary
                # For simplicity, we'll just report the failure
                return {
                    "status": "failed", 
                    "message": f"No suitable MIP solver available. Please install GLPK_MI, CBC, or SCIP. Error details: {solver_error}"
                }
            
            if problem.status != cp.OPTIMAL:
                logger.warning(f"Optimization problem status: {problem.status}")
                return {"status": "failed", "message": f"Optimization failed with status: {problem.status}"}
            
            # Extract results
            grid_power_result = p_grid.value
            
            # Reconstruct the original battery power from charge and discharge components
            battery_power_result = p_batt_charge.value - p_batt_discharge.value
            soc_result = soc.value
            
            device_schedules = []
            for device in device_params_processed:
                device_id = device['id']
                device_power = p_devices[device_id].value
                
                # Create schedule entries for non-zero power periods
                schedule = []
                current_start = None
                current_power = None
                
                for t in range(time_steps):
                    time_str = forecast_data.index[t].strftime('%H:%M')
                    power = device_power[t]
                    
                    if power > 0.01:  # Non-zero power (with small threshold for numerical issues)
                        if current_start is None:
                            current_start = time_str
                            current_power = power
                        elif abs(power - current_power) > 0.01:
                            # Power level changed, end previous entry and start new one
                            end_time = time_str
                            schedule.append({
                                "start": current_start,
                                "end": end_time,
                                "power": float(current_power)
                            })
                            current_start = time_str
                            current_power = power
                    elif current_start is not None:
                        # Power became zero, end the current entry
                        end_time = time_str
                        schedule.append({
                            "start": current_start,
                            "end": end_time,
                            "power": float(current_power)
                        })
                        current_start = None
                        current_power = None
                
                # Add the last entry if there's an open one
                if current_start is not None:
                    end_time = forecast_data.index[-1].strftime('%H:%M')
                    schedule.append({
                        "start": current_start,
                        "end": end_time,
                        "power": float(current_power)
                    })
                
                device_schedules.append({
                    "id": device_id,
                    "name": device.get('name', f"Device {device_id}"),
                    "type": device.get('type', 'unknown'),
                    "schedule": schedule
                })
            
            # Calculate metrics
            total_load = np.sum(load_forecast) + np.sum([p_devices[d].value for d in p_devices], axis=0).sum() if p_devices else np.sum(load_forecast)
            total_pv = np.sum(pv_forecast)
            total_grid_import = np.sum(np.maximum(grid_power_result, 0))
            total_grid_export = np.sum(np.abs(np.minimum(grid_power_result, 0)))
            
            # Self-consumption percentage
            self_consumption_pct = 100 * (1 - total_grid_export / total_pv) if total_pv > 0 else 0
            
            # Battery cycles
            battery_cycles_value = np.sum(np.abs(battery_power_result)) / (2 * battery_capacity)
            
            # Peak grid power
            peak_grid_power_value = np.max(np.maximum(grid_power_result, 0))
            
            # Cost calculation
            cost_value = np.sum(np.maximum(grid_power_result, 0) * grid_import_price) - np.sum(np.abs(np.minimum(grid_power_result, 0)) * grid_export_price)
            
            # Prepare schedule data for visualization
            schedule_data = []
            for t in range(time_steps):
                time_str = forecast_data.index[t].strftime('%H:%M')
                
                # Calculate optimized load (original load + controllable load)
                optimized_load = load_forecast[t]
                if p_devices:
                    optimized_load += sum(p_devices[d].value[t] for d in p_devices)
                
                schedule_data.append({
                    "time": time_str,
                    "load": float(load_forecast[t]),
                    "solar": float(pv_forecast[t]),
                    "battery": float(battery_power_result[t]),
                    "grid": float(grid_power_result[t]),
                    "optimizedLoad": float(optimized_load)
                })
            
            # Return results
            return {
                "status": "success",
                "id": f"opt-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "cost": float(cost_value),
                "selfConsumption": float(self_consumption_pct),
                "peakGridPower": float(peak_grid_power_value),
                "batteryCycles": float(battery_cycles_value),
                "timestamp": datetime.now().isoformat(),
                "scheduleData": schedule_data,
                "deviceScheduleData": device_schedules
            }
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
            
    def generate_schedule(self):
        """Generate an optimal schedule based on forecasts and device constraints"""
        # Run optimization
        
        # Create dummy dataframe for the old generate_schedule function
        # Create index for the DataFrame
        index = pd.date_range(
            start=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            periods=self.time_periods,
            freq='H'
        )
        
        forecast_data = pd.DataFrame({
            'pv_power': self.forecast_data.get('solar_generation', [0] * self.time_periods),
            'load_power': self.forecast_data.get('base_load', [0.5] * self.time_periods)
        }, index=index)
        
        optimization_results = self.optimize(
            forecast_data=forecast_data,
            battery_params={
                'capacity': self.battery_capacity,
                'max_power': 5.0,
                'efficiency': 0.95,
                'min_soc': 0.1,
                'max_soc': 0.9,
                'initial_soc': 0.5
            },
            grid_params={
                'max_power': 10.0,
                'export_price': 0.05,
                'import_price': 0.15
            },
            device_params=self.devices
        )
        
        if optimization_results.get("status") == "success":
            # Create schedule with timestamps
            start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            schedule = []
            
            # Process device schedules
            for device_data in optimization_results.get("deviceScheduleData", []):
                device_schedule = []
                for schedule_entry in device_data.get("schedule", []):
                    try:
                        start_time_obj = datetime.strptime(schedule_entry["start"], "%H:%M")
                        end_time_obj = datetime.strptime(schedule_entry["end"], "%H:%M")
                        
                        # Calculate the start and end datetime objects for the current day
                        start_datetime = start_time.replace(hour=start_time_obj.hour, minute=start_time_obj.minute)
                        end_datetime = start_time.replace(hour=end_time_obj.hour, minute=end_time_obj.minute)
                        
                        # Handle cases where the end time is before the start time (crosses midnight)
                        if end_datetime <= start_datetime:
                            end_datetime += timedelta(days=1)
                        
                        device_schedule.append({
                            "start": start_datetime.strftime("%Y-%m-%d %H:%M"),
                            "end": end_datetime.strftime("%Y-%m-%d %H:%M"),
                            "power_kw": float(schedule_entry["power"])
                        })
                    except Exception as e:
                        logger.error(f"Error processing schedule entry: {e}")
                        continue
                
                schedule.append({
                    "device_id": device_data["id"],
                    "name": device_data["name"],
                    "schedule": device_schedule,
                    "type": device_data.get("type", "unknown")
                })
            
            # Add grid and battery data
            grid_data = []
            battery_data = []
            for schedule_entry in optimization_results.get("scheduleData", []):
                try:
                    hour, minute = map(int, schedule_entry["time"].split(':'))
                    time_str = start_time.replace(hour=hour, minute=minute).strftime("%Y-%m-%d %H:%M")
                    
                    grid_data.append({
                        "time": time_str,
                        "import_kw": max(schedule_entry["grid"], 0),
                        "export_kw": abs(min(schedule_entry["grid"], 0))
                    })
                    
                    battery_data.append({
                        "time": time_str,
                        "power_kw": schedule_entry["battery"]
                    })
                except Exception as e:
                    logger.error(f"Error processing schedule data entry: {e}")
                    continue
            
            return {
                "status": "success",
                "total_cost": optimization_results["cost"],
                "self_consumption": optimization_results["selfConsumption"],
                "peak_grid_power": optimization_results["peakGridPower"],
                "battery_cycles": optimization_results["batteryCycles"],
                "device_schedules": schedule,
                "grid_data": grid_data,
                "battery_data": battery_data
            }
        else:
            return {
                "status": "failed",
                "error": optimization_results.get("message", "Unknown optimization error")
            }
    
    def get_optimization_mode_weights(self, mode: str) -> Tuple[float, float, float, float]:
        """
        Get objective weights based on optimization mode
        
        Args:
            mode: Optimization mode (cost, self_consumption, grid_independence, battery_life)
            
        Returns:
            Tuple of weights (cost, self_consumption, peak_shaving, battery_cycle)
        """
        if mode == "cost":
            return (1.0, 0.1, 0.1, 0.1)
        elif mode == "self_consumption":
            return (0.1, 1.0, 0.1, 0.1)
        elif mode == "grid_independence":
            return (0.1, 0.3, 1.0, 0.1)
        elif mode == "battery_life":
            return (0.1, 0.1, 0.1, 1.0)
        else:
            # Default to cost optimization
            return (1.0, 0.1, 0.1, 0.1)