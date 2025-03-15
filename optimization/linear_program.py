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
        self.cost_weight = cost_weight
        self.self_consumption_weight = self_consumption_weight
        self.peak_shaving_weight = peak_shaving_weight
        self.battery_cycle_weight = battery_cycle_weight
        
    def optimize(self, 
                 forecast_data: pd.DataFrame,
                 battery_params: Dict[str, Any],
                 grid_params: Dict[str, Any],
                 device_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run optimization using linear programming
        
        Args:
            forecast_data: DataFrame with forecasted load and PV production
            battery_params: Battery parameters (capacity, efficiency, etc.)
            grid_params: Grid parameters (max power, pricing, etc.)
            device_params: List of controllable devices with parameters
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Extract time steps and horizon
            time_steps = len(forecast_data)
            delta_t = 1.0  # Time step in hours
            
            # Extract forecasts
            pv_forecast = forecast_data['pv_power'].values
            load_forecast = forecast_data['load_power'].values
            
            # Extract battery parameters
            battery_capacity = battery_params.get('capacity', 10.0)  # kWh
            battery_max_power = battery_params.get('max_power', 5.0)  # kW
            battery_efficiency = battery_params.get('efficiency', 0.95)
            battery_min_soc = battery_params.get('min_soc', 0.1)
            battery_max_soc = battery_params.get('max_soc', 0.9)
            battery_initial_soc = battery_params.get('initial_soc', 0.5)
            
            # Extract grid parameters
            grid_max_power = grid_params.get('max_power', 10.0)  # kW
            grid_export_price = grid_params.get('export_price', 0.05)  # $/kWh
            
            # Time-of-use pricing if available, otherwise flat rate
            if 'import_price_schedule' in grid_params:
                grid_import_price = np.array(grid_params['import_price_schedule'])
            else:
                grid_import_price = np.ones(time_steps) * grid_params.get('import_price', 0.15)  # $/kWh
            
            # Define optimization variables
            p_grid = cp.Variable(time_steps)  # Grid power (positive = import, negative = export)
            p_batt = cp.Variable(time_steps)  # Battery power (positive = charging, negative = discharging)
            soc = cp.Variable(time_steps + 1)  # Battery state of charge
            
            # Device power variables (if controllable devices are provided)
            p_devices = {}
            device_constraints = []
            
            for device in device_params:
                device_id = device['id']
                p_devices[device_id] = cp.Variable(time_steps)
                
                # Device constraints
                if device['type'] == 'shiftable':
                    # Shiftable load (e.g., washing machine, dishwasher)
                    # Must run for a specific duration within a time window
                    start_window = device.get('start_window', 0)
                    end_window = device.get('end_window', time_steps - 1)
                    duration = device.get('duration', 2)
                    power = device.get('power', 1.0)
                    
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
                    temp_min = device.get('temp_min', 55)  # Minimum temperature
                    temp_max = device.get('temp_max', 65)  # Maximum temperature
                    temp_init = device.get('temp_init', 60)  # Initial temperature
                    temp_ambient = device.get('temp_ambient', 20)  # Ambient temperature
                    thermal_resistance = device.get('thermal_resistance', 0.1)  # K/kW
                    thermal_capacitance = device.get('thermal_capacitance', 0.2)  # kWh/K
                    cop = device.get('cop', 3.0)  # Coefficient of performance
                    max_power = device.get('max_power', 2.0)  # Maximum power
                    
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
                    arrival_time = device.get('arrival_time', 0)
                    departure_time = device.get('departure_time', time_steps - 1)
                    energy_needed = device.get('energy_needed', 10.0)  # kWh
                    max_power = device.get('max_power', 7.2)  # kW
                    
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
            # Positive battery = charging, negative battery = discharging
            power_balance_constraints = []
            for t in range(time_steps):
                power_balance_constraints.append(
                    pv_forecast[t] + p_grid[t] - p_batt[t] == load_forecast[t] + p_controllable_load[t]
                )
            
            # Battery constraints
            battery_constraints = []
            
            # SOC dynamics
            for t in range(time_steps):
                # Charging efficiency when p_batt > 0, discharging efficiency when p_batt < 0
                charge_term = cp.maximum(p_batt[t], 0) * battery_efficiency
                discharge_term = cp.minimum(p_batt[t], 0) / battery_efficiency
                
                soc_next = soc[t] + (charge_term + discharge_term) * delta_t / battery_capacity
                battery_constraints.append(soc[t+1] == soc_next)
            
            # Initial SOC
            battery_constraints.append(soc[0] == battery_initial_soc)
            
            # SOC bounds
            for t in range(time_steps + 1):
                battery_constraints.append(soc[t] >= battery_min_soc)
                battery_constraints.append(soc[t] <= battery_max_soc)
            
            # Battery power bounds
            for t in range(time_steps):
                battery_constraints.append(p_batt[t] >= -battery_max_power)  # Discharge limit
                battery_constraints.append(p_batt[t] <= battery_max_power)   # Charge limit
            
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
            # Approximate by minimizing the sum of absolute battery power
            battery_cycles = cp.sum(cp.abs(p_batt)) / (2 * battery_capacity)
            
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
            problem.solve(solver=cp.ECOS)
            
            if problem.status != 'optimal':
                logger.warning(f"Optimization problem status: {problem.status}")
                return {"status": "failed", "message": f"Optimization failed with status: {problem.status}"}
            
            # Extract results
            grid_power_result = p_grid.value
            battery_power_result = p_batt.value
            soc_result = soc.value
            
            device_schedules = []
            for device in device_params:
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

