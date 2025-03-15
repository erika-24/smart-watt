import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import random
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomEnergyOptimizer:
    """
    Custom energy optimization using stochastic gradient descent and other techniques
    """
    
    def __init__(self, 
                 cost_weight: float = 1.0,
                 self_consumption_weight: float = 0.0,
                 peak_shaving_weight: float = 0.0,
                 battery_cycle_weight: float = 0.0,
                 learning_rate: float = 0.01,
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-4,
                 use_sgd: bool = True):
        """
        Initialize the custom energy optimizer
        
        Args:
            cost_weight: Weight for cost minimization objective
            self_consumption_weight: Weight for self-consumption maximization
            peak_shaving_weight: Weight for peak shaving objective
            battery_cycle_weight: Weight for battery cycle minimization
            learning_rate: Learning rate for SGD
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence threshold for optimization
            use_sgd: Whether to use SGD (True) or L-BFGS-B (False)
        """
        self.cost_weight = cost_weight
        self.self_consumption_weight = self_consumption_weight
        self.peak_shaving_weight = peak_shaving_weight
        self.battery_cycle_weight = battery_cycle_weight
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_sgd = use_sgd
        
    def optimize(self, 
                 forecast_data: pd.DataFrame,
                 battery_params: Dict[str, Any],
                 grid_params: Dict[str, Any],
                 device_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run optimization using custom SGD or L-BFGS-B
        
        Args:
            forecast_data: DataFrame with forecasted load and PV production
            battery_params: Battery parameters (capacity, efficiency, etc.)
            grid_params: Grid parameters (max power, pricing, etc.)
            device_params: List of controllable devices with parameters
            
        Returns:
            Dictionary with optimization results
        """
        try:
            start_time = time.time()
            
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
            
            # Process device parameters
            device_powers = np.zeros((len(device_params), time_steps))
            device_constraints = []
            
            for i, device in enumerate(device_params):
                device_type = device.get('type', 'unknown')
                
                if device_type == 'shiftable':
                    # Shiftable load (e.g., washing machine, dishwasher)
                    start_window = device.get('start_window', 0)
                    end_window = device.get('end_window', time_steps - 1)
                    duration = device.get('duration', 2)
                    power = device.get('power', 1.0)
                    
                    device_constraints.append({
                        'type': 'shiftable',
                        'index': i,
                        'start_window': start_window,
                        'end_window': end_window,
                        'duration': duration,
                        'power': power
                    })
                
                elif device_type == 'ev_charger':
                    # EV charger
                    arrival_time = device.get('arrival_time', 0)
                    departure_time = device.get('departure_time', time_steps - 1)
                    energy_needed = device.get('energy_needed', 10.0)  # kWh
                    max_power = device.get('max_power', 7.2)  # kW
                    
                    device_constraints.append({
                        'type': 'ev_charger',
                        'index': i,
                        'arrival_time': arrival_time,
                        'departure_time': departure_time,
                        'energy_needed': energy_needed,
                        'max_power': max_power
                    })
                
                elif device_type == 'thermal':
                    # Thermal load (e.g., HVAC)
                    temp_min = device.get('temp_min', 68)
                    temp_max = device.get('temp_max', 78)
                    temp_init = device.get('temp_init', 72)
                    max_power = device.get('max_power', 1.5)
                    
                    device_constraints.append({
                        'type': 'thermal',
                        'index': i,
                        'temp_min': temp_min,
                        'temp_max': temp_max,
                        'temp_init': temp_init,
                        'max_power': max_power
                    })
            
            # Define the optimization variables
            # For SGD, we'll use a flat array of variables:
            # [p_grid_0, p_grid_1, ..., p_batt_0, p_batt_1, ..., device_0_0, device_0_1, ...]
            n_variables = 2 * time_steps + len(device_params) * time_steps
            
            # Initialize variables
            if self.use_sgd:
                # For SGD, initialize with a feasible solution
                x = self._initialize_variables(time_steps, len(device_params), 
                                             pv_forecast, load_forecast, 
                                             battery_max_power, grid_max_power,
                                             device_constraints)
                
                # Run SGD
                result = self._run_sgd(x, time_steps, len(device_params),
                                     pv_forecast, load_forecast, grid_import_price, grid_export_price,
                                     battery_capacity, battery_efficiency, battery_max_power,
                                     battery_min_soc, battery_max_soc, battery_initial_soc,
                                     grid_max_power, device_constraints, delta_t)
            else:
                # For L-BFGS-B, use scipy's minimize function
                bounds = self._create_bounds(time_steps, len(device_params), 
                                           battery_max_power, grid_max_power,
                                           device_constraints)
                
                x0 = self._initialize_variables(time_steps, len(device_params), 
                                              pv_forecast, load_forecast, 
                                              battery_max_power, grid_max_power,
                                              device_constraints)
                
                result = minimize(
                    self._objective_function,
                    x0,
                    args=(time_steps, len(device_params),
                         pv_forecast, load_forecast, grid_import_price, grid_export_price,
                         battery_capacity, battery_efficiency, battery_max_power,
                         battery_min_soc, battery_max_soc, battery_initial_soc,
                         grid_max_power, device_constraints, delta_t),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.max_iterations, 'disp': True}
                )
                
                if result.success:
                    logger.info(f"Optimization successful: {result.message}")
                    x = result.x
                else:
                    logger.warning(f"Optimization failed: {result.message}")
                    return {"status": "failed", "message": f"Optimization failed: {result.message}"}
            
            # Extract results
            p_grid = x[:time_steps]
            p_batt = x[time_steps:2*time_steps]
            device_powers = x[2*time_steps:].reshape(len(device_params), time_steps)
            
            # Calculate SOC trajectory
            soc = np.zeros(time_steps + 1)
            soc[0] = battery_initial_soc
            
            for t in range(time_steps):
                # Charging efficiency when p_batt > 0, discharging efficiency when p_batt < 0
                charge_term = max(p_batt[t], 0) * battery_efficiency
                discharge_term = min(p_batt[t], 0) / battery_efficiency
                
                soc[t+1] = soc[t] + (charge_term + discharge_term) * delta_t / battery_capacity
                # Ensure SOC stays within bounds
                soc[t+1] = max(battery_min_soc, min(battery_max_soc, soc[t+1]))
            
            # Create device schedules
            device_schedules = []
            for i, device in enumerate(device_params):
                device_id = device['id']
                device_power = device_powers[i]
                
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
            total_load = np.sum(load_forecast) + np.sum(device_powers)
            total_pv = np.sum(pv_forecast)
            total_grid_import = np.sum(np.maximum(p_grid, 0))
            total_grid_export = np.sum(np.abs(np.minimum(p_grid, 0)))
            
            # Self-consumption percentage
            self_consumption_pct = 100 * (1 - total_grid_export / total_pv) if total_pv > 0 else 0
            
            # Battery cycles
            battery_cycles_value = np.sum(np.abs(p_batt)) / (2 * battery_capacity)
            
            # Peak grid power
            peak_grid_power_value = np.max(np.maximum(p_grid, 0))
            
            # Cost calculation
            cost_value = np.sum(np.maximum(p_grid, 0) * grid_import_price) - np.sum(np.abs(np.minimum(p_grid, 0)) * grid_export_price)
            
            # Prepare schedule data for visualization
            schedule_data = []
            for t in range(time_steps):
                time_str = forecast_data.index[t].strftime('%H:%M')
                
                # Calculate optimized load (original load + controllable load)
                optimized_load = load_forecast[t]
                if len(device_params) > 0:
                    optimized_load += sum(device_powers[d][t] for d in range(len(device_params)))
                
                schedule_data.append({
                    "time": time_str,
                    "load": float(load_forecast[t]),
                    "solar": float(pv_forecast[t]),
                    "battery": float(p_batt[t]),
                    "grid": float(p_grid[t]),
                    "optimizedLoad": float(optimized_load)
                })
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Return results
            return {
                "status": "success",
                "id": f"opt-custom-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "cost": float(cost_value),
                "selfConsumption": float(self_consumption_pct),
                "peakGridPower": float(peak_grid_power_value),
                "batteryCycles": float(battery_cycles_value),
                "timestamp": datetime.now().isoformat(),
                "scheduleData": schedule_data,
                "deviceScheduleData": device_schedules,
                "executionTime": execution_time,
                "method": "SGD" if self.use_sgd else "L-BFGS-B"
            }
            
        except Exception as e:
            logger.error(f"Error in custom optimization: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _initialize_variables(self, time_steps, n_devices, 
                             pv_forecast, load_forecast, 
                             battery_max_power, grid_max_power,
                             device_constraints):
        """Initialize optimization variables with a feasible solution"""
        # Initialize arrays
        p_grid = np.zeros(time_steps)
        p_batt = np.zeros(time_steps)
        device_powers = np.zeros((n_devices, time_steps))
        
        # Initialize device powers based on constraints
        for constraint in device_constraints:
            device_idx = constraint['index']
            
            if constraint['type'] == 'shiftable':
                # For shiftable loads, schedule at the earliest possible time
                start_time = constraint['start_window']
                duration = constraint['duration']
                power = constraint['power']  # This will be in kW (e.g., 0.035 for 35W)
                
                # For very short durations (like servo motors), ensure we have at least one time step
                duration_steps = max(1, int(duration * 60 / 60))  # Convert hours to time steps
                
                for t in range(start_time, start_time + duration_steps):
                    if t < time_steps:
                        device_powers[device_idx, t] = power
            
            elif constraint['type'] == 'ev_charger':
                # For EV chargers, distribute charging evenly
                arrival = constraint['arrival_time']
                departure = constraint['departure_time']
                energy = constraint['energy_needed']
                max_power = constraint['max_power']
                
                # Calculate available hours
                if arrival <= departure:
                    hours = departure - arrival + 1
                else:
                    hours = (time_steps - arrival) + departure + 1
                
                # Calculate required power (not exceeding max_power)
                power = min(energy / hours, max_power)
                
                # Assign power
                if arrival <= departure:
                    for t in range(arrival, departure + 1):
                        device_powers[device_idx, t] = power
                else:
                    for t in range(arrival, time_steps):
                        device_powers[device_idx, t] = power
                    for t in range(0, departure + 1):
                        device_powers[device_idx, t] = power
            
            elif constraint['type'] == 'thermal':
                # For thermal loads, use a simple on-off pattern
                max_power = constraint['max_power']
                
                # Assign power during peak hours (simplified)
                for t in range(time_steps):
                    hour = t % 24
                    if 7 <= hour < 9 or 17 <= hour < 21:  # Morning and evening peaks
                        device_powers[device_idx, t] = max_power * 0.8
                    else:
                        device_powers[device_idx, t] = max_power * 0.2
        
        # Calculate net load (load + devices - PV)
        net_load = load_forecast.copy()
        for i in range(n_devices):
            net_load += device_powers[i]
        net_load -= pv_forecast
        
        # Allocate power between battery and grid
        for t in range(time_steps):
            if net_load[t] > 0:  # Need to import
                # Use battery first (if we're not in the first few hours)
                if t > 6:
                    p_batt[t] = -min(net_load[t], battery_max_power)
                    p_grid[t] = net_load[t] + p_batt[t]  # Remaining from grid
                else:
                    p_grid[t] = net_load[t]  # All from grid
            else:  # Excess power
                # Charge battery first
                p_batt[t] = min(-net_load[t], battery_max_power)
                p_grid[t] = net_load[t] + p_batt[t]  # Remaining to grid
        
        # Combine all variables into a single array
        x = np.concatenate([p_grid, p_batt, device_powers.flatten()])
        
        return x
    
    def _create_bounds(self, time_steps, n_devices, 
                      battery_max_power, grid_max_power,
                      device_constraints):
        """Create bounds for optimization variables"""
        # Grid power bounds
        grid_bounds = [(-grid_max_power, grid_max_power) for _ in range(time_steps)]
        
        # Battery power bounds
        battery_bounds = [(-battery_max_power, battery_max_power) for _ in range(time_steps)]
        
        # Device power bounds
        device_bounds = []
        for i in range(n_devices):
            # Find the constraint for this device
            constraint = next((c for c in device_constraints if c['index'] == i), None)
            
            if constraint:
                if constraint['type'] == 'shiftable':
                    # Shiftable loads are either on at rated power or off
                    power = constraint['power']
                    for t in range(time_steps):
                        device_bounds.append((0, power))
                
                elif constraint['type'] == 'ev_charger':
                    # EV chargers can modulate between 0 and max power
                    max_power = constraint['max_power']
                    arrival = constraint['arrival_time']
                    departure = constraint['departure_time']
                    
                    for t in range(time_steps):
                        if arrival <= departure:
                            if arrival <= t <= departure:
                                device_bounds.append((0, max_power))
                            else:
                                device_bounds.append((0, 0))  # EV not present
                        else:
                            if t >= arrival or t <= departure:
                                device_bounds.append((0, max_power))
                            else:
                                device_bounds.append((0, 0))  # EV not present
                
                elif constraint['type'] == 'thermal':
                    # Thermal loads can modulate between 0 and max power
                    max_power = constraint['max_power']
                    for t in range(time_steps):
                        device_bounds.append((0, max_power))
            else:
                # Default bounds
                for t in range(time_steps):
                    device_bounds.append((0, 0))
        
        # Combine all bounds
        return grid_bounds + battery_bounds + device_bounds
    
    def _objective_function(self, x, time_steps, n_devices,
                           pv_forecast, load_forecast, grid_import_price, grid_export_price,
                           battery_capacity, battery_efficiency, battery_max_power,
                           battery_min_soc, battery_max_soc, battery_initial_soc,
                           grid_max_power, device_constraints, delta_t):
        """
        Objective function for optimization
        
        Returns the weighted sum of cost, self-consumption, peak power, and battery cycles
        """
        # Extract variables
        p_grid = x[:time_steps]
        p_batt = x[time_steps:2*time_steps]
        device_powers = x[2*time_steps:].reshape(n_devices, time_steps)
        
        # Calculate total controllable load
        p_controllable_load = np.sum(device_powers, axis=0)
        
        # Calculate power balance violation penalty
        power_balance_penalty = 0
        for t in range(time_steps):
            imbalance = pv_forecast[t] + p_grid[t] - p_batt[t] - load_forecast[t] - p_controllable_load[t]
            power_balance_penalty += imbalance**2
        
        # Calculate battery SOC trajectory and constraints violation penalty
        soc = np.zeros(time_steps + 1)
        soc[0] = battery_initial_soc
        soc_penalty = 0
        
        for t in range(time_steps):
            # Charging efficiency when p_batt > 0, discharging efficiency when p_batt < 0
            charge_term = max(p_batt[t], 0) * battery_efficiency
            discharge_term = min(p_batt[t], 0) / battery_efficiency
            
            soc[t+1] = soc[t] + (charge_term + discharge_term) * delta_t / battery_capacity
            
            # Penalize SOC violations
            if soc[t+1] < battery_min_soc:
                soc_penalty += (battery_min_soc - soc[t+1])**2
            elif soc[t+1] > battery_max_soc:
                soc_penalty += (soc[t+1] - battery_max_soc)**2
        
        # Calculate device constraints violation penalty
        device_penalty = 0
        
        for constraint in device_constraints:
            device_idx = constraint['index']
            device_power = device_powers[device_idx]
            
            if constraint['type'] == 'shiftable':
                # Shiftable load constraints
                start_window = constraint['start_window']
                end_window = constraint['end_window']
                duration = constraint['duration']
                power = constraint['power']
                
                # Penalty for running outside window
                for t in range(time_steps):
                    if t < start_window or t > end_window:
                        device_penalty += device_power[t]**2
                
                # Penalty for not running for required duration
                total_on_time = np.sum(device_power > 0.01)
                if total_on_time < duration:
                    device_penalty += (duration - total_on_time)**2
                
                # Penalty for not running at rated power when on
                for t in range(time_steps):
                    if device_power[t] > 0.01 and abs(device_power[t] - power) > 0.01:
                        device_penalty += (device_power[t] - power)**2
            
            elif constraint['type'] == 'ev_charger':
                # EV charger constraints
                arrival_time = constraint['arrival_time']
                departure_time = constraint['departure_time']
                energy_needed = constraint['energy_needed']
                max_power = constraint['max_power']
                
                # Penalty for charging outside availability window
                for t in range(time_steps):
                    if arrival_time <= departure_time:
                        if t < arrival_time or t > departure_time:
                            device_penalty += device_power[t]**2
                    else:
                        if t < arrival_time and t > departure_time:
                            device_penalty += device_power[t]**2
                
                # Penalty for not delivering required energy
                total_energy = np.sum(device_power) * delta_t
                if total_energy < energy_needed:
                    device_penalty += (energy_needed - total_energy)**2
                
                # Penalty for exceeding max power
                for t in range(time_steps):
                    if device_power[t] > max_power:
                        device_penalty += (device_power[t] - max_power)**2
            
            elif constraint['type'] == 'thermal':
                # Thermal load constraints
                max_power = constraint['max_power']
                
                # Penalty for exceeding max power
                for t in range(time_steps):
                    if device_power[t] > max_power:
                        device_penalty += (device_power[t] - max_power)**2
        
        # Calculate objective components
        grid_import = np.maximum(p_grid, 0)
        grid_export = np.minimum(p_grid, 0)
        
        # 1. Cost
        cost = np.sum(grid_import * grid_import_price) - np.sum(np.abs(grid_export) * grid_export_price)
        
        # 2. Self-consumption (minimize grid export)
        self_consumption = np.sum(np.abs(grid_export))
        
        # 3. Peak shaving (minimize maximum grid import)
        peak_power = np.max(grid_import)
        
        # 4. Battery cycle minimization
        battery_cycles = np.sum(np.abs(p_batt)) / (2 * battery_capacity)
        
        # Combined objective with weights
        objective = (
            self.cost_weight * cost +
            self.self_consumption_weight * self_consumption +
            self.peak_shaving_weight * peak_power +
            self.battery_cycle_weight * battery_cycles
        )
        
        # Add constraint violation penalties
        constraint_penalty = 1000 * (power_balance_penalty + soc_penalty + device_penalty)
        
        return objective + constraint_penalty
    
    def _run_sgd(self, x_init, time_steps, n_devices,
                pv_forecast, load_forecast, grid_import_price, grid_export_price,
                battery_capacity, battery_efficiency, battery_max_power,
                battery_min_soc, battery_max_soc, battery_initial_soc,
                grid_max_power, device_constraints, delta_t):
        """
        Run stochastic gradient descent optimization
        
        Returns the optimized variables
        """
        # Create bounds
        bounds = self._create_bounds(time_steps, n_devices, 
                                   battery_max_power, grid_max_power,
                                   device_constraints)
        
        # Initialize variables
        x = x_init.copy()
        best_x = x.copy()
        best_obj = float('inf')
        
        # Initialize learning rate and momentum
        lr = self.learning_rate
        momentum = 0.9
        velocity = np.zeros_like(x)
        
        # Initialize adaptive learning rates (for Adam)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        # SGD with momentum and adaptive learning rate
        for iteration in range(self.max_iterations):
            # Calculate objective and gradient
            obj = self._objective_function(x, time_steps, n_devices,
                                         pv_forecast, load_forecast, grid_import_price, grid_export_price,
                                         battery_capacity, battery_efficiency, battery_max_power,
                                         battery_min_soc, battery_max_soc, battery_initial_soc,
                                         grid_max_power, device_constraints, delta_t)
            
            # Numerical gradient calculation
            grad = np.zeros_like(x)
            h = 1e-6  # Small step for finite difference
            
            # Use mini-batch for efficiency
            batch_size = min(100, len(x))
            indices = np.random.choice(len(x), batch_size, replace=False)
            
            for i in indices:
                x_plus = x.copy()
                x_plus[i] += h
                obj_plus = self._objective_function(x_plus, time_steps, n_devices,
                                                 pv_forecast, load_forecast, grid_import_price, grid_export_price,
                                                 battery_capacity, battery_efficiency, battery_max_power,
                                                 battery_min_soc, battery_max_soc, battery_initial_soc,
                                                 grid_max_power, device_constraints, delta_t)
                grad[i] = (obj_plus - obj) / h
            
            # Adam optimizer update
            t = iteration + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # Update variables
            x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Project back to bounds
            for i in range(len(x)):
                x[i] = max(bounds[i][0], min(bounds[i][1], x[i]))
            
            # Check if this is the best solution so far
            if obj < best_obj:
                best_obj = obj
                best_x = x.copy()
            
            # Check for convergence
            if iteration > 0 and abs(obj - prev_obj) < self.convergence_threshold:
                logger.info(f"SGD converged after {iteration} iterations")
                break
            
            prev_obj = obj
            
            # Adaptive learning rate
            if iteration > 0 and iteration % 50 == 0:
                lr *= 0.95  # Reduce learning rate over time
            
            # Log progress
            if iteration % 100 == 0:
                logger.info(f"Iteration {iteration}, Objective: {obj}")
        
        return best_x
    
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


class GeneticEnergyOptimizer:
    """
    Energy optimization using genetic algorithms
    """
    
    def __init__(self, 
                 cost_weight: float = 1.0,
                 self_consumption_weight: float = 0.0,
                 peak_shaving_weight: float = 0.0,
                 battery_cycle_weight: float = 0.0,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        """
        Initialize the genetic energy optimizer
        
        Args:
            cost_weight: Weight for cost minimization objective
            self_consumption_weight: Weight for self-consumption maximization
            peak_shaving_weight: Weight for peak shaving objective
            battery_cycle_weight: Weight for battery cycle minimization
            population_size: Size of the population
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.cost_weight = cost_weight
        self.self_consumption_weight = self_consumption_weight
        self.peak_shaving_weight = self_shaving_weight
        self.battery_cycle_weight = battery_cycle_weight
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def optimize(self, 
                 forecast_data: pd.DataFrame,
                 battery_params: Dict[str, Any],
                 grid_params: Dict[str, Any],
                 device_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run optimization using genetic algorithms
        
        Args:
            forecast_data: DataFrame with forecasted load and PV production
            battery_params: Battery parameters (capacity, efficiency, etc.)
            grid_params: Grid parameters (max power, pricing, etc.)
            device_params: List of controllable devices with parameters
            
        Returns:
            Dictionary with optimization results
        """
        try:
            start_time = time.time()
            
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
            
            # Create bounds for variables
            bounds = self._create_bounds(time_steps, len(device_params), 
                                       battery_max_power, grid_max_power,
                                       device_params)
            
            # Initialize population
            population = self._initialize_population(self.population_size, bounds)
            
            # Evaluate initial population
            fitness_scores = [self._evaluate_fitness(individual, time_steps, len(device_params),
                                                  pv_forecast, load_forecast, grid_import_price, grid_export_price,
                                                  battery_capacity, battery_efficiency, battery_max_power,
                                                  battery_min_soc, battery_max_soc, battery_initial_soc,
                                                  grid_max_power, device_params, delta_t) 
                             for individual in population]
            
            # Main genetic algorithm loop
            best_individual = None
            best_fitness = float('inf')
            
            for generation in range(self.generations):
                # Select parents
                parents = self._select_parents(population, fitness_scores)
                
                # Create next generation
                next_generation = []
                
                while len(next_generation) < self.population_size:
                    # Select two parents
                    parent1 = random.choice(parents)
                    parent2 = random.choice(parents)
                    
                    # Crossover
                    if random.random() < self.crossover_rate:
                        child1, child2 = self._crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    # Mutation
                    child1 = self._mutate(child1, bounds, self.mutation_rate)
                    child2 = self._mutate(child2, bounds, self.mutation_rate)
                    
                    # Add to next generation
                    next_generation.append(child1)
                    if len(next_generation) < self.population_size:
                        next_generation.append(child2)
                
                # Update population
                population = next_generation
                
                # Evaluate new population
                fitness_scores = [self._evaluate_fitness(individual, time_steps, len(device_params),
                                                      pv_forecast, load_forecast, grid_import_price, grid_export_price,
                                                      battery_capacity, battery_efficiency, battery_max_power,
                                                      battery_min_soc, battery_max_soc, battery_initial_soc,
                                                      grid_max_power, device_params, delta_t) 
                                 for individual in population]
                
                # Track best individual
                min_fitness = min(fitness_scores)
                if min_fitness < best_fitness:
                    best_fitness = min_fitness
                    best_individual = population[fitness_scores.index(min_fitness)].copy()
                
                # Log progress
                if generation % 10 == 0:
                    logger.info(f"Generation {generation}, Best Fitness: {best_fitness}")
            
            # Extract results from best individual
            x = best_individual
            p_grid = x[:time_steps]
            p_batt = x[time_steps:2*time_steps]
            device_powers = x[2*time_steps:].reshape(len(device_params), time_steps)
            
            # Calculate SOC trajectory
            soc = np.zeros(time_steps + 1)
            soc[0] = battery_initial_soc
            
            for t in range(time_steps):
                # Charging efficiency when p_batt > 0, discharging efficiency when p_batt < 0
                charge_term = max(p_batt[t], 0) * battery_efficiency
                discharge_term = min(p_batt[t], 0) / battery_efficiency
                
                soc[t+1] = soc[t] + (charge_term + discharge_term) * delta_t / battery_capacity
                # Ensure SOC stays within bounds
                soc[t+1] = max(battery_min_soc, min(battery_max_soc, soc[t+1]))
            
            # Create device schedules
            device_schedules = []
            for i, device in enumerate(device_params):
                device_id = device['id']
                device_power = device_powers[i]
                
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
            total_load = np.sum(load_forecast) + np.sum(device_powers)
            total_pv = np.sum(pv_forecast)
            total_grid_import = np.sum(np.maximum(p_grid, 0))
            total_grid_export = np.sum(np.abs(np.minimum(p_grid, 0)))
            
            # Self-consumption percentage
            self_consumption_pct = 100 * (1 - total_grid_export / total_pv) if total_pv > 0 else 0
            
            # Battery cycles
            battery_cycles_value = np.sum(np.abs(p_batt)) / (2 * battery_capacity)
            
            # Peak grid power
            peak_grid_power_value = np.max(np.maximum(p_grid, 0))
            
            # Cost calculation
            cost_value = np.sum(np.maximum(p_grid, 0) * grid_import_price) - np.sum(np.abs(np.minimum(p_grid, 0)) * grid_export_price)
            
            # Prepare schedule data for visualization
            schedule_data = []
            for t in range(time_steps):
                time_str = forecast_data.index[t].strftime('%H:%M')
                
                # Calculate optimized load (original load + controllable load)
                optimized_load = load_forecast[t]
                if len(device_params) > 0:
                    optimized_load += sum(device_powers[d][t] for d in range(len(device_params)))
                
                schedule_data.append({
                    "time": time_str,
                    "load": float(load_forecast[t]),
                    "solar": float(pv_forecast[t]),
                    "battery": float(p_batt[t]),
                    "grid": float(p_grid[t]),
                    "optimizedLoad": float(optimized_load)
                })
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Return results
            return {
                "status": "success",
                "id": f"opt-genetic-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "cost": float(cost_value),
                "selfConsumption": float(self_consumption_pct),
                "peakGridPower": float(peak_grid_power_value),
                "batteryCycles": float(battery_cycles_value),
                "timestamp": datetime.now().isoformat(),
                "scheduleData": schedule_data,
                "deviceScheduleData": device_schedules,
                "executionTime": execution_time,
                "method": "Genetic Algorithm"
            }
            
        except Exception as e:
            logger.error(f"Error in genetic optimization: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _create_bounds(self, time_steps, n_devices, 
                      battery_max_power, grid_max_power,
                      device_params):
        """Create bounds for optimization variables"""
        # Grid power bounds
        grid_bounds = [(-grid_max_power, grid_max_power) for _ in range(time_steps)]
        
        # Battery power bounds
        battery_bounds = [(-battery_max_power, battery_max_power) for _ in range(time_steps)]
        
        # Device power bounds
        device_bounds = []
        for i in range(n_devices):
            # Find the device
            device = next((d for d in device_params if d.get('index', i) == i), None)
            
            if device:
                if device.get('type') == 'shiftable':
                    # Shiftable loads are either on at rated power or off
                    power = device.get('power', 1.0)
                    for t in range(time_steps):
                        device_bounds.append((0, power))
                
                elif device.get('type') == 'ev_charger':
                    # EV chargers can modulate between 0 and max power
                    max_power = device.get('max_power', 7.2)
                    arrival = device.get('arrival_time', 0)
                    departure = device.get('departure_time', time_steps - 1)
                    
                    for t in range(time_steps):
                        if arrival <= departure:
                            if arrival <= t <= departure:
                                device_bounds.append((0, max_power))
                            else:
                                device_bounds.append((0, 0))  # EV not present
                        else:
                            if t >= arrival or t <= departure:
                                device_bounds.append((0, max_power))
                            else:
                                device_bounds.append((0, 0))  # EV not present
                
                elif device.get('type') == 'thermal':
                    # Thermal loads can modulate between 0 and max power
                    max_power = device.get('max_power', 1.5)
                    for t in range(time_steps):
                        device_bounds.append((0, max_power))
                else:
                    # Default bounds
                    for t in range(time_steps):
                        device_bounds.append((0, 0))
            else:
                # Default bounds
                for t in range(time_steps):
                    device_bounds.append((0, 0))
        
        # Combine all bounds
        return grid_bounds + battery_bounds + device_bounds
    
    def _initialize_population(self, population_size, bounds):
        """Initialize population with random individuals within bounds"""
        population = []
        
        for _ in range(population_size):
            individual = np.zeros(len(bounds))
            for i in range(len(bounds)):
                individual[i] = random.uniform(bounds[i][0], bounds[i][1])
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(self, individual, time_steps, n_devices,
                         pv_forecast, load_forecast, grid_import_price, grid_export_price,
                         battery_capacity, battery_efficiency, battery_max_power,
                         battery_min_soc, battery_max_soc, battery_initial_soc,
                         grid_max_power, device_params, delta_t):
        """Evaluate fitness of an individual"""
        # Extract variables
        p_grid = individual[:time_steps]
        p_batt = individual[time_steps:2*time_steps]
        device_powers = individual[2*time_steps:].reshape(n_devices, time_steps)
        
        # Calculate total controllable load
        p_controllable_load = np.sum(device_powers, axis=0)
        
        # Calculate power balance violation penalty
        power_balance_penalty = 0
        for t in range(time_steps):
            imbalance = pv_forecast[t] + p_grid[t] - p_batt[t] - load_forecast[t] - p_controllable_load[t]
            power_balance_penalty += imbalance**2
        
        # Calculate battery SOC trajectory and constraints violation penalty
        soc = np.zeros(time_steps + 1)
        soc[0] = battery_initial_soc
        soc_penalty = 0
        
        for t in range(time_steps):
            # Charging efficiency when p_batt > 0, discharging efficiency when p_batt < 0
            charge_term = max(p_batt[t], 0) * battery_efficiency
            discharge_term = min(p_batt[t], 0) / battery_efficiency
            
            soc[t+1] = soc[t] + (charge_term + discharge_term) * delta_t / battery_capacity
            
            # Penalize SOC violations
            if soc[t+1] < battery_min_soc:
                soc_penalty += (battery_min_soc - soc[t+1])**2
            elif soc[t+1] > battery_max_soc:
                soc_penalty += (soc[t+1] - battery_max_soc)**2
        
        # Calculate device constraints violation penalty
        device_penalty = 0
        
        for i, device in enumerate(device_params):
            device_power = device_powers[i]
            
            if device.get('type') == 'shiftable':
                # Shiftable load constraints
                start_window = device.get('start_window', 0)
                end_window = device.get('end_window', time_steps - 1)
                duration = device.get('duration', 2)
                power = device.get('power', 1.0)
                
                # Penalty for running outside window
                for t in range(time_steps):
                    if t < start_window or t > end_window:
                        device_penalty += device_power[t]**2
                
                # Penalty for not running for required duration
                total_on_time = np.sum(device_power > 0.01)
                if total_on_time < duration:
                    device_penalty += (duration - total_on_time)**2
            
            elif device.get('type') == 'ev_charger':
                # EV charger constraints
                arrival_time = device.get('arrival_time', 0)
                departure_time = device.get('departure_time', time_steps - 1)
                energy_needed = device.get('energy_needed', 10.0)
                
                # Penalty for charging outside availability window
                for t in range(time_steps):
                    if arrival_time <= departure_time:
                        if t < arrival_time or t > departure_time:
                            device_penalty += device_power[t]**2
                    else:
                        if t < arrival_time and t > departure_time:
                            device_penalty += device_power[t]**2
                
                # Penalty for not delivering required energy
                total_energy = np.sum(device_power) * delta_t
                if total_energy < energy_needed:
                    device_penalty += (energy_needed - total_energy)**2
        
        # Calculate objective components
        grid_import = np.maximum(p_grid, 0)
        grid_export = np.minimum(p_grid, 0)
        
        # 1. Cost
        cost = np.sum(grid_import * grid_import_price) - np.sum(np.abs(grid_export) * grid_export_price)
        
        # 2. Self-consumption (minimize grid export)
        self_consumption = np.sum(np.abs(grid_export))
        
        # 3. Peak shaving (minimize maximum grid import)
        peak_power = np.max(grid_import)
        
        # 4. Battery cycle minimization
        battery_cycles = np.sum(np.abs(p_batt)) / (2 * battery_capacity)
        
        # Combined objective with weights
        objective = (
            self.cost_weight * cost +
            self.self_consumption_weight * self_consumption +
            self.peak_shaving_weight * peak_power +
            self.battery_cycle_weight * battery_cycles
        )
        
        # Add constraint violation penalties
        constraint_penalty = 1000 * (power_balance_penalty + soc_penalty + device_penalty)
        
        return objective + constraint_penalty
    
    def _select_parents(self, population, fitness_scores):
        """Select parents using tournament selection"""
        parents = []
        
        # Tournament selection
        tournament_size = 3
        
        for _ in range(self.population_size):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select the best individual from tournament
            winner_index = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            parents.append(population[winner_index])
        
        return parents
    
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2
    
    def _mutate(self, individual, bounds, mutation_rate):
        """Mutate an individual"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Gaussian mutation
                sigma = (bounds[i][1] - bounds[i][0]) * 0.1  # 10% of range
                delta = random.gauss(0, sigma)
                mutated[i] += delta
                
                # Ensure within bounds
                mutated[i] = max(bounds[i][0], min(bounds[i][1], mutated[i]))
        
        return mutated

