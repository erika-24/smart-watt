import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
import logging
from database import db

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoadRecommendationEngine:
    """
    Engine for generating recommendations for deferrable loads
    """
    
    def __init__(self):
        """Initialize the recommendation engine"""
        pass
    
    def generate_recommendations(self, 
                               optimization_result: Dict[str, Any], 
                               deferrable_loads: List[Dict[str, Any]],
                               price_forecast: Dict[str, List[float]],
                               pv_forecast: List[float]) -> List[Dict[str, Any]]:
        """
        Generate recommendations for deferrable loads based on optimization results
        
        Args:
            optimization_result: Results from the optimization
            deferrable_loads: List of deferrable loads
            price_forecast: Forecasted electricity prices
            pv_forecast: Forecasted PV generation
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Get timestamps from optimization result
        timestamps = [datetime.fromisoformat(ts) for ts in optimization_result.get("timestamps", [])]
        if not timestamps:
            logger.warning("No timestamps found in optimization result")
            return recommendations
        
        # Get time step in minutes
        if len(timestamps) > 1:
            time_step = int((timestamps[1] - timestamps[0]).total_seconds() / 60)
        else:
            time_step = 15  # Default to 15 minutes
        
        # Get grid power and PV forecast
        grid_power = optimization_result.get("grid_power", [])
        
        # If we don't have enough data, return empty recommendations
        if not grid_power or not pv_forecast or len(grid_power) != len(timestamps):
            logger.warning("Insufficient data for generating recommendations")
            return recommendations
        
        # Process each deferrable load
        for load in deferrable_loads:
            # Skip loads that don't have required runtime
            if not load.get("required_runtime"):
                continue
            
            # Calculate optimal time windows for this load
            optimal_windows = self._find_optimal_windows(
                load=load,
                timestamps=timestamps,
                grid_power=grid_power,
                price_forecast=price_forecast.get("import", []),
                pv_forecast=pv_forecast,
                time_step=time_step
            )
            
            # Generate recommendations for each optimal window
            for window in optimal_windows:
                start_time = window["start_time"]
                end_time = window["end_time"]
                savings = window["savings"]
                reason = window["reason"]
                
                # Create recommendation
                recommendation = {
                    "device_id": load["device_id"],
                    "recommendation_type": "load_shifting",
                    "priority": load.get("priority", 5),
                    "title": f"Run {load['name']} between {start_time.strftime('%H:%M')} and {end_time.strftime('%H:%M')}",
                    "description": f"Running your {load['name']} during this time window will {reason}. " +
                                  f"This could save approximately ${savings:.2f}.",
                    "potential_savings": savings,
                    "start_time": start_time,
                    "end_time": end_time,
                    "created_at": datetime.now(),
                    "implemented": False
                }
                
                recommendations.append(recommendation)
        
        # Sort recommendations by potential savings (highest first)
        recommendations.sort(key=lambda x: x.get("potential_savings", 0), reverse=True)
        
        # Save recommendations to database
        for rec in recommendations:
            db.save_recommendation(rec)
        
        return recommendations
    
    def _find_optimal_windows(self,
                             load: Dict[str, Any],
                             timestamps: List[datetime],
                             grid_power: List[float],
                             price_forecast: List[float],
                             pv_forecast: List[float],
                             time_step: int) -> List[Dict[str, Any]]:
        """
        Find optimal time windows for a deferrable load
        
        Args:
            load: Deferrable load configuration
            timestamps: List of timestamps
            grid_power: Forecasted grid power
            price_forecast: Forecasted electricity prices
            pv_forecast: Forecasted PV generation
            time_step: Time step in minutes
            
        Returns:
            List of optimal time windows
        """
        # Get load parameters
        load_power = load.get("power", 1.0)  # kW
        required_runtime = load.get("required_runtime", 60)  # minutes
        earliest_start = load.get("earliest_start", 0)  # hour of day
        latest_end = load.get("latest_end", 23)  # hour of day
        
        # Calculate number of intervals needed for the required runtime
        intervals_needed = max(1, int(required_runtime / time_step))
        
        # Initialize variables for finding optimal windows
        optimal_windows = []
        
        # Calculate baseline cost (running at default time)
        # Default time is assumed to be the evening peak (18:00-21:00)
        default_start_hour = 18
        default_intervals = []
        
        for i, ts in enumerate(timestamps):
            if ts.hour >= default_start_hour and ts.hour < default_start_hour + 3:
                default_intervals.append(i)
                if len(default_intervals) >= intervals_needed:
                    break
        
        # If we don't have enough default intervals, use the first available intervals
        if len(default_intervals) < intervals_needed:
            default_intervals = list(range(min(intervals_needed, len(timestamps))))
        
        # Calculate baseline cost
        baseline_cost = sum(price_forecast[i] * load_power * (time_step / 60) for i in default_intervals)
        
        # Evaluate each possible starting interval
        for start_idx in range(len(timestamps) - intervals_needed + 1):
            start_time = timestamps[start_idx]
            end_time = timestamps[start_idx + intervals_needed - 1]
            
            # Check if this window is within the allowed time range
            if start_time.hour < earliest_start or end_time.hour > latest_end:
                continue
            
            # Calculate cost for this window
            window_cost = sum(price_forecast[start_idx + i] * load_power * (time_step / 60) 
                             for i in range(intervals_needed))
            
            # Calculate savings compared to baseline
            savings = baseline_cost - window_cost
            
            # Calculate average PV generation during this window
            avg_pv = sum(pv_forecast[start_idx + i] for i in range(intervals_needed)) / intervals_needed
            
            # Calculate average grid power during this window
            avg_grid = sum(grid_power[start_idx + i] for i in range(intervals_needed)) / intervals_needed
            
            # Determine reason for recommendation
            if avg_pv > load_power * 0.8:
                reason = "use excess solar production"
            elif avg_grid < 0:
                reason = "take advantage of energy being exported to the grid"
            else:
                reason = "save money by using electricity during lower price periods"
            
            # Add to optimal windows if savings are positive
            if savings > 0:
                optimal_windows.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "savings": savings,
                    "reason": reason,
                    "avg_pv": avg_pv,
                    "avg_grid": avg_grid
                })
        
        # Sort windows by savings (highest first)
        optimal_windows.sort(key=lambda x: x["savings"], reverse=True)
        
        # Return top 3 windows
        return optimal_windows[:3]
    
    def generate_battery_recommendations(self,
                                       optimization_result: Dict[str, Any],
                                       price_forecast: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """
        Generate recommendations for battery usage
        
        Args:
            optimization_result: Results from the optimization
            price_forecast: Forecasted electricity prices
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Get timestamps from optimization result
        timestamps = [datetime.fromisoformat(ts) for ts in optimization_result.get("timestamps", [])]
        if not timestamps:
            logger.warning("No timestamps found in optimization result")
            return recommendations
        
        # Get battery power and SOC
        battery_power = optimization_result.get("battery_power", [])
        battery_soc = optimization_result.get("battery_soc", [])
        
        # If we don't have enough data, return empty recommendations
        if not battery_power or not battery_soc or len(battery_power) != len(timestamps):
            logger.warning("Insufficient data for generating battery recommendations")
            return recommendations
        
        # Find periods of significant battery charging
        charging_periods = []
        current_period = None
        
        for i, (ts, power) in enumerate(zip(timestamps, battery_power)):
            if power > 0.5:  # Significant charging (> 0.5 kW)
                if current_period is None:
                    current_period = {"start_idx": i, "start_time": ts, "power": [power]}
                else:
                    current_period["power"].append(power)
            elif current_period is not None:
                current_period["end_idx"] = i - 1
                current_period["end_time"] = timestamps[i - 1]
                current_period["avg_power"] = sum(current_period["power"]) / len(current_period["power"])
                charging_periods.append(current_period)
                current_period = None
        
        # Add the last period if it exists
        if current_period is not None:
            current_period["end_idx"] = len(timestamps) - 1
            current_period["end_time"] = timestamps[-1]
            current_period["avg_power"] = sum(current_period["power"]) / len(current_period["power"])
            charging_periods.append(current_period)
        
        # Find periods of significant battery discharging
        discharging_periods = []
        current_period = None
        
        for i, (ts, power) in enumerate(zip(timestamps, battery_power)):
            if power < -0.5:  # Significant discharging (< -0.5 kW)
                if current_period is None:
                    current_period = {"start_idx": i, "start_time": ts, "power": [power]}
                else:
                    current_period["power"].append(power)
            elif current_period is not None:
                current_period["end_idx"] = i - 1
                current_period["end_time"] = timestamps[i - 1]
                current_period["avg_power"] = sum(current_period["power"]) / len(current_period["power"])
                discharging_periods.append(current_period)
                current_period = None
        
        # Add the last period if it exists
        if current_period is not None:
            current_period["end_idx"] = len(timestamps) - 1
            current_period["end_time"] = timestamps[-1]
            current_period["avg_power"] = sum(current_period["power"]) / len(current_period["power"])
            discharging_periods.append(current_period)
        
        # Generate recommendations based on charging periods
        for period in charging_periods:
            # Calculate average price during charging
            avg_price = sum(price_forecast["import"][period["start_idx"]:period["end_idx"]+1]) / \
                       (period["end_idx"] - period["start_idx"] + 1)
            
            # Check if this is a good charging period
            if avg_price < 0.15:  # Low price threshold
                recommendation = {
                    "device_id": "battery",
                    "recommendation_type": "battery_charging",
                    "priority": 7,
                    "title": f"Charge battery between {period['start_time'].strftime('%H:%M')} and {period['end_time'].strftime('%H:%M')}",
                    "description": f"Charging your battery during this low-price period (avg. ${avg_price:.2f}/kWh) " +
                                  f"will save money. The system will charge at approximately {abs(period['avg_power']):.1f} kW.",
                    "potential_savings": (0.25 - avg_price) * abs(period['avg_power']) * \
                                        ((period['end_time'] - period['start_time']).total_seconds() / 3600),
                    "start_time": period['start_time'],
                    "end_time": period['end_time'],
                    "created_at": datetime.now(),
                    "implemented": False
                }
                
                recommendations.append(recommendation)
        
        # Generate recommendations based on discharging periods
        for period in discharging_periods:
            # Calculate average price during discharging
            avg_price = sum(price_forecast["import"][period["start_idx"]:period["end_idx"]+1]) / \
                       (period["end_idx"] - period["start_idx"] + 1)
            
            # Check if this is a good discharging period
            if avg_price > 0.20:  # High price threshold
                recommendation = {
                    "device_id": "battery",
                    "recommendation_type": "battery_discharging",
                    "priority": 8,
                    "title": f"Use battery power between {period['start_time'].strftime('%H:%M')} and {period['end_time'].strftime('%H:%M')}",
                    "description": f"Using battery power during this high-price period (avg. ${avg_price:.2f}/kWh) " +
                                  f"will save money. The system will discharge at approximately {abs(period['avg_power']):.1f} kW.",
                    "potential_savings": (avg_price - 0.15) * abs(period['avg_power']) * \
                                        ((period['end_time'] - period['start_time']).total_seconds() / 3600),
                    "start_time": period['start_time'],
                    "end_time": period['end_time'],
                    "created_at": datetime.now(),
                    "implemented": False
                }
                
                recommendations.append(recommendation)
        
        # Sort recommendations by potential savings (highest first)
        recommendations.sort(key=lambda x: x.get("potential_savings", 0), reverse=True)
        
        # Save recommendations to database
        for rec in recommendations:
            db.save_recommendation(rec)
        
        return recommendations

