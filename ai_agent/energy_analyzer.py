import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
from ai_agent.prompts import ENERGY_ANALYSIS_PROMPT, OPTIMIZATION_PROMPT
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyAnalysisAgent:
    """
    AI agent for analyzing energy data and providing recommendations
    """
    
    def __init__(self, ai_client=None):
        """
        Initialize the energy analysis agent
        
        Args:
            ai_client: Client for AI model API (optional)
        """
        self.ai_client = ai_client
    
    async def analyze_energy_data(self, 
                                 energy_data: Dict[str, Any], 
                                 optimization_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze energy data and provide insights
        
        Args:
            energy_data: Dictionary containing energy data (consumption, production, etc.)
            optimization_results: Optional optimization results
            
        Returns:
            Dictionary with analysis results and recommendations
        """
        try:
            # Extract relevant data
            load_data = energy_data.get("load_power", [])
            pv_data = energy_data.get("pv_power", [])
            battery_data = energy_data.get("battery_power", [])
            grid_data = energy_data.get("grid_power", [])
            timestamps = energy_data.get("timestamps", [])
            
            # Perform basic statistical analysis
            analysis = self._perform_statistical_analysis(load_data, pv_data, battery_data, grid_data)
            
            # Identify patterns and anomalies
            patterns = self._identify_patterns(load_data, pv_data, battery_data, grid_data, timestamps)
            
            # Generate insights using AI
            insights = await self._generate_ai_insights(energy_data, analysis, patterns, optimization_results)
            
            return {
                "analysis": analysis,
                "patterns": patterns,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing energy data: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _perform_statistical_analysis(self, load_data, pv_data, battery_data, grid_data):
        """
        Perform statistical analysis on energy data
        """
        # Convert to numpy arrays for analysis
        load_array = np.array(load_data) if load_data else np.array([])
        pv_array = np.array(pv_data) if pv_data else np.array([])
        battery_array = np.array(battery_data) if battery_data else np.array([])
        grid_array = np.array(grid_data) if grid_data else np.array([])
        
        # Calculate basic statistics
        analysis = {}
        
        if len(load_array) > 0:
            analysis["load"] = {
                "mean": float(np.mean(load_array)),
                "max": float(np.max(load_array)),
                "min": float(np.min(load_array)),
                "std": float(np.std(load_array)),
                "total": float(np.sum(load_array))
            }
        
        if len(pv_array) > 0:
            analysis["pv"] = {
                "mean": float(np.mean(pv_array)),
                "max": float(np.max(pv_array)),
                "min": float(np.min(pv_array)),
                "std": float(np.std(pv_array)),
                "total": float(np.sum(pv_array))
            }
        
        if len(battery_array) > 0:
            # Separate charging and discharging
            charging = battery_array[battery_array > 0]
            discharging = battery_array[battery_array < 0]
            
            analysis["battery"] = {
                "mean": float(np.mean(battery_array)),
                "max_charging": float(np.max(charging)) if len(charging) > 0 else 0,
                "max_discharging": float(np.min(discharging)) if len(discharging) > 0 else 0,
                "charging_time_pct": float(len(charging) / len(battery_array)) if len(battery_array) > 0 else 0,
                "discharging_time_pct": float(len(discharging) / len(battery_array)) if len(battery_array) > 0 else 0
            }
        
        if len(grid_array) > 0:
            # Separate import and export
            importing = grid_array[grid_array > 0]
            exporting = grid_array[grid_array < 0]
            
            analysis["grid"] = {
                "mean": float(np.mean(grid_array)),
                "max_import": float(np.max(importing)) if len(importing) > 0 else 0,
                "max_export": float(np.min(exporting)) if len(exporting) > 0 else 0,
                "import_time_pct": float(len(importing) / len(grid_array)) if len(grid_array) > 0 else 0,
                "export_time_pct": float(len(exporting) / len(grid_array)) if len(grid_array) > 0 else 0,
                "total_import": float(np.sum(importing)) if len(importing) > 0 else 0,
                "total_export": float(np.sum(np.abs(exporting))) if len(exporting) > 0 else 0
            }
        
        # Calculate self-consumption if possible
        if len(pv_array) > 0 and len(grid_array) > 0:
            exporting = np.abs(grid_array[grid_array < 0])
            total_pv = np.sum(pv_array)
            total_export = np.sum(exporting)
            
            if total_pv > 0:
                self_consumption = (total_pv - total_export) / total_pv
                analysis["self_consumption"] = float(self_consumption)
        
        return analysis
    
    def _identify_patterns(self, load_data, pv_data, battery_data, grid_data, timestamps):
        """
        Identify patterns in energy data
        """
        patterns = {}
        
        # Convert timestamps to datetime objects if they're strings
        if timestamps and isinstance(timestamps[0], str):
            timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
        
        # If we don't have timestamps, we can't identify time-based patterns
        if not timestamps:
            return patterns
        
        # Identify load patterns
        if load_data and len(load_data) > 0:
            # Group by hour of day
            hour_groups = {}
            for i, ts in enumerate(timestamps):
                hour = ts.hour
                if hour not in hour_groups:
                    hour_groups[hour] = []
                hour_groups[hour].append(load_data[i])
            
            # Calculate average load by hour
            hourly_load = {hour: np.mean(values) for hour, values in hour_groups.items()}
            
            # Find peak hours
            if hourly_load:
                peak_hours = sorted(hourly_load.items(), key=lambda x: x[1], reverse=True)[:3]
                patterns["peak_load_hours"] = [{"hour": hour, "load": load} for hour, load in peak_hours]
        
        # Identify PV patterns
        if pv_data and len(pv_data) > 0:
            # Group by hour of day
            hour_groups = {}
            for i, ts in enumerate(timestamps):
                hour = ts.hour
                if hour not in hour_groups:
                    hour_groups[hour] = []
                hour_groups[hour].append(pv_data[i])
            
            # Calculate average PV by hour
            hourly_pv = {hour: np.mean(values) for hour, values in hour_groups.items()}
            
            # Find peak production hours
            if hourly_pv:
                peak_hours = sorted(hourly_pv.items(), key=lambda x: x[1], reverse=True)[:3]
                patterns["peak_pv_hours"] = [{"hour": hour, "pv": pv} for hour, pv in peak_hours]
        
        # Identify grid import/export patterns
        if grid_data and len(grid_data) > 0:
            # Group by hour of day
            import_hour_groups = {}
            export_hour_groups = {}
            
            for i, ts in enumerate(timestamps):
                hour = ts.hour
                if grid_data[i] > 0:  # Import
                    if hour not in import_hour_groups:
                        import_hour_groups[hour] = []
                    import_hour_groups[hour].append(grid_data[i])
                elif grid_data[i] < 0:  # Export
                    if hour not in export_hour_groups:
                        export_hour_groups[hour] = []
                    export_hour_groups[hour].append(abs(grid_data[i]))
            
            # Calculate average import/export by hour
            hourly_import = {hour: np.mean(values) for hour, values in import_hour_groups.items()}
            hourly_export = {hour: np.mean(values) for hour, values in export_hour_groups.items()}
            
            # Find peak import/export hours
            if hourly_import:
                peak_import_hours = sorted(hourly_import.items(), key=lambda x: x[1], reverse=True)[:3]
                patterns["peak_import_hours"] = [{"hour": hour, "import": imp} for hour, imp in peak_import_hours]
            
            if hourly_export:
                peak_export_hours = sorted(hourly_export.items(), key=lambda x: x[1], reverse=True)[:3]
                patterns["peak_export_hours"] = [{"hour": hour, "export": exp} for hour, exp in peak_export_hours]
        
        return patterns
    
    async def _generate_ai_insights(self, energy_data, analysis, patterns, optimization_results):
        """
        Generate insights using AI
        """
        if not self.ai_client:
            # Return basic insights without AI
            return self._generate_basic_insights(analysis, patterns, optimization_results)
        
        try:
            # Prepare data for AI analysis
            data_for_analysis = {
                "energy_data": energy_data,
                "analysis": analysis,
                "patterns": patterns,
                "optimization_results": optimization_results
            }
            
            # Convert to JSON string
            data_json = json.dumps(data_for_analysis, default=str)
            
            # Generate analysis using AI
            prompt = ENERGY_ANALYSIS_PROMPT.format(data=data_json)
            
            response = await self.ai_client.generate_text(prompt)
            
            # Parse the response
            try:
                insights = json.loads(response)
            except json.JSONDecodeError:
                # If the response is not valid JSON, use it as a string
                insights = {"general": response}
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            # Fall back to basic insights
            return self._generate_basic_insights(analysis, patterns, optimization_results)
    
    def _generate_basic_insights(self, analysis, patterns, optimization_results):
        """
        Generate basic insights without AI
        """
        insights = {}
        
        # Load insights
        if "load" in analysis:
            load_insights = []
            
            # Check for high load
            if analysis["load"]["max"] > 5:
                load_insights.append(f"Peak load of {analysis['load']['max']:.2f} kW detected. Consider load shifting to reduce peak demand.")
            
            # Check for load variability
            if analysis["load"]["std"] > 1:
                load_insights.append(f"High load variability detected (std: {analysis['load']['std']:.2f} kW). Consider smoothing consumption patterns.")
            
            if patterns.get("peak_load_hours"):
                peak_hours = [f"{h['hour']}:00" for h in patterns["peak_load_hours"]]
                load_insights.append(f"Peak load hours: {', '.join(peak_hours)}. Consider shifting non-essential loads to off-peak hours.")
            
            insights["load"] = load_insights
        
        # PV insights
        if "pv" in analysis:
            pv_insights = []
            
            # Check for PV production
            if analysis["pv"]["total"] > 0:
                pv_insights.append(f"Total PV production: {analysis['pv']['total']:.2f} kWh.")
            
            if patterns.get("peak_pv_hours"):
                peak_hours = [f"{h['hour']}:00" for h in patterns["peak_pv_hours"]]
                pv_insights.append(f"Peak PV production hours: {', '.join(peak_hours)}. Consider scheduling high-consumption activities during these hours.")
            
            insights["pv"] = pv_insights
        
        # Battery insights
        if "battery" in analysis:
            battery_insights = []
            
            # Check battery usage
            if analysis["battery"]["charging_time_pct"] > 0.6:
                battery_insights.append(f"Battery spent {analysis['battery']['charging_time_pct']*100:.1f}% of time charging. Consider optimizing discharge periods.")
            
            if analysis["battery"]["discharging_time_pct"] > 0.6:
                battery_insights.append(f"Battery spent {analysis['battery']['discharging_time_pct']*100:.1f}% of time discharging. Consider optimizing charging periods.")
            
            insights["battery"] = battery_insights
        
        # Grid insights
        if "grid" in analysis:
            grid_insights = []
            
            # Check grid usage
            if analysis["grid"]["import_time_pct"] > 0.7:
                grid_insights.append(f"Grid import used {analysis['grid']['import_time_pct']*100:.1f}% of time. Consider increasing self-consumption.")
            
            if analysis["grid"]["total_import"] > 10:
                grid_insights.append(f"High grid import: {analysis['grid']['total_import']:.2f} kWh. Consider optimizing battery usage or increasing PV capacity.")
            
            if patterns.get("peak_import_hours"):
                peak_hours = [f"{h['hour']}:00" for h in patterns["peak_import_hours"]]
                grid_insights.append(f"Peak grid import hours: {', '.join(peak_hours)}. Consider reducing consumption during these hours.")
            
            insights["grid"] = grid_insights
        
        # Self-consumption insights
        if "self_consumption" in analysis:
            self_consumption_insights = []
            
            sc_pct = analysis["self_consumption"] * 100
            if sc_pct < 50:
                self_consumption_insights.append(f"Low self-consumption: {sc_pct:.1f}%. Consider adding battery storage or shifting loads to match PV production.")
            elif sc_pct > 80:
                self_consumption_insights.append(f"Excellent self-consumption: {sc_pct:.1f}%. Your system is efficiently using PV production.")
            else:
                self_consumption_insights.append(f"Moderate self-consumption: {sc_pct:.1f}%. Consider optimizing load scheduling to match PV production.")
            
            insights["self_consumption"] = self_consumption_insights
        
        # Optimization insights
        if optimization_results:
            opt_insights = []
            
            if "cost" in optimization_results:
                opt_insights.append(f"Optimized cost: ${optimization_results['cost']:.2f}.")
            
            if "self_consumption" in optimization_results:
                opt_insights.append(f"Optimized self-consumption: {optimization_results['self_consumption']:.1f}%.")
            
            if "peak_grid_power" in optimization_results:
                opt_insights.append(f"Optimized peak grid power: {optimization_results['peak_grid_power']:.2f} kW.")
            
            if "battery_cycles" in optimization_results:
                opt_insights.append(f"Optimized battery cycles: {optimization_results['battery_cycles']:.2f} cycles.")
            
            insights["optimization"] = opt_insights
        
        # General insights
        general_insights = []
        
        # Add a general summary
        if "load" in analysis and "pv" in analysis:
            if analysis["load"]["total"] > 0 and analysis["pv"]["total"] > 0:
                pv_load_ratio = analysis["pv"]["total"] / analysis["load"]["total"]
                if pv_load_ratio < 0.3:
                    general_insights.append(f"PV production covers only {pv_load_ratio*100:.1f}% of your energy consumption. Consider increasing PV capacity.")
                elif pv_load_ratio > 1.0:
                    general_insights.append(f"PV production exceeds your energy consumption by {(pv_load_ratio-1)*100:.1f}%. Consider adding battery storage or exporting excess energy.")
                else:
                    general_insights.append(f"PV production covers {pv_load_ratio*100:.1f}% of your energy consumption.")
        
        if "self_consumption" in analysis:
            if analysis["self_consumption"] < 0.5:
                general_insights.append("Increasing self-consumption should be a priority. Consider adding battery storage or smart load control.")
        
        if "grid" in analysis and "import_time_pct" in analysis["grid"] and analysis["grid"]["import_time_pct"] > 0.8:
            general_insights.append("Your system relies heavily on grid imports. Consider strategies to increase energy independence.")
        
        insights["general"] = general_insights
        
        return insights
    
    async def generate_optimization_recommendations(self, energy_data, analysis_results):
        """
        Generate optimization recommendations based on energy data and analysis
        
        Args:
            energy_data: Dictionary containing energy data
            analysis_results: Results from energy analysis
            
        Returns:
            Dictionary with optimization recommendations
        """
        if not self.ai_client:
            # Return basic recommendations without AI
            return self._generate_basic_recommendations(energy_data, analysis_results)
        
        try:
            # Prepare data for AI
            data_for_optimization = {
                "energy_data": energy_data,
                "analysis_results": analysis_results
            }
            
            # Convert to JSON string
            data_json = json.dumps(data_for_optimization, default=str)
            
            # Generate recommendations using AI
            prompt = OPTIMIZATION_PROMPT.format(data=data_json)
            
            response = await self.ai_client.generate_text(prompt)
            
            # Parse the response
            try:
                recommendations = json.loads(response)
            except json.JSONDecodeError:
                # If the response is not valid JSON, use it as a string
                recommendations = {"general": response}
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating AI optimization recommendations: {e}")
            # Fall back to basic recommendations
            return self._generate_basic_recommendations(energy_data, analysis_results)
    
    def _generate_basic_recommendations(self, energy_data, analysis_results):
        """
        Generate basic optimization recommendations without AI
        """
        recommendations = {}
        
        # Extract insights from analysis results
        insights = analysis_results.get("insights", {})
        patterns = analysis_results.get("patterns", {})
        analysis = analysis_results.get("analysis", {})
        
        # Load shifting recommendations
        load_shift_recs = []
        
        if patterns.get("peak_load_hours") and patterns.get("peak_pv_hours"):
            peak_load_hours = [h["hour"] for h in patterns["peak_load_hours"]]
            peak_pv_hours = [h["hour"] for h in patterns["peak_pv_hours"]]
            
            # Check if peak load doesn't align with peak PV
            misaligned_hours = [h for h in peak_load_hours if h not in peak_pv_hours]
            if misaligned_hours:
                load_shift_recs.append({
                    "title": "Shift loads to match PV production",
                    "description": f"Consider shifting loads from hours {', '.join([f'{h}:00' for h in misaligned_hours])} to peak PV production hours {', '.join([f'{h}:00' for h in peak_pv_hours])}.",
                    "potential_savings": "Medium",
                    "implementation_difficulty": "Medium"
                })
        
        if patterns.get("peak_import_hours"):
            peak_import_hours = [h["hour"] for h in patterns["peak_import_hours"]]
            load_shift_recs.append({
                "title": "Reduce consumption during peak grid import hours",
                "description": f"Consider reducing or shifting consumption during hours {', '.join([f'{h}:00' for h in peak_import_hours])} to minimize grid imports.",
                "potential_savings": "High",
                "implementation_difficulty": "Medium"
            })
        
        recommendations["load_shifting"] = load_shift_recs
        
        # Battery optimization recommendations
        battery_recs = []
        
        if "battery" in analysis:
            if analysis["battery"]["charging_time_pct"] < 0.3:
                battery_recs.append({
                    "title": "Increase battery charging",
                    "description": "Your battery is underutilized for charging. Consider increasing charging during periods of excess PV production.",
                    "potential_savings": "Medium",
                    "implementation_difficulty": "Low"
                })
            
            if analysis["battery"]["discharging_time_pct"] < 0.3:
                battery_recs.append({
                    "title": "Increase battery discharging",
                    "description": "Your battery is underutilized for discharging. Consider increasing discharging during peak consumption or high grid import price periods.",
                    "potential_savings": "Medium",
                    "implementation_difficulty": "Low"
                })
        
        if patterns.get("peak_import_hours") and not battery_recs:
            battery_recs.append({
                "title": "Optimize battery discharge schedule",
                "description": "Schedule battery discharge during peak grid import hours to reduce grid dependency and potentially save on electricity costs.",
                "potential_savings": "High",
                "implementation_difficulty": "Medium"
            })
        
        recommendations["battery_optimization"] = battery_recs
        
        # System sizing recommendations
        system_recs = []
        
        if "pv" in analysis and "load" in analysis:
            pv_load_ratio = analysis["pv"]["total"] / analysis["load"]["total"] if analysis["load"]["total"] > 0 else 0
            
            if pv_load_ratio < 0.5:
                system_recs.append({
                    "title": "Increase PV capacity",
                    "description": "Your PV system covers less than 50% of your energy consumption. Consider adding more solar panels to increase self-generation.",
                    "potential_savings": "High",
                    "implementation_difficulty": "High"
                })
        
        if "grid" in analysis and analysis["grid"]["export_time_pct"] > 0.4 and ("battery" not in analysis or analysis["battery"]["charging_time_pct"] < 0.3):
            system_recs.append({
                "title": "Add or increase battery storage",
                "description": "You're exporting a significant amount of energy to the grid. Adding battery storage could increase self-consumption and provide backup power.",
                "potential_savings": "Medium",
                "implementation_difficulty": "High"
            })
        
        recommendations["system_sizing"] = system_recs
        
        # Energy efficiency recommendations
        efficiency_recs = []
        
        if "load" in analysis and analysis["load"]["mean"] > 2.0:
            efficiency_recs.append({
                "title": "Reduce base load",
                "description": "Your average power consumption is relatively high. Consider energy-efficient appliances and identifying always-on devices that could be optimized.",
                "potential_savings": "Medium",
                "implementation_difficulty": "Medium"
            })
        
        recommendations["energy_efficiency"] = efficiency_recs
        
        # General recommendations
        general_recs = []
        
        general_recs.append({
            "title": "Regular system monitoring",
            "description": "Regularly monitor your energy system performance to identify optimization opportunities and ensure everything is working correctly.",
            "potential_savings": "Low",
            "implementation_difficulty": "Low"
        })
        
        if "self_consumption" in analysis and analysis["self_consumption"] < 0.7:
            general_recs.append({
                "title": "Implement smart home energy management",
                "description": "Consider a smart home energy management system that can automatically optimize device operation based on PV production and electricity prices.",
                "potential_savings": "Medium",
                "implementation_difficulty": "Medium"
            })
        
        recommendations["general"] = general_recs
        
        return recommendations

