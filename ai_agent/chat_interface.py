import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatInterface:
    """
    Chat interface for interacting with the energy management system
    """
    
    def __init__(self, ai_client=None, energy_analyzer=None):
        """
        Initialize the chat interface
        
        Args:
            ai_client: Client for AI model API (optional)
            energy_analyzer: Energy analysis agent (optional)
        """
        self.ai_client = ai_client
        self.energy_analyzer = energy_analyzer
        self.conversation_history = []
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation history
        
        Args:
            role: Role of the message sender ("user" or "assistant")
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    async def process_query(self, query: str, system_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a user query and generate a response
        
        Args:
            query: User query
            system_data: Current system data (optional)
            
        Returns:
            Response to the user query
        """
        # Add user message to history
        self.add_message("user", query)
        
        # Generate response
        response = await self._generate_response(query, system_data)
        
        # Add assistant message to history
        self.add_message("assistant", response)
        
        return response
    
    async def _generate_response(self, query: str, system_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response to a user query
        
        Args:
            query: User query
            system_data: Current system data (optional)
            
        Returns:
            Response to the user query
        """
        from ai_agent.prompts import CHAT_RESPONSE_PROMPT
        
        if not self.ai_client:
            # Generate a basic response without AI
            return self._generate_basic_response(query, system_data)
        
        try:
            # Prepare context for AI
            context = self._prepare_context(query, system_data)
            
            # Generate response using AI
            prompt = CHAT_RESPONSE_PROMPT.format(
                question=query,
                context=context
            )
            
            response = await self.ai_client.generate_text(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            # Fall back to basic response
            return self._generate_basic_response(query, system_data)
    
    def _prepare_context(self, query: str, system_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Prepare context for AI response generation
        
        Args:
            query: User query
            system_data: Current system data (optional)
            
        Returns:
            Context string for AI
        """
        context_parts = []
        
        # Add system data if available
        if system_data:
            # Add current energy status
            if "sensor_data" in system_data:
                sensor_data = system_data["sensor_data"]
                context_parts.append("Current Energy Status:")
                if "load_power" in sensor_data:
                    context_parts.append(f"- Current Load: {sensor_data['load_power']:.2f} kW")
                if "pv_power" in sensor_data:
                    context_parts.append(f"- Solar Production: {sensor_data['pv_power']:.2f} kW")
                if "battery_power" in sensor_data:
                    context_parts.append(f"- Battery Power: {sensor_data['battery_power']:.2f} kW")
                if "battery_soc" in sensor_data:
                    context_parts.append(f"- Battery State of Charge: {sensor_data['battery_soc']:.1f}%")
                if "grid_power" in sensor_data:
                    context_parts.append(f"- Grid Power: {sensor_data['grid_power']:.2f} kW")
            
            # Add optimization results if available
            if "optimization_results" in system_data:
                opt_results = system_data["optimization_results"]
                context_parts.append("\nOptimization Results:")
                if "cost" in opt_results:
                    context_parts.append(f"- Optimized Cost: ${opt_results['cost']:.2f}")
                if "self_consumption" in opt_results:
                    context_parts.append(f"- Self-Consumption: {opt_results['self_consumption']:.1f}%")
                if "peak_grid_power" in opt_results:
                    context_parts.append(f"- Peak Grid Power: {opt_results['peak_grid_power']:.2f} kW")
                if "battery_cycles" in opt_results:
                    context_parts.append(f"- Battery Cycles: {opt_results['battery_cycles']:.2f}")
            
            # Add analysis results if available
            if "analysis_results" in system_data:
                analysis = system_data["analysis_results"]
                if "insights" in analysis:
                    insights = analysis["insights"]
                    context_parts.append("\nSystem Insights:")
                    
                    # Add general insights
                    if "general" in insights and insights["general"]:
                        for insight in insights["general"]:
                            context_parts.append(f"- {insight}")
                    
                    # Add specific insights based on query keywords
                    query_lower = query.lower()
                    if "load" in query_lower and "load" in insights:
                        context_parts.append("\nLoad Insights:")
                        for insight in insights["load"]:
                            context_parts.append(f"- {insight}")
                    
                    if any(kw in query_lower for kw in ["pv", "solar", "panel"]) and "pv" in insights:
                        context_parts.append("\nSolar PV Insights:")
                        for insight in insights["pv"]:
                            context_parts.append(f"- {insight}")
                    
                    if any(kw in query_lower for kw in ["battery", "storage"]) and "battery" in insights:
                        context_parts.append("\nBattery Insights:")
                        for insight in insights["battery"]:
                            context_parts.append(f"- {insight}")
                    
                    if any(kw in query_lower for kw in ["grid", "import", "export"]) and "grid" in insights:
                        context_parts.append("\nGrid Insights:")
                        for insight in insights["grid"]:
                            context_parts.append(f"- {insight}")
            
            # Add recommendations if available
            if "recommendations" in system_data:
                recs = system_data["recommendations"]
                context_parts.append("\nRecommendations:")
                
                # Add recommendations based on query keywords
                query_lower = query.lower()
                
                if any(kw in query_lower for kw in ["shift", "schedule", "timing"]) and "load_shifting" in recs:
                    context_parts.append("\nLoad Shifting Recommendations:")
                    for rec in recs["load_shifting"]:
                        context_parts.append(f"- {rec['title']}: {rec['description']}")
                
                if any(kw in query_lower for kw in ["battery", "storage"]) and "battery_optimization" in recs:
                    context_parts.append("\nBattery Optimization Recommendations:")
                    for rec in recs["battery_optimization"]:
                        context_parts.append(f"- {rec['title']}: {rec['description']}")
                
                if any(kw in query_lower for kw in ["system", "size", "capacity"]) and "system_sizing" in recs:
                    context_parts.append("\nSystem Sizing Recommendations:")
                    for rec in recs["system_sizing"]:
                        context_parts.append(f"- {rec['title']}: {rec['description']}")
                
                if any(kw in query_lower for kw in ["efficiency", "save", "reduce"]) and "energy_efficiency" in recs:
                    context_parts.append("\nEnergy Efficiency Recommendations:")
                    for rec in recs["energy_efficiency"]:
                        context_parts.append(f"- {rec['title']}: {rec['description']}")
        
        # Add conversation history (last 5 messages)
        if self.conversation_history:
            context_parts.append("\nRecent Conversation:")
            for msg in self.conversation_history[-5:]:
                context_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def _generate_basic_response(self, query: str, system_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a basic response without AI
        
        Args:
            query: User query
            system_data: Current system data (optional)
            
        Returns:
            Basic response to the user query
        """
        query_lower = query.lower()
        
        # Check for greetings
        if any(greeting in query_lower for greeting in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm your energy management assistant. How can I help you today?"
        
        # Check for status inquiries
        if any(status in query_lower for status in ["status", "current", "now", "today"]):
            if system_data and "sensor_data" in system_data:
                sensor_data = system_data["sensor_data"]
                response = "Current system status:\n"
                if "load_power" in sensor_data:
                    response += f"- Load: {sensor_data['load_power']:.2f} kW\n"
                if "pv_power" in sensor_data:
                    response += f"- Solar: {sensor_data['pv_power']:.2f} kW\n"
                if "battery_power" in sensor_data:
                    response += f"- Battery: {sensor_data['battery_power']:.2f} kW\n"
                if "battery_soc" in sensor_data:
                    response += f"- Battery SOC: {sensor_data['battery_soc']:.1f}%\n"
                if "grid_power" in sensor_data:
                    response += f"- Grid: {sensor_data['grid_power']:.2f} kW\n"
                return response
            else:
                return "I don't have current system status information available."
        
        # Check for optimization inquiries
        if any(opt in query_lower for opt in ["optimize", "optimization", "improve", "better"]):
            if system_data and "recommendations" in system_data:
                recs = system_data["recommendations"]
                if "general" in recs and recs["general"]:
                    rec = recs["general"][0]
                    return f"Here's an optimization recommendation: {rec['title']} - {rec['description']}"
                else:
                    return "I have optimization recommendations available, but need more specific information about what you'd like to optimize."
            else:
                return "I don't have optimization recommendations available at the moment."
        
        # Check for battery inquiries
        if any(battery in query_lower for battery in ["battery", "storage", "charge"]):
            if system_data and "sensor_data" in system_data and "battery_soc" in system_data["sensor_data"]:
                soc = system_data["sensor_data"]["battery_soc"]
                if "battery_power" in system_data["sensor_data"]:
                    power = system_data["sensor_data"]["battery_power"]
                    status = "charging" if power > 0 else "discharging" if power < 0 else "idle"
                    return f"Your battery is currently at {soc:.1f}% and is {status}."
                else:
                    return f"Your battery is currently at {soc:.1f}%."
            else:
                return "I don't have battery information available at the moment."
        
        # Check for solar inquiries
        if any(solar in query_lower for solar in ["solar", "pv", "panel", "sun"]):
            if system_data and "sensor_data" in system_data and "pv_power" in system_data["sensor_data"]:
                power = system_data["sensor_data"]["pv_power"]
                return f"Your solar panels are currently producing {power:.2f} kW."
            else:
                return "I don't have solar production information available at the moment."
        
        # Default response
        return "I'm not sure how to answer that. Could you please provide more details or ask about your energy system status, optimization recommendations, or specific components like battery or solar?"

