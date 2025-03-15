import aiohttp
import asyncio
import json
import os
import logging
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIClient:
    """
    Client for interacting with AI models
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the AI client
        
        Args:
            api_key: API key for the AI service (optional, will use environment variable if not provided)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
        if not self.api_key:
            logger.warning("No API key provided for AI client. Some features may not work.")
    
    async def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text using the AI model
        
        Args:
            prompt: Prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        if not self.api_key:
            logger.warning("No API key available. Returning empty response.")
            return "AI analysis not available. Please configure an API key."
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error from AI service: {error_text}")
                        return "Error generating AI response."
                    
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating AI response: {str(e)}"
    
    async def analyze_chart(self, chart_data: Dict[str, Any], chart_type: str, time_period: str) -> Dict[str, Any]:
        """
        Analyze chart data using AI
        
        Args:
            chart_data: Chart data to analyze
            chart_type: Type of chart (e.g., "line", "bar")
            time_period: Time period covered by the chart
            
        Returns:
            Analysis results
        """
        from ai_agent.prompts import CHART_ANALYSIS_PROMPT
        
        try:
            # Convert chart data to JSON string
            data_json = json.dumps(chart_data, default=str)
            
            # Generate analysis using AI
            prompt = CHART_ANALYSIS_PROMPT.format(
                data=data_json,
                chart_type=chart_type,
                time_period=time_period
            )
            
            response = await self.generate_text(prompt)
            
            # Parse the response
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # If the response is not valid JSON, use it as a string
                analysis = {"general": response}
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing chart: {e}")
            return {"error": str(e)}

