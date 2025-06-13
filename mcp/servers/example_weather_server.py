"""
Example Weather MCP Server

A proper MCP server using FastMCP that provides weather-related tools.
This demonstrates how to create an MCP server that can be registered
with the MCP registry and used by MCP clients.
"""

import asyncio
import random
from typing import Dict, Any, List
from datetime import datetime

from fastmcp import FastMCP, Context


# Mock weather data for demonstration
MOCK_WEATHER_DATA = {
    "New York": {"temp": 72, "condition": "Partly Cloudy", "humidity": 65},
    "London": {"temp": 59, "condition": "Rainy", "humidity": 80},
    "Tokyo": {"temp": 68, "condition": "Clear", "humidity": 55},
    "Sydney": {"temp": 77, "condition": "Sunny", "humidity": 70},
    "Paris": {"temp": 63, "condition": "Cloudy", "humidity": 75},
}

# Create FastMCP server instance
mcp = FastMCP(
    name="Weather Server",
    description="Provides weather information and forecasts"
)


# MCP Tools
@mcp.tool()
async def get_current_weather(city: str) -> Dict[str, Any]:
    """Get current weather for a specific city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        Dictionary containing temperature, condition, humidity and timestamp
    """
    # Simulate API delay
    await asyncio.sleep(0.5)
    
    # Get mock data or generate random
    if city in MOCK_WEATHER_DATA:
        weather = MOCK_WEATHER_DATA[city]
    else:
        weather = {
            "temp": random.randint(50, 85),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Clear"]),
            "humidity": random.randint(40, 90)
        }
    
    return {
        "city": city,
        "temperature": weather["temp"],
        "condition": weather["condition"],
        "humidity": weather["humidity"],
        "unit": "fahrenheit",
        "timestamp": datetime.utcnow().isoformat()
    }


@mcp.tool()
async def get_weather_forecast(city: str, days: int = 3) -> Dict[str, Any]:
    """Get weather forecast for the next N days.
    
    Args:
        city: The name of the city to get forecast for
        days: Number of days to forecast (1-7)
        
    Returns:
        Dictionary containing city, days and forecast data
    """
    if days < 1 or days > 7:
        raise ValueError("Days must be between 1 and 7")
    
    # Simulate API delay
    await asyncio.sleep(0.8)
    
    # Generate forecast
    forecast = []
    base_temp = MOCK_WEATHER_DATA.get(city, {}).get("temp", 70)
    
    for i in range(days):
        forecast.append({
            "day": i + 1,
            "high": base_temp + random.randint(-5, 10),
            "low": base_temp - random.randint(10, 20),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Clear", "Partly Cloudy"]),
            "precipitation_chance": random.randint(0, 100)
        })
    
    return {
        "city": city,
        "days": days,
        "forecast": forecast,
        "generated_at": datetime.utcnow().isoformat()
    }


@mcp.tool()
async def compare_weather(cities: List[str], ctx: Context) -> Dict[str, Any]:
    """Compare weather between multiple cities.
    
    Args:
        cities: List of city names to compare
        ctx: MCP context for progress reporting
        
    Returns:
        Dictionary with weather comparison data
    """
    if not cities:
        raise ValueError("Cities list cannot be empty")
    if len(cities) > 5:
        raise ValueError("Maximum 5 cities can be compared at once")
    
    # Report initial progress
    await ctx.info(f"Comparing weather for {len(cities)} cities...")
    
    # Get weather for each city
    comparison = {}
    for i, city in enumerate(cities):
        # Report progress
        await ctx.report_progress(i, len(cities))
        
        # Simulate API delay
        await asyncio.sleep(0.3)
        
        if city in MOCK_WEATHER_DATA:
            weather = MOCK_WEATHER_DATA[city]
        else:
            weather = {
                "temp": random.randint(50, 85),
                "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Clear"]),
                "humidity": random.randint(40, 90)
            }
        
        comparison[city] = {
            "temperature": weather["temp"],
            "condition": weather["condition"],
            "humidity": weather["humidity"]
        }
    
    # Report completion
    await ctx.report_progress(len(cities), len(cities))
    
    # Find warmest and coolest
    temps = {city: data["temperature"] for city, data in comparison.items()}
    warmest = max(temps, key=temps.get)
    coolest = min(temps, key=temps.get)
    
    return {
        "cities": comparison,
        "summary": {
            "warmest": {"city": warmest, "temperature": temps[warmest]},
            "coolest": {"city": coolest, "temperature": temps[coolest]},
            "average_temperature": sum(temps.values()) / len(temps)
        },
        "compared_at": datetime.utcnow().isoformat()
    }


# MCP Resources
@mcp.resource("weather://current/all-cities")
async def get_all_cities_weather() -> Dict[str, Any]:
    """Get current weather for all available cities.
    
    This resource provides a snapshot of weather data for all cities
    we have information about.
    """
    all_weather = {}
    for city, weather in MOCK_WEATHER_DATA.items():
        all_weather[city] = {
            "temperature": weather["temp"],
            "condition": weather["condition"],
            "humidity": weather["humidity"],
            "unit": "fahrenheit"
        }
    
    return {
        "cities": all_weather,
        "timestamp": datetime.utcnow().isoformat(),
        "total_cities": len(all_weather)
    }


@mcp.resource("weather://alerts/{city}")
async def get_weather_alerts(city: str) -> Dict[str, Any]:
    """Get weather alerts for a specific city.
    
    Args:
        city: The city to get alerts for
        
    Returns:
        Dictionary containing any active weather alerts
    """
    # Mock implementation - in real server would check actual alerts
    has_alert = random.choice([True, False])
    
    if has_alert:
        alert_types = ["Thunderstorm Warning", "Heat Advisory", "Flood Watch", "Winter Storm Warning"]
        return {
            "city": city,
            "alerts": [
                {
                    "type": random.choice(alert_types),
                    "severity": random.choice(["Minor", "Moderate", "Severe"]),
                    "expires": "2024-12-31T23:59:59Z"
                }
            ],
            "has_alerts": True
        }
    else:
        return {
            "city": city,
            "alerts": [],
            "has_alerts": False
        }


# MCP Prompts
@mcp.prompt()
def weather_report_prompt(city: str, include_forecast: bool = False) -> str:
    """Generate a weather report prompt for a city.
    
    Args:
        city: The city to generate a report for
        include_forecast: Whether to include forecast in the report
        
    Returns:
        A prompt string for generating a weather report
    """
    base_prompt = f"Please provide a detailed weather report for {city}. Include current conditions, temperature, humidity, and any relevant observations."
    
    if include_forecast:
        base_prompt += f" Also include a {3}-day forecast with expected conditions and temperature ranges."
    
    base_prompt += " Format the report in a clear, readable manner suitable for a weather bulletin."
    
    return base_prompt


@mcp.prompt()
def travel_weather_prompt(cities: List[str]) -> str:
    """Generate a travel weather comparison prompt.
    
    Args:
        cities: List of cities to compare for travel planning
        
    Returns:
        A prompt for travel weather advice
    """
    cities_str = ", ".join(cities)
    return f"""I'm planning to travel and need to compare weather conditions between these cities: {cities_str}.

Please analyze the current weather in each city and provide:
1. A comparison of current conditions
2. Which city has the most favorable weather for travel
3. Any weather-related concerns or recommendations
4. Best time of day for outdoor activities in each location

Help me make an informed decision about my travel plans."""


# Main entry point
def main():
    """Run the MCP server."""
    # Default to stdio transport for MCP compatibility
    mcp.run()


if __name__ == "__main__":
    main() 