"""
Example 4: Structured Communication Between LLMs and External Systems

This example demonstrates how to integrate external systems with Frizz agents,
using a weather API as an example of how to connect to external services.

Note: This example uses a mock API for simplicity, but you could replace it
with a real weather API integration.
"""
import asyncio
import os
import sys
from datetime import datetime

from aikernel import Conversation, LLMMessagePart, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import BaseModel, Field

# Add the parent directory to the path so we can import the custom_agent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_agent import Agent

from frizz import tool


# Models for the weather tool
class Coordinates(BaseModel):
    """Geographic coordinates."""
    latitude: float
    longitude: float


class WeatherRequestParams(BaseModel):
    """Parameters for requesting weather information."""
    location: str = Field(..., description="City name or address")
    forecast_days: int | None = Field(0, description="Days to forecast (0 = current weather only)", ge=0, le=7)


class WeatherCondition(BaseModel):
    """Weather condition information."""
    temperature: float
    feels_like: float
    humidity: int
    wind_speed: float
    description: str
    date: datetime


class WeatherResponse(BaseModel):
    """Response from the weather service."""
    location: str
    coordinates: Coordinates
    current: WeatherCondition
    forecast: list[WeatherCondition] | None = None
    units: str = "metric"


# Mock external weather service
class WeatherService:
    """External weather service integration (mocked for demonstration)."""
    
    async def get_weather(self, location: str, forecast_days: int = 0) -> WeatherResponse:
        """Get weather data from the external API."""
        # In a real implementation, this would make an HTTP request to a weather API
        # For this example, we'll return mock data
        
        # Simulate API latency
        await asyncio.sleep(0.5)
        
        # Mock API response based on location
        if "london" in location.lower():
            coords = Coordinates(latitude=51.5074, longitude=-0.1278)
            current = WeatherCondition(
                temperature=12.5,
                feels_like=10.2,
                humidity=76,
                wind_speed=15.3,
                description="Cloudy with light rain",
                date=datetime.utcnow()
            )
        elif "tokyo" in location.lower():
            coords = Coordinates(latitude=35.6762, longitude=139.6503)
            current = WeatherCondition(
                temperature=21.2,
                feels_like=22.1,
                humidity=65,
                wind_speed=8.7,
                description="Partly cloudy",
                date=datetime.utcnow()
            )
        elif "new york" in location.lower() or "ny" in location.lower():
            coords = Coordinates(latitude=40.7128, longitude=-74.0060)
            current = WeatherCondition(
                temperature=18.9,
                feels_like=19.4,
                humidity=58,
                wind_speed=12.8,
                description="Clear skies",
                date=datetime.utcnow()
            )
        else:
            # Default weather for unknown locations
            coords = Coordinates(latitude=0.0, longitude=0.0)
            current = WeatherCondition(
                temperature=25.0,
                feels_like=26.2,
                humidity=60,
                wind_speed=10.0,
                description="Sunny",
                date=datetime.utcnow()
            )
        
        response = WeatherResponse(
            location=location,
            coordinates=coords,
            current=current,
            units="metric"
        )
        
        # Add forecast if requested
        if forecast_days > 0:
            # In a real implementation, this would be actual forecast data
            response.forecast = [
                WeatherCondition(
                    temperature=current.temperature + (i - 2),  # Slight variations
                    feels_like=current.feels_like + (i - 1.5),
                    humidity=max(30, min(90, current.humidity + (i * 2))),
                    wind_speed=max(5, current.wind_speed + (i - 2)),
                    description=current.description,
                    date=datetime(
                        current.date.year, 
                        current.date.month, 
                        current.date.day + i
                    )
                )
                for i in range(1, forecast_days + 1)
            ]
        
        return response


# Context with external system connections
class ExternalSystemsContext:
    """Context containing connections to external systems."""
    def __init__(self):
        # Initialize connections to external systems
        self.weather_service = WeatherService()
        
        # In a real application, you might have other services:
        # self.database = DatabaseClient()
        # self.analytics = AnalyticsService()
        # self.payment_processor = PaymentGateway()


# Define the weather tool that integrates with the external service
@tool(name="get_weather")
async def get_weather(
    *, context: ExternalSystemsContext, parameters: WeatherRequestParams, conversation: Conversation
) -> WeatherResponse:
    """Get current weather and optional forecast for a location."""
    # Call the external service through our context
    weather_data = await context.weather_service.get_weather(
        parameters.location, 
        parameters.forecast_days
    )
    return weather_data


async def main():
    # Create context with external system connections
    systems_context = ExternalSystemsContext()
    
    # Create an agent with the weather tool
    agent = Agent(
        tools=[get_weather],
        context=systems_context,
        system_message=LLMSystemMessage(parts=[LLMMessagePart(content="""
            You are a helpful weather assistant that can provide weather information.
            When asked about weather, use the get_weather tool to fetch real-time data.
            Always interpret the weather data and provide a friendly, conversational response.
            Convert units if the user asks for non-metric units.
        """)])
    )
    
    # Create a router for the LLM API
    router = get_router(models=("gemini-2.0-flash",))
    
    # Example conversation
    print("Starting conversation with the weather assistant...\n")
    
    # First interaction: Check current weather
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="What's the weather like in London?")])
    print(f"User: {user_message.parts[0].content}")
    
    result = await agent.step(
        user_message=user_message,
        model="gemini-2.0-flash",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")
    
    # Second interaction: Get a forecast
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="Can you give me a 3-day forecast for Tokyo?")])
    print(f"\nUser: {user_message.parts[0].content}")
    
    result = await agent.step(
        user_message=user_message,
        model="gemini-2.0-flash",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")
    
    # Third interaction: Ask about a different location
    user_message = LLMUserMessage(parts=[LLMMessagePart(content="How about New York? And can you give me the temperature in Fahrenheit?")])
    print(f"\nUser: {user_message.parts[0].content}")
    
    result = await agent.step(
        user_message=user_message,
        model="gemini-2.0-flash",
        router=router
    )
    
    print(f"Assistant: {result.assistant_message.parts[0].content}")


if __name__ == "__main__":
    asyncio.run(main())
