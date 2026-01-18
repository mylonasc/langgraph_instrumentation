import time
import random
from typing import List
from langchain_core.tools import BaseTool, tool
import asyncio
class DemoAgentToolsFacade:
    """A registry of demonstration tools for the ReAct agent.
    
    This facade groups together various mock tools designed to demonstrate
    different tracing capabilities (e.g., long-running tasks, structured inputs).
    """

    @staticmethod
    @tool
    def get_weather(city: str) -> str:
        """Fetches the current weather for a specified city.
        
        Args:
            city (str): The name of the city (e.g., "Zurich", "Tokyo").
            
        Returns:
            str: A descriptive string of the weather conditions.
        """
        # Simulate network latency slightly for realism in trace
        time.sleep(0.1) 
        conditions = ["Sunny", "Rainy", "Cloudy", "Snowy"]
        temp = random.randint(-5, 35)
        return f"The weather in {city} is {random.choice(conditions)} with a temperature of {temp}°C."

    @staticmethod
    @tool
    async def complex_calculation(base: int, exponent: int) -> int:
        """Performs a heavy calculation. (Async Version)"""
        # Use asyncio.sleep instead of time.sleep to be non-blocking
        await asyncio.sleep(1.0) 
        return base ** exponent

    @staticmethod
    @tool
    def sum_numbers(num_list : list[int | float]) -> float:
        """This tool allows summing a list of numbers. 
        """
        return sum(num_list)

    @staticmethod
    @tool
    def reverse_text(text: str) -> str:
        """Reverses the input text string.
        
        Args:
            text (str): The string to reverse.
            
        Returns:
            str: The reversed string.
        """
        return text[::-1]

    def get_tools(self) -> List[BaseTool]:
        """Returns the list of all available tools.

        Returns:
            List[BaseTool]: A list of LangChain tool objects ready for binding.
        """
        return [
            self.get_weather,
            self.complex_calculation,
            self.reverse_text,
            self.sum_numbers
        ]