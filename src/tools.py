"""
Tools should follow PydanticAI's tool calling convention.
https://ai.pydantic.dev/api/tools/

Weather api call (WEATHER_API_KEY is in the env):
http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q=Sofia&aqi=no

"""

import os
import re
import requests
import json
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def calculator(expression: str) -> str:
    """
    A simple calculator that can evaluate basic arithmetic expressions.
    Supports addition (+), subtraction (-), multiplication (*), and division (/).
    Example: calculator("2 + 2 * 3 - 1 / 2") should return "7.5"
    """
    try:
        # Basic security: 
        # Allow numbers, operators (+, -, *, /), parentheses, and spaces.
        # Disallow consecutive operators like ++, --, **, //, etc.
        # This is still naive and not fully secure.
        if not re.match(r"^(?!.*[\+\-\*\/]{2})[0-9\.\+\-\*\/\(\)\s]+$", expression):
            return "Error: Invalid characters or format in expression."
        
        # Using eval is generally unsafe with arbitrary input.
        # For a real application, a safer parsing method is required.
        # Given the context of an LLM tool, we assume the LLM provides valid expressions.
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def weather(location: str) -> str:
    """
    A tool to get the current weather information for a given location.
    Requires WEATHER_API_KEY to be set in the environment.
    """
    if not WEATHER_API_KEY:
        return "Error: WEATHER_API_KEY not configured."
    if not location:
        return "Error: Location not specified."

    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": WEATHER_API_KEY,
        "q": location,
        "aqi": "no"
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        
        # Extract relevant information (example)
        current_weather = data.get("current", {})
        condition = current_weather.get("condition", {}).get("text", "N/A")
        temp_c = current_weather.get("temp_c", "N/A")
        humidity = current_weather.get("humidity", "N/A")
        
        return json.dumps({
            "location": data.get("location", {}).get("name", location),
            "condition": condition,
            "temperature_celsius": temp_c,
            "humidity_percent": humidity
        })
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"
    except json.JSONDecodeError:
        return "Error: Could not parse weather API response."
    except Exception as e:
        return f"An unexpected error occurred while fetching weather: {e}"

def pdf_reader(file_path: str) -> str:
    """
    A tool to read and extract text content from a PDF file.
    (Placeholder implementation)
    """
    # In a real implementation, use a library like PyPDF2 or pdfplumber
    # For example:
    # try:
    #     import PyPDF2
    #     text = ""
    #     with open(file_path, "rb") as f:
    #         reader = PyPDF2.PdfReader(f)
    #         for page_num in range(len(reader.pages)):
    #             text += reader.pages[page_num].extract_text()
    #     if not text:
    #         return "Error: Could not extract text from PDF or PDF is empty."
    #     return text
    # except ImportError:
    #     return "Error: PyPDF2 library is not installed. Cannot read PDF."
    # except Exception as e:
    #     return f"Error reading PDF {file_path}: {e}"
    return f"Placeholder: PDF reader for {file_path}. Text extraction not yet implemented."
