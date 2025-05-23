
"""
Tools following PydanticAI's tool calling convention.
https://ai.pydantic.dev/api/tools/

Weather API call (WEATHER_API_KEY is in the env):
http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q=Sofia&aqi=no
"""

def calculator(expression: str):
    """A simple calculator that can add, subtract, multiply, and divide."""
    try:
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as e:
        return {"error": f"Error calculating expression: {str(e)}"}

def weather(location: str):
    """A tool to get the current weather information for a specified location."""
    import os
    import requests
    
    api_key = os.getenv("WEATHER_API_KEY", "")
    if not api_key:
        return {"location": "", "temperature": 0.0, "conditions": "Weather API key not found"}
    
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            location_name = data["location"]["name"]
            temp_c = float(data["current"]["temp_c"])
            condition = data["current"]["condition"]["text"]
            return {"location": location_name, "temperature": temp_c, "conditions": condition}
        else:
            return {"location": "", "temperature": 0.0, "conditions": f"Error fetching weather data: {response.status_code}"}
    except Exception as e:
        return {"location": "", "temperature": 0.0, "conditions": f"Error fetching weather data: {str(e)}"}

def pdf_reader(file_path: str):
    """A tool to read and extract information from a PDF file."""
    try:
        import PyPDF2
        
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return {"summary": text[:500] + "..." if len(text) > 500 else text}
    except Exception as e:
        return {"error": f"Error reading PDF file: {str(e)}"}

# Define tools in PydanticAI format for integration with AI helper
TOOLS = [
    {
        "name": "calculator",
        "description": "A simple calculator that can perform basic arithmetic operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                }
            },
            "required": ["expression"]
        },
        "function": calculator
    },
    {
        "name": "weather",
        "description": "Get current weather information for a specified location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather information for (e.g., 'London')"
                }
            },
            "required": ["location"]
        },
        "function": weather
    },
    {
        "name": "pdf_reader",
        "description": "Extract text content from a PDF file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the PDF file to read"
                }
            },
            "required": ["file_path"]
        },
        "function": pdf_reader
    }
]
