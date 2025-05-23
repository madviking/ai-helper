
"""
Tools should follow PydanticAI's tool calling convention.
https://ai.pydantic.dev/api/tools/

Weather api call (WEATHER_API_KEY is in the env):
http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q=Sofia&aqi=no

"""

def calculator(expression: str):
    """A simple calculator that can add, subtract, multiply, and divide."""
    try:
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

def weather(location: str):
    """A tool to get the current weather information."""
    import os
    import requests
    import json
    
    api_key = os.getenv("WEATHER_API_KEY", "")
    if not api_key:
        return json.dumps({"location": "", "temperature": 0.0, "conditions": "Weather API key not found"})
    
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            location_name = data["location"]["name"]
            temp_c = float(data["current"]["temp_c"])
            condition = data["current"]["condition"]["text"]
            return json.dumps({"location": location_name, "temperature": temp_c, "conditions": condition})
        else:
            return json.dumps({"location": "", "temperature": 0.0, "conditions": f"Error fetching weather data: {response.status_code}"})
    except Exception as e:
        return json.dumps({"location": "", "temperature": 0.0, "conditions": f"Error fetching weather data: {str(e)}"})

def pdf_reader(file_path: str):
    """A tool to read and extract information from a PDF file."""
    try:
        import PyPDF2
        
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return f"PDF content summary: {text[:500]}..." if len(text) > 500 else text
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"
