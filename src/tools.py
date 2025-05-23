"""
Tools should follow PydanticAI's tool calling convention.
https://ai.pydantic.dev/api/tools/

Weather api call (WEATHER_API_KEY is in the env):
http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q=Sofia&aqi=no

"""
import os
import requests
import pypdf
from typing import Dict, Any


def calculator(expression: str) -> float:
    """A simple calculator that can add, subtract, multiply, and divide."""
    try:
        # Use eval safely for mathematical expressions only
        # Remove any non-mathematical characters for safety
        allowed_chars = "0123456789+-*/()., "
        cleaned_expr = ''.join(c for c in expression if c in allowed_chars)
        
        # Evaluate the expression
        result = eval(cleaned_expr)
        return float(result)
    except Exception as e:
        raise Exception(f"Invalid expression: {expression}. Error: {str(e)}")


def weather(location: str) -> Dict[str, Any]:
    """A tool to get the current weather information."""
    api_key = os.environ.get('WEATHER_API_KEY')
    if not api_key:
        raise Exception("WEATHER_API_KEY environment variable is not set")
    
    url = "http://api.weatherapi.com/v1/current.json"
    params = {
        'key': api_key,
        'q': location,
        'aqi': 'no'
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            error_data = response.json()
            raise Exception(f"Weather API error: {error_data.get('error', {}).get('message', 'Unknown error')}")
        
        data = response.json()
        
        # Extract relevant information
        result = {
            'location': f"{data['location']['name']}, {data['location']['country']}",
            'temperature': data['current']['temp_c'],
            'conditions': data['current']['condition']['text']
        }
        
        return result
        
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch weather data: {str(e)}")


def pdf_reader(file_path: str) -> str:
    """A tool to read and extract information from a PDF file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    if not file_path.lower().endswith('.pdf'):
        raise Exception(f"File is not a PDF: {file_path}")
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            # Extract text from all pages
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text_content.append(f"Page {page_num + 1}:\n{page_text}")
            
            if not text_content:
                return "No text content found in the PDF"
            
            return "\n\n".join(text_content)
            
    except Exception as e:
        raise Exception(f"Error reading PDF file: {str(e)}")
