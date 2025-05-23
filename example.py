"""
Testing suite for the AiHelper class.
"""

import os
from src.ai_helper import AiHelper
from py_models.weather_model import WeatherModel
from py_models.general_example_model import GeneralExampleModel

# Initialize AiHelper with a model
ai_helper = AiHelper('openrouter:openai/gpt-4o')

# Define tools using actual implementations from src.tools
from src.tools import calculator, weather, pdf_reader

calculator_tool = {"name": "calculator", "description": "A simple calculator that can add, subtract, multiply, and divide.", "function": calculator}
weather_tool = {"name": "weather", "description": "A tool to get the current weather information.", "function": weather}

ai_helper.add_tool(calculator_tool)
ai_helper.add_tool(weather_tool)

# Test with weather tool
result = ai_helper.ask("What is the weather like today in Sofia, Bulgaria?", tools=[weather_tool], output_model=WeatherModel)
print("Weather result:", result)

# List of models to test
models_to_test = [
    'google:gemini-2.5-flash-preview-05-20',
    'openrouter:google/gemini-2.5-flash-preview-05-20',
    'anthropic:claude-3',
    'openrouter:anthropic/claude-3',
    'openai:gpt-4o',
    'openrouter:openai/gpt-4o',
]

for model in models_to_test:
    print(f"\nTesting model: {model}")
    ai_helper = AiHelper(model)
    # Test with a simple prompt and output model
    result = ai_helper.ask("This is a test prompt. Key is 'dog' and value for that is 'Roger'", output_model=GeneralExampleModel)
    print("Text prompt result:", result)
    
    # Test with a file if the model supports it (anthropic models don't support file reading)
    if 'anthropic' not in model.lower():
        test_file_path = "tests/files/test.pdf"
        if os.path.exists(test_file_path):
            result = ai_helper.ask("Please read this PDF and summarize it.", output_model=GeneralExampleModel, file_path=test_file_path)
            print("PDF file result:", result)
        else:
            print(f"Test file {test_file_path} not found, skipping file test.")
        

    test_image_path = "tests/files/test.png"
    if os.path.exists(test_image_path):
        result = ai_helper.ask("Please analyze this image.", output_model=GeneralExampleModel, file_path=test_image_path)
        print("Image file result:", result)
    else:
        print(f"Test image {test_image_path} not found, skipping image test.")
