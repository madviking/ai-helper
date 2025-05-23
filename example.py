"""
Testing suite for the AiHelper class.
"""

import os
from src.ai_helper import AiHelper
from py_models.weather_model import WeatherModel
from py_models.general_example_model import GeneralExampleModel

# Initialize AiHelper with a model
ai_helper = AiHelper('openrouter:openai/gpt-4o')

# Define tools using the TOOLS list from src.tools for PydanticAI compatibility
from src.tools import TOOLS

for tool in TOOLS:
    ai_helper.add_tool(tool)

# Test with weather tool
result = ai_helper.ask("What is the weather like today in Sofia, Bulgaria?", tools=[tool for tool in TOOLS if tool['name'] == 'weather'], output_model=WeatherModel)
print("Weather result:", result)

# List of models to test
models_to_test = [
    'google:gemini-1.5-pro',
    'openrouter:google/gemini-pro',
    'anthropic:claude-3-opus-20240229',
    'openrouter:anthropic/claude-3-opus-20240229',
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
