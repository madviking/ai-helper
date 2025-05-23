import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.ai_helper import AiHelper
from src.cost_tracker import CostTracker
from src.tools import calculator, weather, pdf_reader
from py_models.weather_model import WeatherModel # Assuming WeatherModel is a Pydantic model

# Initialize Cost Tracker
cost_tracker = CostTracker()

# Initialize AiHelper (choose a model identifier, e.g., "openai", "anthropic", "google", "openrouter")
# Ensure you have the corresponding API key in your .env file
# Example using OpenAI:
helper = AiHelper("openai", cost_tracker=cost_tracker)

# Add available tools to the helper
helper.add_tool("calculator", "A simple calculator that can add, subtract, multiply, and divide.", calculator)
helper.add_tool("weather", "A tool to get the current weather information.", weather)
helper.add_tool("pdf_reader", "A tool to read and extract information from a PDF file.", pdf_reader)


# --- Example Usage ---

# 1. Basic Ask
print("--- Basic Ask ---")
prompt_basic = "What is the capital of Canada?"
response_basic = helper.ask(prompt_basic)
print(f"Prompt: {prompt_basic}")
print(f"Response: {response_basic}")
print("-" * 20)

# 2. Ask with Tools (assuming the model can interpret and call tools)
# Note: This requires the adapter and LLM to support tool calling.
# The current adapter implementations are basic and may not trigger tool calls.
print("--- Ask with Tools ---")
prompt_tool = "Calculate 15 + 7 and tell me the result."
tools_list = ["calculator"]
response_tool = helper.ask(prompt_tool, tools=tools_list)
print(f"Prompt: {prompt_tool}")
print(f"Response: {response_tool}") # Expected to contain tool results if tool calling is implemented
print("-" * 20)

# 3. Ask with Pydantic Model (assuming the model can generate structured output)
# Note: This requires the adapter and LLM to support generating output conforming to a schema.
print("--- Ask with Pydantic Model ---")
prompt_model = "Provide the current weather information for London in a structured format."
# Assuming WeatherModel is defined in py_models/weather_model.py
response_model = helper.ask(prompt_model, pydantic_model=WeatherModel)
print(f"Prompt: {prompt_model}")
print(f"Response: {response_model}") # Expected to contain a WeatherModel instance and filled percentage
print("-" * 20)

# 4. Ask with File (assuming the model can process file content)
# Note: This requires the adapter and LLM to support file input.
print("--- Ask with File ---")
# Create a dummy file for demonstration
dummy_file_path = "tests/files/example_dummy_file.txt"
try:
    os.makedirs(os.path.dirname(dummy_file_path), exist_ok=True)
    with open(dummy_file_path, "w") as f:
        f.write("This is some example content in a dummy file.")

    prompt_file = "Summarize the content of the provided file."
    response_file = helper.ask(prompt_file, file=dummy_file_path)
    print(f"Prompt: {prompt_file}")
    print(f"File: {dummy_file_path}")
    print(f"Response: {response_file}")
    print("-" * 20)

finally:
    # Clean up the dummy file
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)


# Print Total Cost
print("--- Total Cost ---")
print(f"Total estimated cost: ${cost_tracker.get_total_cost():.6f}")
print("-" * 20)
