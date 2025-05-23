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
# Use a model known to support tool calling and response_format="json_object"
helper_openai = AiHelper("openai", cost_tracker=cost_tracker)

# Add available tools to the helper (for OpenAI example)
helper_openai.add_tool("calculator", "A simple calculator that can add, subtract, multiply, and divide.", calculator)
helper_openai.add_tool("weather", "A tool to get the current weather information.", weather)
helper_openai.add_tool("pdf_reader", "A tool to read and extract information from a PDF file.", pdf_reader)


# --- Example Usage (OpenAI) ---

print("--- OpenAI Example ---")

# 1. Basic Ask
print("--- Basic Ask ---")
prompt_basic = "What is the capital of Canada?"
response_basic = helper_openai.ask(prompt_basic)
print(f"Prompt: {prompt_basic}")
print(f"Response: {response_basic}")
print("-" * 20)

# 2. Ask with Tools (assuming the model can interpret and call tools)
print("--- Ask with Tools ---")
prompt_tool = "Calculate 15 + 7 and tell me the result."
tools_list = ["calculator"]
response_tool = helper_openai.ask(prompt_tool, tools=tools_list)
print(f"Prompt: {prompt_tool}")
print(f"Response: {response_tool}") # Expected to contain tool results if tool calling is implemented
print("-" * 20)

# 3. Ask with Pydantic Model (assuming the model can generate structured output)
print("--- Ask with Pydantic Model ---")
prompt_model = "Provide the current weather information for London in a structured format."
# Assuming WeatherModel is defined in py_models/weather_model.py
response_model = helper_openai.ask(prompt_model, pydantic_model=WeatherModel)
print(f"Prompt: {prompt_model}")
print(f"Response: {response_model}") # Expected to contain a WeatherModel instance and filled percentage
print("-" * 20)

# 4. Ask with File (assuming the model can process file content)
print("--- Ask with File ---")
# Create a dummy file for demonstration
dummy_file_path = "tests/files/example_dummy_file.txt"
try:
    os.makedirs(os.path.dirname(dummy_file_path), exist_ok=True)
    with open(dummy_file_path, "w") as f:
        f.write("This is some example content in a dummy file.")

    prompt_file = "Summarize the content of the provided file."
    response_file = helper_openai.ask(prompt_file, file=dummy_file_path)
    print(f"Prompt: {prompt_file}")
    print(f"File: {dummy_file_path}")
    print(f"Response: {response_file}")
    print("-" * 20)

finally:
    # Clean up the dummy file
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)

print("=" * 30) # Separator for different adapter examples

# Example using OpenRouter:
# Use a model known to support tool calling and response_format="json_object" if needed
# OpenRouter uses the OpenAI SDK format, so similar models apply
helper_openrouter = AiHelper("openrouter", cost_tracker=cost_tracker)

# Add available tools to the helper (for OpenRouter example)
helper_openrouter.add_tool("calculator", "A simple calculator that can add, subtract, multiply, and divide.", calculator)
helper_openrouter.add_tool("weather", "A tool to get the current weather information.", weather)
helper_openrouter.add_tool("pdf_reader", "A tool to read and extract information from a PDF file.", pdf_reader)

# --- Example Usage (OpenRouter) ---

print("--- OpenRouter Example ---")

# 1. Basic Ask
print("--- Basic Ask ---")
prompt_basic_or = "What is the highest mountain in Africa?"
response_basic_or = helper_openrouter.ask(prompt_basic_or)
print(f"Prompt: {prompt_basic_or}")
print(f"Response: {response_basic_or}")
print("-" * 20)

# 2. Ask with Tools (assuming the model can interpret and call tools)
print("--- Ask with Tools ---")
prompt_tool_or = "Use the calculator to multiply 8 by 9."
tools_list_or = ["calculator"]
response_tool_or = helper_openrouter.ask(prompt_tool_or, tools=tools_list_or)
print(f"Prompt: {prompt_tool_or}")
print(f"Response: {response_tool_or}") # Expected to contain tool results if tool calling is implemented
print("-" * 20)

# 3. Ask with Pydantic Model (assuming the model can generate structured output)
print("--- Ask with Pydantic Model ---")
prompt_model_or = "Provide the current weather information for Tokyo in a structured format."
response_model_or = helper_openrouter.ask(prompt_model_or, pydantic_model=WeatherModel)
print(f"Prompt: {prompt_model_or}")
print(f"Response: {response_model_or}") # Expected to contain a WeatherModel instance and filled percentage
print("-" * 20)

# 4. Ask with File (assuming the model can process file content)
print("--- Ask with File ---")
# Reusing the dummy file created earlier
dummy_file_path = "tests/files/example_dummy_file.txt"
print(f"Attempting to use file: {dummy_file_path}") # Added print
print(f"Does file exist before ask? {os.path.exists(dummy_file_path)}") # Added print

prompt_file_or = "Count the number of words in the provided file."
response_file_or = helper_openrouter.ask(prompt_file_or, file=dummy_file_path)
print(f"Prompt: {prompt_file_or}")
print(f"File: {dummy_file_path}")
print(f"Response: {response_file_or}")
print("-" * 20)


print("=" * 30) # Separator

# TODO: Add examples for Anthropic and Google adapters once their implementations are enhanced


# Print Total Cost
print("--- Total Cost ---")
print(f"Total estimated cost: ${cost_tracker.get_total_cost():.6f}")
print("-" * 20)
