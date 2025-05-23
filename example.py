"""
Example usage of the AIHelper class.
"""
import os
from dotenv import load_dotenv

from src.ai_helper import AiHelper
from py_models.general_example_model import GeneralExampleModel
from py_models.weather_model import WeatherModel
from src.tools import calculator as calculator_func, weather as weather_func
# The pdf_reader tool is a placeholder, so we won't use it in this example directly
# from src.tools import pdf_reader as pdf_reader_func

# Load .env file. AiHelper and tools.py also do this, but good practice here too.
load_dotenv() 

print("===================================")
print("AI Helper Example Script")
print("===================================")

# --- Example 1: Tool Usage ---
print("\\n--- Example 1: Tool Usage ---")
# Ensure you have relevant API keys (e.g., OPENROUTER_API_KEY, WEATHER_API_KEY) in your .env file.
# Using OpenRouter with a free model as an example.
# If WEATHER_API_KEY is not set, the weather tool will return an error message.
try:
    # Using a model known for good tool use, via OpenRouter (many are free)
    ai_helper_tools = AiHelper('openrouter:mistralai/mistral-7b-instruct') 
    
    ai_helper_tools.add_tool(
        name="calculator", 
        description="A simple calculator that can add, subtract, multiply, and divide.", 
        func=calculator_func
    )
    ai_helper_tools.add_tool(
        name="weather", 
        description="A tool to get the current weather information for a given location. Requires WEATHER_API_KEY.", 
        func=weather_func
    )
    
    # Example 1a: Calculator
    calc_query = "What is 125 + 375 / 2?"
    print(f"\\nAsking (Calculator): {calc_query}")
    calc_result = ai_helper_tools.ask(calc_query, tools=["calculator"]) # No Pydantic model, expect string
    print(f"Calculator Result: {calc_result}")

    # Example 1b: Weather
    weather_query = "What is the weather like in Paris?"
    print(f"\\nAsking (Weather): {weather_query}")
    # The 'tools' parameter tells AiHelper which tools are available for the LLM.
    weather_result_obj = ai_helper_tools.ask(weather_query, tools=["weather"], pydantic_model=WeatherModel)

    print("Weather Result:")
    if isinstance(weather_result_obj, WeatherModel):
        # Check if fields are populated before accessing
        loc = getattr(weather_result_obj, 'location', "N/A")
        temp_c = getattr(weather_result_obj, 'temperature_celsius', "N/A")
        cond = getattr(weather_result_obj, 'condition', "N/A")
        hum = getattr(weather_result_obj, 'humidity_percent', "N/A")
        filled_perc = getattr(weather_result_obj, 'model_fields_filled_percentage', 0.0)

        print(f"  Location: {loc}")
        print(f"  Temperature: {temp_c}°C")
        print(f"  Condition: {cond}")
        print(f"  Humidity: {hum}%")
        print(f"  Filled: {filled_perc:.2f}%")
        if filled_perc < 10.0: # Arbitrary threshold to indicate likely parsing/mock issue
            print(f"  (Note: Low fill percentage, LLM response might not have matched model: {weather_result_obj.model_dump_json(indent=2)})")
    elif isinstance(weather_result_obj, str) and "Error: WEATHER_API_KEY not configured" in weather_result_obj:
        print(f"  Weather tool error: {weather_result_obj}")
        print("  INFO: To use the weather tool, set WEATHER_API_KEY in your .env file.")
    else:
        print(f"  Raw Response: {weather_result_obj}")

except Exception as e:
    print(f"Error in Example 1 (Tool Usage): {e}")
    print("Ensure your API keys are set in .env and models are accessible.")

print("\\n-----------------------------\\n")


# --- Example 2: File Processing & Model Variations ---
print("--- Example 2: File Processing & Model Variations ---")
# This part tests various models with file inputs.
# Note: Real file processing (PDF, PNG) depends on actual adapter implementations.
# The current adapter placeholders will just mock responses.

# Using a smaller, more focused list for demonstration.
# Add your desired models and ensure API keys are set in .env
models_to_test = [
    'openrouter:openai/gpt-3.5-turbo',       # General purpose, good with text
    # 'openrouter:google/gemini-flash',      # Fast, multimodal - uncomment if API key available
    # 'anthropic:claude-3-haiku',            # Fast, good text - uncomment if API key available
    # 'openai:gpt-4-turbo-preview',          # Powerful, vision - uncomment if API key available
]

# Check if placeholder files exist
pdf_file_exists = os.path.exists("files/example.pdf")
png_file_exists = os.path.exists("files/example.png")

if not pdf_file_exists:
    print("WARNING: files/example.pdf not found. Skipping PDF tests.")
if not png_file_exists:
    print("WARNING: files/example.png not found. Skipping PNG tests.")


for model_id_str in models_to_test:
    print(f"\\n--- Testing Model: {model_id_str} ---")
    try:
        ai_helper_files = AiHelper(model_id_str)

        # Test with PDF file
        if pdf_file_exists:
            print(f"Asking to summarize PDF: files/example.pdf")
            pdf_summary_result = ai_helper_files.ask(
                "Please read this PDF and summarize it.", 
                pydantic_model=GeneralExampleModel, 
                file_path="files/example.pdf"
            )
            print("PDF Summary Result:")
            if isinstance(pdf_summary_result, GeneralExampleModel):
                summary = getattr(pdf_summary_result, 'summary', "N/A")
                keywords = getattr(pdf_summary_result, 'keywords', [])
                language = getattr(pdf_summary_result, 'language', "N/A")
                filled_perc_pdf = getattr(pdf_summary_result, 'model_fields_filled_percentage', 0.0)

                print(f"  Summary: {summary}")
                print(f"  Keywords: {keywords}")
                print(f"  Language: {language}")
                print(f"  Filled: {filled_perc_pdf:.2f}%")
                if filled_perc_pdf < 10.0:
                    print(f"  (Note: Low fill percentage, LLM response might not have matched model: {pdf_summary_result.model_dump_json(indent=2)})")
            else:
                print(f"  Raw Response: {pdf_summary_result}")
        
        # Test with PNG file (only for models that are not explicitly Anthropic, as per original example logic)
        # This also depends on the specific model's vision capabilities.
        # The 'anthropic' string check is a simplification.
        if png_file_exists and 'anthropic' not in model_id_str: # Basic check from original example
            print(f"\\nAsking to describe PNG: files/example.png")
            png_desc_result = ai_helper_files.ask(
                "Please describe this image.", 
                pydantic_model=GeneralExampleModel, # Using GeneralExampleModel for image description too
                file_path="files/example.png"
            )
            print("PNG Description Result:")
            if isinstance(png_desc_result, GeneralExampleModel):
                description = getattr(png_desc_result, 'summary', "N/A") # Using summary for description
                keywords_png = getattr(png_desc_result, 'keywords', [])
                filled_perc_png = getattr(png_desc_result, 'model_fields_filled_percentage', 0.0)

                print(f"  Description: {description}") 
                print(f"  Keywords: {keywords_png}")
                print(f"  Filled: {filled_perc_png:.2f}%")
                if filled_perc_png < 10.0:
                     print(f"  (Note: Low fill percentage, LLM response might not have matched model: {png_desc_result.model_dump_json(indent=2)})")
            else:
                print(f"  Raw Response: {png_desc_result}")
        elif 'anthropic' in model_id_str and png_file_exists:
            print("\\nSkipping PNG test for Anthropic model as per original example logic (may not support direct image file input this way).")


        # Test with a simple text prompt
        print(f"\\nAsking with simple text prompt...")
        text_result = ai_helper_files.ask(
            "The key is 'dog' and the value for that is 'Roger'. Extract this.", 
            pydantic_model=GeneralExampleModel
        )
        print("Text Prompt Result:")
        if isinstance(text_result, GeneralExampleModel):
            summary_text = getattr(text_result, 'summary', "N/A")
            keywords_text = getattr(text_result, 'keywords', [])
            language_text = getattr(text_result, 'language', "N/A")
            filled_perc_text = getattr(text_result, 'model_fields_filled_percentage', 0.0)

            print(f"  Summary: {summary_text}")
            print(f"  Keywords: {keywords_text}")
            print(f"  Language: {language_text}")
            print(f"  Filled: {filled_perc_text:.2f}%")
            if filled_perc_text < 10.0:
                print(f"  (Note: Low fill percentage, LLM response might not have matched model: {text_result.model_dump_json(indent=2)})")
        else:
            print(f"  Raw Response: {text_result}")

    except Exception as e:
        print(f"Error testing model {model_id_str}: {e}")
        print("  Ensure API key for this model provider is set in .env and the model is accessible.")
    print("-------------------------")

print("\\n===================================")
print("Example Script Finished")
print("===================================")
