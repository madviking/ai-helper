import unittest
import os
from dotenv import load_dotenv

# Load environment variables from .env file for integration tests
load_dotenv()

from src.ai_helper import AiHelper
from src.cost_tracker import CostTracker
from src.tools import calculator, weather, pdf_reader
from py_models.weather_model import WeatherModel
from py_models.pd_reader_model import PDReaderModel

# Note: These are integration tests and may require actual API keys
# and internet access. They can incur costs.

class TestIntegrations(unittest.TestCase):
    def setUp(self):
        # Initialize a CostTracker for each test
        self.cost_tracker = CostTracker()

    # TODO: Add tests for each adapter integration
    # These tests will require valid API keys in the .env file

    # @unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OpenAI API key not set")
    # def test_openai_integration(self):
    #     print("\n--- Testing OpenAI Integration ---")
    #     helper = AiHelper("openai", cost_tracker=self.cost_tracker)
    #     prompt = "Tell me a short story about a brave knight."
    #     response = helper.ask(prompt)
    #     print(f"Prompt: {prompt}")
    #     print(f"Response: {response}")
    #     self.assertIsInstance(response, str) # Assuming basic string response
    #     self.assertGreater(len(response), 0)
    #     self.assertGreater(self.cost_tracker.get_total_cost(), 0) # Check if cost was tracked

    # @unittest.skipUnless(os.environ.get("ANTHROPIC_API_KEY"), "Anthropic API key not set")
    # def test_anthropic_integration(self):
    #     print("\n--- Testing Anthropic Integration ---")
    #     helper = AiHelper("anthropic", cost_tracker=self.cost_tracker)
    #     prompt = "Write a haiku about a sunset."
    #     response = helper.ask(prompt)
    #     print(f"Prompt: {prompt}")
    #     print(f"Response: {response}")
    #     self.assertIsInstance(response, str)
    #     self.assertGreater(len(response), 0)
    #     self.assertGreater(self.cost_tracker.get_total_cost(), 0)

    # @unittest.skipUnless(os.environ.get("GOOGLE_API_KEY"), "Google API key not set")
    # def test_google_integration(self):
    #     print("\n--- Testing Google Integration ---")
    #     helper = AiHelper("google", cost_tracker=self.cost_tracker)
    #     prompt = "Explain the concept of recursion in programming."
    #     response = helper.ask(prompt)
    #     print(f"Prompt: {prompt}")
    #     print(f"Response: {response}")
    #     self.assertIsInstance(response, str)
    #     self.assertGreater(len(response), 0)
    #     # TODO: Check for cost if Google adapter implements cost tracking

    # @unittest.skipUnless(os.environ.get("OPENROUTER_API_KEY"), "OpenRouter API key not set")
    # def test_openrouter_integration(self):
    #     print("\n--- Testing OpenRouter Integration ---")
    #     helper = AiHelper("openrouter", cost_tracker=self.cost_tracker)
    #     prompt = "What is the tallest mountain in the world?"
    #     response = helper.ask(prompt)
    #     print(f"Prompt: {prompt}")
    #     print(f"Response: {response}")
    #     self.assertIsInstance(response, str)
    #     self.assertGreater(len(response), 0)
    #     self.assertGreater(self.cost_tracker.get_total_cost(), 0)


    # TODO: Add tests for tool usage through AiHelper
    # These tests will require the adapters to support tool calling

    # def test_tool_integration_calculator(self):
    #     print("\n--- Testing Tool Integration (Calculator) ---")
    #     helper = AiHelper("openai", cost_tracker=self.cost_tracker) # Choose an adapter that supports tools
    #     helper.add_tool("calculator", "A simple calculator", calculator)
    #     prompt = "Use the calculator to find the sum of 5 and 3."
    #     response = helper.ask(prompt, tools=["calculator"])
    #     print(f"Prompt: {prompt}")
    #     print(f"Response: {response}")
    #     # TODO: Assert that the response indicates tool use and contains the calculation result

    # TODO: Add tests for Pydantic model output through AiHelper
    # These tests will require the adapters to support structured output

    # def test_model_integration_weather(self):
    #     print("\n--- Testing Model Integration (WeatherModel) ---")
    #     helper = AiHelper("openai", cost_tracker=self.cost_tracker) # Choose an adapter that supports structured output
    #     prompt = "Provide the weather details for New York City."
    #     response = helper.ask(prompt, pydantic_model=WeatherModel)
    #     print(f"Prompt: {prompt}")
    #     print(f"Response: {response}")
    #     self.assertIsInstance(response, dict)
    #     self.assertIn("model", response)
    #     self.assertIn("filled_percentage", response)
    #     self.assertIsInstance(response["model"], WeatherModel)
    #     self.assertGreater(response["filled_percentage"], 0) # Expecting some fields to be filled

    # TODO: Add tests for file input through AiHelper
    # These tests will require the adapters to support file input

    # def test_file_integration_pdf_reader(self):
    #     print("\n--- Testing File Integration (PDF Reader) ---")
    #     helper = AiHelper("openai", cost_tracker=self.cost_tracker) # Choose an adapter that supports file input
    #     helper.add_tool("pdf_reader", "A tool to read PDFs", pdf_reader)
    #     pdf_file_path = "tests/files/test.pdf" # Assuming this file exists
    #     prompt = "Summarize the content of the PDF file."
    #     response = helper.ask(prompt, file=pdf_file_path, tools=["pdf_reader"])
    #     print(f"Prompt: {prompt}")
    #     print(f"File: {pdf_file_path}")
    #     print(f"Response: {response}")
    #     # TODO: Assert that the response indicates file processing or tool use and contains a summary

if __name__ == '__main__':
    unittest.main()
