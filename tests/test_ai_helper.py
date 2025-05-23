import unittest
from unittest.mock import Mock, patch # Import Mock and patch directly
from src.ai_helper import AiHelper
from src.cost_tracker import CostTracker
from src.adapters.base_adapter import BaseAdapter # Assuming BaseAdapter is needed for mocking

import unittest
from unittest.mock import Mock, patch # Import Mock and patch directly
from src.ai_helper import AiHelper
from src.cost_tracker import CostTracker
from src.adapters.base_adapter import BaseAdapter # Assuming BaseAdapter is needed for mocking
from py_models.weather_model import WeatherModel # Import WeatherModel for testing
from pydantic import BaseModel # Import BaseModel for creating a dummy model

# Create a mock adapter for testing
class MockAdapter(BaseAdapter):
    def process(self, input_data):
        # Simulate returning a response based on input_data
        response = {}
        if "prompt" in input_data:
            response["model_output"] = f"Mocked response for: {input_data['prompt']}"
        if "tools" in input_data:
             # Simulate a tool call response if tools are available
             response["tool_calls"] = [{"name": "mock_tool", "args": {"arg": "test"}}] # Assuming 'mock_tool' is the tool being tested
        if "output_model" in input_data:
             # Simulate returning data for the Pydantic model
             response["model_output"] = {"field1": "value", "field2": 123} # Assuming DummyOutputModel structure

        response["cost_details"] = {"cost": 0.001, "output_tokens": 10, "model": "mock_model"} # Include dummy cost details

        return response

class TestAiHelper(unittest.TestCase):
    @patch('src.ai_helper.AiHelper._initialize_adapter') # Patch _initialize_adapter during init
    def test_ai_helper_creation(self, mock_initialize_adapter):
        # Test initialization with model identifier
        helper = AiHelper("openai") # Use a recognized model identifier
        self.assertEqual(helper.model_identifier, "openai")
        self.assertIsInstance(helper.cost_tracker, CostTracker)
        self.assertEqual(helper.available_tools, {})
        self.assertIsNone(helper.adapter) # Should be None initially as _initialize_adapter is patched

        # Test initialization with model identifier and provided cost tracker
        custom_tracker = CostTracker()
        helper_with_tracker = AiHelper("google", cost_tracker=custom_tracker) # Use a recognized model identifier
        self.assertEqual(helper_with_tracker.model_identifier, "google")
        self.assertIs(helper_with_tracker.cost_tracker, custom_tracker)
        self.assertEqual(helper_with_tracker.available_tools, {})
        self.assertIsNone(helper_with_tracker.adapter)

    @patch('src.ai_helper.AiHelper._initialize_adapter') # Patch _initialize_adapter during init
    def test_add_tool(self, mock_initialize_adapter):
        helper = AiHelper("openai") # Use a recognized model identifier
        def dummy_tool_func(arg):
            return f"tool called with {arg}"

        helper.add_tool("dummy_tool", "A dummy tool", dummy_tool_func)

        self.assertIn("dummy_tool", helper.available_tools)
        tool_info = helper.available_tools["dummy_tool"]
        self.assertEqual(tool_info["description"], "A dummy tool")
        self.assertIs(tool_info["func"], dummy_tool_func)

    # Note: Comprehensive tests for the 'ask' method will require significant mocking
    # of the adapter and potentially tool execution. These tests provide a basic structure.

    @patch('src.ai_helper.AiHelper._initialize_adapter') # Patch _initialize_adapter during init
    @patch('src.ai_helper.AiHelper._track_cost') # Patch _track_cost
    def test_ask_method_basic(self, mock_track_cost, mock_initialize_adapter):
        helper = AiHelper("openai") # Use a recognized model identifier
        # Mock the adapter's process method to return a dictionary
        helper.adapter = MockAdapter()
        helper.adapter.process = Mock(return_value={"model_output": "Mocked response for basic ask", "cost_details": {"cost": 0.001, "output_tokens": 10, "model": "mock_model"}})

        prompt = "What is the capital of France?"
        response = helper.ask(prompt)

        # Assert that process was called with the input_data dictionary
        expected_input_data = {"prompt": prompt}
        helper.adapter.process.assert_called_once_with(expected_input_data)
        self.assertEqual(response, "Mocked response for basic ask")
        mock_track_cost.assert_called_once_with({"cost": 0.001, "output_tokens": 10, "model": "mock_model"})


    @patch('src.ai_helper.AiHelper._initialize_adapter') # Patch _initialize_adapter during init
    @patch('src.ai_helper.AiHelper._track_cost') # Patch _track_cost
    def test_ask_method_with_tools(self, mock_track_cost, mock_initialize_adapter):
        helper = AiHelper("openai") # Use a recognized model identifier
        helper.adapter = MockAdapter()
        # Mock the adapter to return a dictionary indicating a tool call and cost details
        helper.adapter.process = Mock(return_value={"tool_calls": [{"name": "mock_tool", "args": {"arg": "test"}}], "cost_details": {"cost": 0.002, "output_tokens": 20, "model": "mock_model"}})

        def mock_tool_func(arg):
            return f"mock tool executed with {arg}"

        helper.add_tool("mock_tool", "A mock tool", mock_tool_func)

        # Mock the internal _execute_tool method
        helper._execute_tool = Mock(return_value="mock tool executed result")

        prompt = "Use the mock_tool with argument 'test'."
        tools_list = ["mock_tool"]
        response = helper.ask(prompt, tools=tools_list)

        # Assert that process was called with the input_data dictionary including tool info
        expected_input_data = {
            "prompt": prompt,
            "tools": [{"name": "mock_tool", "description": "A mock tool"}]
        }
        helper.adapter.process.assert_called_once_with(expected_input_data)
        # Assert that _execute_tool was called with the correct arguments
        helper._execute_tool.assert_called_once_with("mock_tool", {"arg": "test"})
        # Assert that the response contains the tool results
        self.assertIsInstance(response, dict)
        self.assertIn("tool_results", response)
        self.assertIn("mock_tool", response["tool_results"])
        self.assertEqual(response["tool_results"]["mock_tool"], "mock tool executed result")
        mock_track_cost.assert_called_once_with({"cost": 0.002, "output_tokens": 20, "model": "mock_model"})


    @patch('src.ai_helper.AiHelper._initialize_adapter') # Patch _initialize_adapter during init
    @patch('src.ai_helper.AiHelper._track_cost') # Patch _track_cost
    def test_ask_method_with_model(self, mock_track_cost, mock_initialize_adapter):
        helper = AiHelper("openai") # Use a recognized model identifier
        helper.adapter = MockAdapter()
        # Mock the adapter to return data that can be validated by a Pydantic model and cost details
        class DummyOutputModel(BaseModel): # Use BaseModel here for actual Pydantic behavior
             field1: str
             field2: int

        mock_model_instance = DummyOutputModel(field1="value", field2=123)
        # Mock the adapter to return a dictionary with model_output data and cost details
        helper.adapter.process = Mock(return_value={"model_output": {"field1": "value", "field2": 123}, "cost_details": {"cost": 0.003, "output_tokens": 30, "model": "mock_model"}})

        # Mock the Pydantic model itself to return a mock instance when called
        # We don't need to mock __fields__ or validate if we use a real BaseModel in the test
        # mock_pydantic_model_class = Mock(return_value=mock_model_instance)
        # mock_pydantic_model_class.__name__ = "DummyOutputModel" # Ensure the mock class has __name__
        # mock_pydantic_model_class.__fields__ = DummyOutputModel.__fields__ # Copy fields to the mock class

        response = helper.ask("Prompt for model", pydantic_model=DummyOutputModel) # Use the real BaseModel class

        # Assert that process was called with the input_data dictionary including model info
        expected_input_data = {
            "prompt": "Prompt for model",
            "output_model": "DummyOutputModel"
        }
        helper.adapter.process.assert_called_once_with(expected_input_data)

        # Assert that the response is a dictionary with 'model' and 'filled_percentage'
        self.assertIsInstance(response, dict)
        self.assertIn("model", response)
        self.assertIn("filled_percentage", response)
        # Assert that the returned model instance is an instance of the expected model
        self.assertIsInstance(response["model"], DummyOutputModel)
        self.assertEqual(response["model"].field1, "value")
        self.assertEqual(response["model"].field2, 123)
        # Assert the filled percentage (assuming 2 out of 2 fields are valid)
        self.assertEqual(response["filled_percentage"], 100.0)
        mock_track_cost.assert_called_once_with({"cost": 0.003, "output_tokens": 30, "model": "mock_model"})


    @patch('src.ai_helper.AiHelper._initialize_adapter') # Patch _initialize_adapter during init
    @patch('builtins.open', new_callable=Mock) # Mock the open function
    @patch('os.remove', new_callable=Mock) # Mock os.remove
    @patch('src.ai_helper.AiHelper._track_cost') # Patch _track_cost
    def test_ask_method_with_file(self, mock_track_cost, mock_os_remove, mock_open, mock_initialize_adapter):
        helper = AiHelper("openai") # Use a recognized model identifier
        helper.adapter = MockAdapter()
        # Mock the adapter to return a dictionary with model_output data and cost details
        helper.adapter.process = Mock(return_value={"model_output": "Mocked response using file content", "cost_details": {"cost": 0.004, "output_tokens": 40, "model": "mock_model"}})

        # Configure the mock open to simulate reading a file
        mock_file = Mock()
        mock_file.read.return_value = "This is dummy file content."
        # Configure the mock returned by open to be a context manager
        mock_open.return_value.__enter__ = Mock(return_value=mock_file)
        mock_open.return_value.__exit__ = Mock(return_value=None)


        dummy_file_path = "tests/files/dummy_test_file.txt"

        prompt = "Analyze the content of the file."
        file_path = dummy_file_path
        response = helper.ask(prompt, file=file_path)

        # Check if open was called with the correct file path and mode
        mock_open.assert_called_once_with(file_path, "r")
        # The adapter's process should be called with the input_data dictionary including file content
        expected_input_data = {
            "prompt": prompt,
            "file_content": "This is dummy file content."
        }
        helper.adapter.process.assert_called_once_with(expected_input_data)
        # For now, just check that something is returned
        self.assertIsNotNone(response)
        self.assertEqual(response, "Mocked response using file content") # Assert the returned model_output
        mock_track_cost.assert_called_once_with({"cost": 0.004, "output_tokens": 40, "model": "mock_model"})

        # Check if os.remove was called to clean up the dummy file
        # Note: The original test created and removed the file directly.
        # With mocking, we just check if the mock was called.
        # mock_os_remove.assert_called_once_with(dummy_file_path) # This assertion is for the cleanup logic in the test itself, not the ask method

    @patch('src.ai_helper.AiHelper._initialize_adapter') # Patch _initialize_adapter during init
    @patch('src.ai_helper.CostTracker.track_cost') # Patch the actual track_cost method
    def test_track_cost_method(self, mock_cost_tracker_track_cost, mock_initialize_adapter):
        helper = AiHelper("openai") # Use a recognized model identifier
        # We are testing the _track_cost method directly here, not through ask
        cost_details = {"cost": 0.01, "output_tokens": 100, "model": "test_model"}
        helper._track_cost(cost_details)

        # Assert that the CostTracker's track_cost method was called with the correct arguments
        mock_cost_tracker_track_cost.assert_called_once_with(0.01, 100, "test_model")

if __name__ == '__main__':
    unittest.main()
