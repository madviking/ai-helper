import unittest
from unittest.mock import patch, MagicMock
from src.ai_helper import AiHelper
from src.cost_tracker import CostTracker
from src.adapters.base_adapter import BaseAdapter
from pydantic import BaseModel, Field
import os

class MockAdapter(BaseAdapter):
    def process(self, input_data):
        return {"content": "Mocked response", "input_tokens": 10, "output_tokens": 20}

class TestModel(BaseModel):
    name: str = Field(default="")
    age: int = Field(default=0)
    description: str = Field(default="")

class TestAiHelper(unittest.TestCase):
    def setUp(self):
        self.model_identifier = "test_model"
        self.cost_tracker = CostTracker()
        self.adapter = MockAdapter()
        self.helper = AiHelper(model_identifier=self.model_identifier, cost_tracker=self.cost_tracker, adapter=self.adapter)
        self.test_file_path = "tests/files/test.pdf"

    def test_ai_helper_creation(self):
        # Test initialization with model identifier only
        helper = AiHelper(model_identifier=self.model_identifier)
        self.assertEqual(helper.model_identifier, self.model_identifier)
        self.assertIsNone(helper.cost_tracker)
        self.assertIsNotNone(helper.adapter)  # Adapter should be initialized automatically if not provided

        # Test initialization with cost tracker and adapter
        helper_with_tracker = AiHelper(model_identifier=self.model_identifier, cost_tracker=self.cost_tracker, adapter=self.adapter)
        self.assertEqual(helper_with_tracker.model_identifier, self.model_identifier)
        self.assertEqual(helper_with_tracker.cost_tracker, self.cost_tracker)
        self.assertEqual(helper_with_tracker.adapter, self.adapter)

    def test_add_tool(self):
        # Test adding a tool to the helper
        tool = {"name": "test_tool", "description": "A test tool", "function": lambda x: f"Tool response: {x}"}
        self.helper.add_tool(tool)
        self.assertIn(tool, self.helper.tools)
        self.assertEqual(self.helper.tools[0]["name"], "test_tool")

    def test_ask_method_basic(self):
        # Test the ask method with a basic prompt without cost tracking
        helper = AiHelper(model_identifier=self.model_identifier, adapter=self.adapter)
        response = helper.ask("Hello, how are you?")
        self.assertEqual(response, {"content": "Mocked response", "input_tokens": 10, "output_tokens": 20})

    def test_ask_method_with_cost_tracking(self):
        # Test the ask method with cost tracking
        with patch.object(self.cost_tracker, 'track_cost') as mock_track_cost:
            response = self.helper.ask("Hello, how are you?")
            self.assertEqual(response, {"content": "Mocked response", "input_tokens": 10, "output_tokens": 20})
            mock_track_cost.assert_called_once_with({"input_tokens": 10, "output_tokens": 20, "model": self.model_identifier})

    def test_ask_method_with_tools(self):
        # Test the ask method when tools are specified
        tool = {"name": "test_tool", "description": "A test tool", "function": lambda x: f"Tool response: {x}"}
        self.helper.add_tool(tool)
        with patch.object(self.helper, '_execute_tool', return_value="Tool executed") as mock_execute_tool:
            response = self.helper.ask("Use the tool", tools=[tool])
            self.assertEqual(response, {"content": "Mocked response", "input_tokens": 10, "output_tokens": 20})
            mock_execute_tool.assert_not_called()  # Since adapter is mocked, tool execution might not be triggered in test

    def test_ask_method_with_output_model(self):
        # Test the ask method with an output model, including field validation and percentage filled
        response_data = {"content": '{"name": "John", "age": 30, "description": ""}', "input_tokens": 10, "output_tokens": 20}
        with patch.object(self.adapter, 'process', return_value=response_data):
            result = self.helper.ask("Return a model", output_model=TestModel)
            self.assertIsInstance(result, dict)
            self.assertIn("model", result)
            self.assertIn("completion_percentage", result)
            self.assertEqual(result["model"].name, "John")
            self.assertEqual(result["model"].age, 30)
            self.assertEqual(result["model"].description, "")
            self.assertAlmostEqual(result["completion_percentage"], 66.67, places=2)  # 2 out of 3 fields filled

    def test_ask_method_with_invalid_model_data(self):
        # Test the ask method with invalid model data, should discard invalid fields
        response_data = {"content": '{"name": "John", "age": "invalid", "description": "A person"}', "input_tokens": 10, "output_tokens": 20}
        with patch.object(self.adapter, 'process', return_value=response_data):
            result = self.helper.ask("Return a model", output_model=TestModel)
            self.assertIsInstance(result, dict)
            self.assertIn("model", result)
            self.assertIn("completion_percentage", result)
            self.assertEqual(result["model"].name, "John")
            self.assertEqual(result["model"].age, 0)  # Invalid field should be discarded and set to default
            self.assertEqual(result["model"].description, "A person")
            self.assertAlmostEqual(result["completion_percentage"], 66.67, places=2)  # 2 out of 3 fields filled

    def test_ask_method_with_file(self):
        # Test the ask method when a file is provided
        if os.path.exists(self.test_file_path):
            response = self.helper.ask("Analyze this file", file_path=self.test_file_path)
            self.assertEqual(response, {"content": "Mocked response", "input_tokens": 10, "output_tokens": 20})
        else:
            self.skipTest(f"Test file {self.test_file_path} not found")

    def test_execute_tool(self):
        # Test the internal _execute_tool method
        tool = {"name": "test_tool", "description": "A test tool", "function": lambda x: f"Tool response: {x}"}
        self.helper.add_tool(tool)
        result = self.helper._execute_tool("test_tool", {"input": "test input"})
        self.assertEqual(result, "Tool response: test input")

    def test_validate_model(self):
        # Test the internal _validate_model method
        response = '{"name": "John", "age": 25, "description": "Test"}'
        result = self.helper._validate_model(response, TestModel)
        self.assertIsInstance(result, dict)
        self.assertIn("model", result)
        self.assertIn("completion_percentage", result)
        self.assertEqual(result["model"].name, "John")
        self.assertEqual(result["model"].age, 25)
        self.assertEqual(result["model"].description, "Test")
        self.assertAlmostEqual(result["completion_percentage"], 100.0, places=2)

if __name__ == '__main__':
    unittest.main()
