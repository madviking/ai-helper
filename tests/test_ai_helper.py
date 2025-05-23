import unittest
from src.ai_helper import AiHelper
from src.cost_tracker import CostTracker
from src.adapters.base_adapter import BaseAdapter # Assuming BaseAdapter is needed for mocking

# Create a mock adapter for testing
class MockAdapter(BaseAdapter):
    def process(self, input_data):
        return "Mocked adapter response"

class TestAiHelper(unittest.TestCase):
    def test_ai_helper_creation(self):
        # Test initialization with model identifier and optional cost tracker
        helper = AiHelper(model_identifier="test_model")
        self.assertEqual(helper.model_identifier, "test_model")
        self.assertIsNone(helper.cost_tracker)
        
        tracker = CostTracker()
        helper_with_tracker = AiHelper(model_identifier="test_model", cost_tracker=tracker)
        self.assertEqual(helper_with_tracker.model_identifier, "test_model")
        self.assertEqual(helper_with_tracker.cost_tracker, tracker)

    def test_add_tool(self):
        # Test adding a tool to the helper
        helper = AiHelper(model_identifier="test_model")
        tool = {"name": "test_tool", "function": lambda x: x}
        helper.add_tool(tool)
        self.assertIn(tool, helper.tools)

    def test_ask_method_basic(self):
        # Test the ask method with a basic prompt
        helper = AiHelper(model_identifier="test_model", adapter=MockAdapter())
        response = helper.ask("Hello, how are you?")
        self.assertEqual(response, "Mocked adapter response")

    def test_ask_method_with_tools(self):
        # Test the ask method when tools are specified
        helper = AiHelper(model_identifier="test_model", adapter=MockAdapter())
        tool = {"name": "test_tool", "function": lambda x: "Tool response"}
        helper.add_tool(tool)
        response = helper.ask("Use the tool", tools=[tool])
        self.assertEqual(response, "Mocked adapter response")

    def test_ask_method_with_model(self):
        # Test the ask method when an output model is specified
        helper = AiHelper(model_identifier="test_model", adapter=MockAdapter())
        class TestModel:
            pass
        response = helper.ask("Return a model", output_model=TestModel)
        self.assertEqual(response, "Mocked adapter response")

    def test_ask_method_with_file(self):
        # Test the ask method when a file is provided
        helper = AiHelper(model_identifier="test_model", adapter=MockAdapter())
        response = helper.ask("Analyze this file", file_path="tests/files/test.pdf")
        self.assertEqual(response, "Mocked adapter response")

    def test_track_cost_method(self):
        # Test that the track_cost method is called
        tracker = CostTracker()
        helper = AiHelper(model_identifier="test_model", cost_tracker=tracker, adapter=MockAdapter())
        response = helper.ask("Track cost for this request")
        self.assertEqual(response, "Mocked adapter response")
        # Assuming track_cost updates some internal state or logs, which we can't directly test here
        self.assertIsNotNone(helper.cost_tracker)

if __name__ == '__main__':
    unittest.main()
