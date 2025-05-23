import unittest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field
from typing import Optional, Any
import tempfile # For creating temporary files in tests
import os # For path joining

from src.ai_helper import AiHelper
from src.cost_tracker import CostTracker
from src.adapters.base_adapter import BaseAdapter
# Import actual adapter classes for isinstance checks and mocking their instantiation
from src.adapters.openai import OpenAIAdapter
from src.adapters.google import GoogleAdapter
from src.adapters.anthropic import AnthropicAdapter
from src.adapters.openrouter import OpenRouterAdapter
from src.tools import calculator # Import a real tool for testing

# Base Pydantic model for testing percentage filled property
class PercentageFilledBaseModel(BaseModel):
    @property
    def model_fields_filled_percentage(self) -> float:
        # Access model_fields from the class, not the instance
        model_fields = self.__class__.model_fields
        if not model_fields:
            return 0.0
        
        filled_count = 0
        # A field is considered "filled" if its value is not None.
        for field_name in model_fields.keys():
            if getattr(self, field_name, None) is not None: # Added default to getattr
                filled_count += 1
        
        return (filled_count / len(model_fields)) * 100.0

# A simple Pydantic model for testing basic successful parsing
class SimpleTestModel(PercentageFilledBaseModel):
    name: str
    value: Optional[int] = None
    description: str

# A Pydantic model with all optional fields to test discarding and partial filling
class OptionalFieldsModel(PercentageFilledBaseModel):
    name: Optional[str] = None
    count: Optional[int] = None
    flag: Optional[bool] = None

# Mock adapter for testing 'ask' method behavior
class MockAdapterForAskTests(BaseAdapter):
    def __init__(self, model_name: str = "mock_model", cost_tracker: Optional[CostTracker] = None):
        # Call super().__init__ if BaseAdapter's __init__ does actual work
        super().__init__(model_name, cost_tracker) 
        self.model_name = model_name # Redundant if super does it, but fine for mock
        self.cost_tracker = cost_tracker 
        self.process_fn = MagicMock(return_value={
            "content": {"text": "Default mock adapter response"},
            "cost_info": {"tokens_used": 10, "cost": 0.0001}
        })

    def process(self, input_data: Any) -> Any:
        return self.process_fn(input_data)

class TestAiHelper(unittest.TestCase):
    def setUp(self):
        self.cost_tracker = MagicMock(spec=CostTracker)

    # Adapter Initialization Tests (Integration-style: testing real adapter instantiation)
    def test_ai_helper_creation_openai(self):
        helper = AiHelper("openai:gpt-3.5-turbo", cost_tracker=self.cost_tracker)
        self.assertIsInstance(helper.adapter, OpenAIAdapter)
        self.assertEqual(helper.adapter.model_name, "gpt-3.5-turbo")
        self.assertEqual(helper.adapter.cost_tracker, helper.cost_tracker)

    def test_ai_helper_creation_google(self):
        helper = AiHelper("google:gemini-1.5-flash", cost_tracker=self.cost_tracker)
        self.assertIsInstance(helper.adapter, GoogleAdapter)
        self.assertEqual(helper.adapter.model_name, "gemini-1.5-flash")
        self.assertEqual(helper.adapter.cost_tracker, helper.cost_tracker)

    def test_ai_helper_creation_anthropic(self):
        helper = AiHelper("anthropic:claude-3-opus", cost_tracker=self.cost_tracker)
        self.assertIsInstance(helper.adapter, AnthropicAdapter)
        self.assertEqual(helper.adapter.model_name, "claude-3-opus")
        self.assertEqual(helper.adapter.cost_tracker, helper.cost_tracker)

    def test_ai_helper_creation_openrouter(self):
        helper = AiHelper("openrouter:google/gemini-1.5-pro", cost_tracker=self.cost_tracker)
        self.assertIsInstance(helper.adapter, OpenRouterAdapter)
        self.assertEqual(helper.adapter.model_name, "google/gemini-1.5-pro")
        self.assertEqual(helper.adapter.cost_tracker, helper.cost_tracker)

    def test_ai_helper_creation_invalid_identifier_format(self):
        with self.assertRaisesRegex(ValueError, "Model identifier 'invalididentifier' is not in 'provider:model_name' format."):
            AiHelper("invalididentifier", cost_tracker=self.cost_tracker)

    def test_ai_helper_creation_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "Unknown provider: unknownprovider"):
            AiHelper("unknownprovider:model-x", cost_tracker=self.cost_tracker)

    # Utility to get AiHelper with a fully mocked adapter for 'ask' tests
    def _get_helper_with_mocked_adapter(self):
        patcher = patch.object(AiHelper, '_initialize_adapter', autospec=True)
        mock_init_adapter = patcher.start()
        self.addCleanup(patcher.stop)

        helper = AiHelper("dummy:dummy", cost_tracker=self.cost_tracker)
        helper.adapter = MockAdapterForAskTests(cost_tracker=self.cost_tracker)
        mock_init_adapter.assert_called_once()
        return helper

    # Tool Management Tests
    def test_add_tool(self):
        helper = self._get_helper_with_mocked_adapter()
        mock_tool_func = MagicMock()
        helper.add_tool(name="test_tool", description="A test tool.", func=mock_tool_func)
        self.assertIn("test_tool", helper.available_tools)
        self.assertEqual(helper.available_tools["test_tool"]["description"], "A test tool.")
        self.assertEqual(helper.available_tools["test_tool"]["func"], mock_tool_func)

    # Ask Method Tests
    def test_ask_method_basic_no_model(self):
        helper = self._get_helper_with_mocked_adapter()
        expected_response_text = "LLM response for basic ask"
        helper.adapter.process_fn.return_value = {
            "content": {"text": expected_response_text},
            "cost_info": {"tokens_used": 20, "cost": 0.0002}
        }
        prompt = "Hello, world!"
        response = helper.ask(prompt)
        
        helper.adapter.process_fn.assert_called_once()
        call_args = helper.adapter.process_fn.call_args[0][0]
        
        self.assertEqual(len(call_args["messages"]), 1)
        self.assertEqual(call_args["messages"][0]["role"], "user")
        self.assertEqual(call_args["messages"][0]["content"], prompt)
        self.assertIsNone(call_args.get("pydantic_model_json_schema"))
        self.assertEqual(call_args.get("tools_details"), []) 
        self.assertIsNone(call_args.get("file_content"))
        
        self.assertEqual(response, expected_response_text)
        self.cost_tracker.track_cost.assert_called_once_with({"tokens_used": 20, "cost": 0.0002})

    def test_ask_method_with_pydantic_model_success(self):
        helper = self._get_helper_with_mocked_adapter()
        llm_output_json_str = '{"name": "Test Name", "value": 123, "description": "Test Description"}'
        helper.adapter.process_fn.return_value = {
            "content": {"text": llm_output_json_str},
            "cost_info": {"tokens_used": 30, "cost": 0.0003}
        }
        result = helper.ask("Get SimpleTestModel data", pydantic_model=SimpleTestModel)
        
        self.assertIsInstance(result, SimpleTestModel)
        self.assertEqual(result.name, "Test Name")
        self.assertEqual(result.value, 123)
        self.assertEqual(result.description, "Test Description")
        self.assertAlmostEqual(result.model_fields_filled_percentage, 100.0)
        self.cost_tracker.track_cost.assert_called_once_with({"tokens_used": 30, "cost": 0.0003})

    def test_ask_method_with_pydantic_model_partial_validation_discard_fields(self):
        helper = self._get_helper_with_mocked_adapter()
        llm_output_json_str_partial = '{"name": "Optional Test", "count": "not_a_number", "extra_field": "ignored_value"}'
        helper.adapter.process_fn.return_value = {
            "content": {"text": llm_output_json_str_partial},
            "cost_info": {"tokens_used": 40, "cost": 0.0004}
        }
        result = helper.ask("Get partial OptionalFieldsModel data", pydantic_model=OptionalFieldsModel)

        self.assertIsInstance(result, OptionalFieldsModel)
        self.assertEqual(result.name, "Optional Test", "Field 'name' should be correctly parsed.")
        self.assertIsNone(result.count, "Field 'count' with invalid type should be discarded (become None).")
        self.assertIsNone(result.flag, "Missing optional field 'flag' should be None.")
        self.assertAlmostEqual(result.model_fields_filled_percentage, (1/3) * 100.0, places=5)
        self.cost_tracker.track_cost.assert_called_once_with({"tokens_used": 40, "cost": 0.0004})

    def test_ask_method_pydantic_model_empty_llm_response(self):
        helper = self._get_helper_with_mocked_adapter()
        helper.adapter.process_fn.return_value = {
            "content": {"text": ""}, 
            "cost_info": {"tokens_used": 5, "cost": 0.00005}
        }
        result = helper.ask("Query", pydantic_model=OptionalFieldsModel)
        self.assertIsInstance(result, OptionalFieldsModel)
        self.assertIsNone(result.name)
        self.assertIsNone(result.count)
        self.assertIsNone(result.flag)
        self.assertAlmostEqual(result.model_fields_filled_percentage, 0.0)

    def test_ask_method_pydantic_model_malformed_json_response(self):
        helper = self._get_helper_with_mocked_adapter()
        helper.adapter.process_fn.return_value = {
            "content": {"text": "{'name': 'Test"}, 
            "cost_info": {"tokens_used": 6, "cost": 0.00006}
        }
        result = helper.ask("Query", pydantic_model=OptionalFieldsModel)
        self.assertIsInstance(result, OptionalFieldsModel)
        self.assertIsNone(result.name)
        self.assertIsNone(result.count)
        self.assertIsNone(result.flag)
        self.assertAlmostEqual(result.model_fields_filled_percentage, 0.0)
        self.cost_tracker.track_cost.assert_called_once_with({"tokens_used": 6, "cost": 0.00006})

    def test_ask_method_with_tool_calling_flow(self):
        helper = self._get_helper_with_mocked_adapter()
        helper.add_tool(name="calculator", description="A simple calculator.", func=calculator)

        mock_tool_call_id = "call_123"
        llm_response_turn1 = {
            "content": {
                "text": None, 
                "tool_calls": [{
                    "id": mock_tool_call_id, "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression": "2 + 2"}'}
                }]
            }, "cost_info": {"tokens_used": 50, "cost": 0.0005}
        }
        final_text_response = "The result of 2 + 2 is 4."
        llm_response_turn2 = {
            "content": {"text": final_text_response, "tool_calls": []}, 
            "cost_info": {"tokens_used": 25, "cost": 0.00025}
        }
        helper.adapter.process_fn.side_effect = [llm_response_turn1, llm_response_turn2]
        
        prompt = "What is 2 + 2 using the calculator?"
        response = helper.ask(prompt, tools=["calculator"])

        self.assertEqual(helper.adapter.process_fn.call_count, 2)
        
        call1_args = helper.adapter.process_fn.call_args_list[0][0][0]
        self.assertEqual(call1_args["messages"][0]["role"], "user")
        self.assertEqual(call1_args["messages"][0]["content"], prompt)
        self.assertTrue(len(call1_args["tools_details"]) == 1)
        self.assertEqual(call1_args["tools_details"][0]["function"]["name"], "calculator")
        self.assertIn("expression", call1_args["tools_details"][0]["function"]["parameters"]["properties"])

        call2_args = helper.adapter.process_fn.call_args_list[1][0][0]
        self.assertEqual(len(call2_args["messages"]), 3)
        self.assertEqual(call2_args["messages"][2]["role"], "tool")
        self.assertEqual(call2_args["messages"][2]["tool_call_id"], mock_tool_call_id)
        self.assertEqual(call2_args["messages"][2]["name"], "calculator")
        self.assertEqual(call2_args["messages"][2]["content"], "4")

        self.assertEqual(response, final_text_response)
        self.assertEqual(self.cost_tracker.track_cost.call_count, 2)
        self.cost_tracker.track_cost.assert_any_call({"tokens_used": 50, "cost": 0.0005})
        self.cost_tracker.track_cost.assert_any_call({"tokens_used": 25, "cost": 0.00025})

    def test_ask_method_with_file_input(self):
        helper = self._get_helper_with_mocked_adapter()
        file_content_str = "This is the content of the test file."
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(file_content_str)
            tmp_file_path = tmp_file.name
        
        self.addCleanup(os.remove, tmp_file_path)

        expected_response_text = "LLM response regarding the file."
        helper.adapter.process_fn.return_value = {
            "content": {"text": expected_response_text},
            "cost_info": {"tokens_used": 70, "cost": 0.0007}
        }
        prompt = "Summarize this file."
        response = helper.ask(prompt, file_path=tmp_file_path)

        helper.adapter.process_fn.assert_called_once()
        call_args = helper.adapter.process_fn.call_args[0][0]

        self.assertEqual(call_args["messages"][0]["content"], prompt)
        self.assertIsNotNone(call_args.get("file_content"))
        self.assertEqual(call_args["file_content"]["filename"], os.path.basename(tmp_file_path))
        self.assertEqual(call_args["file_content"]["content_bytes"], file_content_str.encode('utf-8'))
        
        self.assertEqual(response, expected_response_text)
        self.cost_tracker.track_cost.assert_called_once_with({"tokens_used": 70, "cost": 0.0007})

    # More tests to be added later for:
    # - Complex Pydantic model structures (nested models, lists)
    # - Error handling from adapters (e.g. if LLM fails)
    # - Specific adapter behaviors for file inputs (e.g. image summarization)

if __name__ == '__main__':
    unittest.main()
