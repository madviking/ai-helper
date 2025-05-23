import unittest
from unittest.mock import patch, MagicMock, mock_open

from src.ai_helper import AiHelper, MissingEnvVariableError
from src.adapters.base_adapter import BaseAdapter
from src.adapters.openai import OpenAIAdapter
# Assuming these mock adapters are defined in src.ai_helper for testing this module,
# or they are proper classes in their respective adapter files.
# For this test, we will mock them if they are not OpenAIAdapter.
from src.openrouter_client import OpenRouterClient
from src.cost_tracker import CostTracker

# Define placeholder adapters locally for testing if not available from src.adapters
class MockGoogleAdapter(BaseAdapter):
    def __init__(self, api_key: str, model: str): self.api_key, self.model = api_key, model
    def process(self, input_data): raise NotImplementedError("MockGoogleAdapter.process")
class MockAnthropicAdapter(BaseAdapter):
    def __init__(self, api_key: str, model: str): self.api_key, self.model = api_key, model
    def process(self, input_data): raise NotImplementedError("MockAnthropicAdapter.process")
class MockOpenRouterAdapter(BaseAdapter):
    def __init__(self, api_key: str, model: str, openrouter_client): self.api_key, self.model, self.client = api_key, model, openrouter_client
    def process(self, input_data): raise NotImplementedError("MockOpenRouterAdapter.process")

class TestAiHelperTools(unittest.TestCase):
    def setUp(self):
        self.mock_orc = MagicMock(spec=OpenRouterClient)
        self.mock_ct = MagicMock(spec=CostTracker)
        
        # Patch _initialize_adapter for tool tests to avoid dealing with its complexities here
        with patch.object(AiHelper, '_initialize_adapter', return_value=None) as mock_init_adapter:
            self.helper = AiHelper(model_identifier="some:model", cost_tracker=self.mock_ct, openrouter_client=self.mock_orc)
        self.mock_init_adapter_in_setup = mock_init_adapter # Store for assertions if needed

    def test_add_single_tool(self):
        tool_name = "calculator"
        tool_description = "Calculates mathematical expressions."
        def tool_func(expression): return eval(expression)
        self.helper.add_tool(name=tool_name, description=tool_description, func=tool_func)
        self.assertIn(tool_name, self.helper.available_tools)
        self.assertEqual(self.helper.available_tools[tool_name]["description"], tool_description)
        self.assertEqual(self.helper.available_tools[tool_name]["func"], tool_func)
        self.assertEqual(self.helper.available_tools[tool_name]["func"]("2+2"), 4)

    def test_add_multiple_tools(self):
        def calc_func(expr): return eval(expr)
        def weather_func(city): return f"Weather in {city} is sunny."
        tools_to_add = [
            {"name": "calculator", "description": "Math tool", "func": calc_func},
            {"name": "weather", "description": "Weather tool", "func": weather_func}
        ]
        for tool in tools_to_add:
            self.helper.add_tool(name=tool["name"], description=tool["description"], func=tool["func"])
        self.assertEqual(len(self.helper.available_tools), 2)
        for tool in tools_to_add:
            self.assertIn(tool["name"], self.helper.available_tools)

    def test_add_tool_overwrites_existing(self):
        tool_name = "shared_tool"
        def old_func(x): return x
        def new_func(x): return x * 2
        self.helper.add_tool(name=tool_name, description="Old description", func=old_func)
        self.helper.add_tool(name=tool_name, description="New description", func=new_func) # Overwrite
        self.assertEqual(len(self.helper.available_tools), 1)
        self.assertEqual(self.helper.available_tools[tool_name]["description"], "New description")
        self.assertEqual(self.helper.available_tools[tool_name]["func"], new_func)

@patch('src.ai_helper.GoogleAdapter', new_callable=lambda: MockGoogleAdapter)
@patch('src.ai_helper.AnthropicAdapter', new_callable=lambda: MockAnthropicAdapter)
@patch('src.ai_helper.OpenRouterAdapter', new_callable=lambda: MockOpenRouterAdapter)
class TestAiHelperInitializeAdapter(unittest.TestCase):
    def setUp(self):
        self.mock_openrouter_client_instance = MagicMock(spec=OpenRouterClient)
        # Patch OpenRouterClient constructor to return our mock instance
        self.patch_orc_constructor = patch('src.ai_helper.OpenRouterClient', return_value=self.mock_openrouter_client_instance)
        self.MockORCConstructor = self.patch_orc_constructor.start()
        
        self.mock_cost_tracker_instance = MagicMock(spec=CostTracker)
        self.patch_ct_constructor = patch('src.ai_helper.CostTracker', return_value=self.mock_cost_tracker_instance)
        self.MockCTConstructor = self.patch_ct_constructor.start()

    def tearDown(self):
        self.patch_orc_constructor.stop()
        self.patch_ct_constructor.stop()

    @patch('src.utils.get_env_variable')
    @patch('src.ai_helper.OpenAIAdapter') # Mock the real OpenAIAdapter
    def test_openai_success(self, MockOpenAIAdapter, mock_get_env_var, MockORAdapter, MockAnthAdapter, MockGoogAdapter):
        mock_get_env_var.return_value = "test_openai_key"
        mock_openai_adapter_instance = MockOpenAIAdapter.return_value
        
        helper = AiHelper("openai:gpt-3.5-turbo", openrouter_client=self.mock_openrouter_client_instance)
        
        mock_get_env_var.assert_called_once_with("OPENAI_API_KEY")
        MockOpenAIAdapter.assert_called_once_with(api_key="test_openai_key", model="gpt-3.5-turbo")
        self.assertIsInstance(helper.adapter, mock_openai_adapter_instance.__class__) # Check instance type

    @patch('src.utils.get_env_variable')
    def test_openrouter_success(self, mock_get_env_var, MockORAdapter, MockAnthAdapter, MockGoogAdapter):
        mock_get_env_var.return_value = "test_or_key"
        self.mock_openrouter_client_instance.get_model_details.return_value = {"id": "vendor/model", "name": "Vendor Model"}
        mock_or_adapter_instance = MockORAdapter.return_value

        helper = AiHelper("openrouter:vendor/model", openrouter_client=self.mock_openrouter_client_instance)
        
        mock_get_env_var.assert_called_once_with("OPENROUTER_API_KEY")
        self.mock_openrouter_client_instance.get_model_details.assert_called_once_with("vendor/model")
        MockORAdapter.assert_called_once_with(api_key="test_or_key", model="vendor/model", openrouter_client=self.mock_openrouter_client_instance)
        self.assertIsInstance(helper.adapter, mock_or_adapter_instance.__class__)

    @patch('src.utils.get_env_variable')
    def test_openrouter_model_not_found(self, mock_get_env_var, MockORAdapter, MockAnthAdapter, MockGoogAdapter):
        mock_get_env_var.return_value = "test_or_key"
        self.mock_openrouter_client_instance.get_model_details.return_value = None # Model not found
        
        with self.assertRaisesRegex(ValueError, "Model 'vendor/nonexistent' not found or supported by OpenRouter."):
            AiHelper("openrouter:vendor/nonexistent", openrouter_client=self.mock_openrouter_client_instance)
        mock_get_env_var.assert_called_once_with("OPENROUTER_API_KEY")
        self.mock_openrouter_client_instance.get_model_details.assert_called_once_with("vendor/nonexistent")

    @patch('src.utils.get_env_variable')
    def test_unsupported_provider(self, mock_get_env_var, MockORAdapter, MockAnthAdapter, MockGoogAdapter):
        with self.assertRaisesRegex(ValueError, "Unsupported model provider: unknown"):
            AiHelper("unknown:model", openrouter_client=self.mock_openrouter_client_instance)
        mock_get_env_var.assert_not_called()

    @patch('src.utils.get_env_variable', side_effect=MissingEnvVariableError("OPENAI_API_KEY not found"))
    def test_missing_api_key(self, mock_get_env_var, MockORAdapter, MockAnthAdapter, MockGoogAdapter):
        with self.assertRaises(MissingEnvVariableError) as context:
            AiHelper("openai:gpt-3.5-turbo", openrouter_client=self.mock_openrouter_client_instance)
        self.assertIn("API key error for openai: OPENAI_API_KEY not found", str(context.exception))
        mock_get_env_var.assert_called_once_with("OPENAI_API_KEY")

    @patch('src.utils.get_env_variable')
    def test_invalid_identifier_format_no_colon(self, mock_get_env_var, MockORAdapter, MockAnthAdapter, MockGoogAdapter):
        with self.assertRaisesRegex(ValueError, "Invalid model_identifier format. Expected 'provider:model_name'."):
            AiHelper("openaigpt-3.5-turbo", openrouter_client=self.mock_openrouter_client_instance)
    
    @patch('src.utils.get_env_variable')
    def test_invalid_identifier_format_empty_model(self, mock_get_env_var, MockORAdapter, MockAnthAdapter, MockGoogAdapter):
        with self.assertRaisesRegex(ValueError, "Invalid model_identifier format. Model name cannot be empty after provider."):
            AiHelper("openai:", openrouter_client=self.mock_openrouter_client_instance)

    @patch('src.utils.get_env_variable')
    def test_google_adapter_mocked_success(self, mock_get_env_var, MockORAdapter, MockAnthAdapter, MockGoogAdapter):
        mock_get_env_var.return_value = "test_google_key"
        mock_goog_adapter_instance = MockGoogAdapter.return_value

        helper = AiHelper("google:gemini-pro", openrouter_client=self.mock_openrouter_client_instance)
        
        mock_get_env_var.assert_called_once_with("GOOGLE_API_KEY")
        MockGoogAdapter.assert_called_once_with(api_key="test_google_key", model="gemini-pro")
        self.assertIsInstance(helper.adapter, mock_goog_adapter_instance.__class__)

    @patch('src.utils.get_env_variable')
    def test_anthropic_adapter_mocked_success(self, mock_get_env_var, MockORAdapter, MockAnthAdapter, MockGoogAdapter):
        mock_get_env_var.return_value = "test_anthropic_key"
        mock_anth_adapter_instance = MockAnthAdapter.return_value
        
        helper = AiHelper("anthropic:claude-2", openrouter_client=self.mock_openrouter_client_instance)
        
        mock_get_env_var.assert_called_once_with("ANTHROPIC_API_KEY")
        MockAnthAdapter.assert_called_once_with(api_key="test_anthropic_key", model="claude-2")
        self.assertIsInstance(helper.adapter, mock_anth_adapter_instance.__class__)


if __name__ == '__main__':
    unittest.main()
