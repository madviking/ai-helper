import unittest
from unittest.mock import patch, MagicMock, call
import os
from pydantic import BaseModel
from typing import Optional

from src.ai_helper import AIHelper
from src.cost_tracker import CostTracker
from src.tools import calculator, weather, pdf_reader

# Test Pydantic model
class TestModel(BaseModel):
    name: Optional[str] = None
    value: Optional[int] = None
    description: Optional[str] = None

class TestAIHelper(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.cost_tracker = MagicMock(spec=CostTracker)
        
    def test_ai_helper_creation_with_openai(self):
        """Test AIHelper initialization with OpenAI model"""
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        
        self.assertEqual(helper.model_identifier, 'openai:gpt-3.5-turbo')
        self.assertEqual(helper.cost_tracker, self.cost_tracker)
        self.assertIsNotNone(helper.adapter)
        self.assertEqual(helper.provider, 'openai')
        self.assertEqual(helper.model, 'gpt-3.5-turbo')
    
    def test_ai_helper_creation_with_anthropic(self):
        """Test AIHelper initialization with Anthropic model"""
        helper = AIHelper('anthropic:claude-3', self.cost_tracker)
        
        self.assertEqual(helper.provider, 'anthropic')
        self.assertEqual(helper.model, 'claude-3')
    
    def test_ai_helper_creation_with_google(self):
        """Test AIHelper initialization with Google model"""
        helper = AIHelper('google:gemini-2.5-flash-preview-05-20', self.cost_tracker)
        
        self.assertEqual(helper.provider, 'google')
        self.assertEqual(helper.model, 'gemini-2.5-flash-preview-05-20')
    
    def test_ai_helper_creation_with_openrouter(self):
        """Test AIHelper initialization with OpenRouter model"""
        helper = AIHelper('openrouter:openai/gpt-3.5-turbo', self.cost_tracker)
        
        self.assertEqual(helper.provider, 'openrouter')
        self.assertEqual(helper.model, 'openai/gpt-3.5-turbo')
    
    def test_ai_helper_creation_invalid_format(self):
        """Test AIHelper initialization with invalid model format"""
        with self.assertRaises(ValueError) as context:
            AIHelper('invalid-format', self.cost_tracker)
        
        self.assertIn("Invalid model identifier format", str(context.exception))
    
    def test_ai_helper_creation_unknown_provider(self):
        """Test AIHelper initialization with unknown provider"""
        with self.assertRaises(ValueError) as context:
            AIHelper('unknown:model', self.cost_tracker)
        
        self.assertIn("Unknown provider", str(context.exception))
    
    def test_add_tool(self):
        """Test adding a tool to the helper"""
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        
        # Add tool
        helper.add_tool("calculator", "A simple calculator", calculator)
        
        # Verify tool was added
        self.assertIn("calculator", helper.available_tools)
        self.assertEqual(helper.available_tools["calculator"]["description"], "A simple calculator")
        self.assertEqual(helper.available_tools["calculator"]["func"], calculator)
    
    def test_add_tool_by_name(self):
        """Test adding a tool by name only"""
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        
        # Add tool by name
        helper.add_tool("calculator", "A simple calculator")
        
        # Verify tool was added with correct function
        self.assertIn("calculator", helper.available_tools)
        self.assertEqual(helper.available_tools["calculator"]["func"], calculator)
    
    @patch('src.ai_helper.OpenAIAdapter')
    def test_ask_method_basic(self, mock_adapter_class):
        """Test the ask method with a basic prompt"""
        # Mock adapter
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        mock_adapter.process.return_value = {
            'response': TestModel(name="test", value=42),
            'input_tokens': 10,
            'output_tokens': 20,
            'model_used': 'gpt-3.5-turbo'
        }
        
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        
        # Ask question
        result = helper.ask("Test prompt", model=TestModel)
        
        # Verify adapter was called correctly
        mock_adapter.process.assert_called_once()
        call_args = mock_adapter.process.call_args[0][0]
        self.assertEqual(call_args['prompt'], "Test prompt")
        self.assertEqual(call_args['model'], 'gpt-3.5-turbo')
        self.assertEqual(call_args['pydantic_model'], TestModel)
        
        # Verify result
        self.assertIsInstance(result, TestModel)
        self.assertEqual(result.name, "test")
        self.assertEqual(result.value, 42)
        
        # Verify cost tracking
        self.cost_tracker.add_cost.assert_called_once_with(
            input_tokens=10,
            output_tokens=20,
            pydantic_model='TestModel',
            llm_model='gpt-3.5-turbo'
        )
    
    @patch('src.ai_helper.OpenAIAdapter')
    def test_ask_method_with_tools(self, mock_adapter_class):
        """Test the ask method when tools are specified"""
        # Mock adapter
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        mock_adapter.process.return_value = {
            'response': TestModel(name="calc", value=4),
            'input_tokens': 15,
            'output_tokens': 25,
            'model_used': 'gpt-3.5-turbo'
        }
        
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        helper.add_tool("calculator", "A simple calculator", calculator)
        
        # Ask with tools
        result = helper.ask("Calculate 2+2", tools=["calculator"], model=TestModel)
        
        # Verify adapter was called with tools
        call_args = mock_adapter.process.call_args[0][0]
        self.assertIn('tools', call_args)
        self.assertIn('calculator', call_args['tools'])
        
        # Verify result
        self.assertIsInstance(result, TestModel)
    
    @patch('src.ai_helper.OpenAIAdapter')
    def test_ask_method_with_file(self, mock_adapter_class):
        """Test the ask method when a file is provided"""
        # Mock adapter
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        mock_adapter.process.return_value = {
            'response': TestModel(description="PDF content"),
            'input_tokens': 50,
            'output_tokens': 30,
            'model_used': 'gpt-3.5-turbo'
        }
        
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        
        # Ask with file
        result = helper.ask("Summarize this PDF", model=TestModel, file="test.pdf")
        
        # Verify adapter was called with file
        call_args = mock_adapter.process.call_args[0][0]
        self.assertEqual(call_args['file'], 'test.pdf')
        
        # Verify result
        self.assertIsInstance(result, TestModel)
        self.assertEqual(result.description, "PDF content")
    
    @patch('src.ai_helper.OpenAIAdapter')
    def test_ask_method_without_model(self, mock_adapter_class):
        """Test the ask method without specifying a Pydantic model"""
        # Mock adapter
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        mock_adapter.process.return_value = {
            'response': "Plain text response",
            'input_tokens': 10,
            'output_tokens': 20,
            'model_used': 'gpt-3.5-turbo'
        }
        
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        
        # Ask without model
        result = helper.ask("Test prompt")
        
        # Verify adapter was called without pydantic_model
        call_args = mock_adapter.process.call_args[0][0]
        self.assertIsNone(call_args['pydantic_model'])
        
        # Verify result is plain text
        self.assertEqual(result, "Plain text response")
    
    @patch('src.ai_helper.OpenAIAdapter')
    def test_ask_method_with_validation_errors(self, mock_adapter_class):
        """Test the ask method handles partial validation errors"""
        # Mock adapter returns invalid data
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        mock_adapter.process.return_value = {
            'response': {
                'name': 'test',
                'value': 'not_a_number',  # Invalid
                'description': 'Valid description',
                'extra_field': 'ignored'  # Extra field
            },
            'input_tokens': 10,
            'output_tokens': 20,
            'model_used': 'gpt-3.5-turbo',
            'fill_percentage': 66.67  # 2 out of 3 fields valid
        }
        
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        
        # Ask with model that has validation
        result = helper.ask("Test prompt", model=TestModel)
        
        # Verify result has valid fields and invalid ones are None
        self.assertIsInstance(result, TestModel)
        self.assertEqual(result.name, 'test')
        self.assertIsNone(result.value)  # Invalid field should be None
        self.assertEqual(result.description, 'Valid description')
    
    def test_execute_tool(self):
        """Test executing a tool"""
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        
        # Add mock tool
        mock_tool = MagicMock(return_value="Tool result")
        helper.add_tool("test_tool", "Test tool", mock_tool)
        
        # Execute tool
        result = helper._execute_tool("test_tool", {"arg1": "value1"})
        
        # Verify tool was called
        mock_tool.assert_called_once_with(arg1="value1")
        self.assertEqual(result, "Tool result")
    
    def test_execute_tool_not_found(self):
        """Test executing a non-existent tool"""
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        
        # Try to execute non-existent tool
        with self.assertRaises(ValueError) as context:
            helper._execute_tool("non_existent", {})
        
        self.assertIn("Tool 'non_existent' not found", str(context.exception))
    
    def test_track_cost_method(self):
        """Test that the track_cost method delegates to cost tracker"""
        helper = AIHelper('openai:gpt-3.5-turbo', self.cost_tracker)
        
        cost_details = {
            'model': 'gpt-3.5-turbo',
            'tokens': 100,
            'cost': 0.001
        }
        
        # Track cost
        helper._track_cost(cost_details)
        
        # Verify delegation
        self.cost_tracker.track_cost.assert_called_once_with(cost_details)
    
    @patch('src.ai_helper.OpenAIAdapter')
    def test_model_identifier_variations(self, mock_adapter_class):
        """Test that different model identifier formats work correctly"""
        test_cases = [
            ('openrouter:openai/gpt-3.5-turbo', 'openrouter', 'openai/gpt-3.5-turbo'),
            ('google:gemini-2.5-flash-preview-05-20', 'google', 'gemini-2.5-flash-preview-05-20'),
            ('anthropic:claude-3-opus-20240229', 'anthropic', 'claude-3-opus-20240229'),
            ('openai:gpt-4-vision-preview', 'openai', 'gpt-4-vision-preview')
        ]
        
        for model_id, expected_provider, expected_model in test_cases:
            helper = AIHelper(model_id, self.cost_tracker)
            self.assertEqual(helper.provider, expected_provider)
            self.assertEqual(helper.model, expected_model)
    
    def test_default_cost_tracker(self):
        """Test that a default CostTracker is created if not provided"""
        helper = AIHelper('openai:gpt-3.5-turbo')
        
        self.assertIsInstance(helper.cost_tracker, CostTracker)
        self.assertIsNotNone(helper.cost_tracker)

if __name__ == '__main__':
    unittest.main()
